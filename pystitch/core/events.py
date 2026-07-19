"""이벤트 엔진: 대형(포메이션) 트랙 + 호각 융합 → 킥오프 판정.

킥오프는 규칙이 공간 배치를 강제하는 순간(경기 규칙 8조: 전원 자기
진영, 상대는 센터서클 밖) — 학습 없이 기하 조건으로 읽는다.

  킥오프 = 호각 시각 t 에서
    (1) 직전 수 초간 두 팀이 반대 진영으로 분리 (분리도 지속)
    (2) 센터서클 안 인원 소수 (킥커만)
    (3) t 이후 짧은 시간 안에 대형 붕괴(하프라인 교차) 또는 공 센터 이탈

전반 시작·후반 시작·득점 후 재개가 모두 같은 패턴이므로, 경기 중간의
킥오프는 직전에 골이 있었다는 역추론도 가능하다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .field import CENTER_CIRCLE_R, pano_to_field


def formation_track(analysis, teams, calib, min_players=4):
    """샘플별 대형 지표: {t, sep, circle_n, ball_r, n0, n1}.

    sep: 두 팀이 반대 진영으로 갈린 정도 = min(팀별 자기 진영 비율),
         진영은 팀 X 중앙값의 부호 (전/후반 진영 교대에 무가정).
         같은 쪽이면 0. 팀 인원이 min_players 미만이면 NaN.
    circle_n: 센터서클(9.15m) 안 인원 (팀 무관, 미분류 포함).
    ball_r: 공의 센터 마크 거리 (m, 미검출 NaN).
    역할 3/4(GK)는 팀 0/1 로 접는다. id<0(갭필 주입)은 팀 미상 —
    circle_n 에만 기여.
    """
    fps = analysis["fps"]
    frames = np.asarray(analysis["frames"])
    n = len(frames)
    sep = np.full(n, np.nan)
    circle_n = np.zeros(n, dtype=np.int32)
    ball_r = np.full(n, np.nan)
    n0 = np.zeros(n, dtype=np.int32)
    n1 = np.zeros(n, dtype=np.int32)
    team_of = {int(t): (0 if r in (0, 3) else 1)
               for t, r in teams.items() if r in (0, 1, 3, 4)}
    for si in range(n):
        rows = analysis["players"][si]
        feet = [(p[0], p[1] + p[3] / 2.0,
                 int(p[4]) if len(p) >= 5 else -1)
                for p in rows if len(p) >= 4]
        if feet:
            fxy = pano_to_field(calib, [(a, b) for a, b, _ in feet])
            xs0, xs1 = [], []
            for (gx, gy), (_, _, tid) in zip(fxy, feet):
                if not np.isfinite(gx):
                    continue
                if np.hypot(gx, gy) <= CENTER_CIRCLE_R:
                    circle_n[si] += 1
                tm = team_of.get(tid)
                if tm == 0:
                    xs0.append(gx)
                elif tm == 1:
                    xs1.append(gx)
            n0[si], n1[si] = len(xs0), len(xs1)
            if len(xs0) >= min_players and len(xs1) >= min_players:
                s0 = np.sign(np.median(xs0)) or 1.0
                s1 = np.sign(np.median(xs1)) or 1.0
                if s0 == s1:
                    sep[si] = 0.0
                else:
                    f0 = np.mean(np.sign(xs0) == s0)
                    f1 = np.mean(np.sign(xs1) == s1)
                    sep[si] = float(min(f0, f1))
        bb = analysis["balls"][si] if si < len(analysis["balls"]) else None
        if bb is not None:
            g = pano_to_field(calib, [[bb[0], bb[1]]])[0]
            if np.isfinite(g[0]):
                ball_r[si] = float(np.hypot(g[0], g[1]))
    return {"t": frames / fps, "sep": sep, "circle_n": circle_n,
            "ball_r": ball_r, "n0": n0, "n1": n1}


def _window(track, t0, t1):
    m = (track["t"] >= t0) & (track["t"] <= t1)
    return m


def detect_kickoffs(track, whistles, min_db=15.0,
                    pre_s=12.0, post_s=20.0,
                    sep_hi=0.8, sep_lo=0.55, circle_max=4,
                    ball_leave_m=8.0, merge_s=30.0):
    """호각 이벤트 × 대형 트랙 → 킥오프 [(t, score, detail), ...].

    각 호각(피크 ≥ min_db)에서:
      pre  = [t-pre_s, t]: 분리도 중앙값 ≥ sep_hi 이고 서클 인원 중앙값
             ≤ circle_max (규칙이 강제하는 배치)
      post = [t, t+post_s]: 분리도 최소 ≤ sep_lo (대형 붕괴 = 경기 시작)
             또는 공 센터 거리 최대 ≥ ball_leave_m (공 이탈)
    둘 다 만족하면 킥오프. merge_s 내 중복은 최고 점수만.
    score = pre 분리도 중앙값 (0.8~1.0) — 신뢰도 지표.
    """
    out = []
    for t0_, t1_, db in whistles:
        if db < min_db:
            continue
        pre = _window(track, t0_ - pre_s, t0_ - 0.5)
        post = _window(track, t1_ + 0.5, t1_ + post_s)
        sep_pre = track["sep"][pre]
        sep_pre = sep_pre[np.isfinite(sep_pre)]
        if len(sep_pre) < 5:
            continue
        med_sep = float(np.median(sep_pre))
        med_circ = float(np.median(track["circle_n"][pre]))
        if med_sep < sep_hi or med_circ > circle_max:
            continue
        sep_post = track["sep"][post]
        sep_post = sep_post[np.isfinite(sep_post)]
        broke = len(sep_post) >= 3 and float(np.min(sep_post)) <= sep_lo
        br = track["ball_r"][post]
        br = br[np.isfinite(br)]
        ball_left = len(br) > 0 and float(np.max(br)) >= ball_leave_m
        if not (broke or ball_left):
            continue
        out.append((float(t0_), med_sep,
                    {"whistle": (t0_, t1_, db), "sep_pre": round(med_sep, 2),
                     "circle_pre": med_circ,
                     "broke": bool(broke), "ball_left": bool(ball_left),
                     "long_whistle": bool(t1_ - t0_ >= 0.8)}))
    # 병합: merge_s 내에서는 최고 점수(동점이면 롱 휘슬 우선)
    out.sort(key=lambda e: e[0])
    merged = []
    for e in out:
        if merged and e[0] - merged[-1][0] < merge_s:
            prev = merged[-1]
            key = (e[1], e[2]["long_whistle"])
            key_p = (prev[1], prev[2]["long_whistle"])
            if key > key_p:
                merged[-1] = e
        else:
            merged.append(e)
    return merged


def events_json_path(video_path) -> Path:
    return Path(video_path).with_suffix(".events.json")


def save_events(video_path, kickoffs):
    doc = {"kickoffs": [{"t": round(t, 2), "score": round(s, 3), **d}
                        for t, s, d in kickoffs]}
    p = events_json_path(video_path)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(p)
    return p


def load_events(video_path):
    p = events_json_path(video_path)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("kickoffs", [])
