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


def classify_referees(analysis, teams, calib, min_det=40):
    """비팀 트랙릿의 위치 통계로 심판 역할 제안 (주심=5, 선심=6).

    주심: 필드 내부 상주 + 넓은 활동 범위 (경기를 따라다님).
    선심(부심): 사이드라인 부근 상주(경계 ±3.5m, Y 요동 작음) +
    터치라인 방향으로 이동. 근측/원측 각각. 한 사람이 여러 트랙릿으로
    갈라져도 각 조각이 조건을 만족하면 모두 잡힌다.

    반환: (suggest {tid: 5}, info {main, ar_near, ar_far}).
    사용자 roles 오버라이드는 호출부에서 우선 적용할 것.
    """
    hw = calib["width"] / 2.0
    hl = calib["length"] / 2.0
    team_ids = {t for t, r in teams.items() if r in (0, 1, 3, 4)}
    pts: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        feet = [(p[0], p[1] + p[3] / 2.0, int(p[4]))
                for p in prow if len(p) >= 5 and p[4] >= 0
                and int(p[4]) not in team_ids]
        if not feet:
            continue
        fxy = pano_to_field(calib, [(a, b) for a, b, _ in feet])
        for (gx, gy), (_, _, tid) in zip(fxy, feet):
            if np.isfinite(gx):
                pts.setdefault(tid, []).append((gx, gy))
    suggest = {}
    info = {"main": [], "ar_near": [], "ar_far": []}
    for tid, ps in pts.items():
        if len(ps) < min_det:
            continue
        P = np.asarray(ps)
        x, y = P[:, 0], P[:, 1]
        my = float(np.median(y))
        y_iqr = float(np.percentile(y, 80) - np.percentile(y, 20))
        x_span = float(np.percentile(x, 95) - np.percentile(x, 5))
        d_side = min(abs(my - hw), abs(my + hw))
        inside = float(np.mean((np.abs(x) < hl + 2) & (np.abs(y) < hw - 2)))
        if d_side <= 3.5 and y_iqr <= 3.0 and x_span >= 8.0:
            suggest[tid] = 6                      # 선심 (AR)
            info["ar_near" if my < 0 else "ar_far"].append(tid)
        elif inside >= 0.75 and x_span >= 15.0 and y_iqr >= 6.0:
            suggest[tid] = 5                      # 주심
            info["main"].append(tid)
    return suggest, info


# ------------------------------------------------------------- 선심 포즈
# COCO 키포인트: 5/6 어깨, 9/10 손목, 11/12 엉덩이

def arm_pose_scores(kpts, kconf=None, min_conf=0.3):
    """포즈 키포인트 (17,2) → (올림, 뻗음) 점수 — 기 신호 원자료.

    올림(raise) = max(어깨y − 손목y) / 몸통 — 1 근처면 손목이 머리 위.
    뻗음(extend) = 그 팔의 |손목x − 어깨x| / 몸통 — 수평 지시(오프사이드
    방향 지시)면 1 근처, 팔을 몸에 붙이면 0 근처.
    두 점수는 같은 팔(더 올라간 쪽) 기준. 계산 불가면 (NaN, NaN).
    """
    k = np.asarray(kpts, dtype=np.float64)
    if k.shape[0] < 13:
        return float("nan"), float("nan")
    c = (np.asarray(kconf, dtype=np.float64) if kconf is not None
         else np.ones(len(k)))
    sho = [i for i in (5, 6) if c[i] >= min_conf]
    hip = [i for i in (11, 12) if c[i] >= min_conf]
    if not sho or not hip:
        return float("nan"), float("nan")
    sho_y = float(np.mean([k[i, 1] for i in sho]))
    torso = float(np.mean([k[i, 1] for i in hip])) - sho_y
    if torso <= 1e-6:
        return float("nan"), float("nan")
    best = (float("nan"), float("nan"))
    for s_i, w_i in ((5, 9), (6, 10)):
        if c[s_i] < min_conf or c[w_i] < min_conf:
            continue
        rise = (k[s_i, 1] - k[w_i, 1]) / torso
        ext = abs(k[w_i, 0] - k[s_i, 0]) / torso
        if not np.isfinite(best[0]) or rise > best[0]:
            best = (float(rise), float(ext))
    return best


def arm_raise_score(kpts, kconf=None, min_conf=0.3):
    """(호환) 팔 올림 점수만."""
    return arm_pose_scores(kpts, kconf, min_conf)[0]


def classify_flag_signal(track, raise_hi=0.75, ext_hi=0.6,
                         min_up_s=0.8, min_point_s=0.8):
    """선심 팔 시계열 [(t, raise, extend)] → 기 신호 분류.

    - "offside": 기를 올렸다가(올림 ≥ raise_hi 구간 ≥ min_up_s ×0.5)
      내려서 수평 지시 유지 (올림 ≈ 0, 뻗음 ≥ ext_hi 가 min_point_s 이상,
      올림 구간 이후).
    - "foul": 올림 ≥ raise_hi 지속 ≥ min_up_s (수평 전환 없이 흔듦 포함).
    - "none": 그 외.
    반환: (종류, {up_s, point_s, max_raise}).
    """
    if not track:
        return "none", {}
    t = np.array([p[0] for p in track], dtype=np.float64)
    r = np.array([p[1] for p in track], dtype=np.float64)
    e = np.array([p[2] if len(p) > 2 else np.nan for p in track],
                 dtype=np.float64)
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1
                                    else 0.1))
    dt = np.clip(dt, 0.0, 0.5)             # 샘플 결손 구간 과대 계상 방지
    up = np.isfinite(r) & (r >= raise_hi)
    point = (np.isfinite(r) & np.isfinite(e)
             & (np.abs(r) <= 0.4) & (e >= ext_hi))
    up_s = float(dt[up].sum())
    max_raise = float(np.nanmax(r)) if np.isfinite(r).any() else float("nan")
    detail = {"up_s": round(up_s, 2), "max_raise": round(max_raise, 2)}
    # 오프사이드: 올림 이후의 수평 지시 지속
    if up.any():
        t_up_end = t[up][-1]
        pt_after = point & (t >= t[up][0])
        point_s = float(dt[pt_after].sum())
        detail["point_s"] = round(point_s, 2)
        if up_s >= min_up_s * 0.5 and point_s >= min_point_s:
            return "offside", detail
        if up_s >= min_up_s:
            return "foul", detail
    return "none", detail


def linesman_arm_track(pano_path, analysis, ar_tids, t0, t1,
                       weights="yolo11n-pose.pt", pad=2.2, log=print):
    """[t0, t1] 창에서 선심(ar_tids) 주변 포즈 추정 → 팔 올림 시계열.

    각 분석 샘플에서 선심 검출 박스 중심의 정사각 크롭(키 ×pad,
    네이티브)을 포즈 모델에 넣고, 크롭 중심에 가장 가까운 사람의
    (올림, 뻗음) 점수를 기록. 반환: [(t, raise, extend), ...].
    classify_flag_signal 로 파울(들고 흔듦)/오프사이드(들었다 수평
    지시)를 구분한다. 원측 선심은 해상도가 낮아 근측 우선.
    """
    import cv2 as _cv2

    from ultralytics import YOLO
    model = YOLO(str(weights))
    frames = np.asarray(analysis["frames"])
    fps = analysis["fps"]
    sis = np.where((frames / fps >= t0) & (frames / fps <= t1))[0]
    cap = _cv2.VideoCapture(str(pano_path))
    out = []
    pos = -10 ** 9
    for si in sis:
        row = next((p for p in analysis["players"][si]
                    if len(p) >= 5 and int(p[4]) in ar_tids), None)
        if row is None:
            continue
        F = int(frames[si])
        if 0 <= F - pos <= 90:
            for _ in range(F - pos - 1):
                cap.grab()
        else:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, F)
        ok, frame = cap.read()
        pos = F
        if not ok:
            continue
        H, W = frame.shape[:2]
        half = int(np.clip(row[3] * pad / 2, 64, 500))
        x0 = int(np.clip(row[0] - half, 0, max(W - 2 * half, 0)))
        y0 = int(np.clip(row[1] - half, 0, max(H - 2 * half, 0)))
        crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
        r = model.predict(crop, imgsz=max(192, (2 * half) // 32 * 32),
                          conf=0.25, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        # 크롭 중심에 가장 가까운 사람
        centers = r.boxes.xywh[:, :2].cpu().numpy()
        j = int(np.argmin(((centers - half) ** 2).sum(axis=1)))
        kx = r.keypoints.xy[j].cpu().numpy()
        kc = (r.keypoints.conf[j].cpu().numpy()
              if r.keypoints.conf is not None else None)
        rise, ext = arm_pose_scores(kx, kc)
        if np.isfinite(rise):
            out.append((round(float(frames[si] / fps), 2),
                        round(rise, 3),
                        round(ext, 3) if np.isfinite(ext) else None))
    cap.release()
    return out


def events_json_path(video_path) -> Path:
    return Path(video_path).with_suffix(".events.json")


def save_events(video_path, kickoffs=None, **extra):
    """이벤트 문서 갱신 — 기존 키 보존, 주어진 키만 교체."""
    p = events_json_path(video_path)
    doc = json.loads(p.read_text()) if p.exists() else {}
    if kickoffs is not None:
        doc["kickoffs"] = [{"t": round(t, 2), "score": round(s, 3), **d}
                           for t, s, d in kickoffs]
    doc.update(extra)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(p)
    return p


def load_events(video_path):
    p = events_json_path(video_path)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("kickoffs", [])


def load_events_doc(video_path) -> dict:
    """이벤트 문서 전체 (kickoffs, airborne, linesman_signals, highlights…)."""
    p = events_json_path(video_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())
