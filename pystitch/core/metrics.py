"""경기 지표 (P08-1): 공-선수 근접 소유 추정 → 점유율.

원칙 (P08): 관측된 샘플에서만 계산하고 커버리지를 항상 병기 —
공백을 보간으로 창작하지 않는다. 입력은 필드 좌표 (m), 호출부가
캘리브레이션으로 변환해 넘긴다.
"""
from __future__ import annotations

import numpy as np

#: 소유 판정 반경 (m) — 공과 최근접 선수 거리
POSSESS_DIST = 2.0
#: 경합 마진 (m) — 상대 팀 최근접이 이 차이 안이면 "경합"
CONTEST_MARGIN = 1.0


def possession_samples(ball_xy, players_xy, players_team,
                       max_dist=POSSESS_DIST, margin=CONTEST_MARGIN):
    """샘플 하나의 소유 판정.

    ball_xy: (2,) 또는 None/NaN, players_xy: (M,2), players_team: (M,)
    (0/1, 기타는 음수). 반환 (state, tid_idx):
    state ∈ {"unobserved", "loose", "contested", 0, 1} — 정수는 팀.
    """
    if ball_xy is None or not np.all(np.isfinite(ball_xy)):
        return "unobserved", None
    p = np.asarray(players_xy, float).reshape(-1, 2)
    team = np.asarray(players_team)
    ok = np.isfinite(p).all(1) & (team >= 0)
    if ok.sum() == 0:
        return "loose", None
    d = np.linalg.norm(p[ok] - np.asarray(ball_xy, float)[None], axis=1)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return "loose", None
    own = int(team[ok][i])
    other = d[(team[ok] != own)]
    if len(other) and float(np.min(other)) - float(d[i]) < margin:
        return "contested", int(np.flatnonzero(ok)[i])
    return own, int(np.flatnonzero(ok)[i])


def possession_spans(t, states, tids, min_dur=0.5, max_gap=1.0):
    """샘플열 → 소유 구간 [{"t0","t1","team","tid"}].

    같은 팀이 이어지는 런 (선수 교체는 팀 유지로 연속, tid 는 최빈값).
    "contested"/"loose" 는 런을 끊지 않되 max_gap 초과면 끊는다.
    min_dur 미만 런은 버린다 (스침).
    """
    spans = []
    cur = None
    for i, (ti, st) in enumerate(zip(t, states)):
        if isinstance(st, int):
            if cur is not None and cur["team"] == st \
                    and ti - cur["t1"] <= max_gap:
                cur["t1"] = ti
                cur["tids"].append(tids[i])
            else:
                if cur is not None:
                    spans.append(cur)
                cur = {"t0": ti, "t1": ti, "team": st, "tids": [tids[i]]}
        elif st == "unobserved" and cur is not None \
                and ti - cur["t1"] > max_gap:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    out = []
    for s in spans:
        if s["t1"] - s["t0"] < min_dur:
            continue
        vals, cnt = np.unique([x for x in s["tids"] if x is not None],
                              return_counts=True)
        out.append({"t0": float(s["t0"]), "t1": float(s["t1"]),
                    "team": s["team"],
                    "tid": int(vals[np.argmax(cnt)]) if len(vals) else None})
    return out


def possession_summary(t, states, pauses=None):
    """점유율 + 신뢰도 병기.

    반환 {"team0_s", "team1_s", "contested_s", "loose_s",
          "unobserved_s", "share0", "share1", "coverage"}.
    share 는 팀 배정 시간만의 비 (경합/미관측 제외 — 창작 금지 원칙).
    pauses: [(t0, t1)] 중단 구간은 전체에서 제외.
    """
    t = np.asarray(t, float)
    if len(t) < 2:
        return None
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])
    dt = np.clip(dt, 0.0, np.median(dt) * 3)   # 하프 경계 점프 가드
    in_play = np.ones(len(t), bool)
    for p0, p1 in pauses or []:
        in_play &= ~((t >= p0) & (t <= p1))
    acc = {"team0_s": 0.0, "team1_s": 0.0, "contested_s": 0.0,
           "loose_s": 0.0, "unobserved_s": 0.0}
    for st, d, ip in zip(states, dt, in_play):
        if not ip:
            continue
        if st == 0:
            acc["team0_s"] += d
        elif st == 1:
            acc["team1_s"] += d
        elif st == "contested":
            acc["contested_s"] += d
        elif st == "loose":
            acc["loose_s"] += d
        else:
            acc["unobserved_s"] += d
    total = sum(acc.values())
    owned = acc["team0_s"] + acc["team1_s"]
    acc["share0"] = acc["team0_s"] / owned if owned > 0 else float("nan")
    acc["share1"] = acc["team1_s"] / owned if owned > 0 else float("nan")
    acc["coverage"] = (total - acc["unobserved_s"]) / total if total else 0.0
    return {k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in acc.items()}
