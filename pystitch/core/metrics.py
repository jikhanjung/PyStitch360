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


# ---------------------------------------------------------------- P08-2 패스
def kick_instant(t, ball_xy, t0, window=1.0):
    """t0 근처 공 속도/방향 급변 시점으로 스냅 (패스 시작 정밀화).

    소유 경계는 근접 판정이라 ±0.5s 무디다 — 궤적 가속도 피크가 킥.
    관측 부족이면 t0 그대로.
    """
    t = np.asarray(t, float)
    b = np.asarray(ball_xy, float).reshape(-1, 2)
    m = np.isfinite(b).all(1) & (np.abs(t - t0) <= window)
    if m.sum() < 4:
        return float(t0)
    ts, bs = t[m], b[m]
    v = np.diff(bs, axis=0) / np.clip(np.diff(ts), 1e-6, None)[:, None]
    a = np.linalg.norm(np.diff(v, axis=0), axis=1) \
        / np.clip(np.diff(ts)[1:], 1e-6, None)
    if not len(a):
        return float(t0)
    return float(ts[int(np.argmax(a)) + 1])


def extract_passes(spans, t=None, ball_xy=None, states=None,
                   max_gap=3.0, unobs_frac=0.5):
    """소유 구간열 → 패스/턴오버/미관측 전이 (P08-2).

    같은 팀 다른 선수 = 패스, 팀 변경 = 턴오버. 두 구간 사이가
    max_gap 초과이거나 미관측 비율이 크면 "미관측 전이" 로 분리 집계
    — 없는 패스를 만들지 않는다 (P08 원칙). t/ball_xy 가 있으면 킥
    시점을 궤적 급변으로 스냅.
    """
    passes, turnovers, unobserved = [], [], 0
    t_arr = None if t is None else np.asarray(t, float)
    for a, b in zip(spans, spans[1:]):
        gap = b["t0"] - a["t1"]
        blind = False
        if states is not None and t_arr is not None and gap > 0.2:
            m = (t_arr > a["t1"]) & (t_arr < b["t0"])
            if m.sum():
                un = sum(1 for s in np.asarray(states, object)[m]
                         if s == "unobserved")
                blind = un / m.sum() > unobs_frac
        if gap > max_gap or (blind and gap > 1.0):
            unobserved += 1
            continue
        tk = a["t1"]
        if t_arr is not None and ball_xy is not None:
            tk = kick_instant(t_arr, ball_xy, a["t1"])
        ev = {"t": round(float(tk), 2), "from_tid": a["tid"],
              "to_tid": b["tid"], "team": a["team"]}
        if b["team"] == a["team"]:
            if b["tid"] != a["tid"] and b["tid"] is not None:
                passes.append(ev)
        else:
            ev["to_team"] = b["team"]
            turnovers.append(ev)
    return {"passes": passes, "turnovers": turnovers,
            "unobserved_transitions": unobserved}


def pass_matrix(passes, numbers=None):
    """패스 목록 → {(from, to): count} — 패스맵 화살표 데이터.

    numbers: {tid: 등번호 문자열} 있으면 라벨 치환.
    """
    lab = (lambda tid: numbers.get(tid, str(tid))) if numbers \
        else (lambda tid: str(tid))
    out: dict = {}
    for p in passes:
        k = (lab(p["from_tid"]), lab(p["to_tid"]))
        out[k] = out.get(k, 0) + 1
    return out


def match_metrics(analysis, calib, role_of, rep_of, pauses=None):
    """분석+캘리브레이션 → 경기 지표 일괄 (P08-3 통계 화면 데이터).

    role_of/rep_of: PtzTab 관례 (역할 0/3=팀0, 1/4=팀1, 그 외 제외).
    반환 {"summary", "spans", "passes", "turnovers",
          "unobserved_transitions", "n_samples"} 또는 None (캘리브레이션
    없음 — 필드 좌표가 없으면 지표를 만들지 않는다).
    """
    if calib is None or analysis is None:
        return None
    from .field import pano_to_field
    frames = analysis["frames"]
    fps = float(analysis["fps"])
    t = np.asarray(frames, float) / fps
    ball_f = np.full((len(t), 2), np.nan)
    states, tids = [], []
    team_of_role = {0: 0, 3: 0, 1: 1, 4: 1}
    for si, (b, prow) in enumerate(zip(analysis["balls"],
                                       analysis["players"])):
        if b is not None:
            xy = pano_to_field(calib, [(b[0], b[1])])[0]
            if np.all(np.isfinite(xy)):
                ball_f[si] = xy
        pts, teams_, reps = [], [], []
        for p in prow:
            if len(p) < 5 or p[4] < 0:
                continue
            role = role_of(int(p[4]))
            if role not in team_of_role:
                continue
            fx = pano_to_field(calib, [(p[0], p[1] + p[3] / 2.0)])[0]
            if not np.all(np.isfinite(fx)):
                continue
            pts.append(fx)
            teams_.append(team_of_role[role])
            reps.append(rep_of(int(p[4])))
        st, idx = possession_samples(
            ball_f[si] if np.all(np.isfinite(ball_f[si])) else None,
            pts or np.zeros((0, 2)), teams_ or [])
        states.append(st)
        tids.append(reps[idx] if idx is not None and idx < len(reps)
                    else None)
    spans = possession_spans(t, states, tids)
    ev = extract_passes(spans, t, ball_f, states)
    return {"summary": possession_summary(t, states, pauses),
            "spans": spans, "n_samples": len(t), **ev}
