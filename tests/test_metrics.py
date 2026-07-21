"""core/metrics.py — 소유/점유율 (P08-1) 합성 검증."""
import numpy as np

from pystitch.core.metrics import (
    possession_samples, possession_spans, possession_summary,
)


def _scripted():
    """10Hz, 30초: 0~10s 팀0 #7 소유, 10~12s 미관측(공 결측),
    12~20s 팀1 #22 소유, 20~22s 경합, 22~30s 팀0 #9 소유."""
    t = np.arange(0, 30, 0.1)
    states, tids = [], []
    p7, p22, p9 = (10.0, 5.0), (-20.0, -8.0), (30.0, 20.0)
    for ti in t:
        if ti < 10:
            ball, players, teams = p7, [p7, p22, p9], [0, 1, 0]
        elif ti < 12:
            ball, players, teams = (np.nan, np.nan), [p7, p22], [0, 1]
        elif ti < 20:
            ball, players, teams = p22, [p7, p22, p9], [0, 1, 0]
        elif ti < 22:                       # 경합: 두 팀 선수 다 1m 안
            ball = (0.0, 0.0)
            players, teams = [(0.3, 0.0), (-0.4, 0.1)], [0, 1]
        else:
            ball, players, teams = p9, [p7, p22, p9], [0, 1, 0]
        st, ti_idx = possession_samples(ball, players, teams)
        states.append(st)
        tids.append(ti_idx)
    return t, states, tids


def test_possession_states_and_spans():
    t, states, tids = _scripted()
    assert states[0] == 0 and states[110] == "unobserved"
    assert states[150] == 1 and states[210] == "contested"
    spans = possession_spans(t, states, tids)
    assert [s["team"] for s in spans] == [0, 1, 0]
    assert abs(spans[0]["t1"] - spans[0]["t0"] - 9.9) < 0.2
    assert abs(spans[1]["t0"] - 12.0) < 0.15


def test_possession_summary_shares():
    t, states, _ = _scripted()
    s = possession_summary(t, states)
    # 팀0: 10+8=18s, 팀1: 8s → share0 = 18/26
    assert abs(s["share0"] - 18.0 / 26.0) < 0.02, s
    assert abs(s["unobserved_s"] - 2.0) < 0.3
    assert abs(s["contested_s"] - 2.0) < 0.3
    assert 0.9 < s["coverage"] <= 1.0


def test_pause_excluded():
    t, states, _ = _scripted()
    s = possession_summary(t, states, pauses=[(0.0, 10.0)])
    # 팀0 첫 구간 제외 → 8 vs 8
    assert abs(s["share0"] - 0.5) < 0.03, s


def test_loose_and_empty():
    st, tid = possession_samples((0.0, 0.0), [(30.0, 0.0)], [0])
    assert st == "loose" and tid is None
    st, _ = possession_samples(None, [], [])
    assert st == "unobserved"
    assert possession_summary([0.0], ["loose"]) is None


def test_pass_extraction_and_kick_snap():
    from pystitch.core.metrics import extract_passes, kick_instant, pass_matrix
    # 소유: 팀0 #7 (0~10s) → 팀0 #9 (10.8~15) → 팀1 #22 (17~20, 턴오버)
    spans = [{"t0": 0.0, "t1": 10.0, "team": 0, "tid": 7},
             {"t0": 10.8, "t1": 15.0, "team": 0, "tid": 9},
             {"t0": 17.0, "t1": 20.0, "team": 1, "tid": 22}]
    # 공 궤적: 10.2s 에 방향 급변 (킥)
    t = np.arange(0, 21, 0.1)
    ball = np.zeros((len(t), 2))
    ball[:, 0] = np.where(t < 10.2, t * 1.0, 10.2 + (t - 10.2) * 8.0)
    ball[:, 1] = np.where(t < 10.2, 0.0, (t - 10.2) * 5.0)
    states = ["observed"] * len(t)
    r = extract_passes(spans, t, ball, states)
    assert len(r["passes"]) == 1 and len(r["turnovers"]) == 1
    assert abs(r["passes"][0]["t"] - 10.2) < 0.15   # 킥 스냅
    assert r["passes"][0]["from_tid"] == 7 and r["passes"][0]["to_tid"] == 9
    assert r["turnovers"][0]["to_team"] == 1
    mat = pass_matrix(r["passes"], numbers={7: "7", 9: "9"})
    assert mat[("7", "9")] == 1
    assert abs(kick_instant(t, ball, 10.0) - 10.2) < 0.15


def test_pass_unobserved_transition():
    from pystitch.core.metrics import extract_passes
    spans = [{"t0": 0.0, "t1": 5.0, "team": 0, "tid": 7},
             {"t0": 7.5, "t1": 9.0, "team": 0, "tid": 9}]
    t = np.arange(0, 10, 0.1)
    states = ["unobserved" if 5.0 < ti < 7.5 else "observed" for ti in t]
    r = extract_passes(spans, t, None, states)
    assert r["unobserved_transitions"] == 1 and not r["passes"]
    # 긴 공백 (>max_gap) 도 미관측 전이
    spans2 = [{"t0": 0.0, "t1": 5.0, "team": 0, "tid": 7},
              {"t0": 9.0, "t1": 10.0, "team": 1, "tid": 22}]
    r2 = extract_passes(spans2, t, None, ["observed"] * len(t))
    assert r2["unobserved_transitions"] == 1 and not r2["turnovers"]
