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
