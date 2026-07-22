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


def test_match_metrics_end_to_end():
    """합성 analysis + 항등 캘리브레이션으로 지표 일괄 산출."""
    from pystitch.core.metrics import match_metrics

    class _IdCalib:                       # pano_to_field 대체 불가 —
        pass                              # 실제 calib 인터페이스 필요

    # pano_to_field 를 흉내내기 위해 실제 fit 대신 monkeypatch
    import pystitch.core.metrics as M
    orig = None
    try:
        from pystitch.core import field as F
        orig = F.pano_to_field
        F.pano_to_field = lambda c, pts: np.asarray(pts, float)
        ana = {"frames": list(range(0, 300, 3)), "fps": 30.0,
               "balls": [], "players": []}
        for i, f in enumerate(ana["frames"]):
            t = f / 30.0
            if t < 5:                     # 팀0 #7 소유
                ana["balls"].append([10.0, 5.0, 1.0])
            else:                         # 팀1 #22 소유 (턴오버)
                ana["balls"].append([-20.0, -8.0, 1.0])
            ana["players"].append([[10.0, 5.0, 2.0, 2.0, 7],
                                   [-20.0, -8.0, 2.0, 2.0, 22]])
        roles = {7: 0, 22: 1}
        r = match_metrics(ana, object(), roles.get, lambda t: t)
        assert r is not None
        assert len(r["spans"]) == 2
        assert len(r["turnovers"]) == 1 and not r["passes"]
        assert abs(r["summary"]["share0"] - 0.5) < 0.05
        assert match_metrics(ana, None, roles.get, lambda t: t) is None
    finally:
        if orig is not None:
            F.pano_to_field = orig


def test_render_passmap():
    from pystitch.core.metrics import render_passmap
    passes = [{"from_tid": 7, "to_tid": 9, "team": 0, "t": 1.0}] * 4
    img = render_passmap(passes, {7: (-20.0, 5.0), 9: (15.0, -10.0)},
                         numbers={7: "7", 9: "10"}, title="팀1")
    assert img.shape[0] > 400 and img.shape[2] == 3
    assert img.mean() > 40                # 필드+마킹이 그려짐


def test_write_match_report(tmp_path):
    from pystitch.core.metrics import write_match_report
    m = {"summary": {"share0": 0.6, "share1": 0.4, "team0_s": 900.0,
                     "team1_s": 600.0, "contested_s": 100.0,
                     "loose_s": 200.0, "unobserved_s": 300.0,
                     "coverage": 0.85},
         "passes": [{"from_tid": 7, "to_tid": 9, "team": 0, "t": 1.0}],
         "turnovers": [], "unobserved_transitions": 2, "spans": []}
    files = write_match_report(tmp_path / "r", m, ("홈", "원정"),
                               dist_rows=[("홈", "7", 5400.0, 1.4, 7.2, 0.9)])
    md = (tmp_path / "r" / "match.md").read_text(encoding="utf-8")
    assert "60%" in md and "미관측 전이 2건" in md and "5400" in md
    assert files[0].endswith("match.md")


def test_analysis_summary_cache(tmp_path):
    """파생 요약 캐시: 저장·재사용·분석 변경 시 무효화."""
    import json
    from pystitch.core.ptz import analysis_summary
    ana = {"frames": [0, 3, 6], "fps": 30.0, "balls": [None] * 3,
           "players": [[[10, 20, 4, 8, 7, 45.0, 100.0, 200.0]],
                       [[12, 21, 4, 8, 7, 45.0, 100.0, 200.0]], []]}
    ap = tmp_path / "p.analysis.json"
    ap.write_text(json.dumps(ana))
    s1 = analysis_summary(ap, ana)
    assert s1["spans"][7] == [0, 3, 2] and 7 in s1["colors"]
    cp = tmp_path / "p.analysis.cache.json"
    assert cp.exists()
    # 캐시 적중: 파일 내용을 바꿔치기해도 키 동일하면 캐시 반환
    doc = json.loads(cp.read_text())
    doc["spans"]["7"] = [0, 3, 99]
    cp.write_text(json.dumps(doc))
    assert analysis_summary(ap, ana)["spans"][7][2] == 99
    # 분석 파일 갱신(mtime/size 변경) → 재계산
    import os, time
    ana2 = dict(ana)
    ap.write_text(json.dumps(ana2) + " ")
    os.utime(ap, (time.time() + 5, time.time() + 5))
    assert analysis_summary(ap, ana2)["spans"][7][2] == 2


def test_match_metrics_t_range(tmp_path):
    """구간 한정: 앞 경기 혼입 컷 (pano_5316 사례)."""
    import pystitch.core.metrics as M
    from pystitch.core import field as F
    orig = F.pano_to_field
    F.pano_to_field = lambda c, pts: np.asarray(pts, float)
    try:
        ana = {"frames": list(range(0, 600, 3)), "fps": 30.0,
               "balls": [], "players": []}
        for f in ana["frames"]:
            t = f / 30.0
            if t < 10:                    # "앞 경기": 팀0 만 소유
                ana["balls"].append([10.0, 5.0, 1.0])
            else:                         # "본 경기": 팀1 만 소유
                ana["balls"].append([-20.0, -8.0, 1.0])
            ana["players"].append([[10.0, 5.0, 2.0, 2.0, 7],
                                   [-20.0, -8.0, 2.0, 2.0, 22]])
        roles = {7: 0, 22: 1}
        full = M.match_metrics(ana, object(), roles.get, lambda t: t)
        cut = M.match_metrics(ana, object(), roles.get, lambda t: t,
                              t_range=(10.0, 20.0))
        assert full["summary"]["team0_s"] > 0
        assert cut["summary"]["team0_s"] == 0        # 앞 경기 제외됨
        assert cut["summary"]["share1"] == 1.0
        assert cut["n_samples"] < full["n_samples"]
    finally:
        F.pano_to_field = orig


def test_link_cache_roundtrip(tmp_path):
    import json
    import numpy as np
    from pystitch.core.ptz import link_ball_tracks, link_ball_tracks_cached
    ana = {"frames": list(range(0, 300, 3)), "fps": 30.0,
           "pano_w": 2000, "pano_h": 800,
           "balls": [[100.0 + i * 5, 400.0, 0.9] for i in range(100)],
           "ball_cands": [[[100.0 + i * 5, 400.0, 0.9]] for i in range(100)],
           "players": [[] for _ in range(100)]}
    ap = tmp_path / "p.analysis.json"
    ap.write_text(json.dumps(ana))
    fresh = link_ball_tracks(ana)
    l1 = link_ball_tracks_cached(ap, ana)
    assert (tmp_path / "p.analysis.link.cache.npz").exists()
    l2 = link_ball_tracks_cached(ap, ana)     # 캐시 적중
    assert len(l1["tracks"]) == len(fresh["tracks"]) == len(l2["tracks"])
    for a, b in zip(l1["tracks"], l2["tracks"]):
        assert np.allclose(a["pts"], b["pts"])
        assert np.array_equal(a["i"], b["i"])
    assert np.array_equal(l1["idx"], l2["idx"])


def test_build_plan_excludes_hidden_players():
    """숨긴 선수(관중·오인식)는 공 부재 구간 크롭 목표에서 제외."""
    import math
    from pystitch.core.ptz import build_plan, link_ball_tracks
    n = 300
    ana = {"frames": list(range(0, n * 3, 3)), "fps": 30.0,
           "pano_w": 4000, "pano_h": 1200, "total_frames": 900,
           "balls": [None] * n, "ball_cands": [[] for _ in range(n)],
           "players": []}
    for i in range(n):
        # 진짜 선수들 왼쪽(700 부근), 오인식 관중 무리 오른쪽(3500)
        # 관중 무리(4)가 다수라 중앙값 트리밍이 관중 쪽을 남긴다 —
        # 숨김 제외가 아니면 크롭이 관중을 추종하는 상황 재현
        ana["players"].append(
            [[700.0 + 30 * math.sin(i / 9.0), 600.0, 30.0, 90.0, 1],
             [760.0, 640.0, 30.0, 90.0, 2],
             [640.0, 620.0, 30.0, 90.0, 3],
             [3500.0, 900.0, 30.0, 90.0, 901],
             [3550.0, 880.0, 30.0, 90.0, 902],
             [3600.0, 870.0, 30.0, 90.0, 904],
             [3450.0, 910.0, 30.0, 90.0, 903]])
    linked = link_ball_tracks(ana)
    p0 = build_plan(ana, 4000, 1200, linked=linked, log=None)
    p1 = build_plan(ana, 4000, 1200, linked=linked,
                    exclude_tids={901, 902, 903, 904}, log=None)
    f = 450
    assert p0["cx"][f] > p1["cx"][f] + 300, (p0["cx"][f], p1["cx"][f])
    assert p1["cx"][f] < 1500                 # 진짜 선수 쪽으로
