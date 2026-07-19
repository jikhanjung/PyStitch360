"""core/highlights.py — 규칙 구간·병합·점수·상태 승계 (P03-1)."""
import numpy as np

from pystitch.core.highlights import (
    airborne_box_events, ball_speed_events, build_highlights, carry_states,
)


def _segs():
    return build_highlights(
        1800.0,
        kickoffs=[{"t": 100.0}],
        whistles=[(99.5, 100.6, 25.0),      # 킥오프 휘슬 → 휘슬 규칙 제외
                  (500.0, 501.0, 25.0),     # 롱 휘슬
                  (600.0, 600.3, 25.0),     # 짧은 휘슬 → 미적용
                  (700.0, 701.0, 15.0)],    # 약한 휘슬 → 미적용
        signals=[{"whistle_t": 500.0, "near": {"signal": "foul"}},
                 {"whistle_t": 900.0, "near": {"signal": "none"}}],
        air_events=[(1000.0, 1002.5)],
        speed_events=[(1001.0, 24.0)],      # 공중볼과 겹침 → 병합
        user_events=[(1500.0, "골")])


def test_rule_windows_and_merge():
    segs = _segs()
    assert len(segs) == 4
    assert (segs[0]["t0"], segs[0]["t1"]) == (90.0, 120.0)     # 킥오프
    assert segs[1]["score"] == 4.5                              # 파울+휘슬
    assert set(segs[1]["kinds"]) == {"foul", "whistle"}
    assert segs[1]["label"] == "파울"                           # 가중 우선
    assert set(segs[2]["kinds"]) == {"air", "speed"}
    assert (segs[2]["t0"], segs[2]["t1"]) == (996.0, 1008.5)
    assert segs[3]["label"] == "골" and segs[3]["score"] == 5.0


def test_clamp_to_duration():
    s = build_highlights(60.0, kickoffs=[{"t": 5.0}, {"t": 58.0}])
    assert s[0]["t0"] == 0.0 and s[-1]["t1"] == 60.0


def test_carry_states():
    segs = _segs()
    old = [dict(segs[0], state="accept", t0=92.0, t1=118.0),
           dict(segs[1], state="reject"),
           {"t0": 1700.0, "t1": 1712.0, "kinds": ["user"], "label": "수동",
            "score": 5.0, "state": "accept"}]
    new = build_highlights(1800.0, kickoffs=[{"t": 100.0}],
                           whistles=[(500.0, 501.0, 25.0)],
                           signals=[{"whistle_t": 500.0,
                                     "near": {"signal": "foul"}}])
    out = carry_states(new, old)
    assert out[0]["state"] == "accept"          # 상태 승계
    assert (out[0]["t0"], out[0]["t1"]) == (92.0, 118.0)   # 조정 경계 유지
    assert out[1]["state"] == "reject"
    assert any(h["t0"] == 1700.0 and h["state"] == "accept" for h in out)
    assert all(out[i]["t0"] <= out[i + 1]["t0"] for i in range(len(out) - 1))


def test_ball_speed_events():
    t = np.arange(0, 30, 0.1)
    g = np.zeros((len(t), 2))
    g[:, 0] = 3.0 * t                            # 3 m/s 기본
    g[100:106, 0] += np.cumsum(np.full(6, 2.2))  # 10s: +22 m/s 지속
    g[200, 0] += 2.5                             # 한 샘플 왕복 지터
    g[250, 0] += 10.0                            # 100 m/s 점프 (오인식)
    g[280:283, 1] = np.nan                       # 갭
    ev = ball_speed_events(t, g)
    assert len(ev) == 1                          # 지속 구간만
    assert abs(ev[0][0] - 10.0) < 0.2
    assert 20.0 <= ev[0][1] <= 30.0


def test_airborne_box_gate():
    segs = [
        (0, 1, {"p0": (10.0, 0.0), "v": (15.0, 0.0), "t0": 50.0, "T": 2.0}),
        (2, 3, {"p0": (0.0, 0.0), "v": (2.0, 0.0), "t0": 80.0, "T": 2.0}),
        (4, 5, {"p0": (30.0, 40.0), "v": (5.0, 3.0), "t0": 90.0, "T": 2.0}),
    ]
    assert airborne_box_events(segs, 105.0, 68.0) == [(50.0, 52.0)]
