"""core/sync_multi.py — 멀티캠 동기화 (P06-1) 합성 검증.

완료 기준: 알려진 오프셋+드리프트 주입 → 복원 오차 < 1프레임(33ms),
경기 전 구간에서. 호각만으로는 음속 지터 때문에 안 되고 공 궤적
정밀화까지 거쳐야 달성되는 구조를 그대로 검증한다.
"""
import numpy as np

from pystitch.core.sync_multi import (
    TRANSFORMS, refine_clock_by_ball, refine_sync_by_ball, sync_by_whistles,
)

FPS = 30.0
TRUE_OFFSET = 127.43            # B 가 A 보다 늦게 시작
TRUE_DRIFT = 1.00004            # 40ppm — 30분에 72ms 누적


def _t_a(t_b):
    return TRUE_OFFSET + TRUE_DRIFT * t_b


def make_whistles(rng, n=18, dur=1800.0):
    """A 기준 호각 + B 기준(시계 역변환 + 음속 지터 ±0.3s)."""
    ta = np.sort(rng.uniform(30, dur - 30, n))
    ev_a = [(float(t), float(t + 0.5), 25.0) for t in ta]
    ev_b = []
    for t in ta:
        tb = (t - TRUE_OFFSET) / TRUE_DRIFT
        tb += rng.uniform(-0.3, 0.3)          # 심판 위치별 음속 도달차
        ev_b.append((float(tb), float(tb + 0.5), 25.0))
    return ev_a, ev_b


def make_ball(rng, dur=1800.0, hz=10.0, transform=0):
    """부드러운 합성 공 궤적 (A 시간축) + B 관측 (변환·노이즈·결손)."""
    t = np.arange(0, dur, 1.0 / hz)
    x = 40 * np.sin(2 * np.pi * t / 47.0) + 10 * np.sin(2 * np.pi * t / 7.3)
    y = 25 * np.sin(2 * np.pi * t / 31.0) + 8 * np.cos(2 * np.pi * t / 5.1)
    xy = np.stack([x, y], axis=1)
    # B: 자기 시간축 샘플 → 참 위치는 A 시간으로 환산해 평가
    t_b = np.arange(1.0, dur - 1.0, 1.0 / hz)
    xa_at_b = np.stack([np.interp(_t_a(t_b), t, x),
                        np.interp(_t_a(t_b), t, y)], axis=1)
    xy_b = TRANSFORMS[transform](xa_at_b) + rng.normal(0, 0.3, (len(t_b), 2))
    drop = rng.random(len(t_b)) < 0.25         # 25% 미검출
    xy_b[drop] = np.nan
    return t, xy, t_b, xy_b


def test_whistle_coarse_sync():
    rng = np.random.default_rng(11)
    ev_a, ev_b = make_whistles(rng)
    # 오검출 섞기: A 에만 있는 이벤트 3개, B 에만 있는 이벤트 3개
    ev_a += [(float(t), float(t + 0.3), 22.0)
             for t in rng.uniform(0, 1800, 3)]
    ev_b += [(float(t), float(t + 0.3), 22.0)
             for t in rng.uniform(0, 1700, 3)]
    r = sync_by_whistles(ev_a, ev_b)
    assert r is not None and r["n"] >= 12
    assert abs(r["offset"] - TRUE_OFFSET) < 0.25          # 거친 단계 목표
    assert abs(r["drift"] - TRUE_DRIFT) < 5e-4
    # 음속 지터 때문에 프레임 정밀은 기대하지 않는다 (정밀화가 담당)


def test_whistle_sync_large_offset_with_decoy():
    """참 오프셋이 옛 탐색 창(1800s) 밖 + 미끼 우연 피크 → 정답 복원.

    20241020 실경기 회귀: pano_5316↔C0011 참 오프셋 1871s 가
    max_offset_s=1800 밖이라, 워밍업 잡음 호각이 만든 창 안의 우연
    피크(1383.7s)가 뽑혔다. 상위 K 피크 피팅 선택이 이를 막는다.
    """
    rng = np.random.default_rng(7)
    big = 1871.3                   # A 가 B 보다 훨씬 먼저 시작 (창 밖)
    tb = np.sort(rng.uniform(30, 2200, 20))
    ev_b = [(float(t), float(t + 0.5), 25.0) for t in tb]
    ev_a = [(float(big + 1.00004 * t + rng.uniform(-0.3, 0.3)),
             0.0, 25.0) for t in tb]
    ev_a = [(t0, t0 + 0.5, db) for t0, _z, db in ev_a]
    # 미끼: 좁은 오프셋(487s 차이 흉내)에서 우연히 겹치는 쌍 5개
    decoy = big - 487.0
    td = np.sort(rng.uniform(100, 1500, 5))
    ev_a += [(float(decoy + t), float(decoy + t + 0.3), 20.0) for t in td]
    ev_b += [(float(t), float(t + 0.3), 20.0) for t in td]
    r = sync_by_whistles(ev_a, ev_b)
    assert r is not None
    assert abs(r["offset"] - big) < 0.3, r["offset"]
    assert r["n"] >= 15


def test_ball_refine_reaches_frame_accuracy():
    rng = np.random.default_rng(7)
    ev_a, ev_b = make_whistles(rng)
    coarse = sync_by_whistles(ev_a, ev_b)
    t_a, xy_a, t_b, xy_b = make_ball(rng)
    r = refine_clock_by_ball(t_a, xy_a, t_b, xy_b, coarse)
    assert r is not None and r["transform"] == 0
    ck = r["clock"]
    for tb in (0.0, 900.0, 1650.0):            # 경기 전 구간에서 <1프레임
        err = abs((ck["offset"] + ck["drift"] * tb) - _t_a(tb))
        assert err < 1.0 / FPS, f"t_b={tb}: {err * 1000:.1f}ms"


def test_symmetry_detection():
    rng = np.random.default_rng(3)
    ev_a, ev_b = make_whistles(rng)
    coarse = sync_by_whistles(ev_a, ev_b)
    for ti in range(4):                        # 반대편 카메라 = rot180 등
        t_a, xy_a, t_b, xy_b = make_ball(rng, dur=600.0, transform=ti)
        r = refine_sync_by_ball(t_a, xy_a, t_b, xy_b, coarse)
        assert r is not None
        assert r["transform"] == ti, f"want {ti}, got {r['transform']}"
        assert r["rms_m"] < 1.0                # 정합 후 궤적 거리


def test_to_other_time_roundtrip():
    from pystitch.core.sync_multi import to_other_time
    clock = {"offset": TRUE_OFFSET, "drift": TRUE_DRIFT}
    for tb in (0.0, 500.0, 1800.0):
        assert abs(to_other_time(clock, _t_a(tb)) - tb) < 1e-9


def test_cut_synced_clip(tmp_path):
    """동기화 클립 컷: 프레임 번호를 화소에 새겨 시작점·길이 검증."""
    import shutil

    import cv2
    import pytest

    from pystitch.core.sync_multi import cut_synced_clip
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg 없음")
    src = tmp_path / "other.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (320, 240))
    for f in range(300):                       # 10초, 밝기 = 프레임 번호
        vw.write(np.full((240, 320, 3), min(f, 255), np.uint8))
    vw.release()
    clock = {"offset": 100.0, "drift": 1.0}    # A 시각 = B 시각 + 100
    out = tmp_path / "clip.mp4"
    cut_synced_clip(src, clock, 103.0, 105.0, out, crf=28)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok and abs(n - 60) <= 2             # 2초 분량
    assert abs(float(first.mean()) - 90) <= 3  # B 시각 3.0s = 프레임 90


def test_whistle_sync_insufficient_events():
    ev = [(10.0, 10.5, 25.0), (20.0, 20.5, 25.0)]
    assert sync_by_whistles(ev, ev) is None


def test_ball_refine_no_overlap():
    rng = np.random.default_rng(5)
    t_a = np.arange(0, 100, 0.1)
    xy_a = np.stack([np.sin(t_a), np.cos(t_a)], axis=1)
    xy_b = np.full((100, 2), np.nan)           # B 는 전부 미검출
    r = refine_sync_by_ball(t_a, xy_a, np.arange(0, 10, 0.1), xy_b,
                            {"offset": 0.0, "drift": 1.0})
    assert r is None
