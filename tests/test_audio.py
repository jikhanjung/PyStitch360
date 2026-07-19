"""호각 트랙/이벤트 검출(core.audio) — 합성 오디오."""
import numpy as np

from pystitch.core.audio import SR, whistle_events, whistle_track


def _tone(dur, freq, sr=SR, amp=1.0, vibrato_hz=0.0, vibrato_dev=0.0):
    t = np.arange(int(dur * sr)) / sr
    f = freq + (vibrato_dev * np.sin(2 * np.pi * vibrato_hz * t)
                if vibrato_hz else 0.0)
    return amp * np.sin(2 * np.pi * np.cumsum(f) / sr)


def _mix(total_s, parts, noise=0.15, seed=0):
    """parts = [(시작초, 파형), ...] 를 백색소음 위에 얹음."""
    rng = np.random.default_rng(seed)
    x = noise * rng.standard_normal(int(total_s * SR))
    for t0, w in parts:
        i = int(t0 * SR)
        x[i:i + len(w)] += w
    return x


def test_whistle_bursts_detected_with_timing():
    """4kHz 비브라토 호각 2회 → 이벤트 2개, 시각 일치."""
    x = _mix(30.0, [
        (5.0, _tone(0.6, 4000, amp=0.8, vibrato_hz=30, vibrato_dev=150)),
        (20.0, _tone(0.4, 4200, amp=0.6, vibrato_hz=30, vibrato_dev=150)),
    ])
    ev = whistle_events(whistle_track(x))
    assert len(ev) == 2
    assert abs(ev[0][0] - 5.0) < 0.2 and abs(ev[0][1] - 5.6) < 0.25
    assert abs(ev[1][0] - 20.0) < 0.2


def test_double_whistle_merges():
    """삑-삑 (0.1s 간격 짧은 두 번) → 한 구간으로 병합."""
    w = _tone(0.15, 4000, amp=0.8, vibrato_hz=30, vibrato_dev=150)
    x = _mix(10.0, [(3.0, w), (3.25, w)])
    ev = whistle_events(whistle_track(x))
    assert len(ev) == 1
    assert ev[0][0] < 3.1 and ev[0][1] > 3.3


def test_non_whistle_rejected():
    """1kHz 톤·광대역 잡음 버스트는 호각으로 안 잡힘."""
    rng = np.random.default_rng(1)
    burst = 1.5 * rng.standard_normal(int(0.5 * SR))     # 광대역 (박수/충격음)
    x = _mix(20.0, [
        (4.0, _tone(0.6, 1000, amp=1.0)),                # 저역 톤 (목소리대)
        (12.0, burst),
    ])
    assert whistle_events(whistle_track(x)) == []


def test_track_full_timeline_shape():
    """트랙은 전체 길이를 커버 (32ms 해상도) — 별도 보관용."""
    x = _mix(8.0, [])
    tr = whistle_track(x)
    n_expected = int(8.0 / tr["hop_s"])
    assert abs(len(tr["band_db"]) - n_expected) <= 3
    assert len(tr["tonal"]) == len(tr["band_db"])


def test_save_load_roundtrip(tmp_path):
    """트랙 저장/로드 라운드트립 (0.1 양자화 오차 이내)."""
    from pystitch.core.audio import load_whistle_track, save_whistle_track
    x = _mix(6.0, [(2.0, _tone(0.4, 4000, amp=0.8,
                               vibrato_hz=30, vibrato_dev=150))])
    tr = whistle_track(x)
    ev = whistle_events(tr)
    v = tmp_path / "clip.mp4"
    v.touch()
    save_whistle_track(v, tr, ev)
    tr2, ev2 = load_whistle_track(v)
    assert np.abs(tr2["band_db"] - tr["band_db"]).max() <= 0.06
    assert ev2 == [tuple(e) for e in ev] and len(ev2) == 1
