"""core/match.py — 멀티캠 경기 컨테이너 (P07-1)."""
import json

from pystitch.core.match import (
    alt_coverage, load_match, match_from_sync_sidecars, save_match,
    to_alt_time, to_primary_time,
)


def test_clock_roundtrip():
    clock = {"offset": 1872.26, "drift": 1.000187}
    for t in (0.0, 100.0, 2269.7):
        assert abs(to_primary_time(clock, to_alt_time(clock, t)) - t) < 1e-9
    t0, t1 = alt_coverage(clock, 2269.77)
    assert abs(t0 - 1872.26) < 1e-9
    assert abs(t1 - (1872.26 + 1.000187 * 2269.77)) < 1e-6


def test_save_load_roundtrip_relative_paths(tmp_path):
    pano = tmp_path / "pano_1.mp4"
    alt = tmp_path / "C0011.MP4"
    pano.write_bytes(b"x")
    alt.write_bytes(b"x")
    doc = {"title": "테스트 경기",
           "halves": [{"label": "전반", "primary": str(pano),
                       "alts": [{"video": str(alt),
                                 "clock": {"offset": 10.5, "drift": 1.0},
                                 "stage": "whistle"}]}]}
    mp = tmp_path / "game.match.json"
    save_match(mp, doc)
    raw = json.loads(mp.read_text())
    assert raw["halves"][0]["primary"] == "pano_1.mp4"     # 상대 경로 저장
    d = load_match(mp)
    assert d["halves"][0]["primary"] == str(pano)          # 절대 경로 복원
    assert d["halves"][0]["alts"][0]["video"] == str(alt)
    assert d["halves"][0]["alts"][0]["clock"]["drift"] == 1.0


def test_load_rejects_bad_docs(tmp_path):
    mp = tmp_path / "bad.match.json"
    mp.write_text(json.dumps({"version": 1, "halves": []}))
    try:
        load_match(mp)
        raise AssertionError("halves 없음이 통과됨")
    except ValueError:
        pass
    mp.write_text(json.dumps(
        {"version": 1,
         "halves": [{"label": "전반", "primary": "a.mp4",
                     "alts": [{"video": "b.mp4", "clock": {}}]}]}))
    try:
        load_match(mp)
        raise AssertionError("clock.offset 없음이 통과됨")
    except ValueError:
        pass


def test_from_sync_sidecars(tmp_path):
    pano = tmp_path / "pano_1.mp4"
    alt = tmp_path / "C0011.MP4"
    pano.write_bytes(b"x")
    alt.write_bytes(b"x")
    (tmp_path / "pano_1.events.json").write_text(json.dumps(
        {"sync": {"other": str(alt), "offset": 1872.26,
                  "drift": 1.000187, "stage": "whistle"}}))
    pano2 = tmp_path / "pano_2.mp4"          # sync 사이드카 없음
    pano2.write_bytes(b"x")
    d = match_from_sync_sidecars([pano, pano2], title="t")
    assert d["halves"][0]["label"] == "전반"
    assert d["halves"][0]["alts"][0]["video"] == str(alt)
    assert d["halves"][1]["alts"] == []


def test_relative_clock():
    from pystitch.core.match import (
        half_cameras, relative_clock, to_alt_time, to_primary_time,
    )
    half = {"primary": "p.mp4",
            "alts": [{"video": "a.MP4",
                      "clock": {"offset": 1872.26, "drift": 1.000187}},
                     {"video": "b.MP4",
                      "clock": {"offset": 191.04, "drift": 1.0000676}}]}
    cams = half_cameras(half)
    # 항등: primary 기준 primary
    r = relative_clock(cams, 0, 0)
    assert abs(r["offset"]) < 1e-9 and abs(r["drift"] - 1.0) < 1e-12
    # primary 기준 alt = 저장 규약 그대로
    r = relative_clock(cams, 0, 1)
    assert abs(r["offset"] - 1872.26) < 1e-9
    # 왕복 일관성: a 기준 b 로 변환한 시각을 primary 로 되돌리면 동일
    for t_b in (0.0, 500.0, 1900.0):
        t_p = to_primary_time(cams[2]["clock"], t_b)
        t_a = to_alt_time(cams[1]["clock"], t_p)
        r_ab = relative_clock(cams, 1, 2)
        assert abs(to_primary_time(r_ab, t_b) - t_a) < 1e-6
