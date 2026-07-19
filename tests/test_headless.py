"""헤드리스 모드 — 좌/우 짝 맞추기(크기 기반)와 OCR 후보 게이트 폴백."""
from pathlib import Path

import numpy as np
import pytest

from pystitch.core.ocr import collect_ocr_candidates
from pystitch.core.pairing import pair_directories
from pystitch.headless import _default_out_dir


def _make_chain(d, vid_no, sizes):
    """GOPR<vid_no> + GP01.., GP02.. 챕터 파일을 주어진 크기로 생성."""
    names = [f"GOPR{vid_no}.MP4"] + [
        f"GP{i:02d}{vid_no}.MP4" for i in range(1, len(sizes))]
    for name, size in zip(names, sizes):
        (d / name).write_bytes(b"\0" * size)


def test_pairing_by_size(tmp_path):
    ldir = tmp_path / "L"
    rdir = tmp_path / "R"
    ldir.mkdir()
    rdir.mkdir()
    # 좌: 큰 경기(3챕터)와 작은 클립. 우: 번호는 다르지만 크기가 대응.
    _make_chain(ldir, "0100", [4000, 4000, 1000])
    _make_chain(ldir, "0101", [500])
    _make_chain(rdir, "0007", [520])          # ↔ 0101
    _make_chain(rdir, "0008", [4000, 4000, 1100])   # ↔ 0100
    pairs, un_l, un_r = pair_directories(ldir, rdir)
    assert not un_l and not un_r
    got = {(l[0].name, r[0].name) for l, r, _ in pairs}
    assert got == {("GOPR0100.MP4", "GOPR0008.MP4"),
                   ("GOPR0101.MP4", "GOPR0007.MP4")}


def test_pairing_leftover_unmatched(tmp_path):
    ldir = tmp_path / "L"
    rdir = tmp_path / "R"
    ldir.mkdir()
    rdir.mkdir()
    _make_chain(ldir, "0100", [4000])
    _make_chain(ldir, "0101", [9000])   # 우측에 대응 없음 (크기차 초과)
    _make_chain(rdir, "0007", [4100])
    pairs, un_l, un_r = pair_directories(ldir, rdir)
    assert len(pairs) == 1
    assert pairs[0][0][0].name == "GOPR0100.MP4"
    assert [c[0].name for c in un_l] == ["GOPR0101.MP4"]
    assert not un_r


def test_pairing_empty_dir(tmp_path):
    ldir = tmp_path / "L"
    rdir = tmp_path / "R"
    ldir.mkdir()
    rdir.mkdir()
    _make_chain(rdir, "0007", [100])
    with pytest.raises(RuntimeError):
        pair_directories(ldir, rdir)


def test_default_out_dir():
    base = Path("/data/matches")
    assert _default_out_dir(base / "20260712_GoPro5_L",
                            base / "20260712_GoPro5_R"
                            ).name == "20260712_GoPro5"
    assert _default_out_dir(base / "Left_cam", base / "Right_cam").name == "cam"
    assert _default_out_dir(base / "abc", base / "xyz"
                            ).name == "PyStitch360_headless"


def _analysis(rows_per_frame):
    return {"fps": 30.0, "frames": list(range(0, 90 * len(rows_per_frame), 90)),
            "players": rows_per_frame}


def test_ocr_candidates_without_calib():
    """calib=None 이면 필드 게이트 없이 높이 게이트만 적용."""
    big = [100.0, 500.0, 40.0, 120.0, 1]     # h=120 ≥ min_h
    small = [900.0, 200.0, 10.0, 30.0, 2]    # h=30 < min_h → 제외
    ana = _analysis([[big, small], [big]])
    picked = collect_ocr_candidates(ana, None, lambda t: 0, lambda t: t,
                                    min_h=90.0, per_track=12)
    assert {rep for _, _, rep in picked} == {1}
    assert len(picked) == 2                  # spacing 3s ≥ 2s — 둘 다 선택


def test_ocr_candidates_spacing():
    """같은 트랙릿의 크롭은 spacing_s 이상 시간 간격으로 분산."""
    big = [100.0, 500.0, 40.0, 120.0, 1]
    ana = {"fps": 30.0, "frames": [0, 15, 30], "players": [[big]] * 3}
    picked = collect_ocr_candidates(ana, None, lambda t: 0, lambda t: t,
                                    min_h=90.0, spacing_s=2.0)
    assert len(picked) == 1                  # 0/0.5/1.0s — 전부 2초 이내
    assert np.isfinite(picked[0][1][0])
