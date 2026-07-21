"""headless._StageTimer — 단계 소요 기록 (P09 비교용)."""
import json
from pathlib import Path

from pystitch.headless import _StageTimer


def test_records_and_merges(tmp_path):
    pano = tmp_path / "pano_1.mp4"
    tm = _StageTimer(pano)
    with tm.stage("stitch"):
        pass
    d = json.loads(pano.with_suffix(".timing.json").read_text())
    assert d["runs"]["stitch"]["ok"] and d["runs"]["stitch"]["sec"] >= 0
    # 재실행: 실행된 단계만 갱신, 기존 단계 유지
    tm2 = _StageTimer(pano)
    with tm2.stage("ocr_cached"):
        pass
    d = json.loads(pano.with_suffix(".timing.json").read_text())
    assert set(d["runs"]) == {"stitch", "ocr_cached"}


def test_failure_recorded_and_raised(tmp_path):
    pano = tmp_path / "pano_2.mp4"
    tm = _StageTimer(pano)
    try:
        with tm.stage("analyze"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("예외가 삼켜짐")
    d = json.loads(pano.with_suffix(".timing.json").read_text())
    assert d["runs"]["analyze"]["ok"] is False
