"""core/ptz.py clock_string/_draw_clock — 경기 시계 (P03-2).

시나리오는 실경기 정보 그대로: 후반 30분 하프, 킥오프 +15분에 2분
hydration break (중단 동안 시계 정지), 분:초 누적 표기 (연장 120분).
"""
import numpy as np

from pystitch.core.ptz import _draw_clock, clock_string

FPS = 30.0


def _clk():
    return {"anchor_f": int(60 * FPS), "fps": FPS, "base_s": 30 * 60,
            "tag": "2H",
            "pauses": [[int(16 * 60 * FPS), int(18 * 60 * FPS)]],
            "score": None}


def test_second_half_with_break():
    clk = _clk()
    assert clock_string(clk, 0) == "2H 30:00"                  # 킥오프 전 고정
    assert clock_string(clk, int(60 * FPS)) == "2H 30:00"
    assert clock_string(clk, int((60 + 300) * FPS)) == "2H 35:00"
    assert clock_string(clk, int(16 * 60 * FPS)) == "2H 45:00"  # 브레이크 시작
    assert clock_string(clk, int(17 * 60 * FPS)) == "2H 45:00"  # 정지
    assert clock_string(clk, int(19 * 60 * FPS)) == "2H 46:00"  # 재개
    assert clock_string(clk, int(33 * 60 * FPS)) == "2H 60:00"  # 하프 종료


def test_minutes_accumulate_past_100():
    big = {"anchor_f": 0, "fps": FPS, "base_s": 105 * 60, "tag": "ET",
           "pauses": [], "score": None}
    assert clock_string(big, int(930 * FPS)) == "ET 120:30"


def test_running_score():
    clk = dict(_clk(), score=("HOME", "AWAY",
                              [[int(10 * 60 * FPS), 1],
                               [int(20 * 60 * FPS), 2]]))
    assert clock_string(clk, int(5 * 60 * FPS)).endswith("HOME 0-0 AWAY")
    assert clock_string(clk, int(12 * 60 * FPS)).endswith("HOME 1-0 AWAY")
    assert clock_string(clk, int(25 * 60 * FPS)).endswith("HOME 1-1 AWAY")


def test_draw_clock_renders():
    for w, h in ((1920, 1080), (1280, 720)):
        img = np.full((h, w, 3), 90, np.uint8)
        _draw_clock(img, "2H 120:30  HOME 1-0 AWAY", w, h)
        box = img[:80, :600]
        assert (box == 255).any() and (box < 90).any()   # 흰 글자 + 어두운 박스
