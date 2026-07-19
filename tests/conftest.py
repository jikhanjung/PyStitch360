"""공용 픽스처: 합성 카메라 캘리브레이션·분석 데이터.

실행: 저장소 루트에서 `python -m pytest tests/` (numpy·opencv 필요 —
Windows 실행 환경에 이미 설치돼 있음).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANO_W, PANO_H = 5760, 1920


@pytest.fixture
def calib():
    """직접 구성한 카메라 모델 calib (fit_field_calibration 결과와 동일 키).

    높이 4m, elevation +10°~−38°, 카메라는 근측 터치라인 5m 밖 중앙.
    """
    return {"h": 4.0, "t_top": float(np.tan(np.deg2rad(10.0))),
            "t_bot": float(np.tan(np.deg2rad(-38.0))), "theta": 0.0,
            "ex": 0.0, "ey": -(68 / 2 + 5.0), "pitch": 0.0, "roll": 0.0,
            "length": 105.0, "width": 68.0,
            "pano_w": PANO_W, "pano_h": PANO_H, "warp": None}


def make_team_analysis():
    """팀 분류용: 빨강 팀(1~6)·파랑 팀(11~16)·심판(21)·그늘진 선수(7).

    선수 행 = [cx, cy, w, h, id, h, s, v] (OpenCV HSV, H 0~180).
    7은 채도/명도가 낮아 두 팀 색의 중간쯤 — 오분류 시나리오 재료.
    """
    rows = []
    for tid in range(1, 7):                       # 빨강 (H≈0)
        rows.append([100 * tid, 500, 40, 120, tid,
                     2 + tid % 3, 200 - tid, 170 + tid])
    for tid in range(11, 17):                     # 파랑 (H≈120)
        rows.append([100 * tid, 520, 40, 120, tid,
                     120 + tid % 3, 190 - tid % 5, 160 + tid % 7])
    rows.append([2500, 700, 40, 120, 21, 60, 150, 60])   # 심판 (어두운 초록)
    rows.append([900, 640, 40, 120, 7, 8, 90, 90])       # 그늘진 빨강
    return {"fps": 30.0, "frames": [0], "pano_w": PANO_W, "pano_h": PANO_H,
            "players": [rows] * 30}


def make_ball_analysis(n=300):
    """공 트랙 라벨용: 수락 트랙(0~99) + 주입(100~119, 길이 6 행) +
    무시 구간 미끼(200~299, 움직이는 오검출)."""
    balls, cands, players = [], [], []
    for si in range(n):
        players.append([])
        if si < 100:
            row = [[1000.0 + si * 5, 800.0, 0.6, 14.0, 14.0]]
        elif si < 120:
            row = [[1000.0 + si * 5, 800.0, 0.26, 12.0, 12.0, 0.12]]
        elif si < 200:
            row = [[1000.0 + si * 5, 800.0, 0.6, 14.0, 14.0]]
        else:
            row = [[4000.0 + (si % 7) * 40, 1500.0, 0.5, 16.0, 16.0]]
        cands.append(row)
        balls.append(list(row[0][:5]))
    return {"fps": 30.0, "frames": [si * 3 for si in range(n)],
            "pano_w": PANO_W, "pano_h": PANO_H, "players": players,
            "balls": balls, "ball_cands": cands}
