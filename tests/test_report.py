"""core/report.py — 이동거리/속도·히트맵·리포트 생성 (P03-4)."""
import cv2
import numpy as np

from pystitch.core.report import (
    generate_report, heatmap_grid, movement_stats, player_field_tracks,
)


def test_constant_speed_distance():
    t = np.arange(0, 60, 0.1)
    track = [(tt, 5.0 * tt - 150.0, 0.0) for tt in t]
    st = movement_stats(track)
    assert abs(st["dist_m"] - 300) < 6                 # 300m ±2%
    assert abs(st["avg_mps"] - 5.0) < 0.25
    assert 4.5 <= st["max_mps"] <= 5.6


def test_jitter_does_not_inflate_distance():
    rng = np.random.default_rng(7)
    t = np.arange(0, 60, 0.1)
    track = [(tt, 5.0 * tt - 150.0 + rng.normal(0, 0.3),
              rng.normal(0, 0.3)) for tt in t]
    st = movement_stats(track)
    assert st["dist_m"] < 330                          # raw 합산이면 수십% 부풀음


def test_tracker_jump_excluded():
    t = np.arange(0, 60, 0.1)
    track = [(tt, 5.0 * tt - 150.0, 0.0) for tt in t]
    track[300] = (track[300][0], track[300][1] + 30.0, 0.0)   # 순간이동
    assert movement_stats(track)["dist_m"] < 320


def test_stationary_player():
    assert movement_stats([(tt, 3.0, 4.0)
                           for tt in np.arange(0, 60, 0.1)])["dist_m"] < 5


def test_heatmap_grid_cells():
    g = heatmap_grid([(0.0, 0.0)] * 10 + [(-50.0, 30.0)] * 5, 105, 68)
    assert g.shape == (68, 105)
    assert g[34, 52] == 10 and g[4, 2] == 5


def test_generate_report(calib, tmp_path):
    pw, ph = calib["pano_w"], calib["pano_h"]
    players = []
    for si in range(600):
        row = [[2800 + si, 1200, 40, 120, 1, 5, 200, 170]]
        tid = 2 if si < 300 else 3                     # 3은 2의 병합 조각
        row.append([2400 + si, 1250, 40, 120, tid, 120, 190, 160])
        players.append(row)
    ana = {"fps": 30.0, "frames": list(range(0, 1800, 3)),
           "pano_w": pw, "pano_h": ph, "players": players}
    tracks = player_field_tracks(ana, calib, merges={3: 2})
    assert set(tracks) == {1, 2} and len(tracks[2]) == 600
    r = generate_report(ana, calib, {1: 0, 2: 1}, tmp_path / "out",
                        merges={3: 2}, team_names=("우리팀", "상대팀"),
                        min_det=100, log=lambda s: None)
    assert len(r["files"]) == 5                        # 팀2 + 선수2 + md
    assert any(row["tid"] == 2 and row["k"] == 2 for row in r["rows"])
    md = (tmp_path / "out" / "players.md").read_text(encoding="utf-8")
    assert "우리팀" in md and "(+1)" in md
    img = cv2.imread(str(tmp_path / "out" / "team1_heatmap.png"))
    h, w = img.shape[:2]
    assert abs(w / h - 111 / 74) < 0.02                # 등방 비율 유지
