"""이벤트 엔진(core.events) — 합성 경기로 킥오프 판정."""
import numpy as np

from pystitch.core.events import detect_kickoffs, formation_track
from pystitch.core.field import _project, fit_field_calibration, \
    landmark_positions

PANO_W, PANO_H = 5906, 1662
TRUTH = np.array([4.2, np.tan(np.deg2rad(9.0)), np.tan(np.deg2rad(-36.0)),
                  0.0, 0.0, -40.0, 0.0, 0.0])
FPS, DE = 30.0, 3


def _calib():
    pos = landmark_positions()
    keys = list(pos)
    px = _project(TRUTH, np.array([pos[k] for k in keys]), PANO_W, PANO_H)
    return fit_field_calibration({k: tuple(px[i]) for i, k in enumerate(keys)},
                                 PANO_W, PANO_H)


def _row(fx, fy, tid):
    """필드 좌표의 선수를 파노라마 행 [cx,cy,w,h,id,...] 로 (발=박스 하단)."""
    px = _project(TRUTH, np.array([[fx, fy]]), PANO_W, PANO_H)[0]
    h = 60.0
    return [float(px[0]), float(px[1]) - h / 2, 24.0, h, tid, 0.0, 0.0, 0.0]


def _match(kick_t=30.0, total_s=90.0):
    """킥오프 대형 → kick_t 에 경기 시작(진영 붕괴 + 공 이탈) 합성."""
    rng = np.random.default_rng(0)
    frames = list(range(0, int(total_s * FPS), DE))
    players, balls = [], []
    home = [(-8 - 3 * i, (i % 4 - 1.5) * 12) for i in range(8)]   # X<0
    away = [(8 + 3 * i, (i % 4 - 1.5) * 12) for i in range(8)]    # X>0
    kicker = (-1.5, 0.5)                                          # 서클 안
    for f in frames:
        t = f / FPS
        row = []
        adv = max(0.0, (t - kick_t) * 2.0)        # 킥오프 후 전진 (m/s×2)
        # 실제 경기처럼 절반만 상대 진영으로 — 두 팀이 '섞인다'
        for i, (x, y) in enumerate(home):
            jx, jy = rng.normal(0, .4), rng.normal(0, .4)
            dx = adv if i % 2 == 0 else adv * 0.1
            row.append(_row(min(x + dx + jx, 40), y + jy, 100 + i))
        for i, (x, y) in enumerate(away):
            jx, jy = rng.normal(0, .4), rng.normal(0, .4)
            dx = adv if i % 2 == 0 else adv * 0.1
            row.append(_row(max(x - dx + jx, -40), y + jy, 200 + i))
        row.append(_row(kicker[0] + min(adv, 20), kicker[1], 300))
        players.append(row)
        if t < kick_t:
            bx, by = 0.0, 0.0
        else:
            bx = min((t - kick_t) * 6.0, 45.0)     # 공이 센터에서 굴러감
            by = 0.0
        px = _project(TRUTH, np.array([[bx, by]]), PANO_W, PANO_H)[0]
        balls.append([float(px[0]), float(px[1]), 0.6])
    return {"fps": FPS, "total_frames": int(total_s * FPS),
            "detect_every": DE, "frames": frames,
            "balls": balls, "players": players,
            "pano_w": PANO_W, "pano_h": PANO_H}


TEAMS = {**{100 + i: 0 for i in range(8)}, **{200 + i: 1 for i in range(8)},
         300: 0}


def test_formation_track_metrics():
    a = _match(kick_t=30.0)
    tr = formation_track(a, TEAMS, _calib())
    pre = tr["t"] < 28
    post = tr["t"] > 45
    assert np.nanmedian(tr["sep"][pre]) > 0.9         # 킥오프 전 분리
    assert np.nanmedian(tr["sep"][post]) < 0.7        # 시작 후 붕괴
    assert np.median(tr["circle_n"][pre]) <= 2        # 서클엔 킥커만
    assert np.nanmedian(tr["ball_r"][pre]) < 3.0      # 공 = 센터
    assert np.nanmax(tr["ball_r"]) > 20.0             # 이후 이탈


def test_kickoff_detected_at_whistle():
    a = _match(kick_t=30.0)
    tr = formation_track(a, TEAMS, _calib())
    whistles = [(29.0, 30.2, 25.0),      # 킥오프 롱 휘슬
                (60.0, 60.2, 22.0)]      # 경기 중 파울 휘슬 (대형 붕괴 상태)
    ks = detect_kickoffs(tr, whistles)
    assert len(ks) == 1
    assert abs(ks[0][0] - 29.0) < 0.5
    assert ks[0][2]["long_whistle"] and ks[0][2]["ball_left"]


def test_no_kickoff_without_formation():
    """대형 없이 호각만 있으면 (경기 중 파울) 킥오프 아님."""
    a = _match(kick_t=5.0)               # 5초에 이미 시작 — 이후 대형 없음
    tr = formation_track(a, TEAMS, _calib())
    ks = detect_kickoffs(tr, [(50.0, 50.3, 30.0)])
    assert ks == []
