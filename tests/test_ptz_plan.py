"""2패스 PTZ 계획(build_plan) 유닛 테스트 — ultralytics 불필요 (순수 numpy)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import build_plan  # noqa: E402

PANO_W, PANO_H, FPS = 5906, 1680, 30.0


def _analysis(frames, balls, players=None):
    n = len(frames)
    return {"fps": FPS, "total_frames": int(frames[-1]) + 1,
            "frames": list(frames),
            "balls": balls,
            "players": players if players is not None else [[] for _ in range(n)]}


def test_follows_ball_and_fills_gap():
    """직선 이동하는 공 + 1초 미검출 갭 → 궤적 추종, 갭 보간, 줌 1배."""
    frames = list(range(0, 900, 3))
    balls = []
    for f in frames:
        x = 1500 + 3.0 * f          # 완만한 이동 (90px/s)
        balls.append(None if 300 <= f < 330 else [x, 900.0, 0.5])
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    # 갭 구간 포함 전 구간에서 공 근처 (스무딩 지연 고려 넉넉히)
    err = np.abs(plan["cx"][150:850] - (1500 + 3.0 * np.arange(150, 850)))
    assert err.max() < 120
    assert np.all(plan["crop_w"][150:850] < 1920 * 1.05)  # 공이 있으니 줌인 유지


def test_rejects_teleporting_low_conf_ball():
    """저신뢰 순간이동 검출(장외 공 등)은 게이팅으로 무시."""
    frames = list(range(0, 300, 3))
    balls = [[2000.0, 900.0, 0.5] for _ in frames]
    balls[50] = [5500.0, 600.0, 0.3]   # 3500px 순간이동, conf 0.3
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    assert np.abs(plan["cx"] - 2000).max() < 60


def test_zoom_out_when_no_ball():
    """공 부재 + 선수 산개 → 선수 분포를 덮도록 줌아웃, 중심은 군집 근처."""
    frames = list(range(0, 900, 3))
    rng = np.random.default_rng(3)
    players = [[[float(x), 900.0] for x in rng.uniform(2200, 4800, 20)]
               for _ in frames]
    plan = build_plan(_analysis(frames, [None] * len(frames), players),
                      PANO_W, PANO_H, log=None)
    mid = slice(200, 700)
    # 20명 표본의 p10~p90 기대폭 ~1900px + 마진 520px
    assert np.all(plan["crop_w"][mid] > 2200)
    assert np.abs(plan["cx"][mid] - 3500).max() < 400  # 군집 중심 부근


def test_smoothness_frame_step_bounded():
    """노이즈 낀 검출에도 프레임당 이동량이 작아야 함 (흔들림 방지)."""
    frames = list(range(0, 1800, 3))
    rng = np.random.default_rng(5)
    balls = [[2800.0 + rng.normal(0, 80), 900.0 + rng.normal(0, 40), 0.45]
             for _ in frames]
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    step = np.abs(np.diff(plan["cx"][60:-60]))
    assert step.max() < 4.0            # 정지 상황: 프레임당 4px 미만
    assert np.abs(plan["cx"][60:-60] - 2800).max() < 80


def test_fast_ball_engages_fast_follow():
    """공이 빠르게 관통(2500px/2s)하면 빠른 추종으로 전환되어 크게 뒤처지지 않음."""
    frames = list(range(0, 1800, 3))
    balls = []
    for f in frames:
        if f < 600:
            x = 1200.0
        elif f < 660:
            x = 1200.0 + (f - 600) / 60 * 2500   # 2초에 2500px
        else:
            x = 3700.0
        balls.append([x, 900.0, 0.5])
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H, log=None)
    # 이동 완료 1.5초 뒤에는 목표에 수렴해 있어야 함
    assert abs(plan["cx"][660 + 45] - 3700) < 250
    # 크롭이 파노라마 밖을 요구하지 않음
    half = plan["crop_w"] / 2
    assert np.all(plan["cx"].clip(half, PANO_W - half) - plan["cx"] < 1e-6 + half)


def test_static_isolated_decoy_rejected():
    """장시간 정지 + 선수들로부터 고립된 '공'(낙엽 등)은 기각 → 줌아웃 유지."""
    frames = list(range(0, 900, 3))
    balls = [[4900.0, 1350.0, 0.6] for _ in frames]      # 완전 정지, 높은 conf
    players = [[[float(x), 800.0] for x in np.linspace(1200, 4100, 8)]
               for _ in frames]                           # 전원 700px+ 거리, 넓게 산개
    plan = build_plan(_analysis(frames, balls, players), PANO_W, PANO_H, log=None)
    mid = slice(200, 700)
    assert np.abs(plan["cx"][mid] - 4900).min() > 1200   # 미끼를 안 따라감
    assert np.all(plan["crop_w"][mid] > 1920 * 1.05)     # 선수 산개 커버 줌아웃

    # 같은 정지 공이라도 선수가 근처에 있으면(세트피스) 정상 추적
    players2 = [[[4850.0, 1300.0], [4700.0, 1250.0], [3000.0, 800.0]]
                for _ in frames]
    plan2 = build_plan(_analysis(frames, balls, players2), PANO_W, PANO_H, log=None)
    assert np.abs(plan2["cx"][mid] - 4900).max() < 100


def test_keyframes_override_auto():
    """키프레임은 자동 검출을 덮어쓰고(±1.5s 억제), 멀리서는 자동 추종 유지."""
    frames = list(range(0, 1800, 3))
    balls = [[2000.0, 900.0, 0.5] for _ in frames]          # 자동: 내내 x=2000
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      keyframes=[(900, 4000.0, 1000.0)], log=None)
    assert abs(plan["cx"][900] - 4000) < 350        # 키프레임 지점: 수동 위치
    assert abs(plan["cx"][150] - 2000) < 50         # 먼 곳: 자동 유지
    assert abs(plan["cx"][1700] - 2000) < 50


def test_keyframe_pair_bridges_detection_gap():
    """자동 검출이 전혀 없는 구간도 인접 키프레임(<=8s)끼리 직접 보간."""
    frames = list(range(0, 2700, 3))
    balls = [None] * len(frames)                    # 자동 검출 전무
    kfs = [(300, 1500.0, 900.0), (480, 3000.0, 900.0)]   # 6초 간격
    plan = build_plan(_analysis(frames, balls), PANO_W, PANO_H,
                      keyframes=kfs, log=None)
    mid = plan["cx"][390]                           # 중간 지점 ≈ 평균
    assert abs(mid - 2250) < 300
    assert np.all(plan["crop_w"][360:420] < 1920 * 1.25)  # 구간 중앙: 줌인 유지
