"""공중볼 탄도 보정(core.airborne) — 합성 비행."""
import numpy as np

from pystitch.core.airborne import (
    correct_ball_track, detect_airborne_segments, fit_ballistic,
    project_ballistic,
)

CAM = (0.0, -40.0)
H = 5.0


def _flight(p0=(-10.0, 0.0), v=(12.0, 4.0), t0=2.0, T=1.6,
            total=8.0, dt=0.1, noise=0.0, seed=0):
    """지상 굴림 + 공중 구간 하나가 있는 합성 지면 관측."""
    t = np.arange(0.0, total, dt)
    true_p = np.asarray(p0)[None] + np.asarray(v)[None] * (t - t0)[:, None]
    obs = project_ballistic([*p0, *v, t0, T], t, CAM, H)
    if noise:
        obs = obs + np.random.default_rng(seed).normal(0, noise, obs.shape)
    return t, obs, true_p


def test_projection_displaces_away_from_camera():
    """공중일 때 지면 투영은 카메라 반대쪽으로 밀린다 (배율 h/(h-z))."""
    t, obs, true_p = _flight(noise=0.0)
    i_apex = int(np.argmin(np.abs(t - (2.0 + 0.8))))
    d_true = np.hypot(true_p[i_apex, 0] - CAM[0], true_p[i_apex, 1] - CAM[1])
    d_obs = np.hypot(obs[i_apex, 0] - CAM[0], obs[i_apex, 1] - CAM[1])
    z_apex = 0.5 * 9.81 * 0.8 * 0.8
    assert d_obs > d_true
    assert abs(d_obs / d_true - H / (H - z_apex)) < 0.01


def test_fit_recovers_flight():
    t, obs, _ = _flight(noise=0.1, seed=1)
    m = (t >= 1.9) & (t <= 3.7)
    fit = fit_ballistic(t[m], obs[m], CAM, H)
    assert fit is not None
    assert abs(fit["t0"] - 2.0) < 0.15 and abs(fit["T"] - 1.6) < 0.2
    assert abs(fit["apex_z"] - 0.125 * 9.81 * 1.6 ** 2) < 0.6
    assert np.hypot(fit["v"][0] - 12.0, fit["v"][1] - 4.0) < 1.0


def test_detect_and_straighten():
    """감지된 공중 구간의 보정 XY 는 직선 (관측은 휘어 있음)."""
    t, obs, true_p = _flight(noise=0.12, seed=2)
    segs = detect_airborne_segments(t, obs, CAM, H)
    assert len(segs) == 1
    i0, i1, fit = segs[0]
    assert t[i0] <= 2.25 and t[i1] >= 3.3          # 비행 구간 커버
    corr, z, _ = correct_ball_track(t, obs, CAM, H, segments=segs)

    def max_line_dev(P):
        a, b = P[0], P[-1]
        ux, uy = (b - a) / (np.hypot(*(b - a)) + 1e-9)
        d = P - a
        return float(np.max(np.abs(ux * d[:, 1] - uy * d[:, 0])))

    seg_obs = obs[i0:i1 + 1]
    seg_corr = corr[i0:i1 + 1]
    assert max_line_dev(seg_obs) > 0.8              # 관측: 눈에 띄게 휨
    assert max_line_dev(seg_corr) < 0.35            # 보정: 직선화
    assert np.max(z) > 2.0                          # 높이 복원
    err = np.hypot(*(seg_corr - true_p[i0:i1 + 1]).T)
    assert np.median(err) < 0.5                     # 참 궤적 근접


def test_ground_track_not_flagged():
    """굴러가는 공(직선 등속)은 공중으로 오인하지 않는다."""
    t = np.arange(0.0, 6.0, 0.1)
    p = np.stack([-15 + 9 * t, 5 + 2 * t], axis=1)
    p += np.random.default_rng(3).normal(0, 0.12, p.shape)
    assert detect_airborne_segments(t, p, CAM, H) == []
