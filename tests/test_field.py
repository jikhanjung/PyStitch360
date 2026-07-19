"""경기장 캘리브레이션(core.field) 테스트 — 합성 카메라로 라운드트립."""
import numpy as np

from pystitch.core.field import (
    field_outline, field_to_pano, fit_field_calibration, landmark_positions,
    pano_to_field, _project,
)

PANO_W, PANO_H = 5906, 1662
# 합성 정답 카메라: 높이 4.2m, 필드 중앙에서 3m 오른쪽, 터치라인 6m 뒤,
# 살짝 돌아간 헤딩과 요 오프셋.
TRUTH = np.array([4.2, np.tan(np.deg2rad(9.0)), np.tan(np.deg2rad(-36.0)),
                  0.05, 0.03, 3.0, -40.0])


def _click_all(noise=0.0, seed=0):
    pos = landmark_positions()
    keys = list(pos)
    px = _project(TRUTH, np.array([pos[k] for k in keys]), PANO_W, PANO_H)
    if noise:
        px = px + np.random.default_rng(seed).normal(0, noise, px.shape)
    return {k: tuple(px[i]) for i, k in enumerate(keys)}


def test_fit_recovers_mapping_all_points():
    calib = fit_field_calibration(_click_all(), PANO_W, PANO_H)
    assert calib is not None and calib["rms"] < 1.0
    # 임의 필드 그리드가 파노라마 → 필드 라운드트립으로 복원되는가
    gx, gy = np.meshgrid(np.linspace(-50, 50, 9), np.linspace(-30, 30, 7))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    back = pano_to_field(calib, _project(TRUTH, grid, PANO_W, PANO_H))
    assert np.nanmax(np.abs(back - grid)) < 1.0     # 1m 이내

def test_fit_minimum_four_points():
    pts = _click_all()
    four = {k: pts[k] for k in
            ("corner_far_l", "corner_far_r", "half_near", "circle_far")}
    calib = fit_field_calibration(four, PANO_W, PANO_H)
    assert calib is not None and calib["rms"] < 1.0
    # 안 찍은 랜드마크도 맞는 위치로 투영돼야 함 (일반화)
    pos = landmark_positions()
    unseen = ["corner_near_l", "corner_near_r", "half_far"]
    pred = field_to_pano(calib, [pos[k] for k in unseen])
    true = np.array([pts[k] for k in unseen])
    assert np.abs(pred - true).max() < 40.0         # 픽셀 오차

def test_fit_rejects_too_few():
    pts = _click_all()
    three = {k: pts[k] for k in ("corner_far_l", "corner_far_r", "half_near")}
    assert fit_field_calibration(three, PANO_W, PANO_H) is None

def test_fit_with_click_noise():
    calib = fit_field_calibration(_click_all(noise=4.0), PANO_W, PANO_H)
    assert calib is not None and calib["rms"] < 10.0

def test_fit_with_near_sideline_points():
    """half_near 가 안 보일 때: 위치 점 3개 + 사이드라인 위 점 2개로 해결."""
    from pystitch.core.field import _sideline_rows
    pts = _click_all()
    hw = 34.0
    # 사이드라인 위 임의 지점 (X는 미지라고 가정하고 픽셀만 사용)
    sl = _project(TRUTH, np.array([[-20.0, -hw], [18.0, -hw]]), PANO_W, PANO_H)
    use = {k: pts[k] for k in ("corner_far_l", "corner_far_r", "circle_far")}
    use["sideline_near_l"] = tuple(sl[0])
    use["sideline_near_r"] = tuple(sl[1])
    calib = fit_field_calibration(use, PANO_W, PANO_H)   # 방정식 6+2=8
    assert calib is not None and calib["rms"] < 2.0
    # 사이드라인이 실제로 맞는 위치에 놓였는가: 예측 행 vs 정답 행
    cols = np.linspace(sl[0, 0], sl[1, 0], 7)
    p = np.array([calib["h"], calib["t_top"], calib["t_bot"], calib["phi0"],
                  calib["theta"], calib["ex"], calib["ey"]])
    pred = _sideline_rows(p, cols, PANO_W, PANO_H, 68.0)
    true_rows = _project(TRUTH, np.stack(
        [np.zeros(7), np.full(7, -hw)], axis=1), PANO_W, PANO_H)  # placeholder
    # 정답 행: 각 열의 yaw 광선이 truth 모델에서 Y=-hw 와 만나는 행
    true = _sideline_rows(TRUTH, cols, PANO_W, PANO_H, 68.0)
    assert np.nanmax(np.abs(pred - true)) < 15.0
    # 점이 모자라면 (방정식 7개) 거부
    del use["sideline_near_r"]
    assert fit_field_calibration(use, PANO_W, PANO_H) is None


def test_inverse_consistency_and_outline():
    calib = fit_field_calibration(_click_all(), PANO_W, PANO_H)
    f = np.array([[10.0, 5.0], [-30.0, 20.0], [0.0, -34.0]])
    assert np.abs(pano_to_field(calib, field_to_pano(calib, f)) - f).max() < 0.2
    for line in field_outline():
        assert line.ndim == 2 and line.shape[1] == 2
