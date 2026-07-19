"""경기장 캘리브레이션(core.field) 테스트 — 합성 카메라로 라운드트립."""
import numpy as np

from pystitch.core.field import (
    _params, _project, _sideline_rows, field_outline, field_to_pano,
    fit_field_calibration, landmark_positions, pano_to_field,
)

PANO_W, PANO_H = 5906, 1662
# 합성 정답 카메라: 높이 4.2m, 필드 중앙에서 3m 오른쪽, 터치라인 6m 뒤,
# 헤딩 살짝 + 리그 기울기(pitch/roll — 삼각대 비수평).
# [h, t_top, t_bot, theta, ex, ey, pitch, roll]
TRUTH = np.array([4.2, np.tan(np.deg2rad(9.0)), np.tan(np.deg2rad(-36.0)),
                  0.05, 3.0, -40.0, np.deg2rad(-2.0), np.deg2rad(0.8)])


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
    # 기울기 복원 확인 (10방정식 이상이라 pitch/roll 이 풀림)
    assert abs(np.degrees(calib["pitch"]) - (-2.0)) < 0.5
    assert abs(np.degrees(calib["roll"]) - 0.8) < 0.5
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
    # 8방정식 → 기울기 동결(6파라미터): 기울어진 합성 카메라라 잔차는
    # 남지만 수렴은 해야 한다
    assert calib is not None
    assert calib["pitch"] == 0.0 and calib["roll"] == 0.0


def test_fit_rejects_too_few():
    pts = _click_all()
    three = {k: pts[k] for k in ("corner_far_l", "corner_far_r", "half_near")}
    assert fit_field_calibration(three, PANO_W, PANO_H) is None


def test_fit_with_click_noise():
    calib = fit_field_calibration(_click_all(noise=4.0), PANO_W, PANO_H)
    assert calib is not None and calib["rms"] < 12.0


def test_fit_with_near_sideline_points():
    """half_near 가 안 보일 때: 위치 점 3개 + 사이드라인 위 점 2개로 해결."""
    pts = _click_all()
    hw = 34.0
    sl = _project(TRUTH, np.array([[-20.0, -hw], [18.0, -hw]]), PANO_W, PANO_H)
    use = {k: pts[k] for k in ("corner_far_l", "corner_far_r", "circle_far")}
    use["sideline_near_l"] = tuple(sl[0])
    use["sideline_near_r"] = tuple(sl[1])
    calib = fit_field_calibration(use, PANO_W, PANO_H)   # 방정식 6+2=8
    assert calib is not None
    # 클릭한 두 점이 실제로 그 선(Y=-hw) 위로 매핑되는가 (워프 포함)
    back = pano_to_field(calib, [use["sideline_near_l"],
                                 use["sideline_near_r"]])
    assert np.abs(back[:, 1] + hw).max() < 1.0
    # 점이 모자라면 (방정식 7개) 거부
    del use["sideline_near_r"]
    assert fit_field_calibration(use, PANO_W, PANO_H) is None


def test_warp_pins_clicked_landmarks():
    """모델이 못 잡는 국소 왜곡이 있어도 찍은 점은 정확히 통과 (TPS)."""
    pts = _click_all()
    pos = landmark_positions()
    # 렌즈 잔차 흉내: 몇 점을 수십 px 이동
    shifted = dict(pts)
    for k, (dx, dy) in [("corner_near_l", (-60, 25)),
                        ("corner_near_r", (45, 40)),
                        ("pen_l_box_near", (30, -10))]:
        shifted[k] = (pts[k][0] + dx, pts[k][1] + dy)
    calib = fit_field_calibration(shifted, PANO_W, PANO_H)
    assert calib is not None and calib["warp"] is not None
    keys = list(shifted)
    pred = field_to_pano(calib, [pos[k] for k in keys])
    err = np.abs(pred - np.array([shifted[k] for k in keys]))
    assert err.max() < 1.5          # 워프가 클릭 위치를 사실상 고정
    # 역변환도 일관: 클릭 픽셀 → 필드 좌표 ≈ 실제 랜드마크 위치
    back = pano_to_field(calib, [shifted[k] for k in keys])
    assert np.nanmax(np.abs(back - np.array([pos[k] for k in keys]))) < 1.5


def test_inverse_consistency_and_outline():
    calib = fit_field_calibration(_click_all(), PANO_W, PANO_H)
    f = np.array([[10.0, 5.0], [-30.0, 20.0], [0.0, -34.0]])
    assert np.abs(pano_to_field(calib, field_to_pano(calib, f)) - f).max() < 0.2
    for line in field_outline():
        assert line.ndim == 2 and line.shape[1] == 2
