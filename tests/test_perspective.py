"""pystitch.core.perspective 유닛 테스트."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.perspective import (  # noqa: E402
    PerspectiveWarp, build_perspective_maps,
)

W, H, HORIZON = 590, 168, 38.0


def test_identity_when_disabled():
    mx, my = build_perspective_maps(W, H, HORIZON, k=0.0, m=1.0)
    assert np.allclose(mx, np.arange(W, dtype=np.float32)[None, :])
    assert np.allclose(my, np.arange(H, dtype=np.float32)[:, None])
    warp = PerspectiveWarp(W, H, HORIZON, k=0.0, m=1.0)
    frame = np.random.randint(0, 255, (H, W, 3), np.uint8)
    assert warp.apply(frame) is frame  # no-op 경로


def test_vertical_endpoints_and_monotonic():
    _, my = build_perspective_maps(W, H, HORIZON, k=0.5, m=1.0)
    col = my[:, 0]
    # 수평선 위는 항등, 최하단은 고정
    assert np.allclose(col[: int(HORIZON) + 1], np.arange(int(HORIZON) + 1), atol=1e-4)
    assert col[H - 1] == pytest.approx(H - 1, abs=1e-3)
    assert np.all(np.diff(col) > 0)  # 단조증가 (뒤집힘 없음)


def test_vertical_magnifies_far_compresses_near():
    _, my = build_perspective_maps(W, H, HORIZON, k=0.5, m=1.0)
    col = my[:, 0].astype(np.float64)
    d_far = col[int(HORIZON) + 2] - col[int(HORIZON) + 1]   # 수평선 직하
    d_near = col[H - 1] - col[H - 2]                        # 최하단
    assert d_far < 1.0 < d_near  # 소스 소비량: 원경 <1(확대), 근경 >1(압축)
    assert d_near == pytest.approx(1.5, rel=0.05)           # 1+k


def test_keystone_row_scales():
    mx, my = build_perspective_maps(W, H, HORIZON, k=0.0, m=1.5)
    cx = (W - 1) / 2
    # 하단 행: 배율 1 (항등), 상단 행: 소스 폭 1/m만 소비 (m배 확대)
    assert np.allclose(mx[H - 1], np.arange(W), atol=1e-3)
    top_scale = (mx[0, -1] - mx[0, 0]) / (W - 1)
    assert top_scale == pytest.approx(1 / 1.5, rel=1e-3)
    assert mx[0, W // 2] == pytest.approx(cx, abs=0.51)  # 중심 고정
    assert np.all(np.diff(my[:, 0]) > 0)


def test_combined_maps_match_sequential_application():
    """합성 맵 결과 == 수직 리맵 후 키스톤 순차 적용."""
    import cv2

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (H, W, 3), np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3)  # 보간 차이 민감도 완화
    k, m = 0.3, 1.3

    combined = PerspectiveWarp(W, H, HORIZON, k, m).apply(frame)
    step1 = PerspectiveWarp(W, H, HORIZON, k, 1.0).apply(frame)
    step2 = PerspectiveWarp(W, H, HORIZON, 0.0, m).apply(step1)
    diff = np.abs(combined.astype(int) - step2.astype(int)).mean()
    assert diff < 2.0  # 보간 1회 vs 2회 차이 수준


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        build_perspective_maps(W, H, HORIZON, k=1.0)
    with pytest.raises(ValueError):
        build_perspective_maps(W, H, HORIZON, m=0.9)


def test_renderer_one_pass_matches_two_pass():
    """Renderer(persp) 원패스 == 일반 렌더 후 PerspectiveWarp 투패스."""
    import cv2

    from pystitch.core.lens import LensProfile, builtin_profiles
    from pystitch.core.render import Renderer

    lens = LensProfile.load(next(iter(builtin_profiles().values())))
    rng = np.random.default_rng(7)
    im = rng.integers(0, 255, (lens.height // 8, lens.width // 8, 3), np.uint8)
    im = cv2.resize(im, (lens.width, lens.height))
    # scale=0.2 렌더(5배 다운샘플)에서 앨리어싱이 없도록 충분히 블러
    # — 고주파가 남으면 보간 1회/2회 차이가 증폭돼 합성 검증이 안 됨
    im = cv2.GaussianBlur(im, (0, 0), 8)
    # 같은 이미지·같은 회전 → 심 블렌딩이 diff 에 기여하지 않아
    # 맵 합성(보간 순서)만 순수하게 비교된다
    imgs = [im, im]
    R = np.eye(3)
    args = (lens, R, R, -0.7, 0.7, -0.35, 0.15)
    kw = dict(scale=0.2, feather_px=40)
    k, m = 0.3, 1.3

    r0 = Renderer(*args, **kw)
    pano0 = r0.render(*imgs)
    horizon = (r0.out_h - 1) * np.tan(0.15) / (np.tan(0.15) - np.tan(-0.35))
    two_pass = PerspectiveWarp(r0.out_w, r0.out_h, horizon, k, m).apply(pano0)

    r1 = Renderer(*args, **kw, persp_k=k, persp_m=m)
    assert (r1.out_w, r1.out_h) == (r0.out_w, r0.out_h)
    one_pass = r1.render(*imgs)

    valid = (r1._masks[0] > 0) | (r1._masks[1] > 0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert valid.mean() > 0.5  # 커버리지 자체가 있어야 의미 있는 비교
    diff = np.abs(one_pass.astype(int) - two_pass.astype(int))[valid].mean()
    assert diff < 3.0  # 보간 1회 vs 2회 차이 수준
