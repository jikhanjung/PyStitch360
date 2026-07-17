"""광선/회전/원통 투영 기하 (prototype/stitch_still.py 에서 검증된 구현)."""
from __future__ import annotations

import cv2
import numpy as np

from .lens import LensProfile


def pixel_to_ray(pts: np.ndarray, lens: LensProfile) -> np.ndarray:
    """왜곡된 픽셀 좌표 (N,2) → 단위 광선 (N,3), 카메라 좌표계."""
    und = cv2.fisheye.undistortPoints(
        pts.reshape(-1, 1, 2).astype(np.float64), lens.K, lens.D
    ).reshape(-1, 2)
    rays = np.hstack([und, np.ones((len(und), 1))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def ray_to_pixel(rays: np.ndarray, lens: LensProfile):
    """카메라 좌표계 광선 (...,3) → 왜곡된 픽셀 좌표 (...,2). 벡터화 fisheye 투영."""
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    r_plane = np.sqrt(x * x + y * y)
    theta = np.arctan2(r_plane, z)
    k1, k2, k3, k4 = lens.D.ravel()
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r_plane > 1e-9, theta_d / np.maximum(r_plane, 1e-9), 1.0)
    u = lens.K[0, 0] * x * scale + lens.K[0, 2]
    v = lens.K[1, 1] * y * scale + lens.K[1, 2]
    valid = theta < np.deg2rad(100)
    return np.stack([u, v], axis=-1), valid


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """b ≈ R @ a 가 되는 회전 R (a, b: (N,3) 단위 광선)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def estimate_relative_rotation(rays_l, rays_r, iters=2000, thresh_deg=0.3):
    """RANSAC: rays_l ≈ R @ rays_r 인 R 을 강건하게 추정."""
    rng = np.random.default_rng(0)
    n = len(rays_l)
    best_inliers = None
    thresh = np.deg2rad(thresh_deg)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R = kabsch(rays_r[idx], rays_l[idx])
        err = np.arccos(np.clip(np.sum(rays_l * (rays_r @ R.T), axis=1), -1, 1))
        inliers = err < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    R = kabsch(rays_r[best_inliers], rays_l[best_inliers])
    err = np.arccos(np.clip(np.sum(rays_l * (rays_r @ R.T), axis=1), -1, 1))
    return R, best_inliers, np.rad2deg(err[best_inliers])


def half_rotation(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    Rh, _ = cv2.Rodrigues(rvec * 0.5)
    return Rh


def rot_xz(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """월드 보정 회전: x축(pitch) 후 z축(roll)."""
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rx @ Rz


def build_cylindrical_maps(lens: LensProfile, R_cam, out_w, out_h, yaw0, yaw1, el0, el1):
    """출력 원통 픽셀 → 소스 어안 픽셀 remap 테이블.

    R_cam: 월드 광선 → 카메라 광선 회전.
    수평: 등각 (yaw0~yaw1), 수직: y = tan(elevation) (원통 표면).
    """
    yaw = np.linspace(yaw0, yaw1, out_w, dtype=np.float64)
    t = np.linspace(np.tan(el1), np.tan(el0), out_h, dtype=np.float64)  # 위→아래
    yy_t, xx_yaw = np.meshgrid(t, yaw, indexing="ij")
    rays = np.stack([np.sin(xx_yaw), -yy_t, np.cos(xx_yaw)], axis=-1)  # 이미지 y는 아래 +
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cam_rays = rays @ np.asarray(R_cam).T
    pix, valid = ray_to_pixel(cam_rays, lens)
    map_x = pix[..., 0].astype(np.float32)
    map_y = pix[..., 1].astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y
