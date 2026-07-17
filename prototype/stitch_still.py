"""정지 프레임 원통 파노라마 스티칭 프로토타입.

파이프라인:
  1. Gyroflow 렌즈 프로파일(OpenCV fisheye 모델) 로드
  2. 좌/우 프레임의 겹침 영역에서 SIFT 특징점 매칭
  3. 매칭점을 광선(ray)으로 변환 → RANSAC + Kabsch 로 상대 회전 추정
  4. 출력 원통 좌표 → 각 카메라 어안 픽셀로의 remap 테이블 생성 (왜곡 소스에서 직접 1회 리샘플)
  5. 겹침 영역 게인 보정 + 페더 블렌딩

사용법:
  python stitch_still.py <left.png> <right.png> <lens_profile.json> <out.jpg>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_lens_profile(path):
    with open(path) as f:
        p = json.load(f)
    assert p["use_opencv_fisheye"], "OpenCV fisheye 프로파일만 지원"
    K = np.array(p["fisheye_params"]["camera_matrix"], dtype=np.float64)
    D = np.array(p["fisheye_params"]["distortion_coeffs"], dtype=np.float64)
    dim = (p["calib_dimension"]["w"], p["calib_dimension"]["h"])
    return K, D, dim


def pixel_to_ray(pts, K, D):
    """왜곡된 픽셀 좌표 (N,2) → 단위 광선 (N,3), 카메라 좌표계."""
    und = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D)
    und = und.reshape(-1, 2)
    rays = np.hstack([und, np.ones((len(und), 1))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def ray_to_pixel(rays, K, D):
    """카메라 좌표계 광선 (...,3) → 왜곡된 픽셀 좌표 (...,2). 벡터화된 fisheye 투영."""
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    r_plane = np.sqrt(x * x + y * y)
    theta = np.arctan2(r_plane, z)  # 광축으로부터의 각도
    k1, k2, k3, k4 = D.ravel()
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r_plane > 1e-9, theta_d / np.maximum(r_plane, 1e-9), 1.0)
    u = K[0, 0] * x * scale + K[0, 2]
    v = K[1, 1] * y * scale + K[1, 2]
    # 뒤쪽(θ>100°)은 무효 처리
    valid = theta < np.deg2rad(100)
    return np.stack([u, v], axis=-1), valid


def kabsch(a, b):
    """b ≈ R @ a 가 되는 회전 R 추정 (a, b: (N,3) 단위 광선)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1.0, 1.0, d])
    return Vt.T @ S @ U.T


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


def match_overlap(img_l, img_r, overlap_frac=0.5):
    """L 오른쪽 / R 왼쪽 영역에서 SIFT 매칭. 반환: 원본 픽셀 좌표 (N,2) 쌍."""
    h, w = img_l.shape[:2]
    cut = int(w * (1 - overlap_frac))
    gray_l = cv2.cvtColor(img_l[:, cut:], cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r[:, : w - cut], cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=8000)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(des_l, des_r, k=2)
    good = [m for m, s in raw if m.distance < 0.75 * s.distance]

    pts_l = np.array([kp_l[m.queryIdx].pt for m in good]) + [cut, 0]
    pts_r = np.array([kp_r[m.trainIdx].pt for m in good])
    return pts_l, pts_r


def half_rotation(R):
    """회전 R 의 절반 회전 (rotation vector 스케일링)."""
    rvec, _ = cv2.Rodrigues(R)
    Rh, _ = cv2.Rodrigues(rvec * 0.5)
    return Rh


def build_cylindrical_maps(K, D, R_cam, out_w, out_h, yaw0, yaw1, el0, el1):
    """출력 원통 픽셀 → 소스 어안 픽셀 remap 테이블.

    R_cam: 월드 광선 → 카메라 광선 회전.
    수평: 등각 (yaw0~yaw1), 수직: y = tan(elevation) (원통 표면).
    """
    yaw = np.linspace(yaw0, yaw1, out_w, dtype=np.float64)
    t = np.linspace(np.tan(el1), np.tan(el0), out_h, dtype=np.float64)  # 위→아래
    yy_t, xx_yaw = np.meshgrid(t, yaw, indexing="ij")
    # 월드 광선: x=sin(yaw), y=tan(el) (아래 +), z=cos(yaw)
    rays = np.stack(
        [np.sin(xx_yaw), -yy_t, np.cos(xx_yaw)], axis=-1
    )  # 이미지 y 는 아래가 + 이므로 elevation 부호 반전
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cam_rays = rays @ R_cam.T
    pix, valid = ray_to_pixel(cam_rays, K, D)
    map_x = pix[..., 0].astype(np.float32)
    map_y = pix[..., 1].astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y


def main():
    left_path, right_path, profile_path, out_path = sys.argv[1:5]
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    K, D, dim = load_lens_profile(profile_path)
    assert img_l.shape[1] == dim[0], "프레임 해상도가 프로파일과 다름"

    print("특징점 매칭 중...")
    pts_l, pts_r = match_overlap(img_l, img_r)
    print(f"  매칭 후보: {len(pts_l)}")

    rays_l = pixel_to_ray(pts_l, K, D)
    rays_r = pixel_to_ray(pts_r, K, D)
    R_lr, inliers, errs = estimate_relative_rotation(rays_l, rays_r)
    yaw_split = np.rad2deg(np.linalg.norm(cv2.Rodrigues(R_lr)[0]))
    print(f"  인라이어: {inliers.sum()}/{len(inliers)}, 잔차 중앙값 {np.median(errs):.3f}°")
    print(f"  카메라 간 상대 회전: {yaw_split:.2f}°")

    # 월드 좌표계: 두 카메라의 중간 방향
    # rays_l = R_lr @ rays_r = R_wl @ R_wr^-1 @ rays_r  →  R_wl = sqrt(R_lr), R_wr = sqrt(R_lr)^-1
    Rh = half_rotation(R_lr)
    R_wl = Rh    # 월드 → L 카메라
    R_wr = Rh.T  # 월드 → R 카메라

    # 수평(leveling) 보정: 월드 좌표계를 위로 pitch_deg 만큼 회전
    # (카메라 평균 방향이 아래를 향하므로 지평선을 출력 상단 부근으로 되돌림)
    pitch_deg = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    c, s = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    R_adj = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R_wl = R_wl @ R_adj
    R_wr = R_wr @ R_adj

    f = K[0, 0]
    half_hfov = np.deg2rad(62)
    yaw_range = np.deg2rad(yaw_split / 2) + half_hfov
    yaw0, yaw1 = -yaw_range, yaw_range
    el0, el1 = np.deg2rad(-38), np.deg2rad(10)
    out_w = int((yaw1 - yaw0) * f)
    out_h = int((np.tan(el1) - np.tan(el0)) * f)
    print(f"출력 해상도: {out_w}x{out_h}  (수평 {np.rad2deg(yaw1-yaw0):.0f}°)")

    print("remap 테이블 생성 중...")
    warped, masks = [], []
    for img, R_cam in [(img_l, R_wl), (img_r, R_wr)]:
        mx, my = build_cylindrical_maps(K, D, R_cam, out_w, out_h, yaw0, yaw1, el0, el1)
        w = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
        m = cv2.remap(np.ones(img.shape[:2], np.uint8) * 255, mx, my, cv2.INTER_NEAREST, borderValue=0)
        warped.append(w)
        masks.append(m)

    # 게인 보정: 겹침 영역 채널 평균을 기하평균으로 정렬
    overlap = (masks[0] > 0) & (masks[1] > 0)
    print(f"겹침 픽셀: {overlap.sum():,}")
    if overlap.sum() > 1000:
        mean_l = warped[0][overlap].reshape(-1, 3).mean(axis=0)
        mean_r = warped[1][overlap].reshape(-1, 3).mean(axis=0)
        target = np.sqrt(mean_l * mean_r)
        gain_l, gain_r = target / mean_l, target / mean_r
        print(f"  게인 L={gain_l.round(3)}, R={gain_r.round(3)}")
        warped[0] = np.clip(warped[0] * gain_l, 0, 255).astype(np.uint8)
        warped[1] = np.clip(warped[1] * gain_r, 0, 255).astype(np.uint8)

    # 페더 블렌딩: 경계까지 거리를 FEATHER_PX 로 포화시켜 심 주변만 섞이게 함
    print("블렌딩 중...")
    FEATHER_PX = 120
    dist_l = cv2.distanceTransform((masks[0] > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_r = cv2.distanceTransform((masks[1] > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_l = np.minimum(dist_l, FEATHER_PX)
    dist_r = np.minimum(dist_r, FEATHER_PX)
    wsum = dist_l + dist_r
    wsum[wsum == 0] = 1
    w_l = (dist_l / wsum)[..., None]
    pano = (warped[0].astype(np.float32) * w_l + warped[1].astype(np.float32) * (1 - w_l))
    pano = np.clip(pano, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
