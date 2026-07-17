"""정지 프레임 원통 파노라마 스티칭 프로토타입.

파이프라인:
  1. Gyroflow 렌즈 프로파일(OpenCV fisheye 모델) 로드
  2. 좌/우 프레임의 겹침 영역에서 SIFT 특징점 매칭
  3. 매칭점을 광선(ray)으로 변환 → RANSAC + Kabsch 로 상대 회전 추정
  4. 자동 보정 (+ 사용자 오프셋):
     - 수평(pitch/roll): 먼 쪽 터치라인 검출 → 코사인/사인 기저 피팅
     - 센터링(yaw): 하프라인(중앙 수직 백선) 검출
  5. 출력 원통 좌표 → 각 카메라 어안 픽셀로의 remap (왜곡 소스에서 직접 1회 리샘플)
  6. 겹침 영역 게인 보정 + 페더 블렌딩

사용법:
  python stitch_still.py L.png R.png --profile lens.json -o out.jpg \
      [--pitch d] [--roll d] [--yaw d]   # 자동 추정값에 더해지는 사용자 보정 (도)
"""
import argparse
import json

import cv2
import numpy as np


# ---------------------------------------------------------------- 렌즈/광선

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
    theta = np.arctan2(r_plane, z)
    k1, k2, k3, k4 = D.ravel()
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(r_plane > 1e-9, theta_d / np.maximum(r_plane, 1e-9), 1.0)
    u = K[0, 0] * x * scale + K[0, 2]
    v = K[1, 1] * y * scale + K[1, 2]
    valid = theta < np.deg2rad(100)
    return np.stack([u, v], axis=-1), valid


# ---------------------------------------------------------------- 회전 추정

def kabsch(a, b):
    """b ≈ R @ a 가 되는 회전 R 추정 (a, b: (N,3) 단위 광선)."""
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
    rvec, _ = cv2.Rodrigues(R)
    Rh, _ = cv2.Rodrigues(rvec * 0.5)
    return Rh


def rot_xz(pitch_rad, roll_rad):
    """월드 보정 회전: x축(pitch) 후 z축(roll)."""
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rx @ Rz


# ---------------------------------------------------------------- 렌더링

def build_cylindrical_maps(K, D, R_cam, out_w, out_h, yaw0, yaw1, el0, el1):
    """출력 원통 픽셀 → 소스 어안 픽셀 remap 테이블.

    R_cam: 월드 광선 → 카메라 광선 회전.
    수평: 등각 (yaw0~yaw1), 수직: y = tan(elevation) (원통 표면).
    """
    yaw = np.linspace(yaw0, yaw1, out_w, dtype=np.float64)
    t = np.linspace(np.tan(el1), np.tan(el0), out_h, dtype=np.float64)  # 위→아래
    yy_t, xx_yaw = np.meshgrid(t, yaw, indexing="ij")
    rays = np.stack([np.sin(xx_yaw), -yy_t, np.cos(xx_yaw)], axis=-1)  # 이미지 y는 아래 +
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cam_rays = rays @ R_cam.T
    pix, valid = ray_to_pixel(cam_rays, K, D)
    map_x = pix[..., 0].astype(np.float32)
    map_y = pix[..., 1].astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y


def compute_gains(warped_l, warped_r, mask_l, mask_r):
    overlap = (mask_l > 0) & (mask_r > 0)
    if overlap.sum() < 1000:
        return np.ones(3), np.ones(3)
    mean_l = warped_l[overlap].reshape(-1, 3).mean(axis=0)
    mean_r = warped_r[overlap].reshape(-1, 3).mean(axis=0)
    target = np.sqrt(mean_l * mean_r)
    return target / mean_l, target / mean_r


def feather_weights(mask_l, mask_r, feather_px=120):
    dist_l = np.minimum(cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    dist_r = np.minimum(cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 3), feather_px)
    wsum = dist_l + dist_r
    wsum[wsum == 0] = 1
    return (dist_l / wsum).astype(np.float32)


def render_pano(imgs, Rs, K, D, out_w, out_h, yaw0, yaw1, el0, el1, feather_px=120):
    """두 이미지를 원통 파노라마로 워핑 + 게인 보정 + 페더 블렌딩 (1회용)."""
    warped, masks = [], []
    for img, R_cam in zip(imgs, Rs):
        mx, my = build_cylindrical_maps(K, D, R_cam, out_w, out_h, yaw0, yaw1, el0, el1)
        warped.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0))
        masks.append(cv2.remap(np.ones(img.shape[:2], np.uint8) * 255, mx, my,
                               cv2.INTER_NEAREST, borderValue=0))
    g_l, g_r = compute_gains(warped[0], warped[1], masks[0], masks[1])
    w_l = feather_weights(masks[0], masks[1], feather_px)[..., None]
    pano = (warped[0].astype(np.float32) * (w_l * g_l)
            + warped[1].astype(np.float32) * ((1 - w_l) * g_r))
    return np.clip(pano, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 자동 보정

def detect_far_touchline(pano, yaws, el_of_row):
    """저해상도 파노라마에서 먼 쪽 터치라인 점들 (yaw, elevation) 을 검출.

    조건: 흰색 픽셀 + 위 구간이 수풀(어두움) 또는 트랙(붉음) + 아래 구간이 잔디(녹색).
    근처 터치라인(아래가 트랙)과 트랙 레인 라인(위아래 모두 트랙)을 배제한다.
    """
    h, w = pano.shape[:2]
    hsv = cv2.cvtColor(pano, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 60) & (hsv[..., 2] > 170)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pts = []
    for x in range(0, w, 4):
        for y in np.flatnonzero(white[:, x]):
            if el_of_row(y) < np.deg2rad(-14):
                break  # 이보다 아래는 근거리 영역 — 먼 터치라인일 수 없음
            y0, y1 = max(0, y - 14), min(h, y + 16)
            if y - 2 <= y0 or y + 3 >= y1:
                continue
            above_v = val[y0 : y - 2, x].astype(np.float32)
            above_h, above_s = hue[y0 : y - 2, x], sat[y0 : y - 2, x]
            below_h, below_s, below_v = hue[y + 3 : y1, x], sat[y + 3 : y1, x], val[y + 3 : y1, x]
            dark_above = (above_v < 130).mean() > 0.6
            red_above = ((above_s > 60) & ((above_h < 15) | (above_h > 165))).mean() > 0.5
            grass_below = ((below_h > 35) & (below_h < 95) & (below_s > 40) & (below_v > 50)).mean() > 0.5
            if (dark_above or red_above) and grass_below:
                pts.append((yaws[x], el_of_row(y)))
                break  # 열에서 가장 위 후보만
    return np.array(pts)


def fit_line_coeffs(pts):
    """터치라인 점들의 elevation(yaw) 을 a + b·cos(yaw) + c·sin(yaw) 로 피팅.

    수평이 맞으면 먼 수평선은 elevation 이 거의 상수 → b≈0, c≈0 이 목표.
    강건성: 1차 피팅 후 잔차 큰 점을 제거하고 재피팅.
    """
    yaw, el = pts[:, 0], pts[:, 1]
    A = np.stack([np.ones_like(yaw), np.cos(yaw), np.sin(yaw)], axis=1)
    coef, *_ = np.linalg.lstsq(A, el, rcond=None)
    resid = np.abs(A @ coef - el)
    keep = resid < max(np.deg2rad(0.8), 2.5 * np.median(resid))
    coef, *_ = np.linalg.lstsq(A[keep], el[keep], rcond=None)
    return coef, keep.sum()


def auto_level(imgs, Rs, K, D, f, yaw_range, scale=0.2):
    """먼 쪽 터치라인 기반 pitch/roll 자동 추정.

    보정 부호를 관례로 가정하지 않는다: 시험 회전(+2°)을 가해 (b, c) 계수의
    변화(수치 야코비안)를 측정하고 2x2 선형계를 풀어 보정량을 구한다.
    """
    el0, el1 = np.deg2rad(-45), np.deg2rad(35)  # 검출용 넓은 수직 범위
    out_w = int(2 * yaw_range * f * scale)
    out_h = int((np.tan(el1) - np.tan(el0)) * f * scale)
    t1, t0 = np.tan(el1), np.tan(el0)

    def el_of_row(rows):
        return np.arctan(t1 + (np.asarray(rows, dtype=np.float64) / (out_h - 1)) * (t0 - t1))

    def measure(pitch, roll):
        R_adj = rot_xz(pitch, roll)
        pano = render_pano(imgs, [R @ R_adj for R in Rs], K, D, out_w, out_h,
                           -yaw_range, yaw_range, el0, el1, feather_px=30)
        yaws = np.linspace(-yaw_range, yaw_range, out_w)
        pts = detect_far_touchline(pano, yaws, el_of_row)
        if len(pts) < 30:
            return None, len(pts)
        coef, n_in = fit_line_coeffs(pts)
        return coef, n_in

    pitch, roll = 0.0, 0.0
    delta = np.deg2rad(2.0)
    for it in range(2):
        base, n = measure(pitch, roll)
        if base is None:
            print(f"  [auto-level] 터치라인 점 부족 ({n}) — 보정 중단")
            return pitch, roll
        bc0 = base[1:3]
        if np.linalg.norm(bc0) < np.deg2rad(0.15):
            break  # 이미 수평
        mp, _ = measure(pitch + delta, roll)
        mr, _ = measure(pitch, roll + delta)
        if mp is None or mr is None:
            break
        J = np.stack([(mp[1:3] - bc0) / delta, (mr[1:3] - bc0) / delta], axis=1)
        try:
            step = np.linalg.solve(J, -bc0)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -np.deg2rad(25), np.deg2rad(25))
        pitch += step[0]
        roll += step[1]
        print(f"  [auto-level] 반복 {it+1}: 인라이어 {n} → pitch {np.rad2deg(pitch):+.2f}°, roll {np.rad2deg(roll):+.2f}°")
    return pitch, roll


def find_halfway_line_yaw(imgs, Rs, K, D, f, yaw_range, el0, el1, search_deg=30, scale=0.25):
    """저해상도 파노라마에서 하프라인(중앙 수직 백선)의 yaw 를 찾는다.

    하프라인은 카메라 설치점에서 방사 방향이라 원통 투영에서 거의 수직선이 됨.
    중앙 ±search_deg 구간에서 열별 흰색 픽셀 합의 피크를 찾는다.
    """
    out_w = int(2 * yaw_range * f * scale)
    out_h = int((np.tan(el1) - np.tan(el0)) * f * scale)
    pano = render_pano(imgs, Rs, K, D, out_w, out_h, -yaw_range, yaw_range, el0, el1,
                       feather_px=30)
    hsv = cv2.cvtColor(pano, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).astype(np.float32)
    white = white[out_h // 3:, :]  # 아래쪽 2/3 (잔디 위 라인 영역)만
    yaws = np.linspace(-yaw_range, yaw_range, out_w)
    col_score = white.sum(axis=0)
    col_score[np.abs(yaws) > np.deg2rad(search_deg)] = 0
    col_score = cv2.GaussianBlur(col_score.reshape(1, -1), (1, 31), 0).ravel()
    peak = int(np.argmax(col_score))
    if col_score[peak] <= 0:
        return 0.0
    return float(yaws[peak])


# ---------------------------------------------------------------- 셋업 (공용)

def setup_alignment(img_l, img_r, K, D, pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                    verbose=True):
    """프레임 쌍에서 스티칭 기하 전체를 추정. 반환: dict (video 파이프라인과 공용)."""
    pts_l, pts_r = match_overlap(img_l, img_r)
    rays_l = pixel_to_ray(pts_l, K, D)
    rays_r = pixel_to_ray(pts_r, K, D)
    R_lr, inliers, errs = estimate_relative_rotation(rays_l, rays_r)
    yaw_split = np.rad2deg(np.linalg.norm(cv2.Rodrigues(R_lr)[0]))
    if verbose:
        print(f"  매칭 {len(pts_l)} → 인라이어 {inliers.sum()}, 잔차 중앙값 {np.median(errs):.3f}°")
        print(f"  카메라 간 상대 회전: {yaw_split:.2f}°")

    Rh = half_rotation(R_lr)
    R_wl, R_wr = Rh, Rh.T  # 월드 → 각 카메라

    f = K[0, 0]
    yaw_range = np.deg2rad(yaw_split / 2) + np.deg2rad(62)

    # 자동 수평 + 사용자 보정
    pitch, roll = auto_level([img_l, img_r], [R_wl, R_wr], K, D, f, yaw_range)
    pitch += np.deg2rad(pitch_user)
    roll += np.deg2rad(roll_user)
    R_adj = rot_xz(pitch, roll)
    R_wl, R_wr = R_wl @ R_adj, R_wr @ R_adj

    # 자동 센터링 + 사용자 보정
    el0, el1 = np.deg2rad(-38), np.deg2rad(10)
    yaw_c = find_halfway_line_yaw([img_l, img_r], [R_wl, R_wr], K, D, f, yaw_range, el0, el1)
    yaw_c += np.deg2rad(yaw_user)
    if verbose:
        print(f"  하프라인 yaw {np.rad2deg(yaw_c):+.2f}° → 중앙 배치")

    return {
        "R_wl": R_wl, "R_wr": R_wr, "f": f,
        "yaw0": yaw_c - yaw_range, "yaw1": yaw_c + yaw_range,
        "el0": el0, "el1": el1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--profile", default="presets/lens_profiles/GoPro_HERO5_Black_Wide_4K_16x9.json")
    ap.add_argument("-o", "--out", default="pano.jpg")
    ap.add_argument("--pitch", type=float, default=0.0, help="수평 사용자 보정 (도)")
    ap.add_argument("--roll", type=float, default=0.0, help="기울기 사용자 보정 (도)")
    ap.add_argument("--yaw", type=float, default=0.0, help="센터링 사용자 보정 (도)")
    args = ap.parse_args()

    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)
    K, D, dim = load_lens_profile(args.profile)
    assert img_l.shape[1] == dim[0], "프레임 해상도가 프로파일과 다름"

    print("정합 추정 중...")
    g = setup_alignment(img_l, img_r, K, D, args.pitch, args.roll, args.yaw)

    out_w = int((g["yaw1"] - g["yaw0"]) * g["f"])
    out_h = int((np.tan(g["el1"]) - np.tan(g["el0"])) * g["f"])
    print(f"출력 해상도: {out_w}x{out_h}  (수평 {np.rad2deg(g['yaw1']-g['yaw0']):.0f}°)")

    pano = render_pano([img_l, img_r], [g["R_wl"], g["R_wr"]], K, D,
                       out_w, out_h, g["yaw0"], g["yaw1"], g["el0"], g["el1"])
    cv2.imwrite(args.out, pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
