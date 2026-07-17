"""세션 정합: 특징점 기반 상대 회전 + 자동 수평/센터링 (+ 사용자 오프셋).

Alignment 는 자동 추정 결과를 보존하고, 사용자 오프셋(pitch/roll/yaw)은
rotations()/window() 에서 즉시 적용된다 — 슬라이더 조정 시 SIFT 재실행 불필요.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import (
    estimate_relative_rotation,
    half_rotation,
    pixel_to_ray,
    rot_xz,
)
from .lens import LensProfile
from .render import render_pano

HALF_HFOV_RAD = np.deg2rad(62)   # 카메라 한 대의 사용 반화각
EL0_RAD = np.deg2rad(-38)        # 출력 수직 범위 (아래)
EL1_RAD = np.deg2rad(10)         # 출력 수직 범위 (위)


@dataclass
class Alignment:
    Rh: np.ndarray            # 절반 회전 (월드 → L, 보정 전)
    yaw_split_deg: float      # 카메라 간 상대 회전 크기
    pitch_auto: float         # 자동 수평 (rad)
    roll_auto: float
    yaw_auto: float           # 자동 센터링 (rad)
    n_matches: int = 0
    n_inliers: int = 0
    residual_deg: float = 0.0
    el0: float = EL0_RAD
    el1: float = EL1_RAD

    def rotations(self, pitch_user_deg=0.0, roll_user_deg=0.0):
        """월드→L, 월드→R 회전 (자동 보정 + 사용자 오프셋 적용)."""
        R_adj = rot_xz(self.pitch_auto + np.deg2rad(pitch_user_deg),
                       self.roll_auto + np.deg2rad(roll_user_deg))
        return self.Rh @ R_adj, self.Rh.T @ R_adj

    def window(self, yaw_user_deg=0.0):
        """출력 yaw 범위 (하프라인 센터링 + 사용자 오프셋)."""
        yaw_c = self.yaw_auto + np.deg2rad(yaw_user_deg)
        yaw_range = np.deg2rad(self.yaw_split_deg / 2) + HALF_HFOV_RAD
        return yaw_c - yaw_range, yaw_c + yaw_range


def match_overlap(img_l, img_r, overlap_frac=0.5):
    """L 오른쪽 / R 왼쪽 영역에서 SIFT 매칭. 반환: 원본 픽셀 좌표 (N,2) 쌍."""
    h, w = img_l.shape[:2]
    cut = int(w * (1 - overlap_frac))
    gray_l = cv2.cvtColor(img_l[:, cut:], cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r[:, : w - cut], cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=8000)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(des_l, des_r, k=2)
    good = [m for m, s in raw if m.distance < 0.75 * s.distance]
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pts_l = np.array([kp_l[m.queryIdx].pt for m in good]) + [cut, 0]
    pts_r = np.array([kp_r[m.trainIdx].pt for m in good])
    return pts_l, pts_r


def detect_far_touchline(pano, yaws, el_of_row):
    """먼 쪽 터치라인 점 (yaw, elevation) 검출.

    조건: 흰색 + 위 구간이 수풀(어두움)/트랙(붉음) + 아래 구간이 잔디(녹색).
    """
    h, w = pano.shape[:2]
    hsv = cv2.cvtColor(pano, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 60) & (hsv[..., 2] > 170)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pts = []
    for x in range(0, w, 4):
        for y in np.flatnonzero(white[:, x]):
            if el_of_row(y) < np.deg2rad(-14):
                break  # 이 아래는 근거리 영역
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
                break
    return np.array(pts)


def _fit_line_coeffs(pts):
    """elevation(yaw) ≈ a + b·cosφ + c·sinφ 강건 피팅."""
    yaw, el = pts[:, 0], pts[:, 1]
    A = np.stack([np.ones_like(yaw), np.cos(yaw), np.sin(yaw)], axis=1)
    coef, *_ = np.linalg.lstsq(A, el, rcond=None)
    resid = np.abs(A @ coef - el)
    keep = resid < max(np.deg2rad(0.8), 2.5 * np.median(resid))
    coef, *_ = np.linalg.lstsq(A[keep], el[keep], rcond=None)
    return coef, keep.sum()


def auto_level(imgs, Rs, lens: LensProfile, yaw_range, scale=0.2, log=print):
    """먼 쪽 터치라인 기반 pitch/roll 자동 추정.

    부호를 관례로 가정하지 않는다: 시험 회전(+2°)의 계수 변화(수치 야코비안)를
    측정해 2x2 선형계를 푼다. (해석적 부호 유도는 roll 발산 이력 있음 — devlog 001)
    """
    f = lens.focal
    el0, el1 = np.deg2rad(-45), np.deg2rad(35)
    out_w = int(2 * yaw_range * f * scale)
    out_h = int((np.tan(el1) - np.tan(el0)) * f * scale)
    t1, t0 = np.tan(el1), np.tan(el0)

    def el_of_row(rows):
        return np.arctan(t1 + (np.asarray(rows, dtype=np.float64) / (out_h - 1)) * (t0 - t1))

    def measure(pitch, roll):
        R_adj = rot_xz(pitch, roll)
        pano = render_pano(imgs, [R @ R_adj for R in Rs], lens, out_w, out_h,
                           -yaw_range, yaw_range, el0, el1, feather_px=30)
        yaws = np.linspace(-yaw_range, yaw_range, out_w)
        pts = detect_far_touchline(pano, yaws, el_of_row)
        if len(pts) < 30:
            return None, len(pts)
        coef, n_in = _fit_line_coeffs(pts)
        return coef, n_in

    pitch, roll = 0.0, 0.0
    delta = np.deg2rad(2.0)
    for it in range(2):
        base, n = measure(pitch, roll)
        if base is None:
            log(f"[auto-level] 터치라인 점 부족 ({n}) — 보정 중단")
            return pitch, roll
        bc0 = base[1:3]
        if np.linalg.norm(bc0) < np.deg2rad(0.15):
            break
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
        log(f"[auto-level] 반복 {it+1}: 인라이어 {n} → pitch {np.rad2deg(pitch):+.2f}°, roll {np.rad2deg(roll):+.2f}°")
    if max(abs(pitch), abs(roll)) >= np.deg2rad(24):
        log("[auto-level] 경고: 보정값이 한계(±25°) 부근 — 정합 기하가 비정상일 가능성")
    return pitch, roll


def find_halfway_line_yaw(imgs, Rs, lens: LensProfile, yaw_range, el0, el1,
                          search_deg=30, scale=0.25):
    """하프라인(설치점 방사 방향 → 원통에서 수직선)의 yaw 검출."""
    f = lens.focal
    out_w = int(2 * yaw_range * f * scale)
    out_h = int((np.tan(el1) - np.tan(el0)) * f * scale)
    pano = render_pano(imgs, Rs, lens, out_w, out_h, -yaw_range, yaw_range, el0, el1,
                       feather_px=30)
    hsv = cv2.cvtColor(pano, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).astype(np.float32)
    white = white[out_h // 3:, :]
    yaws = np.linspace(-yaw_range, yaw_range, out_w)
    col_score = white.sum(axis=0)
    col_score[np.abs(yaws) > np.deg2rad(search_deg)] = 0
    col_score = cv2.GaussianBlur(col_score.reshape(1, -1), (1, 31), 0).ravel()
    peak = int(np.argmax(col_score))
    if col_score[peak] <= 0:
        return 0.0
    return float(yaws[peak])


def estimate_alignment(img_l, img_r, lens: LensProfile, log=print,
                       reuse_level: "Alignment | None" = None) -> Alignment:
    """프레임 쌍에서 전체 정합 추정 (상대 회전 + 자동 수평 + 자동 센터링).

    reuse_level 이 주어지면 수평(pitch/roll)·센터링(yaw)은 그 값을 그대로 쓰고
    상대 회전만 재추정한다 — 한 경기 안에서 수평이 바뀌는 일은 거의 없고,
    재추정 노이즈로 세그먼트마다 뷰가 미세하게 달라지는 것을 막는다.
    """
    pts_l, pts_r = match_overlap(img_l, img_r)
    if len(pts_l) < 20:
        raise RuntimeError(f"겹침 영역 매칭 부족 ({len(pts_l)}쌍) — 프레임을 바꿔보세요")
    rays_l = pixel_to_ray(pts_l, lens)
    rays_r = pixel_to_ray(pts_r, lens)
    R_lr, inliers, errs = estimate_relative_rotation(rays_l, rays_r)
    yaw_split = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(R_lr)[0])))
    log(f"[align] 매칭 {len(pts_l)} → 인라이어 {inliers.sum()}, 잔차 {np.median(errs):.3f}°, 상대회전 {yaw_split:.2f}°")
    if inliers.sum() < 30:
        # 인라이어 소수의 RANSAC 해는 겉보기 잔차가 작아도 회전이 엉터리다
        # (실사례: 3/108 인라이어 → 상대회전 99.7°, auto-level ±25° 발산, 검은 화면)
        raise RuntimeError(
            f"정합 실패: 인라이어 {inliers.sum()}개 (최소 30) — 두 카메라가 같은 "
            "장면을 안정적으로 보는 경기 중 프레임에서 다시 시도하세요. "
            "동기화 오프셋이 맞는지도 확인 (1번 탭 '오디오 자동 동기화')")

    Rh = half_rotation(R_lr)
    yaw_range = np.deg2rad(yaw_split / 2) + HALF_HFOV_RAD

    if reuse_level is not None:
        pitch, roll = reuse_level.pitch_auto, reuse_level.roll_auto
        yaw_c = reuse_level.yaw_auto
        log(f"[align] 수평/센터링 기존 값 재사용: pitch {np.rad2deg(pitch):+.2f}° "
            f"roll {np.rad2deg(roll):+.2f}° yaw {np.rad2deg(yaw_c):+.2f}°")
    else:
        pitch, roll = auto_level([img_l, img_r], [Rh, Rh.T], lens, yaw_range, log=log)
        R_adj = rot_xz(pitch, roll)
        yaw_c = find_halfway_line_yaw([img_l, img_r], [Rh @ R_adj, Rh.T @ R_adj],
                                      lens, yaw_range, EL0_RAD, EL1_RAD)
        log(f"[align] 수평 pitch {np.rad2deg(pitch):+.2f}° roll {np.rad2deg(roll):+.2f}°, 하프라인 {np.rad2deg(yaw_c):+.2f}°")

    return Alignment(
        Rh=Rh, yaw_split_deg=yaw_split,
        pitch_auto=float(pitch), roll_auto=float(roll), yaw_auto=float(yaw_c),
        n_matches=len(pts_l), n_inliers=int(inliers.sum()),
        residual_deg=float(np.median(errs)),
    )
