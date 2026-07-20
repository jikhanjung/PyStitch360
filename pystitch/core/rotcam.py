"""회전 카메라 (AX700) 필드 정합 수학 (P06-3).

고정 위치 삼각대 + 팬/틸트(+드문 줌) 카메라의 표준 원근 모델:
  x_cam = R·X_world + t,  K = [[f,0,cx],[0,f,cy],[0,0,1]]
world = 필드 좌표 (X=길이 방향, Y=폭 방향, Z=위, 원점 센터마크).

- 기준 프레임: 랜드마크로 f·자세·설치 위치 추정 (f 는 1D 탐색 + PnP).
- 프레임 간: H = K′·R_rel·K⁻¹ 분해로 회전과 줌 비율 동시 추정
  (삼각대 = 병진 0 → 순수 회전 호모그래피가 정확한 모델).
- 지면 투영: 광선-평면(z=0) 교차.

체인 추적의 누적 드리프트는 랜드마크/흰 라인 재정렬로 주기 보정
(Windows 단계) — 여기는 수학 코어와 합성 검증까지.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_K(f, w, h):
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]])


def calibrate_reference(px_pts, field_pts, img_size, f_frac=(0.4, 3.0),
                        steps=40):
    """기준 프레임 랜드마크 → {f, K, R, t, cam_pos, rms_px}.

    px_pts: 이미지 픽셀 [(u, v)], field_pts: 필드 [(X, Y)] (z=0),
    img_size: (w, h). f 를 [f_frac]×폭 구간에서 탐색(거친 스캔 →
    황금분할)하며 평면 PnP 재투영 오차 최소화. 점 4개 이상.
    """
    w, h = img_size
    obj = np.array([[x, y, 0.0] for x, y in field_pts], np.float64)
    img = np.array(px_pts, np.float64)
    if len(obj) < 4:
        return None

    def solve(f):
        K = make_K(f, w, h)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.inf, None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return rms, (rvec, tvec)

    fs = np.linspace(f_frac[0] * w, f_frac[1] * w, steps)
    errs = [solve(f)[0] for f in fs]
    i = int(np.argmin(errs))
    lo = fs[max(i - 1, 0)]
    hi = fs[min(i + 1, len(fs) - 1)]
    g = (np.sqrt(5) - 1) / 2                     # 황금분할
    a, b = lo, hi
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if solve(c)[0] < solve(d)[0]:
            b = d
        else:
            a = c
    f = 0.5 * (a + b)
    rms, sol = solve(f)
    if sol is None or not np.isfinite(rms):
        return None
    rvec, tvec = sol
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    cam_pos = (-R.T @ t)
    return {"f": float(f), "K": make_K(f, w, h), "R": R, "t": t,
            "cam_pos": cam_pos, "rms_px": rms,
            "img_size": (int(w), int(h))}


def decompose_H(H, K, ratio_range=(0.7, 1.4), steps=36):
    """프레임 간 호모그래피 → (R_rel, f_ratio, 잔차).

    H ≈ K′·R·K⁻¹ (K′ = f×ratio). ratio 를 1D 탐색하며
    G = K′⁻¹HK 의 정규직교 잔차를 최소화, R 은 SVD 최근접 회전.
    """
    H = np.asarray(H, np.float64)
    Kinv_p = None

    def ortho_res(r):
        K2 = K.copy()
        K2[0, 0] *= r
        K2[1, 1] *= r
        G = np.linalg.inv(K2) @ H @ K
        s = np.linalg.svd(G, compute_uv=False)[1]   # 중간 특이값 = 스케일
        Gn = G / max(s, 1e-12)
        return float(np.linalg.norm(Gn @ Gn.T - np.eye(3))), Gn

    rs = np.linspace(ratio_range[0], ratio_range[1], steps)
    errs = [ortho_res(r)[0] for r in rs]
    i = int(np.argmin(errs))
    a, b = rs[max(i - 1, 0)], rs[min(i + 1, len(rs) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(35):
        c, d = b - g * (b - a), a + g * (b - a)
        if ortho_res(c)[0] < ortho_res(d)[0]:
            b = d
        else:
            a = c
    ratio = 0.5 * (a + b)
    res, Gn = ortho_res(ratio)
    U, _s, Vt = np.linalg.svd(Gn)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, float(ratio), res


def pixel_to_field(K, R, cam_pos, pts_px):
    """픽셀 → 지면(z=0) 필드 좌표 (N,2) — 수평선 위는 NaN."""
    p = np.asarray(pts_px, np.float64).reshape(-1, 2)
    ones = np.ones((len(p), 1))
    rays = (np.linalg.inv(K) @ np.hstack([p, ones]).T).T   # 카메라 좌표
    d = rays @ R                                            # = R.T @ ray
    out = np.full((len(p), 2), np.nan)
    m = d[:, 2] < -1e-9                                     # 지면 방향만
    lam = -cam_pos[2] / d[m, 2]
    out[m] = cam_pos[None, :2] + lam[:, None] * d[m, :2]
    return out


def field_to_pixel(K, R, t, pts_field):
    """필드 (X, Y) → 픽셀 (N,2) — 카메라 뒤는 NaN."""
    P = np.asarray(pts_field, np.float64).reshape(-1, 2)
    X = np.hstack([P, np.zeros((len(P), 1))])
    xc = (R @ X.T).T + t[None]
    out = np.full((len(P), 2), np.nan)
    m = xc[:, 2] > 1e-9
    uv = (K @ xc[m].T).T
    out[m] = uv[:, :2] / uv[:, 2:3]
    return out


def anchor_rotation(cam_pos, px_pts, field_pts, f_init, img_size,
                    f_span=0.15):
    """현재 프레임 랜드마크 → (R, f) 절대 재정렬 (설치 위치 고정).

    체인 추적(track_step)의 누적 드리프트 제거용 — world 광선
    (cam_pos→랜드마크)과 카메라 광선(K⁻¹p)을 Kabsch(직교 프로크루스테스)
    로 정렬, f 는 f_init±span 황금분할. 랜드마크 3개 이상.
    반환: {"R", "f", "K", "res_deg"} 또는 None.
    """
    w, h = img_size
    P = np.asarray(field_pts, np.float64).reshape(-1, 2)
    px = np.asarray(px_pts, np.float64).reshape(-1, 2)
    if len(P) < 3:
        return None
    Xw = np.hstack([P, np.zeros((len(P), 1))]) - np.asarray(cam_pos)[None]
    rw = Xw / np.linalg.norm(Xw, axis=1, keepdims=True)

    def solve(f):
        K = make_K(f, w, h)
        rc = (np.linalg.inv(K)
              @ np.hstack([px, np.ones((len(px), 1))]).T).T
        rc = rc / np.linalg.norm(rc, axis=1, keepdims=True)
        M = rc.T @ rw
        U, _s, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        cosang = np.clip(np.sum(rc * (rw @ R.T), axis=1), -1.0, 1.0)
        return float(np.rad2deg(np.mean(np.arccos(cosang)))), R

    a, b = f_init * (1 - f_span), f_init * (1 + f_span)
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if solve(c)[0] < solve(d)[0]:
            b = d
        else:
            a = c
    f = 0.5 * (a + b)
    res, R = solve(f)
    return {"R": R, "f": float(f), "K": make_K(f, w, h),
            "res_deg": res}


def match_frames(img_a, img_b, nfeatures=3000, ratio=0.75):
    """두 프레임 SIFT 매칭 → (N,2), (N,2) — 회전 추적 입력."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    ka, da = sift.detectAndCompute(img_a, None)
    kb, db = sift.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for m, n in bf.knnMatch(da, db, k=2):
        if m.distance < ratio * n.distance:
            pairs.append((ka[m.queryIdx].pt, kb[m.trainIdx].pt))
    if not pairs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return a, b


def track_step(prev_state, pts_prev, pts_cur, min_matches=25,
               max_ortho_res=0.08):
    """이전 상태 {R, f} + 매칭 → 새 상태 (실패 시 None — 품질 플래그).

    H 는 RANSAC 으로 추정, 분해 잔차가 크면(가림·블러·모델 위반) 기각.
    """
    if len(pts_prev) < min_matches:
        return None
    H, mask = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3.0)
    if H is None or mask is None or mask.sum() < min_matches:
        return None
    K = prev_state["K"]
    R_rel, ratio, res = decompose_H(H, K)
    if res > max_ortho_res:
        return None
    f2 = prev_state["f"] * ratio
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = f2
    return {"f": float(f2), "K": K2,
            "R": R_rel @ prev_state["R"],
            "ortho_res": res, "n_inliers": int(mask.sum())}
