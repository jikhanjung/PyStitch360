"""경기장 캘리브레이션: 원통 파노라마 ↔ 경기장 절대 좌표.

랜드마크(코너 플래그 4, 중앙선×사이드라인 2, 중앙선×센터서클 2)를
파노라마에서 찍으면 카메라 모델 7파라미터(높이, 상/하 엘리베이션,
요 오프셋, 헤딩, 필드 내 위치)를 Gauss-Newton(LM 감쇠)으로 피팅한다.
일부 랜드마크는 생략 가능 — 최소 4점이면 풀 수 있다.

경기장 좌표계: 원점 = 센터 마크, X = 터치라인 방향(오른쪽 +, m),
Y = 카메라 반대편(먼 쪽) + (m). 카메라는 근처 터치라인 바깥(Y < -폭/2).

원통 투영 모델은 core.ptz.ground_positions 와 동일: 열 ↔ yaw 선형,
행 ↔ tan(elevation) 선형, f = H / (tan el_top − tan el_bottom).
"""
from __future__ import annotations

import numpy as np

CENTER_CIRCLE_R = 9.15

# (키, 표시 이름, 필수 여부) — 먼쪽 코너 둘은 시야 양 끝을 고정하므로 필수.
LANDMARKS = [
    ("corner_far_l",  "먼쪽 왼쪽 코너", True),
    ("corner_far_r",  "먼쪽 오른쪽 코너", True),
    ("corner_near_l", "가까운 왼쪽 코너", False),
    ("corner_near_r", "가까운 오른쪽 코너", False),
    ("half_far",      "중앙선 × 먼쪽 사이드라인", False),
    ("half_near",     "중앙선 × 가까운 사이드라인", False),
    ("circle_far",    "센터서클 × 중앙선 (먼쪽)", False),
    ("circle_near",   "센터서클 × 중앙선 (가까운쪽)", False),
]


def landmark_positions(length=105.0, width=68.0):
    """랜드마크의 경기장 좌표 {키: (X, Y)} (m)."""
    hl, hw = length / 2.0, width / 2.0
    return {
        "corner_far_l":  (-hl,  hw),
        "corner_far_r":  ( hl,  hw),
        "corner_near_l": (-hl, -hw),
        "corner_near_r": ( hl, -hw),
        "half_far":      (0.0,  hw),
        "half_near":     (0.0, -hw),
        "circle_far":    (0.0,  CENTER_CIRCLE_R),
        "circle_near":   (0.0, -CENTER_CIRCLE_R),
    }


def _project(p, fxy, pano_w, pano_h):
    """필드 좌표 (N,2) → 파노라마 픽셀 (N,2). 카메라 뒤/수평선 위는 NaN."""
    h, t_top, t_bot, phi0, theta, ex, ey = p
    d = fxy - np.array([ex, ey])
    ct, st = np.cos(theta), np.sin(theta)
    Xc = ct * d[:, 0] - st * d[:, 1]
    Yc = st * d[:, 0] + ct * d[:, 1]
    dist = np.hypot(Xc, Yc)
    phi = np.arctan2(Xc, Yc)
    t = np.where(dist > 1e-6, -h / np.maximum(dist, 1e-6), np.nan)
    f = pano_h / (t_top - t_bot)
    span = pano_w / f
    x = ((phi - phi0) / span + 0.5) * (pano_w - 1)
    y = (t_top - t) / (t_top - t_bot) * (pano_h - 1)
    return np.stack([x, y], axis=1)


def fit_field_calibration(points, pano_w, pano_h,
                          length=105.0, width=68.0, iters=200):
    """찍은 랜드마크 {키: (px, py)} 로 카메라 모델 피팅.

    최소 4점 필요 (7 파라미터 / 점당 2 방정식). 반환: calib dict
    (파라미터 + rms 픽셀 잔차) 또는 None (점 부족/발산).
    """
    pos = landmark_positions(length, width)
    keys = [k for k in points if k in pos]
    if len(keys) < 4:
        return None
    fxy = np.array([pos[k] for k in keys], float)
    pxy = np.array([points[k] for k in keys], float)
    p = np.array([4.0, np.tan(np.deg2rad(10.0)), np.tan(np.deg2rad(-38.0)),
                  0.0, 0.0, 0.0, -(width / 2.0 + 5.0)])
    step = np.array([0.01, 1e-4, 1e-4, 1e-4, 1e-4, 0.01, 0.01])
    lam = 1e-3

    def residual(pp):
        r = (_project(pp, fxy, pano_w, pano_h) - pxy).ravel()
        return np.where(np.isfinite(r), r, 1e6)

    r = residual(p)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), len(p)))
        for j in range(len(p)):
            dp = np.zeros_like(p)
            dp[j] = step[j]
            J[:, j] = (residual(p + dp) - r) / step[j]
        A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
        try:
            delta = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            return None
        p_new = p + delta
        # 물리 제약: 높이 0.5~20m, 수평선 위/아래 부호 유지
        p_new[0] = np.clip(p_new[0], 0.5, 20.0)
        p_new[1] = max(p_new[1], p_new[2] + 1e-3)
        r_new = residual(p_new)
        c_new = float(r_new @ r_new)
        if c_new < cost:
            p, r, cost = p_new, r_new, c_new
            lam = max(lam * 0.5, 1e-9)
            if np.abs(delta).max() < 1e-8:
                break
        else:
            lam *= 4.0
            if lam > 1e8:
                break
    rms = float(np.sqrt(cost / len(r)))
    if not np.isfinite(rms) or rms > 200.0:      # 수렴 실패로 간주
        return None
    h, t_top, t_bot, phi0, theta, ex, ey = (float(v) for v in p)
    return {"h": h, "t_top": t_top, "t_bot": t_bot, "phi0": phi0,
            "theta": theta, "ex": ex, "ey": ey,
            "length": float(length), "width": float(width),
            "pano_w": int(pano_w), "pano_h": int(pano_h),
            "n_points": len(keys), "rms": rms}


def _params(calib):
    return np.array([calib["h"], calib["t_top"], calib["t_bot"],
                     calib["phi0"], calib["theta"], calib["ex"], calib["ey"]])


def field_to_pano(calib, fxy):
    """경기장 좌표 (N,2) m → 파노라마 픽셀 (N,2)."""
    return _project(_params(calib), np.asarray(fxy, float),
                    calib["pano_w"], calib["pano_h"])


def pano_to_field(calib, pxy):
    """파노라마 픽셀 (N,2) → 경기장 좌표 (N,2) m. 수평선 위는 NaN."""
    h, t_top, t_bot, phi0, theta, ex, ey = _params(calib)
    pxy = np.asarray(pxy, float)
    W, H = calib["pano_w"], calib["pano_h"]
    f = H / (t_top - t_bot)
    span = W / f
    phi = (pxy[:, 0] / (W - 1) - 0.5) * span + phi0
    t = t_top - pxy[:, 1] / (H - 1) * (t_top - t_bot)
    d = np.where(t < -1e-4, h / np.maximum(-t, 1e-9), np.nan)
    Xc, Yc = d * np.sin(phi), d * np.cos(phi)
    ct, st = np.cos(theta), np.sin(theta)
    return np.stack([ex + ct * Xc + st * Yc,
                     ey - st * Xc + ct * Yc], axis=1)


def field_outline(length=105.0, width=68.0, step=2.0):
    """미리보기 오버레이용 경기장 선 폴리라인 목록 [(N,2) 필드 좌표, ...].

    외곽 사각형, 중앙선, 센터서클 — 원통 투영에서 곡선이 되므로
    step(m) 간격으로 샘플링해 잇는다.
    """
    hl, hw = length / 2.0, width / 2.0
    lines = []
    for (x0, y0), (x1, y1) in [((-hl, -hw), (hl, -hw)), ((hl, -hw), (hl, hw)),
                               ((hl, hw), (-hl, hw)), ((-hl, hw), (-hl, -hw)),
                               ((0.0, -hw), (0.0, hw))]:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
        lines.append(np.stack([np.linspace(x0, x1, n),
                               np.linspace(y0, y1, n)], axis=1))
    a = np.linspace(0, 2 * np.pi, 64)
    lines.append(np.stack([CENTER_CIRCLE_R * np.sin(a),
                           CENTER_CIRCLE_R * np.cos(a)], axis=1))
    return lines
