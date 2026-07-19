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
PEN_HALF_W = 20.16      # 페널티박스 반폭 (40.32m)
PEN_DEPTH = 16.5        # 페널티박스 깊이

# (키, 표시 이름, 필수 여부) — 리스트 순서 = 권장 찍기 순서.
# 1단계(최외곽 선 위, ~10점): 코너 4 → 중앙선×사이드라인 2 → 가까운
# 사이드라인 보조 2 → 골라인 위 페널티박스 모서리 4. 외곽부터 찍어야
# 초기 카메라 모델의 휴리스틱 매칭이 안정적이고, 이후 내부 점들은
# 이미 피팅된 모델로 정확히 매칭된다. 2단계: 센터서클·페널티박스 내부.
LANDMARKS = [
    ("corner_far_l",  "먼쪽 왼쪽 코너", True),
    ("corner_far_r",  "먼쪽 오른쪽 코너", True),
    ("corner_near_l", "가까운 왼쪽 코너", False),
    ("corner_near_r", "가까운 오른쪽 코너", False),
    ("half_far",      "중앙선 × 먼쪽 사이드라인", False),
    ("half_near",     "중앙선 × 가까운 사이드라인", False),
    ("sideline_near_l", "가까운 사이드라인 위 왼쪽 (선 위 아무 점)", False),
    ("sideline_near_r", "가까운 사이드라인 위 오른쪽 (선 위 아무 점)", False),
    ("pen_l_far",   "왼쪽 골라인 × 페널티박스 (먼쪽)", False),
    ("pen_l_near",  "왼쪽 골라인 × 페널티박스 (가까운쪽)", False),
    ("pen_r_far",   "오른쪽 골라인 × 페널티박스 (먼쪽)", False),
    ("pen_r_near",  "오른쪽 골라인 × 페널티박스 (가까운쪽)", False),
    ("circle_far",  "센터서클 × 중앙선 (먼쪽)", False),
    ("circle_near", "센터서클 × 중앙선 (가까운쪽)", False),
    ("pen_l_box_far",  "왼쪽 페널티박스 안 모서리 (먼쪽)", False),
    ("pen_l_box_near", "왼쪽 페널티박스 안 모서리 (가까운쪽)", False),
    ("pen_r_box_far",  "오른쪽 페널티박스 안 모서리 (먼쪽)", False),
    ("pen_r_box_near", "오른쪽 페널티박스 안 모서리 (가까운쪽)", False),
]

# 위치를 모르는 '선 위의 점' 랜드마크: 카메라 앞 중앙선-사이드라인 교차점이
# 안 보일 때, 크게 확대돼 보이는 가까운 사이드라인 위 아무 점 2개로
# 그 선(Y = -폭/2)을 고정한다. 점당 방정식 1개 (해당 열에서 선까지 행 오차).
LINE_LANDMARKS = {"sideline_near_l", "sideline_near_r"}


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
        "pen_l_far":     (-hl,  PEN_HALF_W),
        "pen_l_near":    (-hl, -PEN_HALF_W),
        "pen_r_far":     ( hl,  PEN_HALF_W),
        "pen_r_near":    ( hl, -PEN_HALF_W),
        "pen_l_box_far":  (-hl + PEN_DEPTH,  PEN_HALF_W),
        "pen_l_box_near": (-hl + PEN_DEPTH, -PEN_HALF_W),
        "pen_r_box_far":  ( hl - PEN_DEPTH,  PEN_HALF_W),
        "pen_r_box_near": ( hl - PEN_DEPTH, -PEN_HALF_W),
    }


# 파라미터 벡터: [h, t_top, t_bot, theta, ex, ey, pitch, roll]
# theta = 헤딩(수직축 회전), pitch/roll = 리그 기울기 — 삼각대가 완전
# 수평이 아니면 지평선이 파노라마에서 휘는데, 이를 잡는 자유도.
# (구 모델의 phi0 은 theta 와 완전 축퇴라 제거.)


def _rot(theta, pitch, roll):
    """리그 → 월드 회전. 월드: X=필드X, Y=위, Z=필드Y."""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def _project(p, fxy, pano_w, pano_h):
    """필드 좌표 (N,2) → 파노라마 픽셀 (N,2) (워프 없는 파라메트릭)."""
    h, t_top, t_bot, theta, ex, ey, pitch, roll = p
    fxy = np.asarray(fxy, float)
    w = np.stack([fxy[:, 0] - ex, np.full(len(fxy), -h),
                  fxy[:, 1] - ey], axis=1)
    r = w @ _rot(theta, pitch, roll)      # = R^T · w (리그 프레임)
    d = np.hypot(r[:, 0], r[:, 2])
    yaw = np.arctan2(r[:, 0], r[:, 2])
    t = np.where(d > 1e-6, r[:, 1] / np.maximum(d, 1e-6), np.nan)
    f = pano_h / (t_top - t_bot)
    span = pano_w / f
    x = (yaw / span + 0.5) * (pano_w - 1)
    y = (t_top - t) / (t_top - t_bot) * (pano_h - 1)
    return np.stack([x, y], axis=1)


def _sideline_rows(p, cols, pano_w, pano_h, width):
    """근처 사이드라인(Y=-폭/2)이 각 열(cols)에서 지나는 예측 행 (N,).

    열의 yaw 광선을 t(행 방향)로 매개화하면 월드 방향이 A + t·B 로
    선형 — 지면 교차와 필드 Y=-폭/2 조건에서 t 가 닫힌형으로 풀린다.
    """
    h, t_top, t_bot, theta, ex, ey, pitch, roll = p
    f = pano_h / (t_top - t_bot)
    span = pano_w / f
    yaw = (np.asarray(cols, float) / (pano_w - 1) - 0.5) * span
    R = _rot(theta, pitch, roll)
    A = np.stack([np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)], axis=1) @ R.T
    B = R[:, 1]
    denom = -width / 2.0 - ey
    k = np.where(np.abs(denom) > 1e-6, -h / denom, np.nan)
    t = (k * A[:, 2] - A[:, 1]) / (B[1] - k * B[2])
    return (t_top - t) / (t_top - t_bot) * (pano_h - 1)


# ------------------------------------------------------------------ TPS 워프
# 파라메트릭 모델이 못 잡는 잔류 왜곡(어안 외곽의 렌즈 잔차, 스티칭 심)
# 을 얇은판 스플라인으로 보정 — 찍은 랜드마크는 정확히 통과한다.

def _tps_phi(r):
    return np.where(r > 1e-9, r * r * np.log(r + 1e-12), 0.0)


def _tps_fit(src, dst_delta, scale):
    """제어점 src(N,2)에서 dst_delta(N,2)를 보간하는 TPS 계수."""
    n = len(src)
    s = src / scale
    d2 = np.linalg.norm(s[:, None] - s[None], axis=2)
    K = _tps_phi(d2) + 1e-8 * np.eye(n)
    P = np.hstack([np.ones((n, 1)), s])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 3, 2))
    b[:n] = dst_delta
    try:
        wa = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return {"src": s, "w": wa[:n], "a": wa[n:], "scale": scale}


def _tps_eval(tps, pts):
    s = np.asarray(pts, float) / tps["scale"]
    d = np.linalg.norm(s[:, None] - tps["src"][None], axis=2)
    out = _tps_phi(d) @ tps["w"]
    out += tps["a"][0] + s @ tps["a"][1:]
    return out


def fit_field_calibration(points, pano_w, pano_h,
                          length=105.0, width=68.0, iters=300):
    """찍은 랜드마크 {키: (px, py)} 로 카메라 모델 + 잔차 워프 피팅.

    위치를 아는 점은 방정식 2개, 선 위의 점(LINE_LANDMARKS)은 1개.
    최소: 위치 점 3개 이상이고 총 방정식 8개 이상. 방정식 10개부터는
    리그 기울기(pitch/roll)도 푼다. 코너 플래그는 가중 3배.

    파라메트릭 피팅 뒤 랜드마크 잔차를 TPS 로 보간해 calib["warp"] 에
    저장 — field_to_pano 가 찍은 점을 정확히 통과한다 (렌즈 잔류
    왜곡·스티칭 심 흡수). 반환: calib dict 또는 None.
    """
    pos = landmark_positions(length, width)
    keys = [k for k in points if k in pos]
    ln_keys = [k for k in points if k in LINE_LANDMARKS]
    n_eq = 2 * len(keys) + len(ln_keys)
    if len(keys) < 3 or n_eq < 8:
        return None
    fxy = np.array([pos[k] for k in keys], float)
    pxy = np.array([points[k] for k in keys], float)
    ln = np.array([points[k] for k in ln_keys], float).reshape(-1, 2)
    wgt = np.array([3.0 if k.startswith("corner") else 1.0 for k in keys])
    p = np.array([4.0, np.tan(np.deg2rad(10.0)), np.tan(np.deg2rad(-38.0)),
                  0.0, 0.0, -(width / 2.0 + 5.0), 0.0, 0.0])
    step = np.array([0.01, 1e-4, 1e-4, 1e-4, 0.01, 0.01, 1e-4, 1e-4])
    free = list(range(6 if n_eq < 10 else 8))    # 기울기는 10방정식부터
    lam = 1e-3

    def residual(pp):
        r = ((_project(pp, fxy, pano_w, pano_h) - pxy)
             * wgt[:, None]).ravel()
        if len(ln):
            rl = _sideline_rows(pp, ln[:, 0], pano_w, pano_h,
                                width) - ln[:, 1]
            r = np.concatenate([r, rl])
        return np.where(np.isfinite(r), r, 1e6)

    r = residual(p)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), len(free)))
        for jj, j in enumerate(free):
            dp = np.zeros_like(p)
            dp[j] = step[j]
            J[:, jj] = (residual(p + dp) - r) / step[j]
        A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
        try:
            delta = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            return None
        p_new = p.copy()
        p_new[free] += delta
        # 물리 제약: 높이 0.5~20m, 수평선 위/아래 부호 유지, 기울기 ±10도
        p_new[0] = np.clip(p_new[0], 0.5, 20.0)
        p_new[1] = max(p_new[1], p_new[2] + 1e-3)
        p_new[6:8] = np.clip(p_new[6:8], -0.175, 0.175)
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
    calib = {"h": float(p[0]), "t_top": float(p[1]), "t_bot": float(p[2]),
             "theta": float(p[3]), "ex": float(p[4]), "ey": float(p[5]),
             "pitch": float(p[6]), "roll": float(p[7]),
             "length": float(length), "width": float(width),
             "pano_w": int(pano_w), "pano_h": int(pano_h),
             "n_points": len(keys) + len(ln_keys), "rms": rms,
             "warp": None}
    # 잔차 워프: 예측 위치 → 클릭 위치 차이를 보간 (제어점 4개 이상일 때)
    src = _project(p, fxy, pano_w, pano_h)
    delta = pxy - src
    if len(ln):
        rows = _sideline_rows(p, ln[:, 0], pano_w, pano_h, width)
        m = np.isfinite(rows)
        src = np.vstack([src, np.stack([ln[m, 0], rows[m]], axis=1)])
        delta = np.vstack([delta,
                           np.stack([np.zeros(m.sum()),
                                     ln[m, 1] - rows[m]], axis=1)])
    ok = np.isfinite(src).all(axis=1)
    if ok.sum() >= 4:
        calib["warp"] = _tps_fit(src[ok], delta[ok], float(max(pano_w, 1)))
    return calib


def _params(calib):
    return np.array([calib["h"], calib["t_top"], calib["t_bot"],
                     calib["theta"], calib["ex"], calib["ey"],
                     calib.get("pitch", 0.0), calib.get("roll", 0.0)])


def field_to_pano(calib, fxy):
    """경기장 좌표 (N,2) m → 파노라마 픽셀 (N,2). 워프 보정 포함."""
    out = _project(_params(calib), np.asarray(fxy, float),
                   calib["pano_w"], calib["pano_h"])
    if calib.get("warp") is not None:
        ok = np.isfinite(out).all(axis=1)
        if ok.any():
            out[ok] += _tps_eval(calib["warp"], out[ok])
    return out


def _pano_to_field_raw(calib, pxy):
    h, t_top, t_bot, theta, ex, ey, pitch, roll = _params(calib)
    pxy = np.asarray(pxy, float)
    W, H = calib["pano_w"], calib["pano_h"]
    f = H / (t_top - t_bot)
    span = W / f
    yaw = (pxy[:, 0] / (W - 1) - 0.5) * span
    t = t_top - pxy[:, 1] / (H - 1) * (t_top - t_bot)
    dir_rig = np.stack([np.sin(yaw), t, np.cos(yaw)], axis=1)
    dw = dir_rig @ _rot(theta, pitch, roll).T
    s = np.where(dw[:, 1] < -1e-6, h / np.maximum(-dw[:, 1], 1e-9), np.nan)
    return np.stack([ex + s * dw[:, 0], ey + s * dw[:, 2]], axis=1)


def pano_to_field(calib, pxy):
    """파노라마 픽셀 (N,2) → 경기장 좌표 (N,2) m. 수평선 위는 NaN.

    워프가 있으면 고정점 반복으로 역보정(2~3회면 수렴) 후 역투영.
    """
    pxy = np.asarray(pxy, float)
    if calib.get("warp") is not None:
        u = pxy.copy()
        for _ in range(3):
            u = pxy - _tps_eval(calib["warp"], u)
        pxy = u
    return _pano_to_field_raw(calib, pxy)


def field_outline(length=105.0, width=68.0, step=2.0):
    """미리보기 오버레이용 경기장 선 폴리라인 목록 [(N,2) 필드 좌표, ...].

    외곽 사각형, 중앙선, 센터서클, 페널티박스 — 원통 투영에서 곡선이
    되므로 step(m) 간격으로 샘플링해 잇는다.
    """
    hl, hw = length / 2.0, width / 2.0
    pw, pd = PEN_HALF_W, PEN_DEPTH
    lines = []
    segs = [((-hl, -hw), (hl, -hw)), ((hl, -hw), (hl, hw)),
            ((hl, hw), (-hl, hw)), ((-hl, hw), (-hl, -hw)),
            ((0.0, -hw), (0.0, hw))]
    for sx in (-1, 1):                   # 페널티박스 3변 (좌/우)
        gx, bx = sx * hl, sx * (hl - pd)
        segs += [((gx, -pw), (bx, -pw)), ((bx, -pw), (bx, pw)),
                 ((bx, pw), (gx, pw))]
    for (x0, y0), (x1, y1) in segs:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
        lines.append(np.stack([np.linspace(x0, x1, n),
                               np.linspace(y0, y1, n)], axis=1))
    a = np.linspace(0, 2 * np.pi, 64)
    lines.append(np.stack([CENTER_CIRCLE_R * np.sin(a),
                           CENTER_CIRCLE_R * np.cos(a)], axis=1))
    return lines
