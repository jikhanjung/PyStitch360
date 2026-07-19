"""공중볼 3D: 탄도 피팅으로 지면 투영 왜곡 보정.

카메라(높이 h)가 높이 z 의 공을 보면 지면 투영은 카메라에서 멀어지는
방향으로 배율 h/(h−z) 만큼 밀린다 — 공중볼이 레이더 XY 에서 포물선으로
휘어 보이는 원인. 공중 구간을 감지해 탄도 모델(XY 등속 직선 + Z 중력
포물선)을 피팅하면 참 궤적(XY 직선)과 높이를 복원할 수 있다.

한계: z ≥ h 면 시선이 수평선 위로 넘어가 지면 투영 자체가 사라진다 —
카메라 높이(~5m)를 넘는 높은 공은 검출 공백으로 나타나며 보정 대상이
아니다 (모델은 z ≤ 0.85h 로 클램프).
"""
from __future__ import annotations

import numpy as np

G = 9.81


def _ballistic_z(tau, T, g=G):
    """발사(0)→착지(T) 사이 높이 — 바깥은 0."""
    z = 0.5 * g * tau * (T - tau)
    return np.where((tau >= 0) & (tau <= T), z, 0.0)


def project_ballistic(params, t, cam, h, g=G):
    """탄도 파라미터 → 관측될 지면 투영 (N,2).

    params = [p0x, p0y, vx, vy, t0, T].
    """
    p0 = np.asarray(params[:2])
    v = np.asarray(params[2:4])
    t0, T = params[4], params[5]
    tau = np.asarray(t, dtype=np.float64) - t0
    p = p0[None] + v[None] * tau[:, None]
    z = np.clip(_ballistic_z(tau, max(T, 1e-3), g), 0.0, 0.85 * h)
    scale = h / (h - z)
    cam = np.asarray(cam, dtype=np.float64)
    return cam[None] + (p - cam[None]) * scale[:, None]


def fit_ballistic(t, gxy, cam, h, g=G, iters=120):
    """관측 지면 궤적 → 탄도 피팅 (LM 감쇠 Gauss-Newton).

    반환: {p0, v, t0, T, apex_z, rms} 또는 None (수렴 실패).
    """
    t = np.asarray(t, dtype=np.float64)
    gxy = np.asarray(gxy, dtype=np.float64)
    if len(t) < 6:
        return None
    dur = float(t[-1] - t[0])
    p = np.array([gxy[0, 0], gxy[0, 1],
                  (gxy[-1, 0] - gxy[0, 0]) / max(dur, 1e-3),
                  (gxy[-1, 1] - gxy[0, 1]) / max(dur, 1e-3),
                  float(t[0]), dur])
    step = np.array([0.05, 0.05, 0.05, 0.05, 0.02, 0.02])
    lam = 1e-3

    def residual(pp):
        r = (project_ballistic(pp, t, cam, h, g) - gxy).ravel()
        return np.where(np.isfinite(r), r, 1e6)

    r = residual(p)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), 6))
        for j in range(6):
            dp = np.zeros(6)
            dp[j] = step[j]
            J[:, j] = (residual(p + dp) - r) / step[j]
        A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
        try:
            delta = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            return None
        p_new = p + delta
        p_new[5] = np.clip(p_new[5], 0.2, 5.0)          # 비행시간
        p_new[4] = np.clip(p_new[4], t[0] - 1.0, t[-1])  # 발사 시각
        r_new = residual(p_new)
        c_new = float(r_new @ r_new)
        if c_new < cost:
            p, r, cost = p_new, r_new, c_new
            lam = max(lam * 0.5, 1e-9)
            if np.abs(delta).max() < 1e-9:
                break
        else:
            lam *= 4.0
            if lam > 1e8:
                break
    rms = float(np.sqrt(cost / len(r)))
    T = float(p[5])
    return {"p0": (float(p[0]), float(p[1])),
            "v": (float(p[2]), float(p[3])),
            "t0": float(p[4]), "T": T,
            "apex_z": float(0.125 * g * T * T),   # (g/2)(T/2)^2·… = gT²/8
            "rms": rms}


def _linear_rms(t, gxy):
    """등속 직선(지면) 모델 잔차 — 탄도 개선 판단 기준."""
    t = np.asarray(t, dtype=np.float64)
    A = np.stack([np.ones_like(t), t - t[0]], axis=1)
    r = []
    for d in range(2):
        coef, *_ = np.linalg.lstsq(A, gxy[:, d], rcond=None)
        r.append(gxy[:, d] - A @ coef)
    return float(np.sqrt(np.mean(np.concatenate(r) ** 2)))


def detect_airborne_segments(t, gxy, cam, h, min_dur=0.5, max_dur=3.5,
                             min_improve=1.8, min_apex=0.8, max_rms=2.0,
                             min_speed=4.0, max_speed=32.0):
    """관측 지면 궤적에서 공중 구간 감지 → [(i0, i1, fit), ...].

    유한 샘플 연속 런을 창(0.5~3.5s)으로 훑어, 탄도 피팅이 등속 직선
    대비 min_improve 배 이상 잔차를 줄이고 정점 높이·속도가 그럴듯한
    비겹침 구간만 채택.
    """
    t = np.asarray(t, dtype=np.float64)
    gxy = np.asarray(gxy, dtype=np.float64)
    fin = np.isfinite(gxy[:, 0])
    segs = []
    runs = []
    i = 0
    while i < len(t):
        if not fin[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and fin[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    cands = []
    for r0, r1 in runs:
        i0 = r0
        while i0 < r1:
            i1 = i0
            while i1 < r1 and t[i1 + 1] - t[i0] <= max_dur:
                i1 += 1
            for j1 in range(i1, i0, -1):
                if t[j1] - t[i0] < min_dur or j1 - i0 + 1 < 6:
                    break
                fit = fit_ballistic(t[i0:j1 + 1], gxy[i0:j1 + 1], cam, h)
                if fit is None:
                    continue
                lin = _linear_rms(t[i0:j1 + 1], gxy[i0:j1 + 1])
                spd = float(np.hypot(*fit["v"]))
                if (fit["rms"] < max_rms
                        and lin / max(fit["rms"], 1e-6) >= min_improve
                        and min_apex <= fit["apex_z"] <= 0.85 * h
                        and min_speed <= spd <= max_speed):
                    cands.append((lin / max(fit["rms"], 1e-6), i0, j1, fit))
                    break
            i0 += max(1, (i1 - i0) // 4)
    # 비겹침 그리디 (개선비 큰 순)
    cands.sort(key=lambda c: -c[0])
    used = np.zeros(len(t), dtype=bool)
    for _, i0, i1, fit in cands:
        if used[i0:i1 + 1].any():
            continue
        used[i0:i1 + 1] = True
        segs.append((int(i0), int(i1), fit))
    segs.sort()
    return segs


def correct_ball_track(t, gxy, cam, h, segments=None):
    """공중 구간을 참 XY 로 보정한 사본 + 높이 배열 반환.

    반환: (corrected (N,2), z (N,), segments). 공중 구간 밖은 원본 유지.
    """
    t = np.asarray(t, dtype=np.float64)
    gxy = np.asarray(gxy, dtype=np.float64)
    if segments is None:
        segments = detect_airborne_segments(t, gxy, cam, h)
    out = gxy.copy()
    z_all = np.zeros(len(t))
    cam = np.asarray(cam, dtype=np.float64)
    for i0, i1, fit in segments:
        tau = t[i0:i1 + 1] - fit["t0"]
        z = np.clip(_ballistic_z(tau, fit["T"]), 0.0, 0.85 * h)
        p0 = np.asarray(fit["p0"])
        v = np.asarray(fit["v"])
        out[i0:i1 + 1] = p0[None] + v[None] * tau[:, None]
        z_all[i0:i1 + 1] = z
    return out, z_all, segments
