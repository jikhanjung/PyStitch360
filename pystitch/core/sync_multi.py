"""멀티 카메라 동기화 (P06-1): 호각 거친 동기화 → 공 궤적 정밀 정렬.

시계 모델: t_A = offset + drift · t_B  (drift ≈ 1 ± 수십 ppm — 기기
클럭 차이로 한 시간에 수십 ms 누적).

1) 거친 동기화 — 호각 이벤트 온셋 매칭: 모든 온셋 시각 차의 히스토그램
   투표로 오프셋 후보를 찾고, 매칭 쌍에 강건(IRLS Huber) 선형 피팅.
   음속(343m/s) 때문에 심판 위치에 따라 호각별 ±0.3s 지터가 있으므로
   호각만으로는 프레임 정밀이 안 된다 — 2)로 넘긴다.
2) 정밀 동기화 — 양 카메라의 공 궤적(필드 좌표)을 시계 모델로 겹치고
   추가 오프셋 δ 를 스캔해 거리 중앙값을 최소화. 공간 정합이 미정이면
   직사각형 대칭 4가지(항등/X반전/Y반전/180°)를 함께 시험해 판별.
   창 여러 개로 δ(t) 를 재보 선형 피팅하면 드리프트도 정밀 보정.
"""
from __future__ import annotations

import numpy as np

#: 직사각형 필드의 이산 대칭 — (x, y) → 변환 좌표
TRANSFORMS = (
    lambda xy: xy,                                        # 항등
    lambda xy: np.stack([-xy[..., 0], xy[..., 1]], -1),   # X 반전
    lambda xy: np.stack([xy[..., 0], -xy[..., 1]], -1),   # Y 반전
    lambda xy: -xy,                                       # 180°
)
TRANSFORM_NAMES = ("identity", "flip_x", "flip_y", "rot180")


def to_other_time(clock, t_a):
    """A 시각 → B(다른 카메라) 시각 — t_A = offset + drift·t_B 의 역."""
    return (t_a - clock["offset"]) / clock["drift"]


def cut_synced_clip(other_path, clock, t0_a, t1_a, out_path,
                    codec="libx264", crf=23):
    """A 기준 구간 [t0_a, t1_a] 를 동기화된 다른 카메라에서 잘라 인코딩.

    P06-2 조기 성과물: 하이라이트 구간의 고화질 대체 앵글 (AX700 등).
    -ss 를 입력 앞에 두고 재인코딩 — 트랜스코딩 시크는 프레임 정밀.
    반환: 출력 경로 (실패 시 CalledProcessError).
    """
    import subprocess

    from .encoders import encoder_args, ffmpeg_bin
    b0 = max(0.0, to_other_time(clock, t0_a))
    dur = max(0.1, (t1_a - t0_a) / clock.get("drift", 1.0))
    cmd = ([ffmpeg_bin(), "-y", "-v", "error",
            "-ss", f"{b0:.3f}", "-i", str(other_path),
            "-t", f"{dur:.3f}"]
           + encoder_args(codec, crf)
           + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
              str(out_path)])
    subprocess.run(cmd, check=True)
    return out_path


def sync_by_whistles(events_a, events_b, min_db=15.0, tol_s=0.6,
                     max_offset_s=1800.0, bin_s=0.25, min_matches=4):
    """호각 이벤트 [(t0, t1, db)] 두 벌 → 시계 모델 (거친 동기화).

    온셋 시각 전체 쌍의 차이를 히스토그램 투표해 지배적 오프셋을 찾고,
    그 근방에서 1:1 매칭(가까운 순 그리디) 후 IRLS(Huber) 선형 피팅.
    반환: {"offset", "drift", "n", "rms_s", "pairs"} 또는 None.
    """
    ta = np.array(sorted(t0 for t0, _t1, db in events_a if db >= min_db))
    tb = np.array(sorted(t0 for t0, _t1, db in events_b if db >= min_db))
    if len(ta) < min_matches or len(tb) < min_matches:
        return None
    diffs = (ta[:, None] - tb[None, :]).ravel()
    diffs = diffs[np.abs(diffs) <= max_offset_s]
    if len(diffs) == 0:
        return None
    # 투표: bin_s 격자 + 이웃 bin 합산 (경계 분산 방지)
    q = np.round(diffs / bin_s).astype(np.int64)
    vals, counts = np.unique(q, return_counts=True)
    score = counts.copy()
    for i, v in enumerate(vals):
        for dv in (-1, 1):
            j = np.searchsorted(vals, v + dv)
            if j < len(vals) and vals[j] == v + dv:
                score[i] += counts[j]
    off0 = float(vals[np.argmax(score)] * bin_s)
    # 매칭: 각 b 온셋에 가장 가까운 a 온셋 (tol 내, a 는 1회만)
    pairs = []
    used = set()
    order = sorted(range(len(tb)),
                   key=lambda j: np.min(np.abs(ta - (tb[j] + off0))))
    for j in order:
        d = np.abs(ta - (tb[j] + off0))
        i = int(np.argmin(d))
        if d[i] <= tol_s and i not in used:
            used.add(i)
            pairs.append((float(tb[j]), float(ta[i])))
    if len(pairs) < min_matches:
        return None
    pairs.sort()
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    # IRLS Huber — 오매칭·음속 지터에 강건
    w = np.ones(len(x))
    a, b = off0, 1.0
    for _ in range(5):
        W = np.sum(w)
        mx, my = np.sum(w * x) / W, np.sum(w * y) / W
        vxx = np.sum(w * (x - mx) ** 2)
        if vxx < 1e-9:
            return None
        b = np.sum(w * (x - mx) * (y - my)) / vxx
        a = my - b * mx
        r = y - (a + b * x)
        c = 0.2                                   # Huber 스케일 (s)
        w = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
    r = y - (a + b * x)
    return {"offset": float(a), "drift": float(b), "n": len(pairs),
            "rms_s": float(np.sqrt(np.mean(r ** 2))),
            "pairs": pairs}


def _interp_track(t_query, t_src, xy_src, max_gap_s=0.5):
    """유한 샘플 보간 + 근처에 실측 없는 지점은 NaN (갭 브리지 방지)."""
    fin = np.isfinite(xy_src[:, 0])
    if fin.sum() < 2:
        return np.full((len(t_query), 2), np.nan)
    ts, xs = t_src[fin], xy_src[fin]
    out = np.stack([np.interp(t_query, ts, xs[:, 0]),
                    np.interp(t_query, ts, xs[:, 1])], axis=1)
    # 가장 가까운 실측과의 거리 — max_gap_s 초과면 무효
    idx = np.searchsorted(ts, t_query)
    idx = np.clip(idx, 1, len(ts) - 1)
    near = np.minimum(np.abs(t_query - ts[idx - 1]),
                      np.abs(ts[idx] - t_query))
    out[near > max_gap_s] = np.nan
    return out


def _delta_cost(t_a, xy_a, tb_in_a, xy_b_t, delta, min_overlap):
    """추가 오프셋 δ 의 궤적 거리 중앙값 (겹치는 유한 샘플만)."""
    b_at_a = _interp_track(t_a, tb_in_a + delta, xy_b_t)
    d = np.hypot(*(xy_a - b_at_a).T)
    d = d[np.isfinite(d)]
    if len(d) < min_overlap:
        return np.inf, 0
    return float(np.median(d)), len(d)


def refine_sync_by_ball(t_a, xy_a, t_b, xy_b, clock, transform=None,
                        search_s=0.8, coarse_s=0.05, fine_s=0.01,
                        min_overlap=50):
    """공 궤적으로 δ 정밀 추정 (+공간 대칭 판별).

    clock(호각 결과)으로 B 를 A 시간축에 놓고 δ ∈ ±search_s 스캔 →
    거리 중앙값 최소. transform=None 이면 대칭 4가지 모두 시험해 최적
    선택. 반환: {"delta", "transform", "transform_name", "rms_m",
    "n_overlap", "clock"(오프셋 보정본)} 또는 None.
    """
    t_a = np.asarray(t_a, float)
    xy_a = np.asarray(xy_a, float)
    t_b = np.asarray(t_b, float)
    xy_b = np.asarray(xy_b, float)
    tb_in_a = clock["offset"] + clock["drift"] * t_b
    tries = range(len(TRANSFORMS)) if transform is None else [transform]
    best = None
    for ti in tries:
        xy_b_t = TRANSFORMS[ti](xy_b)
        for d in np.arange(-search_s, search_s + 1e-9, coarse_s):
            c, n = _delta_cost(t_a, xy_a, tb_in_a, xy_b_t, d, min_overlap)
            if best is None or c < best[0]:
                best = (c, float(d), ti, n)
    if best is None or not np.isfinite(best[0]):
        return None
    _, d0, ti, _ = best
    xy_b_t = TRANSFORMS[ti](xy_b)
    for d in np.arange(d0 - coarse_s, d0 + coarse_s + 1e-9, fine_s):
        c, n = _delta_cost(t_a, xy_a, tb_in_a, xy_b_t, d, min_overlap)
        if c < best[0]:
            best = (c, float(d), ti, n)
    cost, delta, ti, n = best
    return {"delta": delta, "transform": ti,
            "transform_name": TRANSFORM_NAMES[ti],
            "rms_m": cost, "n_overlap": n,
            "clock": {"offset": clock["offset"] + delta,
                      "drift": clock["drift"]}}


def refine_clock_by_ball(t_a, xy_a, t_b, xy_b, clock, windows=3,
                         **kw):
    """창 여러 개의 δ(t) 재보로 오프셋+드리프트 동시 정밀화.

    전역 1회로 대칭을 정한 뒤, B 시간축을 windows 등분해 창별 δ 를
    추정하고 δ(τ) = c0 + c1·τ 선형 피팅 → offset += c0, drift += c1.
    창이 부족하면(겹침 없음) 전역 결과만 반환.
    """
    g = refine_sync_by_ball(t_a, xy_a, t_b, xy_b, clock, **kw)
    if g is None:
        return None
    ti = g["transform"]
    fin = np.isfinite(np.asarray(xy_b, float)[:, 0])
    tb = np.asarray(t_b, float)
    lo, hi = tb[fin][0], tb[fin][-1]
    edges = np.linspace(lo, hi, windows + 1)
    taus, deltas = [], []
    for k in range(windows):
        m = (tb >= edges[k]) & (tb <= edges[k + 1])
        r = refine_sync_by_ball(t_a, xy_a, tb[m],
                                np.asarray(xy_b, float)[m], clock,
                                transform=ti, **kw)
        if r is not None:
            taus.append(0.5 * (edges[k] + edges[k + 1]))
            deltas.append(r["delta"])
    if len(taus) < 2:
        return g
    c1, c0 = np.polyfit(taus, deltas, 1)
    return {**g,
            "clock": {"offset": clock["offset"] + float(c0),
                      "drift": clock["drift"] + float(c1)},
            "windows": list(zip(taus, deltas))}
