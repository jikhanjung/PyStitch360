"""A/B: 공 검출을 파노라마 타일 vs 원본 고프로 타일에서 비교.

가설: 원본은 리샘플 1회가 없고 수평선 근처 각해상도가 ~25% 높아
원경 공 검출률이 좋을 수 있다 (devlog 019 논의).

프로토콜 (양쪽 동일):
  정답 공 위치(수락 트랙, 사용자 편집 반영) 중심의 640px 정사각 타일을
  네이티브 해상도로 잘라 yolo11m 공 검출 → 반경 내 검출 여부.
  (a) 파노라마 프레임 타일  (b) 대응 원본(어안) 프레임 타일
  원본 좌표는 렌더와 동일한 build_cylindrical_maps 로 매핑.

파노라마↔원본 시간 오프셋은 모름 → 원본 3프레임에서 사람을 검출해
파노라마로 매핑, 분석의 선수 위치와 가장 잘 겹치는 오프셋을 전수 탐색
(배경이 정적이라 NCC 는 변별력이 없음 — 움직이는 선수가 신호).

사용법:
  python scripts/ab_source_detect.py <project.json> <pano.mp4>
      [--weights yolo11m.pt] [--n-far 120] [--n-near 60] [--n-miss 80]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.chapters import ChapteredVideo          # noqa: E402
from pystitch.core.geometry import build_cylindrical_maps  # noqa: E402
from pystitch.core.lens import LensProfile, builtin_profiles  # noqa: E402
from pystitch.core.project import load_project             # noqa: E402
from pystitch.core.ptz import (                            # noqa: E402
    accept_ball_tracks, link_ball_tracks,
)

CLS_PERSON, CLS_BALL = 0, 32


def _geometry(proj, pano_w, pano_h, user):
    """(pitch, roll, yaw) 사용자값으로 렌더 기하 재구성.

    el_bottom 은 항상 파노라마 종횡비에서 역산 — 렌더 후 프로젝트 설정이
    바뀌었을 수 있어 저장값을 믿지 않는다 (pitch/roll/yaw 도 마찬가지라
    아래 fit_render_params 로 피팅한다).
    """
    a = proj["segments"][0]["alignment"]
    p, r, y = user
    R_wl, R_wr = a.rotations(p, r)
    yaw0, yaw1 = a.window(y)
    u = proj.get("user", {})
    el1 = np.deg2rad(u["el_top_deg"]) if "el_top_deg" in u else a.el1
    el0 = float(np.arctan(np.tan(el1) - (yaw1 - yaw0) * pano_h / pano_w))
    return R_wl, R_wr, yaw0, yaw1, el0, el1


def build_maps(proj, pano_w, pano_h, user):
    lens = LensProfile.load(builtin_profiles()[proj["lens_profile"]])
    R_wl, R_wr, yaw0, yaw1, el0, el1 = _geometry(proj, pano_w, pano_h, user)
    maps = [build_cylindrical_maps(lens, R, pano_w, pano_h,
                                   yaw0, yaw1, el0, el1)
            for R in (R_wl, R_wr)]
    return maps, lens


def src_lookup(maps, cam, x, y):
    mx, my = maps[cam]
    xi = int(np.clip(round(x), 0, mx.shape[1] - 1))
    yi = int(np.clip(round(y), 0, mx.shape[0] - 1))
    return float(mx[yi, xi]), float(my[yi, xi])


def pick_cam(maps, lens, x, y):
    """(x,y) 파노라마 점을 커버하는 카메라 — 둘 다 유효하면 중심에 가까운 쪽."""
    best, best_d = None, None
    for cam in (0, 1):
        u, v = src_lookup(maps, cam, x, y)
        if u < 0 or v < 0 or u >= lens.width or v >= lens.height:
            continue
        d = abs(u - lens.width / 2)
        if best is None or d < best_d:
            best, best_d = cam, d
    return best


def local_scale(maps, cam, x, y):
    """파노라마 → 원본 국소 배율 (px 원본 / px 파노라마)."""
    u0, v0 = src_lookup(maps, cam, x, y)
    u1, v1 = src_lookup(maps, cam, x + 8, y)
    u2, v2 = src_lookup(maps, cam, x, y + 8)
    if min(u1, u2) < 0:
        return 1.0
    return float((np.hypot(u1 - u0, v1 - v0)
                  + np.hypot(u2 - u0, v2 - v0)) / 16.0)


def fit_render_params(model, vid_l, lens, proj, ana, log=print):
    """렌더 당시 사용자 (pitch, roll, yaw) + 시간 오프셋을 동시 피팅.

    프로젝트의 현재 user 값은 렌더 후 바뀌었을 수 있어 못 믿는다.
    원본 L 3프레임에서 선수를 검출해 광선으로 변환해 두면 회전 후보마다
    순방향 투영이 수 ms — (pitch, roll, yaw) 그리드 → 국소 정밀화,
    각 후보에서 오프셋은 전수 탐색으로 흡수한다.
    반환: (user_pry, off, 잔차px).
    """
    from pystitch.core.geometry import pixel_to_ray
    W, H = ana["pano_w"], ana["pano_h"]
    de = ana["detect_every"]
    players = [np.array([[p[0], p[1]] for p in row], dtype=np.float64)
               if row else np.zeros((0, 2)) for row in ana["players"]]
    total_l = vid_l.cum_frames[-1]
    probes = []                          # (lf, rays (N,3) 카메라 좌표)
    for frac in (0.25, 0.5, 0.75):
        lf = int(total_l * frac)
        ok, img = vid_l.read_at(lf)
        if not ok:
            continue
        r = model.predict(img, imgsz=2560, conf=0.3, classes=[CLS_PERSON],
                          verbose=False)[0]
        pts = np.array([[(float(b.xyxy[0][0]) + float(b.xyxy[0][2])) / 2,
                         float(b.xyxy[0][3])] for b in r.boxes])  # 발 위치
        if len(pts) >= 5:
            probes.append((lf, pixel_to_ray(pts, lens)))
        log(f"  probe L{lf}: 사람 {len(pts)}명")
    if not probes:
        raise RuntimeError("probe 에서 사람 검출 실패")

    def project(user):
        R_wl, _, yaw0, yaw1, el0, el1 = _geometry(proj, W, H, user)
        out = []
        for lf, rays in probes:
            w = rays @ np.asarray(R_wl)          # world = R^T · cam
            yaw = np.arctan2(w[:, 0], w[:, 2])
            t = -w[:, 1] / np.hypot(w[:, 0], w[:, 2])
            x = (yaw - yaw0) / (yaw1 - yaw0) * (W - 1)
            y = (np.tan(el1) - t) / (np.tan(el1) - np.tan(el0)) * (H - 1)
            m = (x >= 0) & (x < W) & (y >= -0.2 * H) & (y < 1.2 * H)
            out.append((lf, np.stack([x[m], y[m]], axis=1)))
        return out

    def score(user, offs):
        mapped = project(user)
        best_s, best_o = np.inf, 0
        for off in offs:
            tot, n = 0.0, 0
            for lf, pts in mapped:
                si = int(round((lf - off) / de))
                if not (0 <= si < len(players)) or len(players[si]) == 0 \
                        or len(pts) < 5:
                    tot, n = np.inf, 1
                    break
                d = np.sqrt(((pts[:, None, :] - players[si][None]) ** 2)
                            .sum(2)).min(1)
                tot += float(np.median(d))
                n += 1
            s = tot / max(n, 1)
            if s < best_s:
                best_s, best_o = s, int(off)
        return best_s, best_o

    u0 = proj.get("user", {})
    base = (u0.get("pitch", 0.0), u0.get("roll", 0.0), u0.get("yaw", 0.0))
    offs_all = np.arange(-int(np.asarray(ana["frames"])[-1]), total_l, de)
    t0 = time.perf_counter()
    s0, o0 = score(base, offs_all)       # 전역 오프셋 스캔 (초기 회전)
    log(f"  초기(프로젝트 값): 잔차 {s0:.1f}px @ off={o0}")
    offs_near = np.arange(o0 - 300, o0 + 300 + 1, de)
    best = (s0, base, o0)
    for step, span in ((2.0, 8.0), (0.5, 2.0), (0.15, 0.5)):
        bp, br, by = best[1]
        grid = [(bp + dp, br + dr, by + dy)
                for dp in np.arange(-span, span + 1e-9, step)
                for dr in np.arange(-span, span + 1e-9, step)
                for dy in np.arange(-span / 2, span / 2 + 1e-9, step)]
        for user in grid:
            s, o = score(user, offs_near)
            if s < best[0]:
                best = (s, user, o)
        log(f"  step {step}°: 잔차 {best[0]:.1f}px @ "
            f"pry=({best[1][0]:.2f}, {best[1][1]:.2f}, {best[1][2]:.2f}), "
            f"off={best[2]}")
    # 최종 회전으로 전역 오프셋 재확인
    s_fin, o_fin = score(best[1], offs_all)
    log(f"  피팅 완료 ({time.perf_counter()-t0:.0f}s): 잔차 {s_fin:.1f}px, "
        f"off={o_fin} (프로젝트 저장값 pry=({base[0]:.2f}, {base[1]:.2f}, "
        f"{base[2]:.2f}))")
    return best[1], o_fin, float(s_fin)


def detect_tile(model, frame, x, y, tile=640, conf=0.05):
    """(x,y) 중심 640 타일에서 공 검출 → [(x, y, conf)] 프레임 좌표."""
    H, W = frame.shape[:2]
    x0 = int(np.clip(x - tile / 2, 0, max(W - tile, 0)))
    y0 = int(np.clip(y - tile / 2, 0, max(H - tile, 0)))
    crop = frame[y0:y0 + tile, x0:x0 + tile]
    r = model.predict(crop, imgsz=tile, conf=conf, classes=[CLS_BALL],
                      verbose=False)[0]
    return [((float(b.xyxy[0][0]) + float(b.xyxy[0][2])) / 2 + x0,
             (float(b.xyxy[0][1]) + float(b.xyxy[0][3])) / 2 + y0,
             float(b.conf[0])) for b in r.boxes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("pano")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--n-far", type=int, default=120)
    ap.add_argument("--n-near", type=int, default=60)
    ap.add_argument("--n-miss", type=int, default=80)
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    proj = load_project(args.project)
    pano_p = Path(args.pano)
    ana = json.loads(pano_p.with_suffix(".analysis.json").read_text())
    doc = json.loads(pano_p.with_suffix(".ptz.json").read_text()) \
        if pano_p.with_suffix(".ptz.json").exists() else {}
    W, H = ana["pano_w"], ana["pano_h"]
    de = ana["detect_every"]
    frames = np.asarray(ana["frames"])

    lens = LensProfile.load(builtin_profiles()[proj["lens_profile"]])
    vid_l = ChapteredVideo(proj["left_files"])
    vid_r = ChapteredVideo(proj["right_files"])
    fps = vid_l.fps
    r_shift = int(round(proj.get("offset_sec", 0.0) * fps))

    print("렌더 파라미터(pitch/roll/yaw) + 시간 오프셋 피팅...", flush=True)
    user_fit, off, resid = fit_render_params(model, vid_l, lens, proj, ana)
    print(f"→ off={off}, 잔차 {resid:.1f}px", flush=True)
    if resid > 60:
        print("중단: 매칭 잔차가 너무 큼 — 매핑 신뢰 불가", flush=True)
        return 1
    print("remap 테이블 생성...", flush=True)
    maps, lens = build_maps(proj, W, H, user_fit)

    # 정답 집합: 수락 공 (사용자 편집 반영)
    linked = link_ball_tracks(ana)
    _, acc, _spans = accept_ball_tracks(
        ana, ignore_ranges=[tuple(r) for r in doc.get("ignores", [])],
        force_ranges=[tuple(p) for p in doc.get("promotes", [])],
        linked=linked, log=lambda s: None)
    fin = np.isfinite(acc[:, 0])
    far_band = fin & (acc[:, 1] < 0.58 * H)
    near_band = fin & ~ (acc[:, 1] < 0.58 * H)

    def pick(mask, n):
        idx = np.where(mask)[0]
        if len(idx) <= n:
            return idx
        return idx[np.linspace(0, len(idx) - 1, n).astype(int)]

    tests = [("far", si, acc[si, 0], acc[si, 1]) for si in pick(far_band, args.n_far)]
    tests += [("near", si, acc[si, 0], acc[si, 1]) for si in pick(near_band, args.n_near)]
    # 미검출(보간 가능) 샘플: 양옆 ±4 샘플에 수락 공 → 위치 보간
    miss = []
    fin_idx = np.where(fin)[0]
    for si in range(len(acc)):
        if fin[si]:
            continue
        prev = fin_idx[fin_idx < si][-1:]
        nxt = fin_idx[fin_idx > si][:1]
        if len(prev) and len(nxt) and nxt[0] - prev[0] <= 8:
            w = (si - prev[0]) / (nxt[0] - prev[0])
            x = acc[prev[0], 0] * (1 - w) + acc[nxt[0], 0] * w
            y = acc[prev[0], 1] * (1 - w) + acc[nxt[0], 1] * w
            miss.append((si, x, y))
    if len(miss) > args.n_miss:
        miss = [miss[i] for i in
                np.linspace(0, len(miss) - 1, args.n_miss).astype(int)]
    tests += [("miss", si, x, y) for si, x, y in miss]
    tests.sort(key=lambda t: t[1])
    print(f"테스트 지점: far {sum(1 for t in tests if t[0]=='far')}, "
          f"near {sum(1 for t in tests if t[0]=='near')}, "
          f"miss(보간) {sum(1 for t in tests if t[0]=='miss')}", flush=True)

    cap = cv2.VideoCapture(str(pano_p))
    res = []
    t0 = time.perf_counter()
    for k, (kind, si, x, y) in enumerate(tests):
        F = int(frames[int(si)])
        cap.set(cv2.CAP_PROP_POS_FRAMES, F)
        ok, pano = cap.read()
        if not ok:
            continue
        cam = pick_cam(maps, lens, x, y)
        if cam is None:
            continue
        u, v = src_lookup(maps, cam, x, y)
        vid = vid_l if cam == 0 else vid_r
        sf = F + off + (r_shift if cam == 1 else 0)
        ok2, src = vid.read_at(int(sf))
        if not ok2:
            continue
        sc = local_scale(maps, cam, x, y)
        da = detect_tile(model, pano, x, y)
        db = detect_tile(model, src, u, v)
        ra, rb = 60.0, max(30.0, 60.0 * sc)
        hit_a = [d for d in da if (d[0]-x)**2 + (d[1]-y)**2 <= ra*ra]
        hit_b = [d for d in db if (d[0]-u)**2 + (d[1]-v)**2 <= rb*rb]
        res.append({"kind": kind, "si": int(si), "frame": F, "cam": cam,
                    "scale": round(sc, 3),
                    "a": bool(hit_a), "b": bool(hit_b),
                    "conf_a": round(max((d[2] for d in hit_a), default=0), 3),
                    "conf_b": round(max((d[2] for d in hit_b), default=0), 3)})
        if (k + 1) % 40 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1}/{len(tests)} ({el:.0f}s)", flush=True)
    cap.release()
    vid_l.release()
    vid_r.release()

    def stat(kind):
        rs = [r for r in res if kind in ("all", r["kind"])]
        if not rs:
            return
        na = sum(r["a"] for r in rs)
        nb = sum(r["b"] for r in rs)
        only_b = sum(1 for r in rs if r["b"] and not r["a"])
        only_a = sum(1 for r in rs if r["a"] and not r["b"])
        print(f"{kind:5s} n={len(rs):3d}  파노라마 {na/len(rs)*100:5.1f}%  "
              f"원본 {nb/len(rs)*100:5.1f}%  (원본만 +{only_b}, "
              f"파노라마만 +{only_a})", flush=True)

    print("\n=== 결과 (정답 중심 640 타일, conf 0.05) ===", flush=True)
    for kind in ("far", "near", "miss", "all"):
        stat(kind)
    sc_far = [r["scale"] for r in res if r["kind"] == "far"]
    if sc_far:
        print(f"원경 국소 배율(원본px/파노px): 중앙값 {np.median(sc_far):.2f}",
              flush=True)
    out = args.out or str(Path(__file__).resolve().parents[1]
                          / "ab_source_detect_results.json")
    Path(out).write_text(json.dumps(
        {"offset": off, "resid": resid, "results": res}))
    print(f"저장: {out}", flush=True)


if __name__ == "__main__":
    main()
