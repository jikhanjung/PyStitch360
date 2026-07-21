"""회전 카메라(AX700) 실영상 추적 (P06-3 실영상 적용, devlog 043 잔여).

두 모드:

1. 프로브 (기본) — 캘리브레이션 없이 실영상 추적 가능성 실측:
   샘플 간 SIFT 매칭 → RANSAC H → 분해 잔차/인라이어/팬 속도 프로파일.
   셔터 블러·AGC·급팬이 track_step 게이트에 걸리는 비율을 보고한다.
     python scripts/rotcam_track.py C0011.MP4 --start 400 --dur 60

2. 추적 (--calib) — 기준 프레임 캘리브레이션으로 전체 루프 →
   <video>.rotcam.json 사이드카 (프레임별 회전 rodrigues + f + 품질).
     python scripts/rotcam_track.py C0011.MP4 --calib C0011.rotcam_ref.json

캘리브레이션 JSON (GUI/수동 작성):
   {"frame": 12345, "field_size": [105, 68],
    "points": [{"px": [u, v], "field": [X, Y]}, ...]}   # 4개 이상
   field 좌표는 원점 센터마크, X=길이 방향(m), Y=폭 방향(m).

v1 은 주기 앵커(흰 라인 재정렬) 없이 체인 품질만 기록 — 043 합성
검증에서 확인한 드리프트(20스텝 앵커 권장)는 짧은 구간 사용 전제.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.rotcam import (  # noqa: E402
    calibrate_reference, decompose_H, make_K, match_frames, track_step,
)


def _grab(cap, f_idx, det_w):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok, frame = cap.read()
    if not ok:
        return None, 1.0
    scale = 1.0
    if det_w and frame.shape[1] > det_w:
        scale = det_w / frame.shape[1]
        frame = cv2.resize(frame, (det_w, int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale


def probe(cap, fps, args):
    """캘리브레이션 없는 추적 가능성 프로파일 (f 는 폭 기준 가정치)."""
    f0 = int(args.start * fps)
    n = int(args.dur / args.every)
    step = int(args.every * fps)
    prev, _ = _grab(cap, f0, args.det_w)
    if prev is None:
        sys.exit("시작 프레임 읽기 실패")
    h, w = prev.shape[:2]
    K = make_K(args.f_guess * w, w, h)
    ok_cnt = 0
    rows = []
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        cur, _ = _grab(cap, f0 + i * step, args.det_w)
        if cur is None:
            break
        pa, pb = match_frames(prev, cur)
        row = {"t": args.start + i * args.every, "n_match": len(pa)}
        if len(pa) >= 25:
            H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            if H is not None and mask is not None and mask.sum() >= 25:
                R_rel, ratio, res = decompose_H(H, K)
                ang = np.rad2deg(np.arccos(
                    np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                row.update({"inliers": int(mask.sum()),
                            "ortho_res": round(float(res), 4),
                            "rot_deg": round(float(ang), 2),
                            "zoom": round(float(ratio), 4),
                            "ok": bool(res <= 0.08)})
                ok_cnt += row["ok"]
        rows.append(row)
        prev = cur
    dt = time.perf_counter() - t0
    print(f"프로브: {len(rows)}스텝 ({args.every}s 간격, det_w {args.det_w}, "
          f"{dt / max(len(rows), 1):.2f}s/스텝)")
    good = [r for r in rows if r.get("ok")]
    print(f"게이트 통과 {ok_cnt}/{len(rows)} ({ok_cnt / max(len(rows), 1):.0%})")
    if good:
        inl = [r["inliers"] for r in good]
        res = [r["ortho_res"] for r in good]
        rot = [r["rot_deg"] / args.every for r in good]
        zoom = [abs(r["zoom"] - 1) for r in good]
        print(f"  인라이어 중앙값 {int(np.median(inl))} "
              f"(최소 {min(inl)}), 분해 잔차 중앙값 {np.median(res):.4f}")
        print(f"  팬 속도 중앙값 {np.median(rot):.2f}°/s "
              f"(최대 {max(rot):.2f}), 줌비 편차 최대 {max(zoom):.4f}")
    for r in rows:
        if not r.get("ok"):
            why = ("매칭 부족" if r["n_match"] < 25 else
                   f"잔차 {r.get('ortho_res', float('nan'))}")
            print(f"  [기각] t={r['t']:.0f}s: {why}")
    return rows


def track(cap, fps, args):
    """캘리브레이션 기반 전체 추적 → .rotcam.json."""
    calib_doc = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    px = [p["px"] for p in calib_doc["points"]]
    fld = [p["field"] for p in calib_doc["points"]]
    ref_f = int(calib_doc["frame"])
    ref_img, scale = _grab(cap, ref_f, args.det_w)
    if ref_img is None:
        sys.exit("기준 프레임 읽기 실패")
    h, w = ref_img.shape[:2]
    px = [(u * scale, v * scale) for u, v in px]     # det_w 좌표계로
    cal = calibrate_reference(px, fld, (w, h))
    if cal is None:
        sys.exit("기준 캘리브레이션 실패 (점 4개 이상, 배치 확인)")
    print(f"기준: f {cal['f']:.0f}px, 설치 위치 "
          f"({cal['cam_pos'][0]:+.1f}, {cal['cam_pos'][1]:+.1f}, "
          f"{cal['cam_pos'][2]:.1f})m, rms {cal['rms_px']:.2f}px")
    state = {"R": cal["R"], "f": cal["f"], "K": cal["K"]}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(args.every * fps))
    frames, rvecs, fs, oks, ress = [], [], [], [], []
    prev = ref_img
    t0 = time.perf_counter()
    idx = ref_f
    while idx + step < total:
        idx += step
        cur, _ = _grab(cap, idx, args.det_w)
        if cur is None:
            break
        pa, pb = match_frames(prev, cur)
        new = track_step(state, pa, pb)
        ok = new is not None
        if ok:
            state = {"R": new["R"], "f": new["f"], "K": new["K"]}
        rv, _ = cv2.Rodrigues(state["R"])
        frames.append(idx)
        rvecs.append([round(float(v), 6) for v in rv.ravel()])
        fs.append(round(float(state["f"]), 2))
        oks.append(bool(ok))
        ress.append(round(float(new["ortho_res"]), 4) if ok else None)
        prev = cur
        if len(frames) % 60 == 0:
            el = time.perf_counter() - t0
            print(f"  {idx}/{total} ({idx / total:.0%}) "
                  f"{len(frames) / el:.1f}스텝/s, 기각 "
                  f"{oks.count(False)}/{len(oks)}", flush=True)
    out = {"version": 1, "video": Path(args.video).name,
           "det_w": args.det_w, "every_s": args.every,
           "ref": {"frame": ref_f, "f": round(float(cal["f"]), 2),
                   "cam_pos": [round(float(v), 3) for v in cal["cam_pos"]],
                   "rms_px": round(float(cal["rms_px"]), 3)},
           "frames": frames, "rvec": rvecs, "f": fs,
           "ok": oks, "ortho_res": ress}
    p = Path(args.video).with_suffix(".rotcam.json")
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"저장: {p.name} — 스텝 {len(frames)}개, "
          f"기각 {oks.count(False)}개 (품질 플래그)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default=None,
                    help="기준 캘리브레이션 JSON — 없으면 프로브 모드")
    ap.add_argument("--start", type=float, default=0.0, help="프로브 시작(초)")
    ap.add_argument("--dur", type=float, default=60.0, help="프로브 길이(초)")
    ap.add_argument("--every", type=float, default=0.5,
                    help="스텝 간격(초) — 043 합성검증은 0.5s 기준")
    ap.add_argument("--det-w", type=int, default=1920,
                    help="SIFT 해상도 폭 (4K 원본 축소)")
    ap.add_argument("--f-guess", type=float, default=1.0,
                    help="프로브용 f/폭 가정치 (분해 정규화에만 사용)")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"열기 실패: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.calib:
        track(cap, fps, args)
    else:
        probe(cap, fps, args)
    cap.release()


if __name__ == "__main__":
    main()
