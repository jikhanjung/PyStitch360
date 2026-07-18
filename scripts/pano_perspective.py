"""완성된 파노라마 영상에 원근비 조절 적용 (재스티칭 없이 리맵+재인코딩).

사용법:
  python scripts/pano_perspective.py <pano.mp4> <out.mp4> \
      [--k 0.3] [--m 1.3] [--horizon <px>] [--crf 21] [--codec libx264]

--horizon 생략 시 높이의 23% 지점을 수평선으로 사용.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from pystitch.core.encoders import encoder_args, ffmpeg_bin  # noqa: E402
from pystitch.core.perspective import PerspectiveWarp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("out")
    ap.add_argument("--k", type=float, default=0.3, help="수직 리맵 강도 [0,1)")
    ap.add_argument("--m", type=float, default=1.3, help="키스톤 최상단 배율 (>=1)")
    ap.add_argument("--horizon", type=float, default=None, help="수평선 행 (px)")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--crf", type=int, default=21)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.pano)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {args.pano}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    horizon = args.horizon if args.horizon is not None else 0.23 * h
    print(f"입력: {w}x{h} @ {fps:.2f}fps, {total}프레임 | "
          f"horizon={horizon:.0f}px k={args.k} m={args.m}", flush=True)

    warp = PerspectiveWarp(w, h, horizon, args.k, args.m)
    cmd = ([ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
            "-i", args.pano, "-map", "0:v", "-map", "1:a?"]
           + encoder_args(args.codec, args.crf)
           + ["-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", args.out])
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.perf_counter()
    done = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            enc.stdin.write(warp.apply(frame).tobytes())
            done += 1
            if done % 900 == 0:
                el = time.perf_counter() - t0
                print(f"{done}/{total} @ {done/el:.2f}fps", flush=True)
    finally:
        enc.stdin.close()
        enc.wait()
        cap.release()
    el = time.perf_counter() - t0
    print(f"PERSPECTIVE_OK: {done}프레임 / {el:.0f}s = {done/max(el,1e-9):.2f}fps → {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
