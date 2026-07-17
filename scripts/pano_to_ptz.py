"""완성된 파노라마 영상 → 가상 PTZ 1080p (재스티칭 없이 크롭만).

사용법:
  python scripts/pano_to_ptz.py <pano.mp4> <out.mp4> [--crf 20] [--codec libx264]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from pystitch.core.encoders import encoder_args  # noqa: E402
from pystitch.core.ptz import VirtualPTZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("out")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.pano)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {args.pano}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"입력: {w}x{h} @ {fps:.2f}fps, {total}프레임", flush=True)

    ptz = VirtualPTZ(w, h)
    cmd = (["ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{ptz.out_w}x{ptz.out_h}", "-r", f"{fps}", "-i", "-",
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
            enc.stdin.write(ptz.process(frame).tobytes())
            done += 1
            if done % 900 == 0:
                el = time.perf_counter() - t0
                print(f"{done}/{total} @ {done/el:.2f}fps", flush=True)
    finally:
        enc.stdin.close()
        enc.wait()
        cap.release()
    el = time.perf_counter() - t0
    print(f"PTZ_OK: {done}프레임 / {el:.0f}s = {done/max(el,1e-9):.2f}fps → {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
