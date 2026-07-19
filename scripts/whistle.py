"""호각 트랙 추출: <video> → <video>.whistle.json (전체 타임라인 + 이벤트).

사용법: python scripts/whistle.py <video.mp4> [--hi 7.0] [--lo 4.0]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.audio import (  # noqa: E402
    extract_audio, save_whistle_track, whistle_events, whistle_track,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--hi", type=float, default=15.0)
    ap.add_argument("--lo", type=float, default=8.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("오디오 추출...", flush=True)
    x = extract_audio(args.video)
    print(f"  {len(x)/16000/60:.1f}분 ({time.perf_counter()-t0:.0f}s)",
          flush=True)
    track = whistle_track(x)
    ev = whistle_events(track, hi_db=args.hi, lo_db=args.lo)
    p = save_whistle_track(args.video, track, ev)
    print(f"호각 이벤트 {len(ev)}개 → {p.name} "
          f"(총 {time.perf_counter()-t0:.0f}s)", flush=True)
    for t0_, t1_, db in ev[:40]:
        m, s = divmod(t0_, 60)
        print(f"  {int(m):3d}:{s:04.1f}  {t1_-t0_:.2f}s  {db:+.1f}dB",
              flush=True)
    if len(ev) > 40:
        print(f"  ... 외 {len(ev)-40}개", flush=True)


if __name__ == "__main__":
    main()
