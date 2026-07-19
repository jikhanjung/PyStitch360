"""완주 재분석: 파노라마 → <pano>.analysis.json (GUI 가 읽는 위치).

yolo11m + far_boost + 다중 후보 + ByteTrack 조합. 체크포인트(part.json)
지원 — 중단 후 재실행하면 이어서 돈다. 기존 분석은 .analysis_<태그>.json
으로 백업.

사용법:
  python scripts/reanalyze.py <pano.mp4> [--weights yolo11m.pt]
      [--detect-every 3] [--backup-tag v8n]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import analyze_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--backup-tag", default="v8n",
                    help="기존 분석 백업 접미사 (없으면 백업 안 함)")
    args = ap.parse_args()

    pano = Path(args.pano)
    out = pano.with_suffix(".analysis.json")
    ckpt = pano.with_suffix(".analysis.part.json")
    if out.exists() and args.backup_tag:
        bak = pano.with_suffix(f".analysis_{args.backup_tag}.json")
        if not bak.exists():
            shutil.copy2(out, bak)
            print(f"기존 분석 백업 → {bak.name}", flush=True)

    t0 = time.perf_counter()
    last = [0.0]

    def progress(i, total, fps):
        now = time.perf_counter()
        if now - last[0] >= 30:
            last[0] = now
            rem = (total - i) / max(fps * 3, 1e-6) / 60  # detect_every 감안
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{total} "
                  f"({i/total*100:.1f}%) {fps:.1f}fps 남은 ~{rem:.0f}분",
                  flush=True)

    print(f"재분석 시작: {pano.name}, weights={args.weights}", flush=True)
    d = analyze_video(str(pano), weights=args.weights,
                      detect_every=args.detect_every,
                      checkpoint_path=str(ckpt),
                      progress=progress, log=lambda s: print(s, flush=True))
    if d is None:
        print("취소/실패", flush=True)
        return 1
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)
    el = (time.perf_counter() - t0) / 60
    n_ball = sum(1 for b in d["balls"] if b is not None)
    n_pl = sum(len(p) for p in d["players"])
    print(f"완료 ({el:.0f}분) → {out}", flush=True)
    print(f"샘플 {len(d['frames'])}개, 공 검출 샘플 {n_ball}개 "
          f"({n_ball/len(d['frames'])*100:.1f}%), 선수 검출 총 {n_pl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
