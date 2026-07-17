"""완성된 파노라마 영상 → 가상 PTZ 1080p (2패스: 분석 → 크롭).

1패스에서 전체 공/선수 검출 궤적을 모아 전역(zero-phase) 스무딩하므로
실시간 처리보다 훨씬 부드럽고, 공을 놓친 구간은 선수 분포를 커버하도록
줌아웃한다. 검출 결과는 <out>.analysis.json 에 캐싱 — 스무딩 파라미터만
바꿔 다시 돌리면 1패스를 건너뛴다.

사용법:
  python scripts/pano_to_ptz.py <pano.mp4> <out.mp4> [--crf 20] [--codec libx264]
      [--detect-every 3] [--det-w 2944] [--field-top 0.26]
      [--sigma-slow 1.2] [--sigma-fast 0.35] [--fast-err 400]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import analyze_video, build_plan, render_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("out")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--det-w", type=int, default=2944)
    ap.add_argument("--field-top", type=float, default=0.26)
    ap.add_argument("--sigma-slow", type=float, default=1.2)
    ap.add_argument("--sigma-fast", type=float, default=0.35)
    ap.add_argument("--fast-err", type=float, default=400.0)
    ap.add_argument("--reanalyze", action="store_true", help="분석 캐시 무시")
    args = ap.parse_args()

    cache = Path(args.out).with_suffix(".analysis.json")
    analysis = None
    if cache.exists() and not args.reanalyze:
        d = json.loads(cache.read_text())
        if (d.get("video") == str(args.pano)
                and d.get("detect_every") == args.detect_every
                and d.get("det_w") == args.det_w):
            analysis = d
            print(f"분석 캐시 재사용: {cache}", flush=True)
    if analysis is None:
        t0 = time.perf_counter()
        print("1패스: 공/선수 검출 중...", flush=True)
        analysis = analyze_video(args.pano, detect_every=args.detect_every,
                                 det_w=args.det_w, field_top_frac=args.field_top)
        cache.write_text(json.dumps(analysis))
        print(f"분석 완료 ({time.perf_counter()-t0:.0f}s) → {cache}", flush=True)

    plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                      sigma_slow=args.sigma_slow, sigma_fast=args.sigma_fast,
                      fast_err_px=args.fast_err)
    print("2패스: 크롭·인코딩 중...", flush=True)
    fps = render_plan(args.pano, args.out, plan, codec=args.codec, crf=args.crf)
    print(f"PTZ_OK: {fps:.2f}fps → {args.out}", flush=True)


if __name__ == "__main__":
    main()
