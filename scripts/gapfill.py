"""갭필 2차 패스: 수락 트랙 사이 갭을 저문턱 타일 검출로 메꾼다.

devlog 020 근거 — 운영 패스가 놓친 지점의 ~49% 는 같은 파노라마에서
저문턱(0.06) 중심 타일로 잡힌다. 같은 추론으로 사람도 함께 주입.

사용법:
  python scripts/gapfill.py <pano.mp4> [--weights yolo11m.pt]
      [--max-gap 4.0] [--conf 0.06] [--radius 250]
분석은 <pano>.analysis.json 을 읽어 갱신 (최초 1회 .analysis_pregapfill.json
백업). 사용자 편집(<pano>.ptz.json)의 무시/승격을 반영해 갭을 계산한다.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import (  # noqa: E402
    accept_ball_tracks, gapfill_analysis, gapfill_targets, link_ball_tracks,
)


def coverage(analysis, doc):
    linked = link_ball_tracks(analysis)
    _, acc, spans = accept_ball_tracks(
        analysis, ignore_ranges=[tuple(r) for r in doc.get("ignores", [])],
        force_ranges=[tuple(p) for p in doc.get("promotes", [])],
        linked=linked, log=lambda s: None)
    fin = float(np.isfinite(acc[:, 0]).mean())
    t = sum((f1 - f0) for f0, f1 in spans) / analysis["fps"]
    return fin, t, len(spans), linked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--max-gap", type=float, default=4.0)
    ap.add_argument("--conf", type=float, default=0.06)
    ap.add_argument("--radius", type=float, default=250.0)
    args = ap.parse_args()

    pano = Path(args.pano)
    ap_json = pano.with_suffix(".analysis.json")
    analysis = json.loads(ap_json.read_text())
    sp = pano.with_suffix(".ptz.json")
    doc = json.loads(sp.read_text()) if sp.exists() else {}

    fin0, t0_, n0, linked = coverage(analysis, doc)
    print(f"현재 수락: 트랙 {n0}개, 샘플 {fin0*100:.1f}%, {t0_/60:.1f}분",
          flush=True)
    targets = gapfill_targets(
        analysis, ignore_ranges=[tuple(r) for r in doc.get("ignores", [])],
        force_ranges=[tuple(p) for p in doc.get("promotes", [])],
        linked=linked, max_gap_s=args.max_gap)
    print(f"갭필 목표 {len(targets)}개 (갭 ≤ {args.max_gap}s)", flush=True)
    if not targets:
        return

    bak = pano.with_suffix(".analysis_pregapfill.json")
    if not bak.exists():
        shutil.copy2(ap_json, bak)
        print(f"백업 → {bak.name}", flush=True)

    t0 = time.perf_counter()
    last = [0.0]

    def progress(i, total, fps):
        now = time.perf_counter()
        if now - last[0] >= 20:
            last[0] = now
            print(f"  {i}/{total} ({fps:.1f}지점/s, "
                  f"남은 ~{(total-i)/max(fps,1e-9)/60:.0f}분)", flush=True)

    gapfill_analysis(str(pano), analysis, targets, weights=args.weights,
                     conf=args.conf, ball_radius=args.radius,
                     progress=progress)
    tmp = Path(str(ap_json) + ".tmp")
    tmp.write_text(json.dumps(analysis))
    tmp.replace(ap_json)
    fin1, t1_, n1, _ = coverage(analysis, doc)
    print(f"완료 ({(time.perf_counter()-t0)/60:.1f}분) → {ap_json.name}",
          flush=True)
    print(f"수락 변화: 트랙 {n0}→{n1}개, 샘플 {fin0*100:.1f}→{fin1*100:.1f}%, "
          f"{t0_/60:.1f}→{t1_/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
