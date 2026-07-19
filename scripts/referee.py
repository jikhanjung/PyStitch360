"""심판 자동 분류 + 호각 전 선심 포즈(기 신호) 분석.

1) 비팀 트랙릿 위치 통계 → 주심/근측·원측 선심 분류 (.events.json
   "referees" 저장).
2) 확실한 호각(≥min-db)마다 직전 pre-s 초의 근측 선심 포즈를 추정해
   팔 올림 점수 시계열 저장 ("linesman_signals"). 원측 선심은 해상도
   문제로 --far 옵션일 때만.

사용법: python scripts/referee.py <pano.mp4> [--pre 10] [--min-db 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.audio import load_whistle_track  # noqa: E402
from pystitch.core.events import (  # noqa: E402
    classify_flag_signal, classify_referees, linesman_arm_track, save_events,
)
from pystitch.core.field import fit_field_calibration  # noqa: E402
from pystitch.core.ptz import classify_teams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--pre", type=float, default=10.0)
    ap.add_argument("--post", type=float, default=6.0)   # 오프사이드 지시는 호각 후
    ap.add_argument("--min-db", type=float, default=20.0)
    ap.add_argument("--far", action="store_true", help="원측 선심도 분석")
    ap.add_argument("--pose-weights", default="yolo11n-pose.pt")
    ap.add_argument("--max-events", type=int, default=0,
                    help="0=전체, N=상위 N개 호각만")
    args = ap.parse_args()
    pano = Path(args.pano)
    ana = json.loads(pano.with_suffix(".analysis.json").read_text())
    doc = json.loads(pano.with_suffix(".ptz.json").read_text())
    calib = fit_field_calibration(
        doc["field_points"], ana["pano_w"], ana["pano_h"],
        length=doc.get("field_size", [105, 68])[0],
        width=doc.get("field_size", [105, 68])[1],
        line_points=doc.get("line_points"))
    roles = {int(k): int(v) for k, v in (doc.get("roles") or {}).items()}
    teams = classify_teams(ana, roles=roles)
    teams.update({t: r for t, r in roles.items()})

    sug, info = classify_referees(ana, teams, calib)
    print(f"심판 분류: 주심 {len(info['main'])}트랙릿, 근측 선심 "
          f"{len(info['ar_near'])}, 원측 선심 {len(info['ar_far'])} "
          f"(총 제안 {len(sug)})", flush=True)
    save_events(pano, referees=info)

    _, whistles = load_whistle_track(pano)
    strong = [w for w in whistles if w[2] >= args.min_db]
    if args.max_events:
        strong = sorted(strong, key=lambda w: -w[2])[:args.max_events]
        strong.sort()
    print(f"호각 {len(strong)}개 (≥{args.min_db}dB) 선심 포즈 분석 "
          f"[t-{args.pre}s, t+{args.post}s]", flush=True)
    sides = [("near", info["ar_near"])] + \
        ([("far", info["ar_far"])] if args.far else [])
    signals = []
    t0 = time.perf_counter()
    for k, (w0, w1, db) in enumerate(strong):
        entry = {"whistle_t": w0, "db": db}
        for side, tids in sides:
            if not tids:
                continue
            tr = linesman_arm_track(str(pano), ana, set(tids),
                                    w0 - args.pre, w1 + args.post,
                                    weights=args.pose_weights)
            if tr:
                kind, detail = classify_flag_signal(tr)
                entry[side] = {"n": len(tr), "signal": kind,
                               **detail, "track": tr}
        signals.append(entry)
        got = entry.get("near", {})
        tag = {"offside": "  ← 오프사이드 지시", "foul": "  ← 기 들어 흔듦",
               }.get(got.get("signal"), "")
        print(f"  {int(w0//60):3d}:{w0%60:04.1f} ({db:+.0f}dB)  근측 "
              f"{got.get('n', 0)}샘플 올림max {got.get('max_raise')}"
              f" 들음 {got.get('up_s', 0)}s 지시 {got.get('point_s', 0)}s"
              f"{tag}", flush=True)
    save_events(pano, linesman_signals=signals)
    n_sig = sum(1 for s in signals
                if s.get("near", {}).get("signal") in ("offside", "foul"))
    print(f"완료 ({(time.perf_counter()-t0)/60:.1f}분): 기 신호 "
          f"{n_sig}/{len(signals)}", flush=True)


if __name__ == "__main__":
    main()
