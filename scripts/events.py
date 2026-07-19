"""이벤트 엔진: 호각 × 대형 → 킥오프 검출 → <pano>.events.json.

사용법: python scripts/events.py <pano.mp4>
필요 파일: <pano>.analysis.json, <pano>.whistle.json, <pano>.ptz.json
(랜드마크 캘리브레이션 포함).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.audio import load_whistle_track  # noqa: E402
from pystitch.core.events import (  # noqa: E402
    detect_kickoffs, formation_track, save_events,
)
from pystitch.core.field import fit_field_calibration  # noqa: E402
from pystitch.core.ptz import classify_teams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--min-db", type=float, default=15.0)
    args = ap.parse_args()
    pano = Path(args.pano)
    ana = json.loads(pano.with_suffix(".analysis.json").read_text())
    doc = json.loads(pano.with_suffix(".ptz.json").read_text())
    _, whistles = load_whistle_track(pano)
    if not whistles:
        print("호각 트랙 없음 — scripts/whistle.py 먼저 실행")
        return 1
    calib = fit_field_calibration(
        doc["field_points"], ana["pano_w"], ana["pano_h"],
        length=doc.get("field_size", [105, 68])[0],
        width=doc.get("field_size", [105, 68])[1],
        line_points=doc.get("line_points"))
    if calib is None:
        print("캘리브레이션 실패 — 랜드마크 확인")
        return 1
    roles = {int(k): int(v) for k, v in (doc.get("roles") or {}).items()}
    teams = classify_teams(ana, roles=roles)
    print(f"팀 분류 트랙릿 {len(teams)}개, 호각 {len(whistles)}개", flush=True)
    tr = formation_track(ana, teams, calib)
    ok = np.isfinite(tr["sep"])
    print(f"대형 트랙: 유효 {ok.mean()*100:.0f}%, 분리도 중앙값 "
          f"{np.nanmedian(tr['sep']):.2f}", flush=True)
    ks = detect_kickoffs(tr, whistles, min_db=args.min_db)
    p = save_events(pano, ks)
    print(f"킥오프 {len(ks)}개 → {p.name}")
    for t, s, d in ks:
        m, sec = divmod(t, 60)
        w = d["whistle"]
        print(f"  {int(m):3d}:{sec:04.1f}  분리도 {s:.2f}  서클 "
              f"{d['circle_pre']:.0f}명  호각 {w[1]-w[0]:.2f}s/{w[2]:+.0f}dB"
              f"{'  [롱]' if d['long_whistle'] else ''}"
              f"{'  대형붕괴' if d['broke'] else ''}"
              f"{'  공이탈' if d['ball_left'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
