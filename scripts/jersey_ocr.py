"""등번호 OCR 실험 — 근측 절반 선수만 (헤드리스 실행, devlog 040).

GUI 분석 메뉴 "등번호 OCR (근측 선수)" 와 같은 로직 (core/ocr.py 공유).
결과는 .events.json "ocr_numbers" 에 제안으로 저장 — 적용은 GUI 번호
메뉴의 "OCR 제안" 항목으로.

사용법: python scripts/jersey_ocr.py <pano.mp4>
        [--min-h 90] [--per-track 12] [--min-conf 0.4] [--gpu]
의존성: pip install easyocr
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.events import save_events  # noqa: E402
from pystitch.core.field import fit_field_calibration  # noqa: E402
from pystitch.core.ocr import (  # noqa: E402
    collect_ocr_candidates, run_jersey_ocr,
)
from pystitch.core.ptz import classify_teams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--min-h", type=float, default=90.0)
    ap.add_argument("--per-track", type=int, default=12)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    pano = Path(args.pano)
    ana = json.loads(pano.with_suffix(".analysis.json").read_text())
    doc = json.loads(pano.with_suffix(".ptz.json").read_text())
    calib = fit_field_calibration(
        doc["field_points"], ana["pano_w"], ana["pano_h"],
        length=doc.get("field_size", [105, 68])[0],
        width=doc.get("field_size", [105, 68])[1],
        line_points=doc.get("line_points"))
    if calib is None:
        sys.exit("경기장 캘리브레이션 실패 — 근측 게이트에 필요")
    merges = {int(k): int(v) for k, v in (doc.get("merges") or {}).items()}
    roles = {int(k): int(v) for k, v in (doc.get("roles") or {}).items()}
    known = {int(k): str(v) for k, v in
             (doc.get("player_nums") or {}).items()}
    teams = classify_teams(ana, roles=roles)
    teams.update(roles)

    def rep_of(t):
        return merges.get(t, t)

    def role_of(t):
        r = rep_of(t)
        return roles.get(r, teams.get(r, teams.get(t, 2)))

    picked = collect_ocr_candidates(ana, calib, role_of, rep_of,
                                    min_h=args.min_h,
                                    per_track=args.per_track)
    n_rep = len({r for _, _, r in picked})
    print(f"근측 후보: 트랙릿 {n_rep}개, 크롭 {len(picked)}장 "
          f"(min_h={args.min_h:.0f}px)")
    out = run_jersey_ocr(
        str(pano), ana, picked, min_conf=args.min_conf,
        min_votes=args.min_votes, gpu=args.gpu,
        progress=lambda d, t, f: print(f"  {d}/{t} ({f:.1f}장/s)",
                                       flush=True))
    print("\n제안:")
    for k in sorted(out, key=lambda k: -out[k]["score"]):
        v = out[k]
        cur = known.get(int(k))
        tag = ("  = 지정과 일치" if cur == v["num"] else
               f"  ! 지정({cur})과 충돌" if cur else "")
        print(f"  #{k:>7s} → {v['num']:>2s}번  "
              f"(점수 {v['score']}, 지분 {v['share']:.0%}){tag}")
    save_events(pano, ocr_numbers=out)
    print(f"\n저장: {len(out)}건 → .events.json \"ocr_numbers\"")


if __name__ == "__main__":
    main()
