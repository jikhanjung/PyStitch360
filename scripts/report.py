"""선수 히트맵/활동량 리포트 생성 (P03-4).

.analysis.json + .ptz.json(역할·병합·캘리브레이션)에서 팀/선수 히트맵
PNG 와 players.md 요약을 만든다. 출력: <파노라마 이름>_report/.

사용법: python scripts/report.py <pano.mp4> [--out DIR] [--min-det 150]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.field import fit_field_calibration  # noqa: E402
from pystitch.core.ptz import classify_teams  # noqa: E402
from pystitch.core.report import generate_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-det", type=int, default=150)
    ap.add_argument("--top-n", type=int, default=30)
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
        sys.exit("경기장 캘리브레이션 실패 — GUI 에서 랜드마크를 지정하세요")
    roles = {int(k): int(v) for k, v in (doc.get("roles") or {}).items()}
    merges = {int(k): int(v) for k, v in (doc.get("merges") or {}).items()}
    teams = classify_teams(ana, roles=roles)
    teams.update(roles)

    def role_of(tid):
        rep = merges.get(tid, tid)
        return roles.get(rep, teams.get(rep, teams.get(tid, 2)))

    roles_of = {merges.get(t, t): role_of(t) for t in teams}
    out = args.out or str(pano.with_name(pano.stem + "_report"))
    r = generate_report(ana, calib, roles_of, out, merges=merges,
                        team_names=tuple(doc.get("team_names")
                                         or ("Team1", "Team2")),
                        min_det=args.min_det, top_n=args.top_n)
    print(f"완료: {len(r['files'])}개 파일 → {r['dir']}")


if __name__ == "__main__":
    main()
