"""AX700 기준 캘리브레이션 준비 (P06-3 / P07-4).

PitchWatch 에서 AX700 영상을 직접 열고 경기장 캘리브레이션 UI 로
랜드마크를 찍으면 <video>.ptz.json "field_points" 에 저장된다 — 이를
rotcam_track.py 의 기준 캘리브레이션 JSON 으로 변환한다.

주의: 팬 카메라라 랜드마크는 찍은 그 프레임에서만 유효 — 찍을 때
멈춰 둔 프레임 번호를 --frame 으로 지정하라 (PitchWatch 하단 시각
표시). 선 위의 점(sideline_near_*, center_near)은 위치 미정이라 제외.

사용법:
  python scripts/rotcam_ref_from_ptz.py C0011.MP4 --frame 12345
  → C0011.rotcam_ref.json (rotcam_track.py --calib 입력)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.field import (  # noqa: E402
    LINE_LANDMARKS, VLINE_LANDMARKS, landmark_positions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, required=True,
                    help="랜드마크를 찍은 프레임 번호 (팬 카메라 필수)")
    ap.add_argument("--length", type=float, default=105.0)
    ap.add_argument("--width", type=float, default=68.0)
    args = ap.parse_args()
    sp = Path(args.video).with_suffix(".ptz.json")
    if not sp.exists():
        sys.exit(f"{sp.name} 없음 — PitchWatch 에서 경기장 캘리브레이션 먼저")
    doc = json.loads(sp.read_text(encoding="utf-8"))
    fpts = doc.get("field_points") or {}
    size = doc.get("field_size") or [args.length, args.width]
    pos = landmark_positions(size[0], size[1])
    pts = []
    skipped = []
    for key, px in fpts.items():
        if key in LINE_LANDMARKS or key in VLINE_LANDMARKS:
            skipped.append(key)               # 선 위 점 — 위치 미정
            continue
        if key not in pos:
            skipped.append(key)
            continue
        pts.append({"px": [float(px[0]), float(px[1])],
                    "field": [float(pos[key][0]), float(pos[key][1])],
                    "key": key})
    if len(pts) < 4:
        sys.exit(f"쓸 수 있는 랜드마크 {len(pts)}개 (<4) — 코너/교점류를 "
                 f"더 찍어야 함 (제외됨: {', '.join(skipped) or '없음'})")
    out = {"frame": args.frame, "field_size": size, "points": pts}
    p = Path(args.video).with_suffix(".rotcam_ref.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"저장: {p.name} — 랜드마크 {len(pts)}개"
          + (f" (선 위 점 제외: {', '.join(skipped)})" if skipped else ""))
    print(f"다음: python scripts/rotcam_track.py {args.video} --calib {p}")


if __name__ == "__main__":
    main()
