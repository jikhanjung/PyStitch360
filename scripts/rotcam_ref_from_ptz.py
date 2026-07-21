"""AX700 기준 캘리브레이션 준비 (P06-3 / P07-4).

PitchWatch 에서 AX700 영상을 직접 열고 경기장 캘리브레이션 UI 로
랜드마크를 찍으면 <video>.ptz.json "field_points" 에 저장된다 — 이를
rotcam_track.py 의 기준 캘리브레이션 JSON 으로 변환한다.

팬 카메라라 랜드마크가 한 화면에 다 안 들어온다 — PitchWatch 가
랜드마크마다 찍은 프레임을 기록(field_point_frames)하므로, 여기서
프레임 간 호모그래피 체인(순수 회전 = 시차 없는 정확한 픽셀 이송)으로
전부 기준 프레임에 합쳐 캘리브레이션한다. 서로 다른 프레임에서 찍어도
되고, 시간상 가까울수록(팬 경로가 짧을수록) 이송 비용이 작다.

프레임 기록이 없는 옛 사이드카는 --frame 필수 (그 프레임에서 전부
찍었다고 가정). 선 위의 점(sideline_near_*, center_near)은 제외.

사용법:
  python scripts/rotcam_ref_from_ptz.py C0011.MP4
  → C0011.rotcam_ref.json (rotcam_track.py --calib 입력)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.field import (  # noqa: E402
    LINE_LANDMARKS, VLINE_LANDMARKS, landmark_positions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=None,
                    help="기준 프레임 (기본: 기록된 최빈 프레임 — "
                         "프레임 기록 없는 옛 사이드카에선 필수)")
    ap.add_argument("--det-w", type=int, default=1920)
    ap.add_argument("--length", type=float, default=105.0)
    ap.add_argument("--width", type=float, default=68.0)
    args = ap.parse_args()
    sp = Path(args.video).with_suffix(".ptz.json")
    if not sp.exists():
        sys.exit(f"{sp.name} 없음 — PitchWatch 에서 경기장 캘리브레이션 먼저")
    doc = json.loads(sp.read_text(encoding="utf-8"))
    fpts = doc.get("field_points") or {}
    frames = doc.get("field_point_frames") or {}
    size = doc.get("field_size") or [args.length, args.width]
    pos = landmark_positions(size[0], size[1])
    raw = []
    skipped = []
    for key, px in fpts.items():
        if key in LINE_LANDMARKS or key in VLINE_LANDMARKS \
                or key not in pos:
            skipped.append(key)               # 선 위 점 — 위치 미정
            continue
        raw.append((key, [float(px[0]), float(px[1])], frames.get(key)))
    if not raw:
        sys.exit("랜드마크 없음 — PitchWatch 에서 먼저 찍어야 함")
    # 기준 프레임: 지정 > 기록 최빈 > (기록 없음 → --frame 필수)
    rec = [f for _k, _p, f in raw if f is not None]
    if args.frame is not None:
        ref = int(args.frame)
    elif rec:
        vals, cnt = np.unique(rec, return_counts=True)
        ref = int(vals[np.argmax(cnt)])
    else:
        sys.exit("프레임 기록 없는 옛 사이드카 — --frame 지정 필요")
    # 다른 프레임에서 찍은 점은 호모그래피 체인으로 기준 프레임에 이송
    from pystitch.core.rotcam import chain_homography, transfer_points
    pts = []
    for key, px, f in raw:
        src = int(f) if f is not None else ref
        if src != ref:
            H = chain_homography(args.video, src, ref, det_w=args.det_w)
            if H is None:
                skipped.append(f"{key}(이송 실패 @{src})")
                continue
            px = [float(v) for v in transfer_points(H, [px])[0]]
            print(f"  {key}: 프레임 {src} → {ref} 이송 "
                  f"({px[0]:.0f}, {px[1]:.0f})")
        pts.append({"px": px,
                    "field": [float(pos[key][0]), float(pos[key][1])],
                    "key": key})
    if len(pts) < 4:
        sys.exit(f"쓸 수 있는 랜드마크 {len(pts)}개 (<4) — 코너/교점류를 "
                 f"더 찍어야 함 (제외됨: {', '.join(skipped) or '없음'})")
    out = {"frame": ref, "field_size": size, "points": pts}
    p = Path(args.video).with_suffix(".rotcam_ref.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"저장: {p.name} — 랜드마크 {len(pts)}개"
          + (f" (선 위 점 제외: {', '.join(skipped)})" if skipped else ""))
    print(f"다음: python scripts/rotcam_track.py {args.video} --calib {p}")


if __name__ == "__main__":
    main()
