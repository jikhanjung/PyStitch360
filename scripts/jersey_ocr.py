"""등번호 OCR 실험 — 근측 절반 선수만 (Windows 에서 실행).

방침 (메모리/로드맵): 원경 선수는 수십 px 라 인식 불가 — 카메라 쪽
절반(필드 Y<0)에 있고 박스가 충분히 큰 검출만 시도한다. 트랙릿(병합
대표)마다 가장 큰 박스 몇 장을 골라 숫자 전용 OCR 후 신뢰도 가중
투표로 번호를 제안한다.

결과는 .events.json "ocr_numbers" 에 제안으로 저장 (비파괴 — 적용은
GUI 에서 사용자가 번호 지정으로). 기존 지정(player_nums)과의 일치/충돌
도 표시.

사용법: python scripts/jersey_ocr.py <pano.mp4>
        [--min-h 90] [--per-track 12] [--min-conf 0.4] [--gpu]
의존성: pip install easyocr  (torch 는 ultralytics 환경에 이미 있음)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.events import save_events  # noqa: E402
from pystitch.core.field import fit_field_calibration, pano_to_field  # noqa: E402
from pystitch.core.ptz import classify_teams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--min-h", type=float, default=90.0,
                    help="시도할 최소 박스 높이(px) — 근측 게이트와 병행")
    ap.add_argument("--per-track", type=int, default=12,
                    help="트랙릿(그룹)당 OCR 시도 크롭 수")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    try:
        import easyocr
    except ImportError:
        sys.exit("easyocr 미설치 — pip install easyocr (Windows 환경)")
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

    # 근측 + 큰 박스 후보 수집: {rep: [(높이, si, box), ...]}
    frames = ana["frames"]
    cands: dict[int, list] = {}
    for si, prow in enumerate(ana["players"]):
        rows = [p for p in prow if len(p) >= 5 and p[4] >= 0
                and p[3] >= args.min_h and role_of(int(p[4])) in (0, 1, 3, 4)]
        if not rows:
            continue
        fxy = pano_to_field(calib, [(p[0], p[1] + p[3] / 2.0) for p in rows])
        for (gx, gy), p in zip(fxy, rows):
            if np.isfinite(gy) and gy < 0.0:          # 근측 절반만
                cands.setdefault(rep_of(int(p[4])), []).append(
                    (float(p[3]), si, [float(v) for v in p[:4]]))
    # 트랙릿당 큰 박스 순 + 시간 분산 (같은 순간 중복 방지, ≥2s 간격)
    picked = []
    for rep, lst in cands.items():
        lst.sort(reverse=True)
        chosen, used_t = [], []
        for h, si, box in lst:
            t = frames[si] / ana["fps"]
            if all(abs(t - u) >= 2.0 for u in used_t):
                chosen.append((si, box))
                used_t.append(t)
            if len(chosen) >= args.per_track:
                break
        picked += [(si, box, rep) for si, box in chosen]
    picked.sort()
    print(f"근측 후보: 트랙릿 {len(cands)}개, 크롭 {len(picked)}장 "
          f"(min_h={args.min_h:.0f}px)")

    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)
    cap = cv2.VideoCapture(str(pano))
    votes: dict[int, dict] = {}
    pos = -10 ** 9
    t0 = time.perf_counter()
    for k, (si, box, rep) in enumerate(picked):
        F = int(frames[si])
        if 0 <= F - pos <= 90:
            for _ in range(F - pos - 1):
                cap.grab()
            ok, frame = cap.read()
        elif F == pos:
            ok = False
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, F)
            ok, frame = cap.read()
        if not ok:
            pos = F
            continue
        pos = F
        cx, cy, w, h = box
        # 상반신(등번호 영역) 크롭 + 2배 업스케일
        x0 = int(max(cx - w * 0.65, 0))
        x1 = int(min(cx + w * 0.65, frame.shape[1]))
        y0 = int(max(cy - h * 0.55, 0))
        y1 = int(min(cy + h * 0.15, frame.shape[0]))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                          interpolation=cv2.INTER_CUBIC)
        for _bbox, txt, conf in reader.readtext(
                crop, allowlist="0123456789", detail=1):
            txt = txt.strip()
            if 1 <= len(txt) <= 2 and conf >= args.min_conf:
                v = votes.setdefault(rep, {})
                v[txt] = v.get(txt, 0.0) + float(conf)
        if (k + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {k + 1}/{len(picked)} ({(k + 1) / el:.1f}장/s)",
                  flush=True)
    cap.release()

    out = {}
    print("\n제안 (신뢰도 가중 투표):")
    for rep in sorted(votes, key=lambda r: -max(votes[r].values())):
        best, score = max(votes[rep].items(), key=lambda kv: kv[1])
        total = sum(votes[rep].values())
        if score < args.min_votes * args.min_conf:
            continue
        cur = known.get(rep)
        tag = ("  = 지정과 일치" if cur == best else
               f"  ! 지정({cur})과 충돌" if cur else "")
        out[str(rep)] = {"num": best, "score": round(score, 2),
                         "share": round(score / total, 2)}
        print(f"  #{rep:<7d} → {best:>2s}번  "
              f"(점수 {score:.1f}, 지분 {score / total:.0%}){tag}")
    save_events(pano, ocr_numbers=out)
    print(f"\n저장: {len(out)}건 → .events.json \"ocr_numbers\" "
          f"(적용은 GUI 번호 지정으로)")


if __name__ == "__main__":
    main()
