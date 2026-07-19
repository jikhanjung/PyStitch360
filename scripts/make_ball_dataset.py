"""공 검출 파인튜닝용 타일 데이터셋 생성 (P03-5 준비).

.analysis.json + .ptz.json 의 검수 라벨(export_training_labels)로
파노라마에서 640px 네이티브 타일을 잘라 YOLO 형식 데이터셋을 만든다:

- 양성: 수락 트랙("ball") + 갭필/시드 주입("ball_lowconf") + 사용자
  키프레임("ball_manual") — 타일 중심을 공 주변에서 랜덤 지터 (공이
  항상 정중앙에 오는 편향 방지). 같은 타일 안의 다른 양성도 라벨링.
- 하드 네거티브: 무시 구간 후보("not_ball") 중심 타일, 라벨 없음
  (같은 프레임 양성이 타일에 들어오면 제외).
- 원경 우선: 박스 폭 ≤ far-px 양성은 전부 채택, 나머지는 스트라이드
  샘플로 상한까지 (recall 목표가 원경이므로).
- train/val 분할은 시간 기준 (뒤 val-frac 구간 = val — 인접 프레임
  누출 방지).

출력: <out>/images/{train,val}, labels/{train,val}, dataset.yaml
사용법: python scripts/make_ball_dataset.py <pano.mp4> [--out DIR]
        [--max-pos 1500] [--max-neg 500] [--far-px 14] [--seed 7]

주의: 파인튜닝 모델은 공 1클래스 — 분석의 선수 검출까지 대체하지
않는다 (등록 방식은 A/B 개선 확인 후 결정, devlog 033).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.ptz import export_training_labels  # noqa: E402

TILE = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-pos", type=int, default=1500)
    ap.add_argument("--max-neg", type=int, default=500)
    ap.add_argument("--far-px", type=float, default=14.0,
                    help="이하 폭 양성은 전부 채택 (원경 우선)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--jitter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pano = Path(args.pano)
    ana = json.loads(pano.with_suffix(".analysis.json").read_text())
    doc = json.loads(pano.with_suffix(".ptz.json").read_text())
    labels = export_training_labels(
        ana, keyframes=[tuple(k[:3]) for k in doc.get("keyframes") or []],
        ignore_ranges=[tuple(r) for r in doc.get("ignores") or []],
        force_ranges=[tuple(p) for p in doc.get("promotes") or []])
    pos_all = [r for r in labels if r["label"] != "not_ball"]
    neg_all = [r for r in labels if r["label"] == "not_ball"]
    ws = [r["w"] for r in pos_all if r["w"] > 0]
    est_w = float(np.median(ws)) if ws else 14.0   # ball_manual 박스 추정
    print(f"라벨: 양성 {len(pos_all)} (수동 "
          f"{sum(1 for r in pos_all if r['label'] == 'ball_manual')}, 저신뢰 "
          f"{sum(1 for r in pos_all if r['label'] == 'ball_lowconf')}), "
          f"네거티브 {len(neg_all)}, 공 폭 중앙값 {est_w:.1f}px")

    # 원경 우선 샘플링: far 전부 + 나머지 스트라이드
    far = [r for r in pos_all if 0 < (r["w"] or est_w) <= args.far_px]
    far_ids = {id(r) for r in far}
    near = [r for r in pos_all if id(r) not in far_ids]
    if len(far) > args.max_pos:
        far = far[:: max(1, len(far) // args.max_pos)][:args.max_pos]
    room = max(0, args.max_pos - len(far))
    if len(near) > room and room > 0:
        near = near[:: max(1, len(near) // room)][:room]
    elif room == 0:
        near = []
    pos = sorted(far + near, key=lambda r: r["frame"])
    neg = neg_all[:: max(1, len(neg_all) // args.max_neg)][:args.max_neg]
    print(f"샘플: 양성 {len(pos)} (원경 {len(far)}) + 네거티브 {len(neg)}")

    # 프레임별 양성 위치 (타일 내 추가 라벨·네거티브 오염 검사용)
    by_frame: dict[int, list] = {}
    for r in pos_all:
        by_frame.setdefault(r["frame"], []).append(r)

    out = Path(args.out or pano.with_name(pano.stem + "_ball_dataset"))
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(pano))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    val_from = int(total * (1.0 - args.val_frac))
    tasks = sorted([("pos", r) for r in pos] + [("neg", r) for r in neg],
                   key=lambda x: x[1]["frame"])
    pos_read = -10 ** 9
    last = None                                   # 같은 프레임 다중 라벨용
    n_img = {"train": 0, "val": 0}
    n_neg = 0
    for kind, r in tasks:
        F = int(r["frame"])
        if F == pos_read and last is not None:
            ok, frame = True, last
        elif 0 <= F - pos_read <= 90:             # 순차 grab (시크 회피)
            for _ in range(F - pos_read - 1):
                cap.grab()
            ok, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, F)
            ok, frame = cap.read()
        if not ok:
            pos_read, last = F, None
            continue
        pos_read, last = F, frame
        jx = rng.randint(-args.jitter, args.jitter)
        jy = rng.randint(-args.jitter, args.jitter)
        x0 = int(np.clip(r["x"] + jx - TILE / 2, 0, max(W - TILE, 0)))
        y0 = int(np.clip(r["y"] + jy - TILE / 2, 0, max(H - TILE, 0)))
        in_tile = [q for q in by_frame.get(F, [])
                   if x0 + 8 <= q["x"] <= x0 + TILE - 8
                   and y0 + 8 <= q["y"] <= y0 + TILE - 8]
        if kind == "neg":
            if in_tile:
                continue                          # 진짜 공이 들어옴 — 오염
            lines = []
        else:
            if not in_tile:
                continue                          # 지터로 공이 빠짐
            lines = []
            for q in in_tile:
                bw = (q["w"] or est_w)
                bh = (q["h"] or bw)
                lines.append(f"0 {(q['x'] - x0) / TILE:.6f} "
                             f"{(q['y'] - y0) / TILE:.6f} "
                             f"{min(bw, TILE) / TILE:.6f} "
                             f"{min(bh, TILE) / TILE:.6f}")
        split = "val" if F >= val_from else "train"
        name = f"{pano.stem}_f{F:06d}_{kind}"
        crop = frame[y0:y0 + TILE, x0:x0 + TILE]
        if crop.shape[:2] != (TILE, TILE):
            continue
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_img[split] += 1
        n_neg += kind == "neg"
    cap.release()
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: ball\n")
    print(f"완료: train {n_img['train']} / val {n_img['val']} 타일 "
          f"(네거티브 {n_neg}) → {out}")
    print(f"다음: python scripts/finetune_ball.py {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
