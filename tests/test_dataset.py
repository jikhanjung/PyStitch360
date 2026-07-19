"""make_ball_dataset.py 엔드투엔드 — 합성 mp4 → YOLO 타일 데이터셋.

느린 편(합성 영상 인코딩 포함, 수 초) — 빠른 실행은
`pytest -m "not slow"` 로 제외.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_dataset_end_to_end(tmp_path):
    W, H, N = 1280, 720, 600
    pano = tmp_path / "synth_pano.mp4"
    vw = cv2.VideoWriter(str(pano), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), (30, 90, 40), np.uint8)
        si = f // 3
        if si < 120:
            cv2.circle(img, (100 + si * 5, 400), 7, (255, 255, 255), -1)
        if si >= 145:                  # 미끼 — 갭필 한도(2s) 밖 별도 트랙
            cv2.circle(img, (1100, 600), 8, (200, 200, 220), -1)
        vw.write(img)
    vw.release()
    balls, cands, players = [], [], []
    for si in range(N // 3):
        players.append([])
        if si < 100:
            row = [[100.0 + si * 5, 400.0, 0.6, 14.0, 14.0]]
        elif si < 120:
            row = [[100.0 + si * 5, 400.0, 0.26, 12.0, 12.0, 0.12]]
        elif si >= 145:
            row = [[1100.0, 600.0, 0.5, 16.0, 16.0]]
        else:
            row = []
        cands.append(row)
        balls.append(list(row[0][:5]) if row else None)
    pano.with_suffix(".analysis.json").write_text(json.dumps(
        {"fps": 30.0, "frames": [si * 3 for si in range(N // 3)],
         "pano_w": W, "pano_h": H, "players": players, "balls": balls,
         "ball_cands": cands}))
    pano.with_suffix(".ptz.json").write_text(json.dumps(
        {"keyframes": [[99, 265.0, 400.0]], "ignores": [[430, 597]],
         "promotes": []}))

    out = tmp_path / "ds"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_ball_dataset.py"),
         str(pano), "--out", str(out), "--max-pos", "60", "--max-neg", "15",
         "--jitter", "80", "--seed", "3"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]

    imgs = list((out / "images/train").glob("*.jpg")) \
        + list((out / "images/val").glob("*.jpg"))
    labs = list((out / "labels/train").glob("*.txt")) \
        + list((out / "labels/val").glob("*.txt"))
    assert len(imgs) > 25 and len(labs) == len(imgs)
    assert (out / "dataset.yaml").exists()
    assert cv2.imread(str(imgs[0])).shape[:2] == (640, 640)
    n_pos = 0
    for p in labs:
        txt = p.read_text().strip()
        if "_neg" in p.name:
            assert not txt                             # 네거티브 = 빈 라벨
        else:
            n_pos += 1
            assert txt
            for line in txt.splitlines():
                assert all(0.0 <= float(x) <= 1.0 for x in line.split()[1:])
    assert n_pos > 20
    # 라벨 좌표에 실제 공 (흰 픽셀)
    sample = next(p for p in labs if "_pos" in p.name and p.read_text().strip())
    im = cv2.imread(str(out / "images" / sample.parent.name
                        / (sample.stem + ".jpg")))
    cx, cy = [float(x) for x in sample.read_text().split()[1:3]]
    px, py = int(cx * 640), int(cy * 640)
    assert im[max(py - 10, 0):py + 10, max(px - 10, 0):px + 10].max() > 200
