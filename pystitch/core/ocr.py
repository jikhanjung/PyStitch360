"""등번호 OCR — 근측 절반 선수만 (P03 부속 실험, devlog 040).

원경 선수는 수십 px 라 인식 불가(로드맵 판단) — 필드 Y<0(카메라 쪽)
이고 박스가 충분히 큰 검출만 시도한다. 트랙릿(병합 대표)당 큰 박스
몇 장을 시간 분산해 고르고, 숫자 전용 OCR 결과를 신뢰도 가중 투표로
집계한다. easyocr 의존 (실행 환경에서 pip install easyocr).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .field import pano_to_field


def collect_ocr_candidates(analysis, calib, role_of, rep_of,
                           min_h=90.0, per_track=12, spacing_s=2.0):
    """근측 + 큰 박스 크롭 후보 [(si, box, rep), ...] (프레임 순).

    role_of/rep_of: tid → 유효 역할 / 병합 대표. 팀 역할(0/1/3/4)만.
    """
    frames = analysis["frames"]
    fps = analysis["fps"]
    cands: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        rows = [p for p in prow if len(p) >= 5 and p[4] >= 0
                and p[3] >= min_h and role_of(int(p[4])) in (0, 1, 3, 4)]
        if not rows:
            continue
        fxy = pano_to_field(calib, [(p[0], p[1] + p[3] / 2.0) for p in rows])
        for (gx, gy), p in zip(fxy, rows):
            if np.isfinite(gy) and gy < 0.0:          # 근측 절반만
                cands.setdefault(rep_of(int(p[4])), []).append(
                    (float(p[3]), si, [float(v) for v in p[:4]]))
    picked = []
    for rep, lst in cands.items():
        lst.sort(reverse=True)
        used_t: list = []
        for h, si, box in lst:
            t = frames[si] / fps
            if all(abs(t - u) >= spacing_s for u in used_t):
                picked.append((si, box, rep))
                used_t.append(t)
            if len(used_t) >= per_track:
                break
    picked.sort()
    return picked


def run_jersey_ocr(pano_path, analysis, picked, min_conf=0.4, min_votes=3,
                   gpu=False, progress=None, cancel=None, log=print):
    """크롭 후보 → easyocr 숫자 인식 → 트랙릿별 제안 {rep: {...}}.

    제안 = 신뢰도 가중 최다 득표 (문턱 min_votes×min_conf 미달 제외).
    """
    import easyocr
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    frames = analysis["frames"]
    cap = cv2.VideoCapture(str(pano_path))
    votes: dict[int, dict] = {}
    pos = -10 ** 9
    t0 = time.perf_counter()
    for k, (si, box, rep) in enumerate(picked):
        if cancel is not None and cancel():
            break
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
            if 1 <= len(txt) <= 2 and conf >= min_conf:
                v = votes.setdefault(rep, {})
                v[txt] = v.get(txt, 0.0) + float(conf)
        if progress is not None and (k + 1) % 20 == 0:
            progress(k + 1, len(picked),
                     (k + 1) / max(time.perf_counter() - t0, 1e-9))
    cap.release()
    out = {}
    for rep, v in votes.items():
        best, score = max(v.items(), key=lambda kv: kv[1])
        total = sum(v.values())
        if score >= min_votes * min_conf:
            out[str(rep)] = {"num": best, "score": round(score, 2),
                             "share": round(score / total, 2)}
    log(f"[ocr] 크롭 {len(picked)}장 → 제안 {len(out)}건")
    return out
