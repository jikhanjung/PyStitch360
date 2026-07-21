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
    calib=None 이면 필드 게이트 없이 높이 게이트만 적용 (헤드리스 등
    캘리브레이션 이전 단계 — 원경 선수는 min_h 로 대부분 걸러진다).
    """
    frames = analysis["frames"]
    fps = analysis["fps"]
    cands: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        rows = [p for p in prow if len(p) >= 5 and p[4] >= 0
                and p[3] >= min_h and role_of(int(p[4])) in (0, 1, 3, 4)]
        if not rows:
            continue
        if calib is None:
            near = [True] * len(rows)
        else:
            fxy = pano_to_field(calib,
                                [(p[0], p[1] + p[3] / 2.0) for p in rows])
            near = [bool(np.isfinite(gy) and gy < 0.0) for _gx, gy in fxy]
        for ok, p in zip(near, rows):
            if ok:                                    # 근측 절반만
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
    out = _votes_to_proposals(votes, min_conf, min_votes)
    log(f"[ocr] 크롭 {len(picked)}장 → 제안 {len(out)}건")
    return out


def _votes_to_proposals(votes, min_conf, min_votes):
    """신뢰도 가중 최다 득표 → {rep: {num, score, share}} (공용)."""
    out = {}
    for rep, v in votes.items():
        best, score = max(v.items(), key=lambda kv: kv[1])
        total = sum(v.values())
        if score >= min_votes * min_conf:
            out[str(rep)] = {"num": best, "score": round(score, 2),
                             "share": round(score / total, 2)}
    return out


class OnlineCropCache:
    """P09: 분석 패스 중 OCR 크롭을 온라인 수집 — 재디코드 제거.

    analyze_video 의 crop_hook 로 물려서, 높이 게이트를 넘는 선수
    크롭(상반신, run_jersey_ocr 과 동일 기하)을 tid 별 높이 상위
    per_track 개(간격 spacing_s)로 JPEG 인코딩해 보관한다. 선정
    의미론은 기존 후보 수집(트랙별 최고 높이 + 간격)과 동일 — 차이는
    rep 병합 전 tid 단위라는 것뿐이고, 소비 시 rep 로 합쳐 상위만 쓴다.
    """

    def __init__(self, fps, min_h=90.0, per_track=12, spacing_s=2.0,
                 max_crops=8000):
        self.fps = float(fps)
        self.min_h = float(min_h)
        self.per_track = int(per_track)
        self.spacing_s = float(spacing_s)
        self.max_crops = int(max_crops)
        self.n = 0
        self._by_tid: dict[int, list] = {}   # tid → [(h, t, si, jpeg)]

    def hook(self, frame, si, frame_idx, prow):
        t = frame_idx / self.fps
        for p in prow:
            if len(p) < 5 or p[4] < 0 or p[3] < self.min_h:
                continue
            tid = int(p[4])
            lst = self._by_tid.setdefault(tid, [])
            if any(abs(t - e[1]) < self.spacing_s for e in lst):
                continue
            if len(lst) >= self.per_track and p[3] <= lst[-1][0]:
                continue
            cx, cy, w, h = (float(v) for v in p[:4])
            x0 = int(max(cx - w * 0.65, 0))
            x1 = int(min(cx + w * 0.65, frame.shape[1]))
            y0 = int(max(cy - h * 0.55, 0))
            y1 = int(min(cy + h * 0.15, frame.shape[0]))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", crop,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                continue
            lst.append((float(p[3]), t, si, buf))
            lst.sort(key=lambda e: -e[0])
            while len(lst) > self.per_track:
                lst.pop()
            self.n = sum(len(v) for v in self._by_tid.values())
            if self.n > self.max_crops:      # 전역 상한: 최소 높이부터 퇴출
                small = min(self._by_tid, key=lambda k:
                            self._by_tid[k][-1][0] if self._by_tid[k] else 1e9)
                if self._by_tid[small]:
                    self._by_tid[small].pop()

    def picked(self, role_of, rep_of):
        """rep 병합 + 팀 역할 게이트 → [(si, jpeg, rep)] (기존 선정 대응)."""
        by_rep: dict[int, list] = {}
        for tid, lst in self._by_tid.items():
            if role_of(tid) not in (0, 1, 3, 4):
                continue
            by_rep.setdefault(rep_of(tid), []).extend(lst)
        out = []
        for rep, lst in by_rep.items():
            lst.sort(key=lambda e: -e[0])
            used_t: list = []
            for h, t, si, buf in lst:
                if all(abs(t - u) >= self.spacing_s for u in used_t):
                    out.append((si, buf, rep))
                    used_t.append(t)
                if len(used_t) >= self.per_track:
                    break
        return out


def run_jersey_ocr_cached(picked, min_conf=0.4, min_votes=3, gpu=False,
                          log=print):
    """캐시 크롭 [(si, jpeg, rep)] → 등번호 제안 — 영상 접근 없음 (P09)."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    votes: dict[int, dict] = {}
    for _si, buf, rep in picked:
        crop = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8),
                            cv2.IMREAD_COLOR)
        if crop is None:
            continue
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                          interpolation=cv2.INTER_CUBIC)
        for _bbox, txt, conf in reader.readtext(
                crop, allowlist="0123456789", detail=1):
            txt = txt.strip()
            if 1 <= len(txt) <= 2 and conf >= min_conf:
                v = votes.setdefault(rep, {})
                v[txt] = v.get(txt, 0.0) + float(conf)
    return _votes_to_proposals(votes, min_conf, min_votes)
