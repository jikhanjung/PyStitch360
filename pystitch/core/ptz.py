"""가상 PTZ: 파노라마에서 공/선수를 추적해 16:9 크롭 창을 부드럽게 이동.

PitchStitch/PitchAnalysis.py 의 가중평균 + 슬라이딩 윈도우 스무딩 아이디어를
재구현. YOLO(ultralytics) 는 선택 의존성 — 없으면 PTZ 기능만 비활성화된다.

  pip install ultralytics   (CPU torch 로 충분)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# COCO 클래스
_CLS_PERSON = 0
_CLS_BALL = 32  # sports ball

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "presets" / "yolov8n.pt"


def ptz_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


class Detector:
    """YOLO 래퍼. detect() → (N,4) [cx, cy, weight, is_ball]."""

    def __init__(self, weights: str | Path | None = None, imgsz: int = 960):
        from ultralytics import YOLO
        w = str(weights or _DEFAULT_WEIGHTS)
        self.model = YOLO(w)
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf,
                                 classes=[_CLS_PERSON, _CLS_BALL],
                                 verbose=False)[0]
        out = []
        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            is_ball = cls == _CLS_BALL
            # 공은 압도적 가중치 — 공이 보이면 공을 따라간다
            weight = c * (30.0 if is_ball else 1.0)
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, weight, float(is_ball)))
        return np.array(out) if out else np.zeros((0, 4))


class PTZSmoother:
    """감지 결과의 가중평균 목표점을 EMA + 속도 제한으로 부드럽게 추종."""

    def __init__(self, alpha: float = 0.08, max_speed_px: float = 25.0):
        self.alpha = alpha            # EMA 계수 (프레임당)
        self.max_speed = max_speed_px  # 프레임당 최대 이동 픽셀
        self.pos: np.ndarray | None = None

    def update(self, detections: np.ndarray, fallback: tuple[float, float]):
        """detections: (N,4) [cx, cy, weight, is_ball]. 반환: 부드러운 (x, y)."""
        if len(detections):
            w = detections[:, 2]
            target = (detections[:, :2] * w[:, None]).sum(axis=0) / w.sum()
        elif self.pos is not None:
            target = self.pos          # 감지 없음 → 제자리 유지
        else:
            target = np.array(fallback, dtype=np.float64)
        if self.pos is None:
            self.pos = np.array(target, dtype=np.float64)
        else:
            step = self.alpha * (np.asarray(target) - self.pos)
            n = np.linalg.norm(step)
            if n > self.max_speed:
                step *= self.max_speed / n
            self.pos = self.pos + step
        return float(self.pos[0]), float(self.pos[1])


class VirtualPTZ:
    """파노라마 프레임 → 공 추적 16:9 크롭 (기본 1920x1080).

    감지는 detect_every 프레임마다 축소본에서 수행하고,
    크롭 중심은 매 프레임 스무더로 갱신한다.
    """

    def __init__(self, pano_w: int, pano_h: int, out_w: int = 1920, out_h: int = 1080,
                 detect_every: int = 5, detect_width: int = 1280,
                 weights: str | Path | None = None):
        if pano_w < out_w or pano_h < out_h:
            raise ValueError(
                f"파노라마({pano_w}x{pano_h})가 크롭({out_w}x{out_h})보다 작음 — "
                "내보내기 해상도를 100% 로 하세요")
        self.pano_w, self.pano_h = pano_w, pano_h
        self.out_w, self.out_h = out_w, out_h
        self.detect_every = detect_every
        self.det_scale = detect_width / pano_w
        self.detector = Detector(weights)
        self.smoother = PTZSmoother()
        self._i = 0
        self._last_det = np.zeros((0, 4))

    def process(self, pano: np.ndarray) -> np.ndarray:
        if self._i % self.detect_every == 0:
            small = cv2.resize(pano, (int(self.pano_w * self.det_scale),
                                      int(self.pano_h * self.det_scale)))
            det = self.detector.detect(small)
            if len(det):
                det[:, :2] /= self.det_scale
            self._last_det = det
        self._i += 1

        cx, cy = self.smoother.update(self._last_det,
                                      fallback=(self.pano_w / 2, self.pano_h * 0.45))
        x0 = int(round(cx - self.out_w / 2))
        y0 = int(round(cy - self.out_h / 2))
        x0 = max(0, min(x0, self.pano_w - self.out_w))
        y0 = max(0, min(y0, self.pano_h - self.out_h))
        return np.ascontiguousarray(
            pano[y0 : y0 + self.out_h, x0 : x0 + self.out_w])
