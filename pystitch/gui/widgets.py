"""공용 위젯: 비디오/파노라마 프레임 표시."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class FramePane(QLabel):
    """BGR ndarray 를 종횡비 유지로 표시하는 라벨."""

    def __init__(self, placeholder="영상 없음"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 120)
        self.setStyleSheet("background-color: #202020; color: #808080;")
        self._pixmap: QPixmap | None = None

    def set_frame(self, bgr: np.ndarray | None):
        if bgr is None:
            self._pixmap = None
            self.setText("영상 없음")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img.copy())
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_scaled()
