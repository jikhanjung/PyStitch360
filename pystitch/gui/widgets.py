"""공용 위젯: 비디오/파노라마 프레임 표시."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class FramePane(QLabel):
    """BGR ndarray 를 종횡비 유지로 표시하는 라벨.

    interactive=True 면 드래그를 dragged(dx, dy, shift) 로 방출한다
    (dx/dy 는 표시된 픽스맵 좌표계 픽셀, shift 는 Shift 키 여부).
    """

    dragged = pyqtSignal(float, float, bool)

    def __init__(self, placeholder="영상 없음", interactive=False):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 120)
        self.setStyleSheet("background-color: #202020; color: #808080;")
        self._pixmap: QPixmap | None = None
        self._interactive = interactive
        self._drag_pos = None
        if interactive:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def displayed_width(self) -> int:
        p = self.pixmap()
        return p.width() if p is not None and not p.isNull() else 0

    def displayed_height(self) -> int:
        p = self.pixmap()
        return p.height() if p is not None and not p.isNull() else 0

    def mousePressEvent(self, ev):
        if self._interactive and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None:
            d = ev.position() - self._drag_pos
            self._drag_pos = ev.position()
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.dragged.emit(d.x(), d.y(), shift)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_pos is not None:
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(ev)

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
