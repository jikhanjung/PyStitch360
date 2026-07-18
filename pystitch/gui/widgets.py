"""공용 위젯: 비디오/파노라마 프레임 표시."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class FramePane(QLabel):
    """BGR ndarray 를 종횡비 유지로 표시하는 라벨.

    interactive=True 면 드래그를 dragged(dx, dy, shift) 로 방출한다
    (dx/dy 는 표시된 픽스맵 좌표계 픽셀, shift 는 Shift 키 여부).
    좌표 비율(0~1)로 clicked / context_requested(우클릭) / hover 도 방출한다.
    """

    dragged = pyqtSignal(float, float, bool)
    clicked = pyqtSignal(float, float)   # 표시 픽스맵 기준 좌표 비율 (0~1)
    context_requested = pyqtSignal(float, float, QPoint)   # fx, fy, 전역좌표
    hover = pyqtSignal(float, float)     # 버튼 안 눌린 이동 시 커서 위치 비율

    def __init__(self, placeholder="영상 없음", interactive=False):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 120)
        self.setStyleSheet("background-color: #202020; color: #808080;")
        self._pixmap: QPixmap | None = None
        self._interactive = interactive
        self._drag_pos = None
        self._grid = False
        if interactive:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setMouseTracking(True)   # 버튼 없이도 hover 위치 추적

    def _frac_at(self, pos):
        """위젯 좌표 → 표시 픽스맵 기준 비율 (0~1). 밖이면 None."""
        p = self.pixmap()
        if p is None or p.isNull():
            return None
        x0 = (self.width() - p.width()) / 2
        y0 = (self.height() - p.height()) / 2
        fx = (pos.x() - x0) / p.width()
        fy = (pos.y() - y0) / p.height()
        if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
            return fx, fy
        return None

    def set_grid(self, on: bool):
        """정렬 가이드라인 그리드 표시 (중앙 세로선 강조 — 하프라인 맞춤용)."""
        self._grid = bool(on)
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = self.pixmap()
        if not self._grid or p is None or p.isNull():
            return
        pw, ph = p.width(), p.height()
        x0 = (self.width() - pw) // 2
        y0 = (self.height() - ph) // 2
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        for i in range(1, 8):                       # 세로 8등분
            x = x0 + pw * i // 8
            if i != 4:
                painter.drawLine(x, y0, x, y0 + ph)
        for j in range(1, 4):                       # 가로 4등분
            y = y0 + ph * j // 4
            painter.drawLine(x0, y, x0 + pw, y)
        painter.setPen(QPen(QColor(255, 210, 0, 210), 3))   # 중앙 세로선 강조
        painter.drawLine(x0 + pw // 2, y0, x0 + pw // 2, y0 + ph)
        painter.end()

    def displayed_width(self) -> int:
        p = self.pixmap()
        return p.width() if p is not None and not p.isNull() else 0

    def displayed_height(self) -> int:
        p = self.pixmap()
        return p.height() if p is not None and not p.isNull() else 0

    def mousePressEvent(self, ev):
        if self._interactive and ev.button() == Qt.MouseButton.RightButton:
            fr = self._frac_at(ev.position())
            if fr is not None:
                self.context_requested.emit(
                    fr[0], fr[1], ev.globalPosition().toPoint())
            return
        if self._interactive and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.position()
            self._press_pos = ev.position()
            self._moved = 0.0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None:
            d = ev.position() - self._drag_pos
            self._drag_pos = ev.position()
            self._moved += abs(d.x()) + abs(d.y())
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.dragged.emit(d.x(), d.y(), shift)
        elif self._interactive:              # 버튼 없이 이동 = hover
            fr = self._frac_at(ev.position())
            if fr is not None:
                self.hover.emit(fr[0], fr[1])
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_pos is not None:
            if self._moved < 4.0:        # 이동 없는 프레스+릴리스 = 클릭
                fr = self._frac_at(self._press_pos)
                if fr is not None:
                    self.clicked.emit(fr[0], fr[1])
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
