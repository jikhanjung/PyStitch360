"""
미리보기 위젯
360도 영상 미리보기 및 방향 조정
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent


class PreviewWidget(QWidget):
    """360도 영상 미리보기 위젯"""
    
    orientation_changed = pyqtSignal(float, float, float)  # yaw, pitch, roll
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.mouse_pressed = False
        self.last_mouse_pos = QPoint()
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 이미지 표시 레이블
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("미리보기 영역")
        layout.addWidget(self.image_label)
        
        # 컨트롤 버튼
        control_layout = QHBoxLayout()
        
        reset_btn = QPushButton("방향 초기화")
        reset_btn.clicked.connect(self.reset_orientation)
        control_layout.addWidget(reset_btn)
        
        control_layout.addStretch()
        
        self.info_label = QLabel("Yaw: 0° | Pitch: 0° | Roll: 0°")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
    
    def set_image(self, image: np.ndarray):
        """이미지 설정"""
        self.image = image
        self.update_preview()
    
    def update_preview(self):
        """미리보기 업데이트"""
        if self.image is None:
            return
        
        # 방향 조정 적용
        adjusted_image = self.apply_orientation(self.image)
        
        # numpy 배열을 QPixmap으로 변환
        height, width, channel = adjusted_image.shape
        bytes_per_line = 3 * width
        
        # BGR을 RGB로 변환
        rgb_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 레이블 크기에 맞게 조정
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def apply_orientation(self, image: np.ndarray) -> np.ndarray:
        """방향 조정 적용"""
        h, w = image.shape[:2]
        
        # Roll 회전 적용
        if abs(self.roll) > 0.01:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.roll, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Yaw 조정 (수평 이동)
        if abs(self.yaw) > 0.01:
            shift_x = int(w * self.yaw / 360)
            # 순환 이동
            if shift_x > 0:
                image = np.hstack([image[:, shift_x:], image[:, :shift_x]])
            else:
                image = np.hstack([image[:, shift_x:], image[:, :shift_x]])
        
        # Pitch 조정 (수직 이동)
        if abs(self.pitch) > 0.01:
            shift_y = int(h * self.pitch / 180)
            # 순환 이동
            if shift_y > 0:
                image = np.vstack([image[shift_y:, :], image[:shift_y, :]])
            else:
                image = np.vstack([image[shift_y:, :], image[:shift_y, :]])
        
        return image
    
    def mousePressEvent(self, event: QMouseEvent):
        """마우스 클릭 이벤트"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 릴리즈 이벤트"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 이벤트"""
        if self.mouse_pressed and event.buttons() == Qt.MouseButton.LeftButton:
            # 마우스 이동량 계산
            delta = event.pos() - self.last_mouse_pos
            
            # 방향 업데이트
            self.yaw += delta.x() * 0.5  # 수평 이동 -> Yaw
            self.pitch -= delta.y() * 0.3  # 수직 이동 -> Pitch
            
            # 범위 제한
            self.yaw = self.yaw % 360
            self.pitch = max(-90, min(90, self.pitch))
            
            self.last_mouse_pos = event.pos()
            self.update_info()
            self.update_preview()
            self.orientation_changed.emit(self.yaw, self.pitch, self.roll)
    
    def wheelEvent(self, event):
        """휠 이벤트 (Roll 조정)"""
        delta = event.angleDelta().y() / 120  # 휠 한 칸 = 120
        self.roll += delta
        
        # 범위 제한
        self.roll = max(-180, min(180, self.roll))
        
        self.update_info()
        self.update_preview()
        self.orientation_changed.emit(self.yaw, self.pitch, self.roll)
    
    def reset_orientation(self):
        """방향 초기화"""
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.update_info()
        self.update_preview()
        self.orientation_changed.emit(self.yaw, self.pitch, self.roll)
    
    def update_info(self):
        """정보 레이블 업데이트"""
        self.info_label.setText(
            f"Yaw: {self.yaw:.1f}° | Pitch: {self.pitch:.1f}° | Roll: {self.roll:.1f}°"
        )
    
    def set_orientation(self, yaw: float, pitch: float, roll: float):
        """방향 설정"""
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.update_info()
        self.update_preview()