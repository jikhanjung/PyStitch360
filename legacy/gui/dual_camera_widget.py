"""
듀얼 카메라 뷰어 위젯
좌측/우측 카메라 영상을 나란히 표시
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QSlider, QSpinBox, QGroupBox, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage


class VideoPlayer:
    """단일 비디오 플레이어 클래스"""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
        
    def open(self) -> bool:
        """비디오 파일 열기"""
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                return False
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            return True
            
        except Exception:
            return False
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """특정 프레임 가져오기"""
        if self.cap is None:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame_number
            return frame
        return None
    
    def close(self):
        """비디오 파일 닫기"""
        if self.cap:
            self.cap.release()
            self.cap = None


class DualCameraWidget(QWidget):
    """듀얼 카메라 뷰어 위젯"""
    
    # 시그널 정의
    sync_offset_changed = pyqtSignal(int)  # 동기화 오프셋 변경
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # 비디오 플레이어
        self.left_player = None
        self.right_player = None
        
        # 현재 상태
        self.current_frame = 0
        self.sync_offset = 0  # 우측 카메라 오프셋 (프레임 단위)
        self.is_playing = False
        
        # 타이머 (재생용)
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 상단: 비디오 표시 영역
        video_group = QGroupBox("듀얼 카메라 뷰")
        video_layout = QHBoxLayout(video_group)
        
        # 좌측 영상
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        
        left_title = QLabel("좌측 카메라")
        left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_title.setStyleSheet("font-weight: bold; background-color: #e3f2fd; padding: 5px;")
        left_layout.addWidget(left_title)
        
        self.left_video_label = QLabel()
        self.left_video_label.setMinimumSize(320, 240)
        self.left_video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.left_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_video_label.setText("좌측 영상 없음")
        self.left_video_label.setScaledContents(True)
        left_layout.addWidget(self.left_video_label)
        
        video_layout.addWidget(left_frame)
        
        # 우측 영상
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        
        right_title = QLabel("우측 카메라")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_title.setStyleSheet("font-weight: bold; background-color: #fff3e0; padding: 5px;")
        right_layout.addWidget(right_title)
        
        self.right_video_label = QLabel()
        self.right_video_label.setMinimumSize(320, 240)
        self.right_video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.right_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_video_label.setText("우측 영상 없음")
        self.right_video_label.setScaledContents(True)
        right_layout.addWidget(self.right_video_label)
        
        video_layout.addWidget(right_frame)
        
        layout.addWidget(video_group)
        
        # 중단: 프레임 탐색 슬라이더
        frame_group = QGroupBox("프레임 탐색")
        frame_layout = QVBoxLayout(frame_group)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        frame_layout.addWidget(self.frame_slider)
        
        # 프레임 정보 표시
        info_layout = QHBoxLayout()
        self.frame_info_label = QLabel("프레임: 0 / 0")\n        info_layout.addWidget(self.frame_info_label)
        info_layout.addStretch()
        
        self.fps_label = QLabel("FPS: 0")
        info_layout.addWidget(self.fps_label)
        frame_layout.addLayout(info_layout)
        
        layout.addWidget(frame_group)
        
        # 하단: 동기화 및 재생 컨트롤
        control_group = QGroupBox("재생 및 동기화 컨트롤")
        control_layout = QVBoxLayout(control_group)
        
        # 동기화 컨트롤
        sync_layout = QHBoxLayout()
        sync_layout.addWidget(QLabel("동기화 오프셋:"))
        
        self.sync_slider = QSlider(Qt.Orientation.Horizontal)
        self.sync_slider.setRange(-100, 100)
        self.sync_slider.setValue(0)
        self.sync_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sync_slider.setTickInterval(10)
        self.sync_slider.valueChanged.connect(self.on_sync_changed)
        sync_layout.addWidget(self.sync_slider)
        
        self.sync_spinbox = QSpinBox()
        self.sync_spinbox.setRange(-100, 100)
        self.sync_spinbox.setValue(0)
        self.sync_spinbox.setSuffix(" 프레임")
        self.sync_spinbox.valueChanged.connect(self.sync_slider.setValue)
        sync_layout.addWidget(self.sync_spinbox)
        
        control_layout.addLayout(sync_layout)
        
        # 재생 버튼
        button_layout = QHBoxLayout()
        
        self.play_button = QPushButton("재생")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)
        
        self.prev_button = QPushButton("이전")
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("다음")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        layout.addWidget(control_group)
    
    def load_videos(self, left_files: List[Path], right_files: List[Path]) -> bool:
        """비디오 파일들 로드"""
        if not left_files or not right_files:
            self.logger.warning("좌측 또는 우측 파일이 없습니다")
            return False
        
        # 첫 번째 파일만 사용 (추후 연결된 파일 지원 가능)
        left_path = left_files[0]
        right_path = right_files[0]
        
        self.logger.info(f"비디오 로드: {left_path.name}, {right_path.name}")
        
        # 기존 플레이어 정리
        self.close_videos()
        
        # 새 플레이어 생성
        self.left_player = VideoPlayer(left_path)
        self.right_player = VideoPlayer(right_path)
        
        # 비디오 열기
        if not self.left_player.open():
            self.logger.error(f"좌측 비디오 열기 실패: {left_path}")
            return False
            
        if not self.right_player.open():
            self.logger.error(f"우측 비디오 열기 실패: {right_path}")
            return False
        
        # UI 업데이트
        total_frames = min(self.left_player.total_frames, self.right_player.total_frames)
        self.frame_slider.setMaximum(total_frames - 1)
        
        fps = (self.left_player.fps + self.right_player.fps) / 2
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # 첫 프레임 표시
        self.current_frame = 0
        self.update_frame_display()
        
        # 컨트롤 활성화
        self.play_button.setEnabled(True)
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        
        return True
    
    def close_videos(self):
        """비디오 파일들 닫기"""
        if self.left_player:
            self.left_player.close()
            self.left_player = None
            
        if self.right_player:
            self.right_player.close()
            self.right_player = None
        
        # UI 초기화
        self.left_video_label.clear()
        self.left_video_label.setText("좌측 영상 없음")
        self.right_video_label.clear()
        self.right_video_label.setText("우측 영상 없음")
        
        self.play_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        
        if self.is_playing:
            self.toggle_playback()
    
    def update_frame_display(self):
        """현재 프레임 표시 업데이트"""
        if not self.left_player or not self.right_player:
            return
        
        # 좌측 프레임
        left_frame = self.left_player.get_frame(self.current_frame)
        if left_frame is not None:
            self.display_frame(left_frame, self.left_video_label)
        
        # 우측 프레임 (동기화 오프셋 적용)
        right_frame_number = self.current_frame + self.sync_offset
        right_frame_number = max(0, min(right_frame_number, self.right_player.total_frames - 1))
        right_frame = self.right_player.get_frame(right_frame_number)
        if right_frame is not None:
            self.display_frame(right_frame, self.right_video_label)
        
        # 프레임 정보 업데이트
        total_frames = min(self.left_player.total_frames, self.right_player.total_frames)
        self.frame_info_label.setText(f"프레임: {self.current_frame + 1} / {total_frames}")
        
        # 슬라이더 업데이트 (시그널 차단)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
    
    def display_frame(self, frame: np.ndarray, label: QLabel):
        """프레임을 QLabel에 표시"""
        try:
            # BGR to RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_frame.shape
            
            # QImage 생성
            q_image = QImage(rgb_frame.data, w, h, w * c, QImage.Format.Format_RGB888)
            
            # QPixmap으로 변환하여 표시
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            
        except Exception as e:
            self.logger.error(f"프레임 표시 오류: {e}")
    
    def on_frame_changed(self, frame_number: int):
        """프레임 슬라이더 변경"""
        self.current_frame = frame_number
        self.update_frame_display()
    
    def on_sync_changed(self, offset: int):
        """동기화 오프셋 변경"""
        self.sync_offset = offset
        self.sync_spinbox.blockSignals(True)
        self.sync_spinbox.setValue(offset)
        self.sync_spinbox.blockSignals(False)
        
        self.update_frame_display()
        self.sync_offset_changed.emit(offset)
    
    def toggle_playback(self):
        """재생/일시정지 토글"""
        if not self.left_player or not self.right_player:
            return
        
        if self.is_playing:
            self.play_timer.stop()
            self.play_button.setText("재생")
            self.is_playing = False
        else:
            # FPS에 맞춰 타이머 설정 (milliseconds)
            fps = (self.left_player.fps + self.right_player.fps) / 2
            interval = int(1000 / fps)
            self.play_timer.start(interval)
            self.play_button.setText("일시정지")
            self.is_playing = True
    
    def prev_frame(self):
        """이전 프레임"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame_display()
    
    def next_frame(self):
        """다음 프레임"""
        if not self.left_player or not self.right_player:
            return
        
        total_frames = min(self.left_player.total_frames, self.right_player.total_frames)
        if self.current_frame < total_frames - 1:
            self.current_frame += 1
            self.update_frame_display()
        elif self.is_playing:
            # 끝에 도달하면 재생 중지
            self.toggle_playback()
    
    def get_sync_offset(self) -> int:
        """현재 동기화 오프셋 반환"""
        return self.sync_offset
    
    def set_sync_offset(self, offset: int):
        """동기화 오프셋 설정"""
        self.sync_slider.setValue(offset)