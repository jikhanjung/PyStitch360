"""
진행 상황 다이얼로그
스티칭 작업 진행 상황 표시
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal


class ProgressDialog(QDialog):
    """진행 상황 다이얼로그"""
    
    cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("스티칭 진행 중...")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 현재 작업 표시
        self.task_label = QLabel("초기화 중...")
        self.task_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.task_label)
        
        # 전체 진행률
        self.overall_label = QLabel("전체 진행률")
        layout.addWidget(self.overall_label)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setTextVisible(True)
        layout.addWidget(self.overall_progress)
        
        # 현재 단계 진행률
        self.step_label = QLabel("현재 단계")
        layout.addWidget(self.step_label)
        
        self.step_progress = QProgressBar()
        self.step_progress.setTextVisible(True)
        layout.addWidget(self.step_progress)
        
        # 로그 표시
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
        
        # 버튼
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def set_task(self, task: str):
        """현재 작업 설정"""
        self.task_label.setText(task)
        self.log_text.append(f"[작업] {task}")
    
    def set_overall_progress(self, value: int, maximum: int = 100):
        """전체 진행률 설정"""
        self.overall_progress.setMaximum(maximum)
        self.overall_progress.setValue(value)
    
    def set_step_progress(self, value: int, maximum: int = 100):
        """단계 진행률 설정"""
        self.step_progress.setMaximum(maximum)
        self.step_progress.setValue(value)
    
    def append_log(self, message: str):
        """로그 추가"""
        self.log_text.append(message)
    
    def on_cancel(self):
        """취소 버튼 클릭"""
        self.cancelled.emit()
        self.cancel_btn.setEnabled(False)
        self.append_log("[취소] 작업을 취소하는 중...")
    
    def finish(self, success: bool = True):
        """작업 완료"""
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        if success:
            self.task_label.setText("작업 완료!")
            self.append_log("[완료] 모든 작업이 성공적으로 완료되었습니다.")
        else:
            self.task_label.setText("작업 실패")
            self.append_log("[실패] 작업 중 오류가 발생했습니다.")