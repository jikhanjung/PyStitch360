"""
메인 스티처 윈도우
PyStitch360의 메인 GUI 인터페이스
"""

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QStatusBar, QFileDialog,
    QProgressBar, QGroupBox, QSlider, QSpinBox, QComboBox,
    QTabWidget, QListWidget, QCheckBox, QMenuBar, QMenu,
    QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QAction, QIcon, QPixmap

from core.preprocessor import Preprocessor
from core.project_manager import ProjectManager
from gui.stitching_thread import StitchingThread
from gui.progress_dialog import ProgressDialog
from gui.preview_widget import PreviewWidget


class StitcherWindow(QMainWindow):
    """메인 애플리케이션 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.front_files = []
        self.back_files = []
        self.output_path = None
        
        # 스레드 및 모듈 초기화
        self.stitching_thread = None
        self.preprocessor = Preprocessor()
        self.project_manager = ProjectManager()
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("PyStitch360 - 360도 영상 스티칭 툴")
        self.setGeometry(100, 100, 1400, 900)
        
        # 메뉴바 생성
        self.create_menu_bar()
        
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 헤더 영역
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        
        # 1. 입력 탭
        input_tab = self.create_input_tab()
        self.tab_widget.addTab(input_tab, "입력 설정")
        
        # 2. 스티칭 설정 탭
        stitching_tab = self.create_stitching_tab()
        self.tab_widget.addTab(stitching_tab, "스티칭 설정")
        
        # 3. 출력 설정 탭
        output_tab = self.create_output_tab()
        self.tab_widget.addTab(output_tab, "출력 설정")
        
        # 4. 미리보기 탭
        preview_tab = self.create_preview_tab()
        self.tab_widget.addTab(preview_tab, "미리보기")
        
        main_layout.addWidget(self.tab_widget)
        
        # 진행 상황 영역
        progress_layout = self.create_progress_section()
        main_layout.addLayout(progress_layout)
        
        # 로그 표시 영역
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.append("PyStitch360 초기화 완료")
        main_layout.addWidget(self.log_text)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("준비됨")
        
        self.logger.info("GUI 초기화 완료")
    
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        
        new_project_action = QAction("새 프로젝트", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("프로젝트 열기", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        # 최근 프로젝트 서브메뉴
        self.recent_menu = file_menu.addMenu("최근 프로젝트")
        self.update_recent_projects_menu()
        
        file_menu.addSeparator()
        
        save_project_action = QAction("프로젝트 저장", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        save_as_action = QAction("다른 이름으로 저장", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        save_template_action = QAction("템플릿으로 저장", self)
        save_template_action.triggered.connect(self.save_as_template)
        file_menu.addAction(save_template_action)
        
        file_menu.addSeparator()
        
        import_action = QAction("설정 가져오기", self)
        import_action.triggered.connect(self.import_settings)
        file_menu.addAction(import_action)
        
        export_action = QAction("설정 내보내기", self)
        export_action.triggered.connect(self.export_settings)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도구 메뉴
        tools_menu = menubar.addMenu("도구")
        
        calibration_action = QAction("캘리브레이션 관리", self)
        calibration_action.triggered.connect(self.manage_calibration)
        tools_menu.addAction(calibration_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        
        about_action = QAction("PyStitch360 정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_header(self):
        """헤더 영역 생성"""
        header_layout = QVBoxLayout()
        
        # 제목
        title_label = QLabel("PyStitch360")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        # 설명
        desc_label = QLabel("GoPro 듀얼 카메라 360도 영상 스티칭 툴")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: gray; margin-bottom: 10px;")
        header_layout.addWidget(desc_label)
        
        return header_layout
    
    def create_input_tab(self):
        """입력 설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 파일 선택 영역
        file_group = QGroupBox("GoPro 파일 선택")
        file_layout = QGridLayout(file_group)
        
        # 좌측 카메라 파일
        file_layout.addWidget(QLabel("좌측 카메라:"), 0, 0)
        self.front_list = QListWidget()
        self.front_list.setMaximumHeight(100)
        file_layout.addWidget(self.front_list, 0, 1)
        
        front_btn = QPushButton("전면 파일 선택")
        front_btn.clicked.connect(lambda: self.select_files("front"))
        file_layout.addWidget(front_btn, 0, 2)
        
        # 우측 카메라 파일
        file_layout.addWidget(QLabel("우측 카메라:"), 1, 0)
        self.back_list = QListWidget()
        self.back_list.setMaximumHeight(100)
        file_layout.addWidget(self.back_list, 1, 1)
        
        back_btn = QPushButton("후면 파일 선택")
        back_btn.clicked.connect(lambda: self.select_files("back"))
        file_layout.addWidget(back_btn, 1, 2)
        
        # 자동 감지 버튼
        auto_detect_btn = QPushButton("폴더에서 자동 감지")
        auto_detect_btn.clicked.connect(self.auto_detect_files)
        file_layout.addWidget(auto_detect_btn, 2, 1, 1, 2)
        
        layout.addWidget(file_group)
        
        # 동기화 설정 영역
        sync_group = QGroupBox("동기화 설정")
        sync_layout = QHBoxLayout(sync_group)
        
        sync_layout.addWidget(QLabel("프레임 오프셋:"))
        
        self.sync_slider = QSlider(Qt.Orientation.Horizontal)
        self.sync_slider.setRange(-100, 100)
        self.sync_slider.setValue(0)
        self.sync_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sync_slider.setTickInterval(10)
        sync_layout.addWidget(self.sync_slider)
        
        self.sync_spinbox = QSpinBox()
        self.sync_spinbox.setRange(-100, 100)
        self.sync_spinbox.setValue(0)
        self.sync_spinbox.setSuffix(" 프레임")
        sync_layout.addWidget(self.sync_spinbox)
        
        # 슬라이더와 스핀박스 연결
        self.sync_slider.valueChanged.connect(self.sync_spinbox.setValue)
        self.sync_spinbox.valueChanged.connect(self.sync_slider.setValue)
        
        layout.addWidget(sync_group)
        
        layout.addStretch()
        return widget
    
    def create_stitching_tab(self):
        """스티칭 설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 캘리브레이션 설정
        calib_group = QGroupBox("카메라 캘리브레이션")
        calib_layout = QHBoxLayout(calib_group)
        
        calib_layout.addWidget(QLabel("프리셋:"))
        self.calib_combo = QComboBox()
        self.calib_combo.addItems(["gopro_dual.yaml", "커스텀..."])
        calib_layout.addWidget(self.calib_combo)
        
        calib_btn = QPushButton("캘리브레이션 로드")
        calib_btn.clicked.connect(self.load_calibration)
        calib_layout.addWidget(calib_btn)
        
        layout.addWidget(calib_group)
        
        # 블렌딩 설정
        blend_group = QGroupBox("블렌딩 설정")
        blend_layout = QGridLayout(blend_group)
        
        blend_layout.addWidget(QLabel("블렌드 타입:"), 0, 0)
        self.blend_type_combo = QComboBox()
        self.blend_type_combo.addItems(["Linear", "Feather", "Multi-band"])
        blend_layout.addWidget(self.blend_type_combo, 0, 1)
        
        blend_layout.addWidget(QLabel("Feather 폭:"), 1, 0)
        self.feather_slider = QSlider(Qt.Orientation.Horizontal)
        self.feather_slider.setRange(10, 200)
        self.feather_slider.setValue(50)
        blend_layout.addWidget(self.feather_slider, 1, 1)
        
        self.feather_label = QLabel("50 픽셀")
        blend_layout.addWidget(self.feather_label, 1, 2)
        self.feather_slider.valueChanged.connect(
            lambda v: self.feather_label.setText(f"{v} 픽셀")
        )
        
        layout.addWidget(blend_group)
        
        # 투영 설정
        proj_group = QGroupBox("투영 설정")
        proj_layout = QGridLayout(proj_group)
        
        proj_layout.addWidget(QLabel("투영 타입:"), 0, 0)
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["Equirectangular", "Cylindrical"])
        proj_layout.addWidget(self.proj_combo, 0, 1)
        
        proj_layout.addWidget(QLabel("출력 해상도:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "3840x1920 (4K)",
            "5760x2880 (5.7K)",
            "7680x3840 (8K)",
            "커스텀..."
        ])
        proj_layout.addWidget(self.resolution_combo, 1, 1)
        
        layout.addWidget(proj_group)
        
        layout.addStretch()
        return widget
    
    def create_output_tab(self):
        """출력 설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 인코딩 설정
        encoding_group = QGroupBox("인코딩 설정")
        encoding_layout = QGridLayout(encoding_group)
        
        encoding_layout.addWidget(QLabel("코덱:"), 0, 0)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["H.264 (libx264)", "H.265 (libx265)"])
        encoding_layout.addWidget(self.codec_combo, 0, 1)
        
        encoding_layout.addWidget(QLabel("품질 (CRF):"), 1, 0)
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(15, 35)
        self.crf_slider.setValue(23)
        encoding_layout.addWidget(self.crf_slider, 1, 1)
        
        self.crf_label = QLabel("23 (보통)")
        encoding_layout.addWidget(self.crf_label, 1, 2)
        self.crf_slider.valueChanged.connect(self.update_crf_label)
        
        encoding_layout.addWidget(QLabel("프리셋:"), 2, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "ultrafast", "superfast", "veryfast", "faster",
            "fast", "medium", "slow", "slower", "veryslow"
        ])
        self.preset_combo.setCurrentText("medium")
        encoding_layout.addWidget(self.preset_combo, 2, 1)
        
        layout.addWidget(encoding_group)
        
        # 메타데이터 설정
        metadata_group = QGroupBox("메타데이터")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.metadata_check = QCheckBox("360도 메타데이터 삽입")
        self.metadata_check.setChecked(True)
        metadata_layout.addWidget(self.metadata_check)
        
        self.insta360_check = QCheckBox("Insta360 Studio 호환 포맷")
        metadata_layout.addWidget(self.insta360_check)
        
        layout.addWidget(metadata_group)
        
        # 출력 경로
        output_group = QGroupBox("출력 경로")
        output_layout = QHBoxLayout(output_group)
        
        self.output_label = QLabel("선택되지 않음")
        output_layout.addWidget(self.output_label)
        
        output_btn = QPushButton("출력 경로 선택")
        output_btn.clicked.connect(self.select_output_path)
        output_layout.addWidget(output_btn)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
        return widget
    
    def create_preview_tab(self):
        """미리보기 탭 생성"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # 왼쪽: 미리보기 위젯
        left_layout = QVBoxLayout()
        
        self.preview_widget = PreviewWidget()
        self.preview_widget.orientation_changed.connect(self.on_orientation_changed)
        left_layout.addWidget(self.preview_widget)
        
        # 미리보기 컨트롤
        preview_control_group = QGroupBox("미리보기 컨트롤")
        preview_control_layout = QHBoxLayout(preview_control_group)
        
        generate_preview_btn = QPushButton("미리보기 생성")
        generate_preview_btn.clicked.connect(self.generate_preview)
        preview_control_layout.addWidget(generate_preview_btn)
        
        save_frame_btn = QPushButton("현재 프레임 저장")
        save_frame_btn.clicked.connect(self.save_current_frame)
        preview_control_layout.addWidget(save_frame_btn)
        
        left_layout.addWidget(preview_control_group)
        
        # 오른쪽: 방향 조정 컨트롤
        right_layout = QVBoxLayout()
        
        # 방향 조정 그룹
        orientation_group = QGroupBox("방향 조정")
        orientation_layout = QGridLayout(orientation_group)
        
        # Yaw 조정
        orientation_layout.addWidget(QLabel("Yaw (좌우):"), 0, 0)
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setRange(-180, 180)
        self.yaw_slider.setValue(0)
        self.yaw_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.yaw_slider.setTickInterval(30)
        self.yaw_slider.valueChanged.connect(self.on_yaw_changed)
        orientation_layout.addWidget(self.yaw_slider, 0, 1)
        
        self.yaw_spinbox = QSpinBox()
        self.yaw_spinbox.setRange(-180, 180)
        self.yaw_spinbox.setValue(0)
        self.yaw_spinbox.setSuffix("°")
        self.yaw_spinbox.valueChanged.connect(self.yaw_slider.setValue)
        orientation_layout.addWidget(self.yaw_spinbox, 0, 2)
        
        # Pitch 조정
        orientation_layout.addWidget(QLabel("Pitch (상하):"), 1, 0)
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-90, 90)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pitch_slider.setTickInterval(15)
        self.pitch_slider.valueChanged.connect(self.on_pitch_changed)
        orientation_layout.addWidget(self.pitch_slider, 1, 1)
        
        self.pitch_spinbox = QSpinBox()
        self.pitch_spinbox.setRange(-90, 90)
        self.pitch_spinbox.setValue(0)
        self.pitch_spinbox.setSuffix("°")
        self.pitch_spinbox.valueChanged.connect(self.pitch_slider.setValue)
        orientation_layout.addWidget(self.pitch_spinbox, 1, 2)
        
        # Roll 조정
        orientation_layout.addWidget(QLabel("Roll (기울기):"), 2, 0)
        self.roll_slider = QSlider(Qt.Orientation.Horizontal)
        self.roll_slider.setRange(-180, 180)
        self.roll_slider.setValue(0)
        self.roll_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.roll_slider.setTickInterval(30)
        self.roll_slider.valueChanged.connect(self.on_roll_changed)
        orientation_layout.addWidget(self.roll_slider, 2, 1)
        
        self.roll_spinbox = QSpinBox()
        self.roll_spinbox.setRange(-180, 180)
        self.roll_spinbox.setValue(0)
        self.roll_spinbox.setSuffix("°")
        self.roll_spinbox.valueChanged.connect(self.roll_slider.setValue)
        orientation_layout.addWidget(self.roll_spinbox, 2, 2)
        
        # 슬라이더와 스핀박스 연결
        self.yaw_slider.valueChanged.connect(self.yaw_spinbox.setValue)
        self.pitch_slider.valueChanged.connect(self.pitch_spinbox.setValue)
        self.roll_slider.valueChanged.connect(self.roll_spinbox.setValue)
        
        right_layout.addWidget(orientation_group)
        
        # 프리셋 그룹
        preset_group = QGroupBox("방향 프리셋")
        preset_layout = QVBoxLayout(preset_group)
        
        preset_buttons = [
            ("정면", 0, 0, 0),
            ("우측", 90, 0, 0),
            ("후면", 180, 0, 0),
            ("좌측", -90, 0, 0),
            ("상단", 0, 45, 0),
            ("하단", 0, -45, 0)
        ]
        
        for name, yaw, pitch, roll in preset_buttons:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, y=yaw, p=pitch, r=roll: self.set_orientation_preset(y, p, r))
            preset_layout.addWidget(btn)
        
        right_layout.addWidget(preset_group)
        
        # 안내 텍스트
        help_group = QGroupBox("사용법")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QLabel(
            "• 마우스 드래그: Yaw/Pitch 조정\n"
            "• 마우스 휠: Roll 조정\n"
            "• 슬라이더: 정밀 조정\n"
            "• 프리셋: 빠른 방향 설정"
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        right_layout.addWidget(help_group)
        right_layout.addStretch()
        
        # 레이아웃 구성
        layout.addLayout(left_layout, 2)  # 미리보기가 더 넓게
        layout.addLayout(right_layout, 1)  # 컨트롤은 좁게
        
        return widget
    
    def create_progress_section(self):
        """진행 상황 영역 생성"""
        layout = QVBoxLayout()
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 컨트롤 버튼
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("스티칭 시작")
        self.start_btn.clicked.connect(self.start_stitching)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("일시정지")
        self.pause_btn.clicked.connect(self.pause_stitching)
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_stitching)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        return layout
    
    def update_crf_label(self, value):
        """CRF 레이블 업데이트"""
        if value <= 18:
            quality = "매우 높음"
        elif value <= 23:
            quality = "높음"
        elif value <= 28:
            quality = "보통"
        else:
            quality = "낮음"
        self.crf_label.setText(f"{value} ({quality})")
    
    def select_files(self, camera_type):
        """파일 선택"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"{camera_type} 카메라 파일 선택",
            "",
            "MP4 Files (*.MP4 *.mp4);;All Files (*.*)"
        )
        
        if files:
            if camera_type == "front":
                self.front_files = [Path(f) for f in files]
                self.front_list.clear()
                self.front_list.addItems([f.name for f in self.front_files])
            else:
                self.back_files = [Path(f) for f in files]
                self.back_list.clear()
                self.back_list.addItems([f.name for f in self.back_files])
            
            self.log_text.append(f"{camera_type} 카메라 파일 {len(files)}개 선택됨")
    
    def auto_detect_files(self):
        """폴더에서 자동 감지"""
        folder = QFileDialog.getExistingDirectory(self, "GoPro 파일 폴더 선택")
        
        if folder:
            self.log_text.append(f"폴더 검색 중: {folder}")
            folder_path = Path(folder)
            
            # preprocessor의 detect_gopro_files 호출
            front_files, back_files = self.preprocessor.detect_gopro_files(folder_path)
            
            if front_files or back_files:
                self.front_files = front_files
                self.back_files = back_files
                
                # 리스트 위젯 업데이트
                self.front_list.clear()
                self.front_list.addItems([f.name for f in self.front_files])
                
                self.back_list.clear()
                self.back_list.addItems([f.name for f in self.back_files])
                
                self.log_text.append(f"자동 감지 완료: 전면 {len(front_files)}개, 후면 {len(back_files)}개")
            else:
                self.log_text.append("GoPro 파일을 찾을 수 없습니다.")
    
    def select_output_path(self):
        """출력 경로 선택"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "출력 파일 경로 선택",
            "",
            "MP4 Files (*.mp4);;All Files (*.*)"
        )
        
        if file_path:
            self.output_path = Path(file_path)
            self.output_label.setText(str(self.output_path))
            self.log_text.append(f"출력 경로 설정: {self.output_path}")
    
    def load_calibration(self):
        """캘리브레이션 로드"""
        if self.calib_combo.currentText() == "커스텀...":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "캘리브레이션 파일 선택",
                "",
                "YAML Files (*.yaml *.yml);;All Files (*.*)"
            )
            if file_path:
                self.log_text.append(f"커스텀 캘리브레이션 파일 선택: {file_path}")
        else:
            self.log_text.append(f"캘리브레이션 프리셋 사용: {self.calib_combo.currentText()}")
    
    def start_stitching(self):
        """스티칭 시작"""
        if not self.front_files or not self.back_files:
            QMessageBox.warning(self, "경고", "전면과 우측 카메라 파일을 모두 선택해주세요.")
            return
        
        if not self.output_path:
            QMessageBox.warning(self, "경고", "출력 경로를 선택해주세요.")
            return
        
        # 설정 수집
        settings = self.collect_settings()
        
        # 진행 상황 다이얼로그 생성
        self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.show()
        
        # 스티칭 스레드 생성
        self.stitching_thread = StitchingThread()
        self.stitching_thread.set_parameters(
            self.front_files, self.back_files,
            self.output_path, settings
        )
        
        # 시그널 연결
        self.stitching_thread.progress_update.connect(self.on_progress_update)
        self.stitching_thread.step_update.connect(self.on_step_update)
        self.stitching_thread.log_message.connect(self.on_log_message)
        self.stitching_thread.error_occurred.connect(self.on_error)
        self.stitching_thread.finished.connect(self.on_finished)
        self.stitching_thread.preview_ready.connect(self.on_preview_ready)
        
        # 다이얼로그 취소 시그널 연결
        self.progress_dialog.cancelled.connect(self.on_cancel_clicked)
        
        # 스레드 시작
        self.stitching_thread.start()
        
        self.log_text.append("스티칭 시작...")
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
    
    def pause_stitching(self):
        """스티칭 일시정지"""
        if self.stitching_thread and self.stitching_thread.isRunning():
            if self.pause_btn.text() == "일시정지":
                self.stitching_thread.pause()
                self.pause_btn.setText("재개")
                self.log_text.append("스티칭 일시정지")
            else:
                self.stitching_thread.resume()
                self.pause_btn.setText("일시정지")
                self.log_text.append("스티칭 재개")
    
    def stop_stitching(self):
        """스티칭 중지"""
        if self.stitching_thread and self.stitching_thread.isRunning():
            self.stitching_thread.cancel()
            self.log_text.append("스티칭 중지 요청...")
        else:
            self.reset_controls()
    
    def reset_controls(self):
        """컨트롤 초기화"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("일시정지")
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
    
    def new_project(self):
        """새 프로젝트 생성"""
        # 현재 프로젝트가 수정되었는지 확인
        if self.check_unsaved_changes():
            reply = QMessageBox.question(
                self, "새 프로젝트", 
                "현재 프로젝트의 변경사항이 저장되지 않았습니다.\n새 프로젝트를 생성하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.project_manager.create_new_project()
        self.reset_gui_to_defaults()
        self.update_window_title()
        self.log_text.append("새 프로젝트 생성됨")
    
    def open_project(self):
        """프로젝트 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "프로젝트 열기",
            "",
            "PyStitch360 Project Files (*.pys360);;All Files (*.*)"
        )
        
        if file_path:
            self.load_project_file(Path(file_path))
    
    def load_project_file(self, file_path: Path):
        """프로젝트 파일 로드"""
        project_data = self.project_manager.load_project(file_path)
        
        if project_data:
            self.apply_project_settings(project_data)
            self.project_manager.add_to_recent_projects(
                file_path, 
                project_data["project_info"]["name"]
            )
            self.update_recent_projects_menu()
            self.update_window_title()
            self.log_text.append(f"프로젝트 로드됨: {file_path.name}")
        else:
            QMessageBox.critical(self, "오류", "프로젝트 파일을 열 수 없습니다.")
    
    def save_project(self):
        """프로젝트 저장"""
        if self.project_manager.current_project_path:
            self.save_current_project()
        else:
            self.save_project_as()
    
    def save_project_as(self):
        """다른 이름으로 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "프로젝트 저장",
            f"{self.project_manager.get_current_project_name()}.pys360",
            "PyStitch360 Project Files (*.pys360);;All Files (*.*)"
        )
        
        if file_path:
            file_path = Path(file_path)
            if file_path.suffix != ".pys360":
                file_path = file_path.with_suffix(".pys360")
            
            self.save_project_to_file(file_path)
    
    def save_current_project(self):
        """현재 프로젝트 저장"""
        if self.project_manager.current_project_path:
            self.save_project_to_file(self.project_manager.current_project_path)
        else:
            self.save_project_as()
    
    def save_project_to_file(self, file_path: Path):
        """지정된 파일에 프로젝트 저장"""
        project_data = self.collect_project_data()
        
        if self.project_manager.save_project(file_path, project_data):
            self.project_manager.add_to_recent_projects(
                file_path,
                project_data["project_info"]["name"]
            )
            self.update_recent_projects_menu()
            self.update_window_title()
            self.log_text.append(f"프로젝트 저장됨: {file_path.name}")
            QMessageBox.information(self, "저장 완료", "프로젝트가 저장되었습니다.")
        else:
            QMessageBox.critical(self, "오류", "프로젝트 저장에 실패했습니다.")
    
    def save_as_template(self):
        """템플릿으로 저장"""
        template_name, ok = self.get_template_name()
        if not ok:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "템플릿 저장",
            f"{template_name}.pys360",
            "PyStitch360 Project Files (*.pys360);;All Files (*.*)"
        )
        
        if file_path:
            file_path = Path(file_path)
            if file_path.suffix != ".pys360":
                file_path = file_path.with_suffix(".pys360")
            
            if self.project_manager.save_as_template(file_path, template_name):
                self.log_text.append(f"템플릿 저장됨: {file_path.name}")
                QMessageBox.information(self, "저장 완료", "템플릿이 저장되었습니다.")
            else:
                QMessageBox.critical(self, "오류", "템플릿 저장에 실패했습니다.")
    
    def import_settings(self):
        """설정 가져오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "설정 가져오기",
            "",
            "PyStitch360 Project Files (*.pys360);;JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 파일 경로 제외하고 설정만 적용
                if "stitching" in settings:
                    self.apply_stitching_settings(settings["stitching"])
                if "orientation" in settings:
                    self.apply_orientation_settings(settings["orientation"])
                if "postprocessing" in settings:
                    self.apply_postprocessing_settings(settings["postprocessing"])
                
                self.log_text.append("설정 가져오기 완료")
                QMessageBox.information(self, "가져오기 완료", "설정을 가져왔습니다.")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"설정 가져오기 실패: {str(e)}")
    
    def export_settings(self):
        """설정 내보내기"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "설정 내보내기",
            "pystitch360_settings.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                settings = {
                    "stitching": {
                        "calibration_preset": self.calib_combo.currentText(),
                        "blend_type": self.blend_type_combo.currentText(),
                        "feather_width": self.feather_slider.value(),
                        "projection_type": self.proj_combo.currentText(),
                        "output_resolution": self.resolution_combo.currentText()
                    },
                    "orientation": {
                        "yaw": self.yaw_slider.value() if hasattr(self, 'yaw_slider') else 0,
                        "pitch": self.pitch_slider.value() if hasattr(self, 'pitch_slider') else 0,
                        "roll": self.roll_slider.value() if hasattr(self, 'roll_slider') else 0
                    },
                    "postprocessing": {
                        "codec": self.codec_combo.currentText(),
                        "crf": self.crf_slider.value(),
                        "preset": self.preset_combo.currentText(),
                        "metadata_enabled": self.metadata_check.isChecked(),
                        "insta360_compatible": self.insta360_check.isChecked()
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
                
                self.log_text.append("설정 내보내기 완료")
                QMessageBox.information(self, "내보내기 완료", "설정을 내보냈습니다.")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"설정 내보내기 실패: {str(e)}")
    
    def manage_calibration(self):
        """캘리브레이션 관리"""
        QMessageBox.information(self, "정보", "캘리브레이션 관리 기능은 추후 구현 예정입니다.")
    
    def show_about(self):
        """프로그램 정보 표시"""
        QMessageBox.about(
            self,
            "PyStitch360 정보",
            "PyStitch360 v1.0.0\n\n"
            "GoPro 듀얼 카메라 360도 영상 스티칭 툴\n\n"
            "개발: PyStitch360 Team\n"
            "라이선스: MIT"
        )
    
    def collect_settings(self):
        """현재 GUI 설정을 딕셔너리로 수집"""
        settings = {
            # 동기화 설정
            "sync_offset": self.sync_spinbox.value(),
            
            # 캘리브레이션
            "calibration": self.calib_combo.currentText(),
            
            # 블렌딩 설정
            "blend_type": self.blend_type_combo.currentText(),
            "feather_width": self.feather_slider.value(),
            
            # 투영 설정
            "projection": self.proj_combo.currentText(),
            "resolution": self.resolution_combo.currentText().split()[0],  # "3840x1920 (4K)" -> "3840x1920"
            
            # 인코딩 설정
            "codec": self.codec_combo.currentText(),
            "crf": self.crf_slider.value(),
            "preset": self.preset_combo.currentText(),
            
            # 메타데이터
            "metadata_enabled": self.metadata_check.isChecked(),
            "insta360_compatible": self.insta360_check.isChecked(),
            
            # 방향 조정
            "yaw": self.yaw_slider.value() if hasattr(self, 'yaw_slider') else 0,
            "pitch": self.pitch_slider.value() if hasattr(self, 'pitch_slider') else 0,
            "roll": self.roll_slider.value() if hasattr(self, 'roll_slider') else 0
        }
        
        return settings
    
    # 스레드 시그널 핸들러
    def on_progress_update(self, current, total):
        """진행률 업데이트"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.progress_dialog.set_overall_progress(current, total)
    
    def on_step_update(self, step_name):
        """단계 업데이트"""
        self.progress_dialog.set_task(step_name)
        self.status_bar.showMessage(step_name)
    
    def on_log_message(self, message):
        """로그 메시지"""
        self.log_text.append(message)
        self.progress_dialog.append_log(message)
    
    def on_error(self, error_message):
        """오류 발생"""
        self.log_text.append(f"[오류] {error_message}")
        self.progress_dialog.append_log(f"[오류] {error_message}")
        QMessageBox.critical(self, "오류", error_message)
    
    def on_finished(self, success):
        """작업 완료"""
        self.progress_dialog.finish(success)
        self.reset_controls()
        
        if success:
            self.status_bar.showMessage("스티칭 완료!")
            QMessageBox.information(self, "완료", "360도 영상 스티칭이 완료되었습니다!")
        else:
            self.status_bar.showMessage("스티칭 실패")
    
    def on_preview_ready(self, image):
        """미리보기 이미지 준비됨"""
        self.preview_widget.set_image(image)
        self.log_text.append("미리보기 이미지 생성됨")
        # 미리보기 탭으로 자동 전환
        self.tab_widget.setCurrentIndex(3)
    
    def on_cancel_clicked(self):
        """취소 버튼 클릭"""
        if self.stitching_thread and self.stitching_thread.isRunning():
            self.stitching_thread.cancel()
    
    # 미리보기 및 방향 조정 메서드
    def generate_preview(self):
        """미리보기 생성"""
        if not self.front_files or not self.back_files:
            QMessageBox.warning(self, "경고", "전면과 우측 카메라 파일을 먼저 선택해주세요.")
            return
        
        self.log_text.append("미리보기 생성 시작...")
        
        try:
            import cv2
            from core.stitcher import Stitcher
            
            # 첫 번째 파일의 첫 프레임 읽기
            front_cap = cv2.VideoCapture(str(self.front_files[0]))
            back_cap = cv2.VideoCapture(str(self.back_files[0]))
            
            ret1, front_frame = front_cap.read()
            ret2, back_frame = back_cap.read()
            
            front_cap.release()
            back_cap.release()
            
            if not ret1 or not ret2:
                QMessageBox.warning(self, "오류", "영상 프레임을 읽을 수 없습니다.")
                return
            
            # 스티처 초기화
            stitcher = Stitcher()
            calib_path = Path("presets") / self.calib_combo.currentText()
            stitcher.load_calibration(calib_path)
            
            # 파노라마 생성
            panorama = stitcher.create_panorama(front_frame, back_frame)
            
            if panorama is not None:
                # Equirectangular 투영
                resolution = self.resolution_combo.currentText().split()[0]
                width, height = map(int, resolution.split('x'))
                equirect = stitcher.apply_equirectangular_projection(panorama, width, height)
                
                # 현재 방향 설정 적용
                yaw = self.yaw_slider.value()
                pitch = self.pitch_slider.value()
                roll = self.roll_slider.value()
                final_image = stitcher.apply_orientation(equirect, yaw, pitch, roll)
                
                # 미리보기 위젯에 설정
                self.preview_widget.set_image(final_image)
                self.tab_widget.setCurrentIndex(3)  # 미리보기 탭으로 전환
                self.log_text.append("미리보기 생성 완료")
            else:
                QMessageBox.warning(self, "오류", "파노라마 생성에 실패했습니다.")
                
        except Exception as e:
            self.log_text.append(f"미리보기 생성 오류: {str(e)}")
            QMessageBox.critical(self, "오류", f"미리보기 생성 중 오류 발생: {str(e)}")
    
    def save_current_frame(self):
        """현재 프레임 저장"""
        if self.preview_widget.image is None:
            QMessageBox.warning(self, "경고", "저장할 미리보기 이미지가 없습니다.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "프레임 저장",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import cv2
                cv2.imwrite(file_path, self.preview_widget.image)
                self.log_text.append(f"프레임 저장됨: {file_path}")
                QMessageBox.information(self, "저장 완료", "현재 프레임이 저장되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 중 오류 발생: {str(e)}")
    
    def on_orientation_changed(self, yaw, pitch, roll):
        """미리보기 위젯에서 방향이 변경됨"""
        # 슬라이더 업데이트 (시그널 방지를 위해 일시적으로 차단)
        self.yaw_slider.blockSignals(True)
        self.pitch_slider.blockSignals(True)
        self.roll_slider.blockSignals(True)
        
        self.yaw_slider.setValue(int(yaw))
        self.pitch_slider.setValue(int(pitch))
        self.roll_slider.setValue(int(roll))
        
        self.yaw_slider.blockSignals(False)
        self.pitch_slider.blockSignals(False)
        self.roll_slider.blockSignals(False)
    
    def on_yaw_changed(self, value):
        """Yaw 슬라이더 변경"""
        self.preview_widget.set_orientation(value, self.pitch_slider.value(), self.roll_slider.value())
    
    def on_pitch_changed(self, value):
        """Pitch 슬라이더 변경"""
        self.preview_widget.set_orientation(self.yaw_slider.value(), value, self.roll_slider.value())
    
    def on_roll_changed(self, value):
        """Roll 슬라이더 변경"""
        self.preview_widget.set_orientation(self.yaw_slider.value(), self.pitch_slider.value(), value)
    
    def set_orientation_preset(self, yaw, pitch, roll):
        """방향 프리셋 설정"""
        self.yaw_slider.setValue(yaw)
        self.pitch_slider.setValue(pitch)
        self.roll_slider.setValue(roll)
        self.preview_widget.set_orientation(yaw, pitch, roll)
    
    # 프로젝트 관리 헬퍼 메서드
    def collect_project_data(self):
        """현재 GUI 상태를 프로젝트 데이터로 수집"""
        settings = self.collect_settings()
        
        project_data = {
            "project_info": {
                "name": self.project_manager.get_current_project_name(),
                "created_date": self.project_manager.project_data.get("project_info", {}).get("created_date", ""),
                "modified_date": "",  # ProjectManager에서 설정
                "version": "1.0.0"
            },
            "input_files": {
                "left_camera": [str(f) for f in self.front_files],
                "right_camera": [str(f) for f in self.back_files]
            },
            "preprocessing": {
                "sync_offset_frames": settings["sync_offset"],
                "concat_method": "demuxer"
            },
            "stitching": {
                "calibration_preset": settings["calibration"],
                "blend_type": settings["blend_type"],
                "feather_width": settings["feather_width"],
                "projection_type": settings["projection"],
                "output_resolution": settings["resolution"]
            },
            "orientation": {
                "yaw": float(settings["yaw"]),
                "pitch": float(settings["pitch"]),
                "roll": float(settings["roll"])
            },
            "postprocessing": {
                "encoding": {
                    "codec": settings["codec"],
                    "crf": settings["crf"],
                    "preset": settings["preset"]
                },
                "metadata": {
                    "enabled": settings["metadata_enabled"],
                    "projection": "equirectangular",
                    "insta360_compatible": settings["insta360_compatible"]
                }
            },
            "output": {
                "path": str(self.output_path) if self.output_path else "",
                "format": "mp4"
            }
        }
        
        return project_data
    
    def apply_project_settings(self, project_data):
        """프로젝트 데이터를 GUI에 적용"""
        try:
            # 입력 파일
            if "input_files" in project_data:
                self.front_files = [Path(f) for f in project_data["input_files"].get("left_camera", [])]
                self.back_files = [Path(f) for f in project_data["input_files"].get("right_camera", [])]
                
                self.front_list.clear()
                self.front_list.addItems([f.name for f in self.front_files])
                self.back_list.clear()
                self.back_list.addItems([f.name for f in self.back_files])
            
            # 전처리 설정
            if "preprocessing" in project_data:
                sync_offset = project_data["preprocessing"].get("sync_offset_frames", 0)
                self.sync_slider.setValue(sync_offset)
            
            # 스티칭 설정
            if "stitching" in project_data:
                self.apply_stitching_settings(project_data["stitching"])
            
            # 방향 설정
            if "orientation" in project_data:
                self.apply_orientation_settings(project_data["orientation"])
            
            # 후처리 설정
            if "postprocessing" in project_data:
                self.apply_postprocessing_settings(project_data["postprocessing"])
            
            # 출력 경로
            if "output" in project_data and project_data["output"]["path"]:
                self.output_path = Path(project_data["output"]["path"])
                self.output_label.setText(str(self.output_path))
            
        except Exception as e:
            self.logger.error(f"프로젝트 설정 적용 실패: {e}")
            QMessageBox.warning(self, "경고", "일부 설정을 적용하지 못했습니다.")
    
    def apply_stitching_settings(self, settings):
        """스티칭 설정 적용"""
        if "calibration_preset" in settings:
            index = self.calib_combo.findText(settings["calibration_preset"])
            if index >= 0:
                self.calib_combo.setCurrentIndex(index)
        
        if "blend_type" in settings:
            index = self.blend_type_combo.findText(settings["blend_type"])
            if index >= 0:
                self.blend_type_combo.setCurrentIndex(index)
        
        if "feather_width" in settings:
            self.feather_slider.setValue(settings["feather_width"])
        
        if "projection_type" in settings:
            index = self.proj_combo.findText(settings["projection_type"])
            if index >= 0:
                self.proj_combo.setCurrentIndex(index)
        
        if "output_resolution" in settings:
            for i in range(self.resolution_combo.count()):
                if settings["output_resolution"] in self.resolution_combo.itemText(i):
                    self.resolution_combo.setCurrentIndex(i)
                    break
    
    def apply_orientation_settings(self, settings):
        """방향 설정 적용"""
        if hasattr(self, 'yaw_slider'):
            self.yaw_slider.setValue(int(settings.get("yaw", 0)))
            self.pitch_slider.setValue(int(settings.get("pitch", 0)))
            self.roll_slider.setValue(int(settings.get("roll", 0)))
    
    def apply_postprocessing_settings(self, settings):
        """후처리 설정 적용"""
        if "encoding" in settings:
            encoding = settings["encoding"]
            
            if "codec" in encoding:
                index = self.codec_combo.findText(encoding["codec"])
                if index >= 0:
                    self.codec_combo.setCurrentIndex(index)
            
            if "crf" in encoding:
                self.crf_slider.setValue(encoding["crf"])
            
            if "preset" in encoding:
                index = self.preset_combo.findText(encoding["preset"])
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
        
        if "metadata" in settings:
            metadata = settings["metadata"]
            self.metadata_check.setChecked(metadata.get("enabled", True))
            self.insta360_check.setChecked(metadata.get("insta360_compatible", False))
    
    def reset_gui_to_defaults(self):
        """GUI를 기본값으로 리셋"""
        # 파일 목록 초기화
        self.front_files = []
        self.back_files = []
        self.output_path = None
        self.front_list.clear()
        self.back_list.clear()
        self.output_label.setText("선택되지 않음")
        
        # 설정 초기화
        self.sync_slider.setValue(0)
        self.calib_combo.setCurrentIndex(0)
        self.blend_type_combo.setCurrentIndex(0)
        self.feather_slider.setValue(50)
        self.proj_combo.setCurrentIndex(0)
        self.resolution_combo.setCurrentIndex(0)
        self.codec_combo.setCurrentIndex(0)
        self.crf_slider.setValue(23)
        self.preset_combo.setCurrentText("medium")
        self.metadata_check.setChecked(True)
        self.insta360_check.setChecked(False)
        
        # 방향 초기화
        if hasattr(self, 'yaw_slider'):
            self.yaw_slider.setValue(0)
            self.pitch_slider.setValue(0)
            self.roll_slider.setValue(0)
            self.preview_widget.reset_orientation()
    
    def update_window_title(self):
        """윈도우 제목 업데이트"""
        project_name = self.project_manager.get_current_project_name()
        title = f"PyStitch360 - {project_name}"
        
        if self.project_manager.current_project_path:
            title += f" [{self.project_manager.current_project_path.name}]"
        else:
            title += " [저장되지 않음]"
        
        self.setWindowTitle(title)
    
    def update_recent_projects_menu(self):
        """최근 프로젝트 메뉴 업데이트"""
        self.recent_menu.clear()
        
        recent_projects = self.project_manager.get_recent_projects()
        
        if recent_projects:
            for project in recent_projects:
                action = QAction(project["name"], self)
                action.setToolTip(project["path"])
                action.triggered.connect(
                    lambda checked, path=project["path"]: self.load_project_file(Path(path))
                )
                self.recent_menu.addAction(action)
            
            self.recent_menu.addSeparator()
            clear_action = QAction("목록 지우기", self)
            clear_action.triggered.connect(self.clear_recent_projects)
            self.recent_menu.addAction(clear_action)
        else:
            no_recent_action = QAction("최근 프로젝트 없음", self)
            no_recent_action.setEnabled(False)
            self.recent_menu.addAction(no_recent_action)
    
    def clear_recent_projects(self):
        """최근 프로젝트 목록 지우기"""
        settings_file = Path.home() / ".pystitch360" / "recent_projects.json"
        if settings_file.exists():
            settings_file.unlink()
        self.update_recent_projects_menu()
    
    def check_unsaved_changes(self):
        """저장되지 않은 변경사항 확인"""
        current_settings = self.collect_project_data()
        return self.project_manager.is_project_modified(current_settings)
    
    def get_template_name(self):
        """템플릿 이름 입력 다이얼로그"""
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self,
            "템플릿 이름",
            "템플릿 이름을 입력하세요:",
            text="새 템플릿"
        )
        
        return name.strip(), ok and bool(name.strip())
    
    def closeEvent(self, event):
        """윈도우 닫기 이벤트"""
        if self.check_unsaved_changes():
            reply = QMessageBox.question(
                self, "종료", 
                "현재 프로젝트의 변경사항이 저장되지 않았습니다.\n저장하지 않고 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # 스티칭 스레드 정리
        if self.stitching_thread and self.stitching_thread.isRunning():
            self.stitching_thread.cancel()
            self.stitching_thread.wait(3000)  # 3초 대기
        
        event.accept()