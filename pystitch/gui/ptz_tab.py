"""4번 탭: 가상 PTZ — 자동 공 검출을 기본으로 깔고 클릭으로 키프레임 교정.

워크플로우: 완성 파노라마 열기 → 자동 분석(캐시) → 타임라인을 훑으며
공이 아닌 곳을 보는 구간에서 화면 클릭(=키프레임) → PTZ 내보내기.
키프레임은 <파노라마>.ptz_keyframes.json 에 자동 저장된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QCheckBox
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMenu, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..core.encoders import available_encoders
from ..core.ptz import (
    accept_ball_tracks, analyze_video, build_plan, classify_teams,
    ground_positions, link_ball_tracks, ptz_available, render_plan,
    same_spot_spans,
)
from .widgets import FramePane


TEAM_COLORS = [(60, 60, 230), (230, 140, 40), (60, 200, 230)]   # BGR: 팀0, 팀1, 기타


class RadarView(QWidget):
    """탑다운 레이더: 카메라 기준 지면 좌표(m)의 선수(팀 색)·공 표시."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(180)
        self.points: list = []      # (X, Y, team)
        self.ball = None            # (X, Y) or None

    def set_data(self, points, ball=None):
        self.points, self.ball = list(points), ball
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 60, 34))
        W, H = self.width(), self.height()
        # 표시 범위: X -55..55m, Y 0..75m (카메라 = 하단 중앙)
        def px(X, Y):
            return (int((X + 55) / 110 * (W - 1)),
                    int((1 - Y / 75) * (H - 1)))
        p.setPen(QColor(255, 255, 255, 45))
        for gx in range(-50, 51, 10):
            p.drawLine(*px(gx, 0), *px(gx, 75))
        for gy in range(0, 76, 10):
            p.drawLine(*px(-55, gy), *px(55, gy))
        p.setPen(QColor(255, 255, 255, 140))
        p.drawLine(*px(0, 0), *px(0, 75))            # 하프라인 방향
        for X, Y, team in self.points:
            b, g, r = TEAM_COLORS[min(max(team, 0), 2)]
            p.setBrush(QColor(r, g, b))
            p.setPen(Qt.PenStyle.NoPen)
            x, y = px(X, Y)
            p.drawEllipse(x - 4, y - 4, 8, 8)
        if self.ball is not None:
            p.setBrush(QColor(255, 255, 255))
            x, y = px(*self.ball)
            p.drawEllipse(x - 3, y - 3, 6, 6)
        p.setPen(QColor(255, 255, 255, 160))
        p.drawText(6, 14, "레이더 (카메라 기준, 10m 격자)")
        p.end()


class TrackBar(QWidget):
    """타임라인 위 공 트랙 표시줄: 초록=수락 트랙, 빨강=무시 구간, 주황=키프레임.

    클릭하면 해당 프레임으로 이동한다.
    """

    seek = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(16)
        self.total = 0
        self.spans: list = []
        self.ignores: list = []
        self.kfs: list = []

    def set_data(self, total, spans, ignores, kfs):
        self.total = max(1, total)
        self.spans, self.ignores, self.kfs = list(spans), list(ignores), list(kfs)
        self.update()

    def _x(self, f):
        return int(f / self.total * (self.width() - 1))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        h = self.height()
        for f0, f1 in self.spans:
            p.fillRect(self._x(f0), 2, max(1, self._x(f1) - self._x(f0)), h - 4,
                       QColor(70, 200, 90))
        for r in self.ignores:
            f0, f1 = r[0], r[1]
            p.fillRect(self._x(f0), 2, max(1, self._x(f1) - self._x(f0)), h - 4,
                       QColor(220, 70, 60))
        for kf in self.kfs:
            p.fillRect(self._x(kf[0]) - 1, 0, 3, h, QColor(255, 190, 0))
        p.end()

    def mousePressEvent(self, ev):
        if self.total > 1:
            self.seek.emit(int(ev.position().x() / max(1, self.width() - 1) * self.total))


class AnalyzeWorker(QThread):
    done = pyqtSignal(dict)
    progress = pyqtSignal(int, int, float)   # done_frames, total, fps
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, weights=None, checkpoint_path=None):
        super().__init__()
        self.video_path = video_path
        self.weights = weights
        self.checkpoint_path = checkpoint_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            d = analyze_video(self.video_path, weights=self.weights,
                              cancel=lambda: self._cancel,
                              progress=lambda i, t, f: self.progress.emit(i, t, f),
                              checkpoint_path=self.checkpoint_path,
                              log=lambda s: self.log.emit(s))
            if d is None:
                self.failed.emit("취소됨")
            else:
                self.done.emit(d)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class LinkWorker(QThread):
    """트랙 연결(느린 단계) 백그라운드 계산 — 분석에만 의존하므로 1회면 됨."""

    done = pyqtSignal(dict)

    def __init__(self, analysis):
        super().__init__()
        self.analysis = analysis

    def run(self):
        linked = link_ball_tracks(self.analysis)
        linked["teams"] = classify_teams(self.analysis)
        self.done.emit(linked)


class PlanWorker(QThread):
    """크롭 계획 미리보기 재계산 (키프레임/무시 변경 시 백그라운드)."""

    done = pyqtSignal(dict, tuple)   # plan, (out_w, out_h)

    def __init__(self, analysis, keyframes, ignores, wide, linked=None,
                 far_zoom=1.0, promotes=None):
        super().__init__()
        self.args = (analysis, keyframes, ignores, wide, linked, far_zoom,
                     promotes or [])

    def run(self):
        analysis, kfs, ignores, wide, linked, far_zoom, promotes = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes, linked=linked,
                              far_zoom=far_zoom,
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0, log=None)
            self.done.emit(plan, (out_w, out_h))
        except Exception:  # noqa: BLE001
            pass


class ProxyWorker(QThread):
    """스크럽 프록시 생성: 1920px, 키프레임 0.5s 간격 → 즉시 탐색용."""

    progress = pyqtSignal(float)         # 0.0 ~ 1.0
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, proxy_path, duration_sec=0.0):
        super().__init__()
        self.pano_path, self.proxy_path = str(pano_path), str(proxy_path)
        self.duration = duration_sec

    def run(self):
        import subprocess

        from ..core.encoders import available_encoders, ffmpeg_bin
        encs = available_encoders().values()
        if "h264_nvenc" in encs:
            venc = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr",
                    "-cq", "27", "-b:v", "0"]
        else:
            venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]
        tmp = self.proxy_path + ".part.mp4"
        cmd = ([ffmpeg_bin(), "-y", "-v", "error", "-nostats",
                "-i", self.pano_path, "-vf", "scale=1920:-2"] + venc
               + ["-g", "15", "-an", "-progress", "pipe:1", tmp])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            tail = []
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms=") and self.duration > 0:
                    try:                     # ffmpeg 의 out_time_ms 는 마이크로초
                        t = int(line.split("=", 1)[1]) / 1e6
                        self.progress.emit(min(t / self.duration, 1.0))
                    except ValueError:
                        pass
                elif line and "=" not in line:
                    tail.append(line)        # 진행 키가 아닌 줄 = 오류 메시지 후보
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(" / ".join(tail[-3:]) or "ffmpeg 실패")
            Path(tmp).rename(self.proxy_path)
            self.finished_ok.emit(self.proxy_path)
        except Exception as e:  # noqa: BLE001
            Path(tmp).unlink(missing_ok=True)
            self.failed.emit(str(e))


class PtzPlayWorker(QThread):
    """파노라마 순차 재생 (자체 디코더) — 오버레이는 GUI 쪽 _redraw 가 담당."""

    frame_ready = pyqtSignal(object, int)

    def __init__(self, path, start_frame, fps, display_fps=15.0):
        super().__init__()
        self.path, self.start_frame = str(path), start_frame
        self.fps, self.display_fps = fps, display_fps
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import time
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            step = max(1, int(round(self.fps / self.display_fps)))
            f = self.start_frame
            t0 = time.perf_counter()
            shown = 0
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_ready.emit(frame, f)
                shown += 1
                lag = shown * step / self.fps - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
                for _ in range(step - 1):
                    cap.grab()
                f += step
        finally:
            cap.release()


class PtzRenderWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_path, analysis, keyframes, codec, crf,
                 wide=False, ignores=None, far_zoom=1.0, promotes=None):
        super().__init__()
        self.args = (pano_path, out_path, analysis, keyframes, codec, crf, wide,
                     ignores or [], far_zoom, promotes or [])
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        (pano, out, analysis, kfs, codec, crf, wide, ignores, far_zoom,
         promotes) = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes,
                              far_zoom=far_zoom,
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0,
                              log=lambda s: self.log.emit(s))
            render_plan(pano, out, plan, out_w=out_w, out_h=out_h,
                        codec=codec, crf=crf,
                        log=lambda s: self.log.emit(s),
                        progress=lambda d, t, f: self.progress.emit(d, t, f),
                        cancel=lambda: self._cancel)
            if self._cancel:
                self.failed.emit("취소됨")
            else:
                self.finished_ok.emit(str(out))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PtzTab(QWidget):
    """가상 PTZ 탭. log_fn 은 메인 윈도우 로그 박스."""

    def __init__(self, log_fn, video_dir_fn=None, remember_dir_fn=None):
        super().__init__()
        self.log = log_fn
        self._video_dir = video_dir_fn or (lambda: "")
        self._remember_dir = remember_dir_fn or (lambda p: None)
        self.pano_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.total = 0
        self.pano_w = self.pano_h = 0
        self.analysis: dict | None = None
        self.keyframes: list[list] = []   # [frame, x, y]
        self.ignores: list[list] = []     # [f0, f1] 사용자 지정 오인식 구간
        self.promotes: list[list] = []    # [f, x, y] 회색 공 → 트랙 강제 수락
        self.track_spans: list = []
        self._accepted_ball = None        # accept_ball_tracks 의 샘플별 수락 공
        self._hover = None                # 커서가 가리키는 오브젝트
        self._hover_key = None            # hover 변경 감지 키
        self._plan_box = None             # 현재 프레임에 그려진 크롭 박스 (x0,y0,w,h)
        self._box_hover = None            # 크롭 박스 hover 존: ("corner",i)|("border",None)
        self._box_edit = None             # 진행 중인 박스 드래그 상태
        self._analyze_worker = None
        self._render_worker = None
        self._plan_worker = None
        self._link_worker = None
        self._linked = None
        self._teams = {}
        self._play_worker = None
        self._playing = False
        self._proxy_worker = None
        self.disp_path = None
        self.disp_scale = 1.0
        self.plan = None
        self.plan_out = (1920, 1080)
        self._build_ui()
        self._plan_timer = QTimer(singleShot=True, interval=150)
        self._plan_timer.timeout.connect(self._run_plan)
        self._save_timer = QTimer(singleShot=True, interval=1500)
        self._save_timer.timeout.connect(self._write_sidecar)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_open = QPushButton("파노라마 영상 열기...")
        self.btn_open.clicked.connect(self._open_pano)
        self.lbl_file = QLabel("완성된 파노라마 mp4 를 열어주세요")
        self.btn_analyze = QPushButton("자동 공/선수 분석")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_analyze)
        self.btn_proxy = QPushButton("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setToolTip("1920px·조밀 키프레임 사본 — 타임라인 클릭 즉시 표시")
        self.btn_proxy.clicked.connect(self._make_proxy)
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        top.addWidget(QLabel("모델"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n (기본·빠름)", "yolov8s", "yolov8m",
                                   "yolo11n", "yolo11s", "yolo11m (정확)",
                                   "사용자 .pt 파일..."])
        self.combo_model.setToolTip(
            "공/선수 검출 모델. GPU 추론이라 큰 모델도 부담 적음 —\n"
            "정확도가 아쉬우면 yolo11s/m 권장. 이름 모델은 최초 1회 자동 다운로드.")
        saved_model = QSettings("PyStitch360", "PyStitch360").value("ptz_model", 0)
        self.combo_model.setCurrentIndex(int(saved_model))
        self.combo_model.currentIndexChanged.connect(self._model_changed)
        top.addWidget(self.combo_model)
        top.addWidget(self.btn_proxy)
        top.addWidget(self.btn_analyze)
        v.addLayout(top)

        # 영상은 전체 폭 사용, 목록·레이더는 하단 스트립으로
        self.pane = FramePane("클릭 = 오브젝트 조작 / 빈 곳 = 키프레임, 우클릭 = 메뉴",
                              interactive=True)
        self.pane.clicked.connect(self._pane_clicked)
        self.pane.context_requested.connect(self._pane_context)
        self.pane.hover.connect(self._pane_hover)
        self.pane.pressed.connect(self._pane_pressed)
        self.pane.drag_moved.connect(self._pane_dragged)
        self.pane.released.connect(self._pane_released)
        v.addWidget(self.pane, 1)

        tl = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_play.setMaximumWidth(80)
        self.btn_play.clicked.connect(self._toggle_play)
        tl.addWidget(self.btn_play)
        btn_prev = QPushButton("◀트랙")
        btn_prev.setMaximumWidth(58)
        btn_prev.clicked.connect(lambda: self._jump_track(-1))
        tl.addWidget(btn_prev)
        btn_next = QPushButton("트랙▶")
        btn_next.setMaximumWidth(58)
        btn_next.clicked.connect(lambda: self._jump_track(1))
        tl.addWidget(btn_next)
        for text, d in [("-10s", -300), ("-1s", -30), ("-1", -1),
                        ("+1", 1), ("+1s", 30), ("+10s", 300)]:
            b = QPushButton(text)
            b.setMaximumWidth(52)
            b.clicked.connect(lambda _, dd=d: self._step(dd))
            tl.addWidget(b)
        # 트랙바는 슬라이더 바로 위, 같은 폭 — 위치가 1:1 로 대응
        self.trackbar = TrackBar()
        self.trackbar.seek.connect(lambda f: self.slider.setValue(f))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._stop_play)
        self.slider.valueChanged.connect(self._on_slider)
        bar_col = QVBoxLayout()
        bar_col.setSpacing(1)
        bar_col.addWidget(self.trackbar)
        bar_col.addWidget(self.slider)
        tl.addLayout(bar_col, 1)
        self.lbl_time = QLabel("--:--.-")
        self.lbl_time.setMinimumWidth(90)
        tl.addWidget(self.lbl_time)
        v.addLayout(tl)
        self._slider_timer = QTimer(singleShot=True, interval=120)
        self._slider_timer.timeout.connect(self._show_frame)

        # 하단 스트립: 공 목록 | 오인식 목록 | 레이더 (로그 위 공간 활용)
        strip = QHBoxLayout()
        col_ball = QVBoxLayout()
        col_ball.addWidget(QLabel("공 — 자동 트랙 + 수동 지정 (↑↓=이동, →/Del=오인식으로)"))
        self.track_list = QListWidget()
        self.track_list.setMaximumHeight(150)
        # 클릭·키보드 화살표 선택 모두에서 이동 (currentRowChanged 는 둘 다 발생)
        self.track_list.currentRowChanged.connect(lambda _: self._goto_track())
        for key in (Qt.Key.Key_Right, Qt.Key.Key_Delete):   # → 또는 Del = 오인식
            QShortcut(QKeySequence(key), self.track_list,
                      activated=self._ignore_selected_track,
                      context=Qt.ShortcutContext.WidgetShortcut)
        col_ball.addWidget(self.track_list, 1)
        strip.addLayout(col_ball, 2)

        # 두 목록 사이 이동 버튼: >> = 오인식으로, << = 복원
        col_move = QVBoxLayout()
        col_move.addStretch(1)
        self.btn_ignore = QPushButton("≫")
        self.btn_ignore.setMaximumWidth(44)
        self.btn_ignore.setToolTip("선택한 공 트랙을 오인식으로 (→/Del)")
        self.btn_ignore.clicked.connect(self._to_ignore)
        col_move.addWidget(self.btn_ignore)
        self.btn_restore = QPushButton("≪")
        self.btn_restore.setMaximumWidth(44)
        self.btn_restore.setToolTip("선택한 오인식을 복원 (←)")
        self.btn_restore.clicked.connect(self._delete_kf)
        col_move.addWidget(self.btn_restore)
        col_move.addStretch(1)
        strip.addLayout(col_move)

        col_ig = QVBoxLayout()
        col_ig.addWidget(QLabel("오인식 — 공 아님 (더블클릭=이동, ←=복원)"))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(150)
        self.kf_list.itemDoubleClicked.connect(self._goto_kf)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.kf_list,
                  activated=self._delete_kf,
                  context=Qt.ShortcutContext.WidgetShortcut)
        col_ig.addWidget(self.kf_list, 1)
        strip.addLayout(col_ig, 2)

        col_radar = QVBoxLayout()
        self.radar = RadarView()
        col_radar.addWidget(self.radar)
        self.check_players = QCheckBox("선수 표시 (팀 색)")
        self.check_players.setChecked(True)
        self.check_players.toggled.connect(lambda _: self._redraw())
        col_radar.addWidget(self.check_players)
        strip.addLayout(col_radar, 1)
        v.addLayout(strip)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("출력"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.currentIndexChanged.connect(lambda _: self._plan_dirty())
        bottom.addWidget(self.combo_mode)
        lbl_fz = QLabel("원경 줌")
        lbl_fz.setToolTip("공이 반대편(원경)에 있을 때 추가 줌인 배율 (1.0=없음)")
        bottom.addWidget(lbl_fz)
        self.spin_far_zoom = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=1.5,
                                            singleStep=0.05)
        self.spin_far_zoom.setValue(float(QSettings("PyStitch360", "PyStitch360")
                                          .value("ptz_far_zoom", 1.0)))
        self.spin_far_zoom.valueChanged.connect(self._far_zoom_changed)
        bottom.addWidget(self.spin_far_zoom)
        bottom.addWidget(QLabel("코덱"))
        self.combo_codec = QComboBox()
        self.encoders = available_encoders()
        self.combo_codec.addItems(list(self.encoders))
        bottom.addWidget(self.combo_codec)
        bottom.addWidget(QLabel("CRF/CQ"))
        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=20)
        bottom.addWidget(self.spin_crf)
        self.btn_export = QPushButton("PTZ 내보내기...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._start_render)
        bottom.addWidget(self.btn_export)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_render)
        bottom.addWidget(self.btn_cancel)
        bottom.addStretch(1)
        v.addLayout(bottom)
        self.progress = QProgressBar()
        v.addWidget(self.progress)

    # ------------------------------------------------------------ 파일/분석
    def _open_pano(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "완성 파노라마 영상", self._video_dir(), "영상 (*.mp4 *.MP4 *.mkv)")
        if path:
            self.open_path(path)

    def open_path(self, path: str, quiet: bool = False):
        """파노라마 열기 (프로젝트 복원 경로 포함). 분석/키프레임 사이드카 자동 로드."""
        if not Path(path).exists():
            from ..core.project import _cross_platform_candidates
            for cand in _cross_platform_candidates(path):
                if Path(cand).exists():
                    path = cand
                    break
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if not quiet:
                QMessageBox.warning(self, "열기 실패", path)
            else:
                self.log(f"[ptz] 파노라마 없음 — 건너뜀: {path}")
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.pano_path = Path(path)
        self._remember_dir(str(self.pano_path.parent))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lbl_file.setText(f"{self.pano_path.name} — {self.pano_w}x{self.pano_h}, "
                              f"{self.total/self.fps/60:.1f}분")
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(0)
        self._use_display_source()
        self.btn_analyze.setEnabled(ptz_available())
        if not ptz_available():
            self.btn_analyze.setToolTip("ultralytics 미설치 (pip install ultralytics)")
        self.analysis = None
        self._load_sidecar()
        self._show_frame()
        self._update_export_enabled()

    def _proxy_path(self) -> Path:
        return self.pano_path.with_suffix(".scrub.mp4")

    def _use_display_source(self):
        """표시/재생용 소스 선택: 프록시가 있으면 프록시 (즉시 탐색)."""
        proxy = self._proxy_path()
        if proxy.exists():
            cap = cv2.VideoCapture(str(proxy))
            if cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                self.cap = cap
                self.disp_path = proxy
                pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.disp_scale = pw / max(self.pano_w, 1)
                self.btn_proxy.setText("프록시 사용 중")
                self.btn_proxy.setEnabled(False)
                self.log(f"[ptz] 스크럽 프록시 사용: {proxy.name} (즉시 탐색)")
                return
        self.disp_path = self.pano_path
        self.disp_scale = 1.0
        self.btn_proxy.setText("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(True)

    def _make_proxy(self):
        if self.pano_path is None:
            return
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setText("프록시 생성 중...")
        self.log("[ptz] 스크럽 프록시 생성 중 (NVENC 가용 시 수 분)...")
        w = ProxyWorker(self.pano_path, self._proxy_path(),
                        duration_sec=self.total / max(self.fps, 1e-9))
        w.progress.connect(self._proxy_progress)
        w.finished_ok.connect(lambda _: (self.log("[ptz] 프록시 완료"),
                                         self._use_display_source(),
                                         self._show_frame()))
        w.failed.connect(lambda m: (self.log(f"[오류] 프록시: {m}"),
                                    self.btn_proxy.setText("스크럽 프록시 생성"),
                                    self.btn_proxy.setEnabled(True)))
        self._proxy_worker = w
        w.start()

    def _proxy_progress(self, p: float):
        self.btn_proxy.setText(f"프록시 생성 {p:.0%}")
        # 내보내기가 진행바를 쓰고 있지 않을 때만 진행바에도 표시
        if self._render_worker is None or not self._render_worker.isRunning():
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(p * 1000))
            self.progress.setFormat(f"프록시 생성 %p%")

    def _sidecar_path(self) -> Path:
        """파노라마당 단일 사이드카: {analysis, keyframes, ignores}."""
        return self.pano_path.with_suffix(".ptz.json")

    def _legacy_analysis_path(self) -> Path:
        return self.pano_path.with_suffix(".analysis.json")

    def _kf_path(self) -> Path:                 # 구버전 (마이그레이션용)
        return self.pano_path.with_suffix(".ptz_keyframes.json")

    def _load_sidecar(self):
        """통합 사이드카 로드. 구버전 두 파일이 있으면 병합 후 이전.

        구형 .analysis.json 이 통합본보다 새로우면(외부에서 분석을 돌린 경우)
        그쪽 분석을 채택한다.
        """
        self.keyframes, self.ignores, self.promotes, self.analysis = \
            [], [], [], None
        sp = self._sidecar_path()
        doc = None
        if sp.exists():
            try:
                doc = json.loads(sp.read_text())
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 사이드카 무시: {e}")
        if doc is not None:
            self.keyframes = [list(k) for k in doc.get("keyframes", [])]
            self.ignores = [list(r) for r in doc.get("ignores", [])]
            self.promotes = [list(p) for p in doc.get("promotes", [])]
            self.analysis = doc.get("analysis")
        migrated = False
        la = self._legacy_analysis_path()
        if la.exists() and (doc is None or
                            la.stat().st_mtime > sp.stat().st_mtime):
            try:
                self.analysis = json.loads(la.read_text())
                migrated = True
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 구형 분석 무시: {e}")
        if doc is None and self._kf_path().exists():
            try:
                d = json.loads(self._kf_path().read_text())
                if isinstance(d, list):
                    self.keyframes = [list(k) for k in d]
                else:
                    self.keyframes = [list(k) for k in d.get("keyframes", [])]
                    self.ignores = [list(r) for r in d.get("ignores", [])]
                migrated = True
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 구형 키프레임 무시: {e}")
        # 분석 유효성 (프레임 수 허용 오차)
        if self.analysis is not None:
            if abs(self.analysis.get("total_frames", 0) - self.total) \
                    > max(60, self.total // 100):
                self.log("[ptz] 사이드카 분석이 다른 영상 기준 — 무시")
                self.analysis = None
        if migrated:
            self._write_sidecar()
            self.log("[ptz] 구버전 사이드카를 통합본(.ptz.json)으로 이전")
        if self.analysis is not None:
            self.log(f"[ptz] 분석 불러옴 (검출 샘플 "
                     f"{len(self.analysis['frames'])}개), 키프레임 "
                     f"{len(self.keyframes)}개, 무시 {len(self.ignores)}개")
            self._start_link()
        self._refresh_lists()

    def _write_sidecar(self):
        if self.pano_path is None:
            return
        sp = self._sidecar_path()
        tmp = sp.with_suffix(".ptz.json.tmp")
        tmp.write_text(json.dumps({"analysis": self.analysis,
                                   "keyframes": self.keyframes,
                                   "ignores": self.ignores,
                                   "promotes": self.promotes}))
        tmp.replace(sp)

    _MODEL_NAMES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", None]

    def _model_changed(self, i):
        st = QSettings("PyStitch360", "PyStitch360")
        if self._MODEL_NAMES[i] is None:      # 사용자 .pt
            path, _ = QFileDialog.getOpenFileName(
                self, "YOLO 가중치 (.pt)", "", "PyTorch 가중치 (*.pt)")
            if path:
                st.setValue("ptz_model_custom", path)
            else:
                self.combo_model.setCurrentIndex(0)
                return
        st.setValue("ptz_model", i)

    def _model_weights(self):
        """선택된 모델의 가중치 경로/이름 (None=내장 기본 yolov8n)."""
        name = self._MODEL_NAMES[self.combo_model.currentIndex()]
        if name is None:
            custom = QSettings("PyStitch360", "PyStitch360").value("ptz_model_custom", "")
            return str(custom) if custom else None
        if name == "yolov8n.pt":
            return None                        # presets/yolov8n.pt (내장)
        local = Path(__file__).resolve().parents[2] / "presets" / name
        return str(local) if local.exists() else name   # 없으면 자동 다운로드

    def _run_analyze(self):
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            self._analyze_worker.cancel()          # 버튼이 취소 역할
            self.log("[ptz] 분석 취소 요청...")
            return
        if self.pano_path is None:
            return
        self.btn_analyze.setText("분석 취소")
        self.progress.setRange(0, 0)
        self.progress.setFormat("분석 중... (로그 참조)")
        weights = self._model_weights()
        self.log(f"[ptz] 자동 분석 시작 (모델: {weights or 'yolov8n(내장)'}) — "
                 "진행은 로그에 표시")
        ckpt = str(self.pano_path.with_suffix(".analysis.part.json"))
        w = AnalyzeWorker(str(self.pano_path), weights=weights,
                          checkpoint_path=ckpt)
        w.progress.connect(self._analyze_progress)
        w.log.connect(self.log)
        w.done.connect(self._analyze_done)
        w.failed.connect(self._analyze_failed)
        self._analyze_worker = w
        w.start()

    def _analyze_progress(self, done, total, fps):
        if self._render_worker is not None and self._render_worker.isRunning():
            return                              # 내보내기가 진행바 사용 중
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(
            f"분석 %p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _analyze_done(self, d):
        self.analysis = d
        self._write_sidecar()
        self._legacy_analysis_path().unlink(missing_ok=True)  # 구형 파일 정리
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("분석 완료")
        n_ball = sum(1 for b in d["balls"] if b is not None)
        self.log(f"[ptz] 분석 완료: 샘플 {len(d['frames'])}개, "
                 f"공 검출 {n_ball}개 ({n_ball/max(len(d['frames']),1):.0%}) → 캐시 저장")
        self._update_export_enabled()
        self._start_link()
        self._show_frame()

    def _analyze_failed(self, msg):
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log(f"[오류] 분석: {msg}")

    # ------------------------------------------------------------ 타임라인/표시
    def _step(self, d):
        self._stop_play()
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    def _on_slider(self, _):
        t = self.slider.value() / self.fps
        cs = round(t * 100)
        m, cs = divmod(cs, 6000)
        self.lbl_time.setText(f"{m:02d}:{cs/100:05.2f}")
        self._slider_timer.start()

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
            return
        if self.cap is None:
            return
        w = PtzPlayWorker(self.disp_path or self.pano_path,
                          self.slider.value(), self.fps,
                          display_fps=30.0 if self.disp_scale < 1.0 else 15.0)
        w.frame_ready.connect(self._play_frame)
        w.finished.connect(self._play_finished)
        self._play_worker = w
        self._playing = True
        self.btn_play.setText("⏸ 정지")
        w.start()

    def _stop_play(self):
        if self._play_worker is not None and self._play_worker.isRunning():
            self._play_worker.stop()

    def _play_frame(self, frame, f):
        self._cur_frame, self._cur_frame_idx = frame, f
        self.slider.setValue(f)     # _show_frame 은 재생 중 가드로 무시됨
        self._redraw()

    def _play_finished(self):
        self._playing = False
        self.btn_play.setText("▶ 재생")

    def _current_sample(self):
        """현재 시각 근처(±0.5s)의 분석 샘플 인덱스."""
        if self.analysis is None:
            return None
        idx = np.asarray(self.analysis["frames"])
        i = int(np.argmin(np.abs(idx - self.slider.value())))
        if abs(idx[i] - self.slider.value()) > 0.5 * self.fps:
            return None
        return i

    def _current_auto_ball(self):
        i = self._current_sample()
        return None if i is None else self.analysis["balls"][i]

    def _ball_in_ignore(self, f, bx, by):
        """(f, bx, by) 공이 어떤 무시 구간에 걸리는가 (자리 있으면 150px)."""
        return any(
            r[0] <= f <= r[1] and (len(r) < 4 or
                                   ((r[2] - bx) ** 2 + (r[3] - by) ** 2)
                                   ** 0.5 <= 150)
            for r in self.ignores)

    def _promote_near(self, f, bx, by):
        """(f, bx, by) 부근(±0.5s, 150px)에 걸린 승격 항목들 (토글/표시용)."""
        return [p for p in self.promotes
                if abs(p[0] - f) <= 0.5 * self.fps
                and (p[1] - bx) ** 2 + (p[2] - by) ** 2 <= 150 ** 2]

    # ---------------------------------------------------- 오브젝트 질의/조작
    def _candidates_at(self, si):
        """샘플 si 의 공 후보 [(x, y, conf), ...] (신형 ball_cands 우선)."""
        if self.analysis is None or si is None:
            return []
        bc = self.analysis.get("ball_cands")
        if bc is not None and si < len(bc):
            return [(float(p[0]), float(p[1]), float(p[2])) for p in bc[si]]
        b = self.analysis["balls"][si]
        return [(float(b[0]), float(b[1]), float(b[2]))] if b else []

    def _track_for(self, si, x, y):
        """샘플 si 에서 (x, y) 를 지나는 링크 트랙 (없으면 None)."""
        if self._linked is None or si is None:
            return None
        best, bestd = None, 80.0 ** 2
        for t in self._linked["tracks"]:
            hits = np.where(t["i"] == si)[0]
            if len(hits) == 0:
                continue
            k = int(hits[0])
            d = (t["pts"][k][0] - x) ** 2 + (t["pts"][k][1] - y) ** 2
            if d <= bestd:
                best, bestd = t, d
        return best

    def _track_span(self, t):
        idx = self._linked["idx"]
        return int(idx[t["i"][0]]), int(idx[t["i"][-1]])

    def _cand_state(self, f, si, x, y):
        """공 후보 상태: 'ignored' | 'promoted' | 'accepted' | 'rejected'."""
        if self._ball_in_ignore(f, x, y):
            return "ignored"
        if self._promote_near(f, x, y):
            return "promoted"
        ab = self._accepted_ball
        if (ab is not None and si is not None and si < len(ab)
                and not np.isnan(ab[si, 0])
                and (ab[si, 0] - x) ** 2 + (ab[si, 1] - y) ** 2 <= 60.0 ** 2):
            return "accepted"
        return "rejected"

    def _objects_at(self):
        """현재 프레임의 조작 가능한 오브젝트 (공 후보 + 근처 키프레임)."""
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        si = self._current_sample()
        objs = []
        for (x, y, conf) in self._candidates_at(si):
            objs.append({"kind": "ball", "x": x, "y": y, "conf": conf,
                         "state": self._cand_state(f, si, x, y), "si": si})
        for i, k in enumerate(self.keyframes):
            if abs(k[0] - f) <= 1.0 * self.fps:
                objs.append({"kind": "kf", "x": float(k[1]), "y": float(k[2]),
                             "i": i})
        return objs

    def _hit(self, x, y, r=80.0):
        """(x, y) 파노라마 좌표에서 반경 r 안 가장 가까운 오브젝트 (없으면 None)."""
        best, bestd = None, r * r
        for o in self._objects_at():
            d = (o["x"] - x) ** 2 + (o["y"] - y) ** 2
            if d <= bestd:
                best, bestd = o, d
        return best

    def _ignore_covers(self, entry):
        """entry(=[f0,f1,x,y]) 와 사실상 동일한 무시가 이미 있는가."""
        f0, f1 = entry[0], entry[1]
        for r in self.ignores:
            if r[0] <= f0 and f1 <= r[1] and (
                    len(r) < 4 or ((r[2] - entry[2]) ** 2
                                   + (r[3] - entry[3]) ** 2) ** 0.5 <= 150):
                return True
        return False

    def _mark_dirty_and_redraw(self):
        self._save_keyframes()
        self._recompute_tracks()
        self._redraw()

    def _promote_ball(self, f, si, x, y):
        """(x,y) 공을 경기 공으로 승격 + 같은 프레임 다른 후보 트랙 자동 무시."""
        if not self._promote_near(f, x, y):
            self.promotes.append([f, round(x, 1), round(y, 1)])
            self.promotes.sort()
        auto = 0
        for (cx, cy, cc) in self._candidates_at(si):
            if (cx - x) ** 2 + (cy - y) ** 2 <= 60.0 ** 2:
                continue                       # 승격 대상 자신
            t = self._track_for(si, cx, cy)
            if t is None:
                continue
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
            # 승격한 공을 함께 잡을 수 있는 무시는 건너뜀 (자기 트랙 보호)
            if (f0 <= f <= f1 and
                    ((med[0] - x) ** 2 + (med[1] - y) ** 2) ** 0.5 <= 150):
                continue
            if not self._ignore_covers(entry):
                self.ignores.append(entry)
                auto += 1
        self.ignores.sort()
        self._mark_dirty_and_redraw()
        extra = f", 경쟁 후보 {auto}개 자동 무시" if auto else ""
        self.log(f"[ptz] 경기 공 승격 {f/self.fps:.1f}s "
                 f"→ ({x:.0f}, {y:.0f}){extra}")

    def _unpromote_at(self, f, x, y):
        near = self._promote_near(f, x, y)
        if not near:
            return
        for p in near:
            self.promotes.remove(p)
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 승격 취소 {f/self.fps:.1f}s")

    def _ignore_track_at(self, f, si, x, y):
        """(x,y) 후보의 트랙을 오인식으로 무시 (링크 없으면 그 시점만)."""
        t = self._track_for(si, x, y)
        if t is not None:
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
        else:
            entry = [int(f), int(f), round(x, 1), round(y, 1)]
        for p in self._promote_near(f, x, y):    # 무시가 승격보다 우선
            self.promotes.remove(p)
        if not self._ignore_covers(entry):
            self.ignores.append(entry)
            self.ignores.sort()
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 검출 무시 {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _restore_at(self, f, x, y):
        """(x,y) 위치에 걸린 무시 구간을 해제 (복원)."""
        keep, removed = [], 0
        for r in self.ignores:
            if (r[0] <= f <= r[1] and
                    (len(r) < 4 or ((r[2] - x) ** 2 + (r[3] - y) ** 2)
                     ** 0.5 <= 150)):
                removed += 1
            else:
                keep.append(r)
        if removed:
            self.ignores = keep
            self._mark_dirty_and_redraw()
            self.log(f"[ptz] 무시 해제 {f/self.fps:.1f}s ({removed}개 구간)")

    def _add_keyframe(self, f, x, y):
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        self.keyframes.append([f, round(x, 1), round(y, 1)])
        self.keyframes.sort()
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes)
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] 키프레임 {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _delete_keyframe_idx(self, i):
        if 0 <= i < len(self.keyframes):
            k = self.keyframes[i]
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_kf_list()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes)
            self._plan_dirty()
            self._redraw()
            self.log(f"[ptz] 키프레임 삭제 {k[0]/self.fps:.1f}s")

    # ------------------------------------------------- 크롭 박스 드래그 편집
    def _disp_to_pano(self):
        """표시 픽셀 1개가 파노라마 몇 px 인가 (히트 톨러런스 환산)."""
        return self.pano_w / max(self.pane.displayed_width(), 1)

    def _box_hit(self, x, y):
        """크롭 박스 히트: ("corner", 0~3) | ("border", None) | None.

        모서리(핸들) = 리사이즈(줌), 테두리 밴드 = 이동. 내부는 공 클릭용으로
        비워 둔다 (공은 대개 박스 안에 있다).
        """
        if self._plan_box is None:
            return None
        x0, y0, w, h = self._plan_box
        k = self._disp_to_pano()
        ct = 18 * k                          # 모서리 히트 반경
        for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                       (x0, y0 + h), (x0 + w, y0 + h)]):
            if abs(x - hx) <= ct and abs(y - hy) <= ct:
                return ("corner", ci)
        bt = 10 * k                          # 테두리 밴드 폭
        on_v = (abs(x - x0) <= bt or abs(x - x0 - w) <= bt) \
            and y0 - bt <= y <= y0 + h + bt
        on_h = (abs(y - y0) <= bt or abs(y - y0 - h) <= bt) \
            and x0 - bt <= x <= x0 + w + bt
        if on_v or on_h:
            return ("border", None)
        return None

    def _pane_pressed(self, fx, fy):
        """좌버튼 프레스: 크롭 박스 테두리/핸들이면 편집 시작."""
        if self.cap is None or self.plan is None or self.analysis is None:
            return
        if self._playing:
            self._stop_play()
        x, y = fx * self.pano_w, fy * self.pano_h
        if self._hit(x, y) is not None:      # 공/키프레임이 우선 (클릭 동작)
            return
        hit = self._box_hit(x, y)
        if hit is None or self._plan_box is None:
            return
        x0, y0, w, h = self._plan_box
        box = [x0 + w / 2, y0 + h / 2, float(w)]
        self._box_edit = {"mode": hit[0], "corner": hit[1],
                          "box": list(box), "orig": list(box),
                          "start": (x, y),
                          "frame": getattr(self, "_cur_frame_idx",
                                           self.slider.value())}

    def _pane_dragged(self, fx, fy):
        e = self._box_edit
        if e is None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        ow, oh = self.plan_out
        cx0, cy0, w0 = e["orig"]
        if e["mode"] == "border":            # 이동
            cx = cx0 + (x - e["start"][0])
            cy = cy0 + (y - e["start"][1])
            w = w0
        else:                                # 모서리 = 중심 기준 리사이즈(줌)
            w = 2 * max(abs(x - cx0), abs(y - cy0) * ow / oh)
            cx, cy = cx0, cy0
        top = int(self.plan.get("top_margin", 0)) if self.plan else 0
        max_w = min(self.pano_w, (self.pano_h - top) * ow / oh)
        w = min(max(w, ow / 6.0), max_w)     # 코어 min_crop_w 와 동일 하한
        h = w * oh / ow
        cx = min(max(cx, w / 2), self.pano_w - w / 2)
        cy = min(max(cy, h / 2), self.pano_h - h / 2)
        e["box"] = [cx, cy, w]
        self._redraw()

    def _pane_released(self, fx, fy):
        e = self._box_edit
        if e is None:
            return
        self._box_edit = None
        cx, cy, w = e["box"]
        cx0, cy0, w0 = e["orig"]
        # 사실상 클릭(수 px 미만 이동)이면 커밋하지 않음
        if abs(cx - cx0) + abs(cy - cy0) + abs(w - w0) < 6 * self._disp_to_pano():
            self._redraw()
            return
        f = e["frame"]
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        self.keyframes.append([f, round(cx, 1), round(cy, 1), round(w, 1)])
        self.keyframes.sort()
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes)
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] 크롭 키프레임 {f/self.fps:.1f}s → "
                 f"중심 ({cx:.0f}, {cy:.0f}), 폭 {w:.0f}px")

    def _show_frame(self):
        """현재 프레임 디코딩 + 오버레이. 디코딩 결과는 캐시된다."""
        if self.cap is None or self._playing:
            return
        f = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._cur_frame, self._cur_frame_idx = frame, f
        self._redraw()

    def _redraw(self):
        """오버레이만 다시 그림 — 키프레임/무시/계획 변경 시 디코딩 없이 즉시."""
        if getattr(self, "_cur_frame", None) is None:
            return
        frame = self._cur_frame.copy()
        f = self._cur_frame_idx
        sc = self.disp_scale                 # 프록시 표시 시 좌표 축소
        # 계획된 크롭 창 (render_plan 과 동일한 클램프) — 결과 미리보기.
        # 드래그(이동)/모서리 핸들(줌)로 편집 가능 — 놓으면 줌 키프레임 커밋.
        self._plan_box = None
        if self.plan is not None and f < len(self.plan["cx"]):
            ow, oh = self.plan_out
            top = int(self.plan.get("top_margin", 0))
            if self._box_edit is not None:      # 편집 중: 미확정 박스
                bcx, bcy, bw = self._box_edit["box"]
                w = int(round(bw)) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(bcx - w / 2))
                y0 = int(round(bcy - h / 2))
                box_color = (0, 255, 255)
            else:
                w = int(round(min(self.plan["crop_w"][f], self.pano_w,
                                  self.pano_h * ow / oh))) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(self.plan["cx"][f] - w / 2))
                y0 = int(round(self.plan["cy"][f] - h / 2))
                x0 = max(0, min(x0, self.pano_w - w))
                y0 = max(min(top, self.pano_h - h), min(y0, self.pano_h - h))
                box_color = (255, 200, 0)
            self._plan_box = (x0, y0, w, h)
            thick = max(2, int(6 * sc))
            if self._box_edit is None and self._box_hover == ("border", None):
                thick += max(2, int(4 * sc))    # 이동 가능 표시: 테두리 두껍게
            cv2.rectangle(frame, (int(x0 * sc), int(y0 * sc)),
                          (int((x0 + w) * sc), int((y0 + h) * sc)),
                          box_color, thick)
            hs = max(6, int(16 * sc))           # 모서리 핸들 (리사이즈 = 줌)
            for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                           (x0, y0 + h), (x0 + w, y0 + h)]):
                hov = (self._box_edit is None
                       and self._box_hover == ("corner", ci))
                r = hs + (max(3, int(8 * sc)) if hov else 0)
                cv2.rectangle(frame, (int(hx * sc) - r, int(hy * sc) - r),
                              (int(hx * sc) + r, int(hy * sc) + r),
                              (0, 255, 255) if hov else box_color, -1)
        # 공 후보 전부 표시 — 상태별 색: 초록=수락, 자홍=승격, 빨강X=무시,
        # 회색=자동 기각. 커서가 가리키는 오브젝트는 노란 링으로 하이라이트.
        si = self._current_sample()
        hv = self._hover_key[0] if self._hover_key else None

        def _is_hover(kind, ox, oy):
            return hv is not None and hv == (kind, round(ox), round(oy))

        rad = max(10, int(28 * sc))
        for (bx, by, conf) in self._candidates_at(si):
            p = (int(bx * sc), int(by * sc))
            st = self._cand_state(f, si, bx, by)
            if st == "ignored":
                cv2.circle(frame, p, rad, (60, 60, 200), 4)
                cv2.line(frame, (p[0] - 24, p[1] - 24), (p[0] + 24, p[1] + 24),
                         (60, 60, 200), 6)
                cv2.line(frame, (p[0] - 24, p[1] + 24), (p[0] + 24, p[1] - 24),
                         (60, 60, 200), 6)
                cv2.putText(frame, "IGNORED", (p[0] + 36, p[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 200), 4)
            elif st == "promoted":
                cv2.circle(frame, p, rad, (200, 0, 255), 6)
            elif st == "accepted":
                cv2.circle(frame, p, rad, (0, 220, 0), 6)
            else:
                cv2.circle(frame, p, rad, (150, 150, 150), 4)
            if _is_hover("ball", bx, by):
                cv2.circle(frame, p, rad + max(6, int(12 * sc)), (0, 255, 255), 3)
        for k in self.keyframes:
            if abs(k[0] - f) <= 1.0 * self.fps:
                kx, ky = float(k[1]), float(k[2])
                p = (int(kx * sc), int(ky * sc))
                cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS,
                               max(20, int(60 * sc)), max(3, int(8 * sc)))
                if _is_hover("kf", kx, ky):
                    cv2.circle(frame, p, max(16, int(34 * sc)), (0, 255, 255), 3)
        # 선수 박스(팀 색) + 레이더
        radar_pts = []
        if si is not None:
            prow = self.analysis["players"][si]
            if self.check_players.isChecked():
                for pp in prow:
                    if len(pp) < 4:
                        continue
                    team = self._teams.get(int(pp[4]), 2) if len(pp) >= 5 else 2
                    color = TEAM_COLORS[min(max(team, 0), 2)]
                    x1 = int((pp[0] - pp[2] / 2) * sc)
                    y1 = int((pp[1] - pp[3] / 2) * sc)
                    x2 = int((pp[0] + pp[2] / 2) * sc)
                    y2 = int((pp[1] + pp[3] / 2) * sc)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                                  max(2, int(4 * sc)))
            for X, Y, tid, j in ground_positions(prow, self.pano_w, self.pano_h):
                radar_pts.append((X, Y, self._teams.get(tid, 2)))
            ball_g = None
            bb = self.analysis["balls"][si]
            if bb is not None:
                g = ground_positions([[bb[0], bb[1], 0.0, 0.0]],
                                     self.pano_w, self.pano_h)
                ball_g = (g[0][0], g[0][1]) if g else None
            self.radar.set_data(radar_pts, ball_g)
        else:
            self.radar.set_data([], None)
        self.pane.set_frame(frame)

    # ------------------------------------------------------ 화면 오브젝트 조작
    def _pane_clicked(self, fx, fy):
        """좌클릭: 오브젝트 상태별 기본 동작 / 빈 곳은 키프레임 추가."""
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        if o is None:
            if self._box_hit(x, y) is not None:
                return                      # 박스 테두리 클릭 = 드래그 미수 — 무동작
            self._add_keyframe(f, x, y)     # 빈 곳 = 키프레임 추가
            return
        if o["kind"] == "kf":               # 키프레임 = 삭제
            self._delete_keyframe_idx(o["i"])
            return
        st = o["state"]                     # 공 후보 = 상태별 기본 동작
        if st == "promoted":
            self._unpromote_at(f, o["x"], o["y"])
        elif st == "ignored":
            self._restore_at(f, o["x"], o["y"])
        else:                               # gray/green → 경기 공 승격
            self._promote_ball(f, o["si"], o["x"], o["y"])

    def _pane_context(self, fx, fy, gpos):
        """우클릭: 오브젝트별 전체 동작 메뉴."""
        if self.cap is None or self.analysis is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        menu = QMenu(self)
        if o is None:
            menu.addAction("여기 키프레임 추가",
                           lambda: self._add_keyframe(f, x, y))
        elif o["kind"] == "kf":
            menu.addAction("키프레임 삭제",
                           lambda: self._delete_keyframe_idx(o["i"]))
        else:
            st, ox, oy, si = o["state"], o["x"], o["y"], o["si"]
            if st == "promoted":
                menu.addAction("승격 취소",
                               lambda: self._unpromote_at(f, ox, oy))
            else:
                menu.addAction("이 공을 경기 공으로 승격",
                               lambda: self._promote_ball(f, si, ox, oy))
            if st == "ignored":
                menu.addAction("무시 해제(복원)",
                               lambda: self._restore_at(f, ox, oy))
            else:
                menu.addAction("이 검출 무시(오인식)",
                               lambda: self._ignore_track_at(f, si, ox, oy))
            menu.addSeparator()
            menu.addAction("여기 키프레임 추가",
                           lambda: self._add_keyframe(f, x, y))
        menu.exec(gpos)

    def _pane_hover(self, fx, fy):
        """커서 근처 오브젝트/크롭 박스를 하이라이트 (바뀔 때만 리드로우)."""
        if self.analysis is None or getattr(self, "_cur_frame", None) is None:
            return
        if self._box_edit is not None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        zone = None if o is not None else self._box_hit(x, y)
        key = ((None if o is None
                else (o["kind"], round(o["x"]), round(o["y"]))), zone)
        if key != self._hover_key:
            self._hover_key = key
            self._hover = o
            self._box_hover = zone
            if o is not None:
                cur = Qt.CursorShape.PointingHandCursor
            elif zone is None:
                cur = Qt.CursorShape.OpenHandCursor
            elif zone[0] == "border":
                cur = Qt.CursorShape.SizeAllCursor
            else:                            # 모서리: 대각 리사이즈 커서
                cur = (Qt.CursorShape.SizeFDiagCursor if zone[1] in (0, 3)
                       else Qt.CursorShape.SizeBDiagCursor)
            self.pane.setCursor(cur)
            self._redraw()

    def _refresh_lists(self):
        """위: 공(자동 트랙+수동 키프레임) / 아래: 오인식(무시 구간)."""
        # 위 목록 — 시간순 병합
        self._top = ([("track", i) for i in range(len(self.track_spans))]
                     + [("kf", i) for i in range(len(self.keyframes))])
        self._top.sort(key=lambda e: (self.track_spans[e[1]][0] if e[0] == "track"
                                      else self.keyframes[e[1]][0]))
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for kind, i in self._top:
            if kind == "track":
                f0, f1 = self.track_spans[i]
                t0, t1 = f0 / self.fps, f1 / self.fps
                self.track_list.addItem(
                    f"{int(t0//60):02d}:{t0%60:04.1f} ~ "
                    f"{int(t1//60):02d}:{t1%60:04.1f}  ({t1-t0:.1f}s) 자동")
            else:
                k = self.keyframes[i]
                kf, kx, ky = k[0], k[1], k[2]
                zoom = f", 폭{k[3]:.0f}" if len(k) > 3 else ""
                t = kf / self.fps
                self.track_list.addItem(
                    f"{int(t//60):02d}:{t%60:04.1f}  ● 수동 "
                    f"({kx:.0f}, {ky:.0f}{zoom})")
        self.track_list.blockSignals(False)
        # 아래 목록 — 오인식만
        self.kf_list.clear()
        for r in self.ignores:
            f0, f1 = r[0], r[1]
            pos = f"  @({r[2]:.0f},{r[3]:.0f})" if len(r) >= 4 else ""
            self.kf_list.addItem(
                f"{int(f0/self.fps//60):02d}:{f0/self.fps%60:04.1f}~"
                f"{int(f1/self.fps//60):02d}:{f1/self.fps%60:04.1f}  공 아님{pos}")

    # 기존 호출부 호환 별칭
    _refresh_kf_list = _refresh_lists
    _refresh_track_list = _refresh_lists

    def _goto_kf(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            self.slider.setValue(int(self.ignores[row][0]))

    def _delete_kf(self):
        """선택 복원 — 오인식 마킹을 철회해 트랙을 되살린다."""
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            del self.ignores[row]
            self._save_keyframes()
            self._recompute_tracks()
            self._redraw()

    def _save_keyframes(self):
        """마킹 변경 저장 (1.5s 디바운스 — 분석 포함 파일이라 통으로 씀)."""
        self._save_timer.start()


    def _far_zoom_changed(self, v):
        QSettings("PyStitch360", "PyStitch360").setValue("ptz_far_zoom", v)
        self._plan_dirty()

    def _plan_dirty(self):
        if self.analysis is not None:
            self._plan_timer.start()

    def _run_plan(self):
        if self.analysis is None:
            return
        if self._plan_worker is not None and self._plan_worker.isRunning():
            self._plan_timer.start()      # 진행 중이면 잠시 뒤 재시도
            return
        w = PlanWorker(self.analysis, [tuple(k) for k in self.keyframes],
                       [tuple(r) for r in self.ignores],
                       self.combo_mode.currentIndex() == 1, linked=self._linked,
                       far_zoom=self.spin_far_zoom.value(),
                       promotes=[tuple(p) for p in self.promotes])
        w.done.connect(self._plan_done)
        self._plan_worker = w
        w.start()

    def _plan_done(self, plan, out):
        self.plan = plan
        self.plan_out = out
        self._redraw()

    def _start_link(self):
        """트랙 연결(느린 단계)을 백그라운드로 1회 계산."""
        if self.analysis is None:
            return
        w = LinkWorker(self.analysis)
        w.done.connect(self._link_done)
        self._link_worker = w
        self.log("[ptz] 트랙 연결 계산 중... (완료 후 클릭 반응이 빨라짐)")
        w.start()

    def _link_done(self, linked):
        self._linked = linked
        self._teams = linked.pop("teams", {}) or {}
        n_team = sum(1 for v in self._teams.values() if v < 2)
        self.log(f"[ptz] 트랙 연결 완료: {len(linked['tracks'])}개"
                 + (f", 팀 분류 선수 ID {n_team}개" if self._teams else
                    " (팀 분류: ID 포함 재분석 필요)"))
        self._recompute_tracks()
        self._plan_dirty()

    def _recompute_tracks(self):
        """수락 트랙 구간 재계산 → 트랙바 갱신 (분석/무시 구간 변경 시).

        linked 캐시가 있으면 수락 단계만 돌아 즉각 반응한다.
        """
        if self.analysis is None:
            self.track_spans = []
            self._accepted_ball = None
        else:
            _, self._accepted_ball, self.track_spans = accept_ball_tracks(
                self.analysis, ignore_ranges=[tuple(r) for r in self.ignores],
                force_ranges=[tuple(p) for p in self.promotes],
                linked=self._linked, log=self.log)
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes)
        self._refresh_track_list()
        self._plan_dirty()

    def _goto_track(self):
        row = self.track_list.currentRow()
        if 0 <= row < len(getattr(self, "_top", [])):
            kind, i = self._top[row]
            f = (self.track_spans[i][0] if kind == "track"
                 else self.keyframes[i][0])
            self.slider.setValue(int(f))

    def _jump_track(self, direction: int):
        """현재 위치 기준 이전(-1)/다음(+1) 수락 트랙 시작으로 이동."""
        self._stop_play()
        f = self.slider.value()
        if direction > 0:
            nxt = [sp for sp in self.track_spans if sp[0] > f]
            if nxt:
                self.slider.setValue(int(nxt[0][0]))
        else:
            prv = [sp for sp in self.track_spans if sp[0] < f - 1]
            if prv:
                self.slider.setValue(int(prv[-1][0]))

    def _to_ignore(self):
        """≫ 버튼: 목록 선택이 있으면 그 항목, 없으면 현재 시각 트랙."""
        if 0 <= self.track_list.currentRow() < len(getattr(self, "_top", [])):
            self._ignore_selected_track()
        else:
            self._ignore_current_track()

    def _ignore_selected_track(self):
        """위 목록에서 Del — 자동 트랙은 오인식으로, 수동 지정은 삭제."""
        row = self.track_list.currentRow()
        if not (0 <= row < len(getattr(self, "_top", []))):
            return
        kind, i = self._top[row]
        if kind == "track":
            self.slider.setValue(int(self.track_spans[i][0]))
            self._ignore_current_track()
        else:
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_lists()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes)
            self._plan_dirty()
            self._redraw()

    def _ignore_current_track(self):
        """현재 시각을 덮는 수락 트랙을 통째로 무시 목록에 추가."""
        f = self.slider.value()
        for f0, f1 in self.track_spans:
            if f0 <= f <= f1:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    spans = [(int(f0), int(f1))]
                    if self._linked is not None:
                        # 같은 자리 반복 정적 트랙 일괄 수집 (낙엽·마킹).
                        # 위치 포함 4-요소 — 같은 시간대의 진짜 공 트랙은 보호
                        spans = same_spot_spans(self._linked, f0, f1) or spans
                    added = 0
                    for sp in spans:
                        lo, hi = sp[0], sp[1]
                        if not any(a <= lo and hi <= b for a, b in
                                   ((r[0], r[1]) for r in self.ignores)):
                            self.ignores.append(list(sp))
                            added += 1
                    self.ignores.sort()
                    self._save_keyframes()
                    self._recompute_tracks()
                    self._refresh_kf_list()
                    self._redraw()
                finally:
                    QApplication.restoreOverrideCursor()
                extra = f" (같은 자리 반복 포함 {added}개 구간)" if added > 1 else ""
                self.log(f"[ptz] 트랙 무시: {f0/self.fps:.1f}s ~ "
                         f"{f1/self.fps:.1f}s{extra}")
                # 검수 흐름: 다음 항목 자동 선택 + 이동
                def _start(e):
                    return (self.track_spans[e[1]][0] if e[0] == "track"
                            else self.keyframes[e[1]][0])
                nxt = [r for r, e in enumerate(self._top) if _start(e) > f0]
                row = nxt[0] if nxt else len(self._top) - 1
                if row >= 0:
                    self.track_list.setCurrentRow(row)
                return
        QMessageBox.information(self, "무시", "현재 시각을 덮는 공 트랙이 없습니다.")

    # ------------------------------------------------------------ 내보내기
    def _update_export_enabled(self):
        self.btn_export.setEnabled(self.analysis is not None)

    def _start_render(self):
        if self.analysis is None or self.pano_path is None:
            return
        self._stop_play()
        wide = self.combo_mode.currentIndex() == 1
        suffix = "_wide.mp4" if wide else "_ptz.mp4"
        default = str(self.pano_path.with_name(self.pano_path.stem + suffix))
        out, _ = QFileDialog.getSaveFileName(self, "PTZ 출력 파일", default,
                                             "MP4 (*.mp4)")
        if not out:
            return
        codec = self.encoders[self.combo_codec.currentText()]
        kfs = [tuple(k) for k in self.keyframes]
        self.log(f"[ptz] 내보내기 시작: {'와이드' if wide else 'PTZ'} 모드, "
                 f"키프레임 {len(kfs)}개 반영")
        w = PtzRenderWorker(str(self.pano_path), out, self.analysis, kfs,
                            codec, self.spin_crf.value(), wide=wide,
                            ignores=[tuple(r) for r in self.ignores],
                            far_zoom=self.spin_far_zoom.value(),
                            promotes=[tuple(p) for p in self.promotes])
        w.log.connect(self.log)
        w.progress.connect(self._render_progress)
        w.finished_ok.connect(self._render_done)
        w.failed.connect(self._render_failed)
        self._render_worker = w
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("준비 중...")
        w.start()

    def _cancel_render(self):
        if self._render_worker is not None and self._render_worker.isRunning():
            self._render_worker.cancel()

    def _render_progress(self, done, total, fps):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _render_done(self, path):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("완료")
        self.log(f"[ptz] 저장: {path}")
        QMessageBox.information(self, "가상 PTZ", f"완료: {path}")

    def _render_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.log(f"[오류] PTZ 내보내기: {msg}")
