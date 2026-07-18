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
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..core.encoders import available_encoders
from ..core.ptz import (
    accept_ball_tracks, analyze_video, build_plan, link_ball_tracks,
    ptz_available, render_plan,
)
from .widgets import FramePane


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
        for f0, f1 in self.ignores:
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
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            d = analyze_video(self.video_path, log=lambda s: self.log.emit(s))
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
        self.done.emit(link_ball_tracks(self.analysis))


class PlanWorker(QThread):
    """크롭 계획 미리보기 재계산 (키프레임/무시 변경 시 백그라운드)."""

    done = pyqtSignal(dict, tuple)   # plan, (out_w, out_h)

    def __init__(self, analysis, keyframes, ignores, wide, linked=None):
        super().__init__()
        self.args = (analysis, keyframes, ignores, wide, linked)

    def run(self):
        analysis, kfs, ignores, wide, linked = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores, linked=linked,
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
                 wide=False, ignores=None):
        super().__init__()
        self.args = (pano_path, out_path, analysis, keyframes, codec, crf, wide,
                     ignores or [])
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, out, analysis, kfs, codec, crf, wide, ignores = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
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
        self.track_spans: list = []
        self._analyze_worker = None
        self._render_worker = None
        self._plan_worker = None
        self._link_worker = None
        self._linked = None
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
        top.addWidget(self.btn_proxy)
        top.addWidget(self.btn_analyze)
        v.addLayout(top)

        mid = QHBoxLayout()
        self.pane = FramePane("클릭 = 현재 시각의 공 위치 키프레임", interactive=True)
        self.pane.clicked.connect(self._pane_clicked)
        mid.addWidget(self.pane, 1)
        side = QVBoxLayout()
        side.addWidget(QLabel("공 트랙 (더블클릭=이동)"))
        self.track_list = QListWidget()
        self.track_list.setMaximumWidth(240)
        self.track_list.itemDoubleClicked.connect(self._goto_track)
        side.addWidget(self.track_list, 1)
        side.addWidget(QLabel("키프레임·무시 구간 (더블클릭=이동)"))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumWidth(240)
        self.kf_list.itemDoubleClicked.connect(self._goto_kf)
        side.addWidget(self.kf_list, 1)
        btn_del = QPushButton("선택 삭제")
        btn_del.clicked.connect(self._delete_kf)
        side.addWidget(btn_del)
        self.btn_ignore = QPushButton("현재 공 트랙 무시 (오인식)")
        self.btn_ignore.clicked.connect(self._ignore_current_track)
        side.addWidget(self.btn_ignore)
        mid.addLayout(side)
        v.addLayout(mid, 1)

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

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("출력"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.currentIndexChanged.connect(lambda _: self._plan_dirty())
        bottom.addWidget(self.combo_mode)
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
        self._load_keyframes()
        self._load_cached_analysis()
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

    def _analysis_cache(self) -> Path:
        return self.pano_path.with_suffix(".analysis.json")

    def _kf_path(self) -> Path:
        return self.pano_path.with_suffix(".ptz_keyframes.json")

    def _load_cached_analysis(self):
        c = self._analysis_cache()
        if c.exists():
            try:
                d = json.loads(c.read_text())
                # 컨테이너 메타데이터와 실제 디코딩 프레임 수는 약간 다를 수 있음
                if abs(d.get("total_frames", 0) - self.total) <= max(60, self.total // 100):
                    self.analysis = d
                    self.log(f"[ptz] 분석 캐시 불러옴: {c.name} "
                             f"(검출 샘플 {len(d['frames'])}개)")
                    self._start_link()
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 분석 캐시 무시: {e}")

    def _run_analyze(self):
        if self.pano_path is None or self._analyze_worker is not None \
                and self._analyze_worker.isRunning():
            return
        self.btn_analyze.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("분석 중... (로그 참조)")
        self.log("[ptz] 자동 분석 시작 — 완주 기준 ~20분, 진행은 로그에 표시")
        w = AnalyzeWorker(str(self.pano_path))
        w.log.connect(self.log)
        w.done.connect(self._analyze_done)
        w.failed.connect(self._analyze_failed)
        self._analyze_worker = w
        w.start()

    def _analyze_done(self, d):
        self.analysis = d
        self._analysis_cache().write_text(json.dumps(d))
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

    def _current_auto_ball(self):
        """현재 시각 근처(±0.5s)의 자동 검출 공."""
        if self.analysis is None:
            return None
        idx = np.asarray(self.analysis["frames"])
        i = int(np.argmin(np.abs(idx - self.slider.value())))
        if abs(idx[i] - self.slider.value()) > 0.5 * self.fps:
            return None
        return self.analysis["balls"][i]

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
        # 계획된 크롭 창 (render_plan 과 동일한 클램프) — 결과 미리보기
        if self.plan is not None and f < len(self.plan["cx"]):
            ow, oh = self.plan_out
            top = int(self.plan.get("top_margin", 0))
            w = int(round(min(self.plan["crop_w"][f], self.pano_w,
                              self.pano_h * ow / oh))) & ~1
            h = int(round(w * oh / ow)) & ~1
            x0 = int(round(self.plan["cx"][f] - w / 2))
            y0 = int(round(self.plan["cy"][f] - h / 2))
            x0 = max(0, min(x0, self.pano_w - w))
            y0 = max(min(top, self.pano_h - h), min(y0, self.pano_h - h))
            cv2.rectangle(frame, (int(x0 * sc), int(y0 * sc)),
                          (int((x0 + w) * sc), int((y0 + h) * sc)),
                          (255, 200, 0), max(2, int(6 * sc)))
        b = self._current_auto_ball()
        if b is not None:
            p = (int(b[0] * sc), int(b[1] * sc))
            in_ignore = any(f0 <= f <= f1 for f0, f1 in
                            ((r[0], r[1]) for r in self.ignores))
            in_track = any(f0 <= f <= f1 for f0, f1 in self.track_spans)
            if in_ignore:
                # 사용자가 취소한 검출: 빨간 X + 라벨
                cv2.circle(frame, p, max(10, int(28 * sc)), (60, 60, 200), 4)
                cv2.line(frame, (p[0] - 24, p[1] - 24), (p[0] + 24, p[1] + 24),
                         (60, 60, 200), 6)
                cv2.line(frame, (p[0] - 24, p[1] + 24), (p[0] + 24, p[1] - 24),
                         (60, 60, 200), 6)
                cv2.putText(frame, "IGNORED", (p[0] + 36, p[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 200), 4)
            elif in_track:
                cv2.circle(frame, p, max(10, int(28 * sc)), (0, 220, 0), 6)     # 수락 트랙의 공
            else:
                cv2.circle(frame, p, max(10, int(28 * sc)), (150, 150, 150), 4)  # 자동 기각된 검출
        for kf, kx, ky in self.keyframes:
            if abs(kf - f) <= 1.0 * self.fps:
                p = (int(kx * sc), int(ky * sc))
                cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS,
                               max(20, int(60 * sc)), max(3, int(8 * sc)))
        self.pane.set_frame(frame)

    # ------------------------------------------------------------ 키프레임
    def _pane_clicked(self, fx, fy):
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        # 같은 시각(±0.5s) 클릭은 교체
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

    def _refresh_kf_list(self):
        self.kf_list.clear()
        self._entries = ([("kf", i) for i in range(len(self.keyframes))]
                         + [("ig", i) for i in range(len(self.ignores))])
        self._entries.sort(key=lambda e: (self.keyframes[e[1]][0] if e[0] == "kf"
                                          else self.ignores[e[1]][0]))
        for kind, i in self._entries:
            if kind == "kf":
                kf, kx, ky = self.keyframes[i]
                t = kf / self.fps
                self.kf_list.addItem(
                    f"{int(t//60):02d}:{t%60:04.1f}  공 ({kx:.0f}, {ky:.0f})")
            else:
                f0, f1 = self.ignores[i]
                self.kf_list.addItem(
                    f"{int(f0/self.fps//60):02d}:{f0/self.fps%60:04.1f}~"
                    f"{int(f1/self.fps//60):02d}:{f1/self.fps%60:04.1f}  무시")

    def _selected_entry(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(getattr(self, "_entries", [])):
            return self._entries[row]
        return None

    def _goto_kf(self):
        e = self._selected_entry()
        if e is None:
            return
        f = self.keyframes[e[1]][0] if e[0] == "kf" else self.ignores[e[1]][0]
        self.slider.setValue(int(f))

    def _delete_kf(self):
        e = self._selected_entry()
        if e is None:
            return
        if e[0] == "kf":
            del self.keyframes[e[1]]
            self._plan_dirty()
        else:
            del self.ignores[e[1]]
            self._recompute_tracks()
        self._save_keyframes()
        self._refresh_kf_list()
        self._redraw()

    def _save_keyframes(self):
        if self.pano_path is not None:
            self._kf_path().write_text(json.dumps(
                {"keyframes": self.keyframes, "ignores": self.ignores}))

    def _load_keyframes(self):
        self.keyframes, self.ignores = [], []
        p = self._kf_path()
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, list):          # 구버전: 키프레임 목록만
                    self.keyframes = [list(k) for k in d]
                else:
                    self.keyframes = [list(k) for k in d.get("keyframes", [])]
                    self.ignores = [list(r) for r in d.get("ignores", [])]
                self.log(f"[ptz] 키프레임 {len(self.keyframes)}개, "
                         f"무시 구간 {len(self.ignores)}개 불러옴")
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 키프레임 파일 무시: {e}")
        self._refresh_kf_list()

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
                       self.combo_mode.currentIndex() == 1, linked=self._linked)
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
        self.log(f"[ptz] 트랙 연결 완료: {len(linked['tracks'])}개")
        self._recompute_tracks()
        self._plan_dirty()

    def _recompute_tracks(self):
        """수락 트랙 구간 재계산 → 트랙바 갱신 (분석/무시 구간 변경 시).

        linked 캐시가 있으면 수락 단계만 돌아 즉각 반응한다.
        """
        if self.analysis is None:
            self.track_spans = []
        else:
            _, _, self.track_spans = accept_ball_tracks(
                self.analysis, ignore_ranges=[tuple(r) for r in self.ignores],
                linked=self._linked, log=self.log)
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes)
        self._refresh_track_list()
        self._plan_dirty()

    def _refresh_track_list(self):
        self.track_list.clear()
        for f0, f1 in self.track_spans:
            t0, t1 = f0 / self.fps, f1 / self.fps
            self.track_list.addItem(
                f"{int(t0//60):02d}:{t0%60:04.1f} ~ {int(t1//60):02d}:{t1%60:04.1f}"
                f"  ({t1-t0:.1f}s)")

    def _goto_track(self):
        row = self.track_list.currentRow()
        if 0 <= row < len(self.track_spans):
            self.slider.setValue(int(self.track_spans[row][0]))

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

    def _ignore_current_track(self):
        """현재 시각을 덮는 수락 트랙을 통째로 무시 목록에 추가."""
        f = self.slider.value()
        for f0, f1 in self.track_spans:
            if f0 <= f <= f1:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self.ignores.append([int(f0), int(f1)])
                    self._save_keyframes()
                    self._recompute_tracks()
                    self._refresh_kf_list()
                    self._redraw()
                finally:
                    QApplication.restoreOverrideCursor()
                self.log(f"[ptz] 트랙 무시: {f0/self.fps:.1f}s ~ {f1/self.fps:.1f}s")
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
                            ignores=[tuple(r) for r in self.ignores])
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
