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
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..core.encoders import available_encoders
from ..core.ptz import analyze_video, build_plan, ptz_available, render_plan
from .widgets import FramePane


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


class PtzRenderWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_path, analysis, keyframes, codec, crf,
                 wide=False):
        super().__init__()
        self.args = (pano_path, out_path, analysis, keyframes, codec, crf, wide)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, out, analysis, kfs, codec, crf, wide = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide,
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
        self._analyze_worker = None
        self._render_worker = None
        self._build_ui()

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
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        top.addWidget(self.btn_analyze)
        v.addLayout(top)

        mid = QHBoxLayout()
        self.pane = FramePane("클릭 = 현재 시각의 공 위치 키프레임", interactive=True)
        self.pane.clicked.connect(self._pane_clicked)
        mid.addWidget(self.pane, 1)
        side = QVBoxLayout()
        side.addWidget(QLabel("키프레임 (더블클릭=이동)"))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumWidth(240)
        self.kf_list.itemDoubleClicked.connect(self._goto_kf)
        side.addWidget(self.kf_list, 1)
        btn_del = QPushButton("선택 삭제")
        btn_del.clicked.connect(self._delete_kf)
        side.addWidget(btn_del)
        mid.addLayout(side)
        v.addLayout(mid, 1)

        tl = QHBoxLayout()
        for text, d in [("-10s", -300), ("-1s", -30), ("-1", -1),
                        ("+1", 1), ("+1s", 30), ("+10s", 300)]:
            b = QPushButton(text)
            b.setMaximumWidth(52)
            b.clicked.connect(lambda _, dd=d: self._step(dd))
            tl.addWidget(b)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        tl.addWidget(self.slider, 1)
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
        self.btn_analyze.setEnabled(ptz_available())
        if not ptz_available():
            self.btn_analyze.setToolTip("ultralytics 미설치 (pip install ultralytics)")
        self.analysis = None
        self._load_keyframes()
        self._load_cached_analysis()
        self._show_frame()
        self._update_export_enabled()

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
        self._show_frame()

    def _analyze_failed(self, msg):
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log(f"[오류] 분석: {msg}")

    # ------------------------------------------------------------ 타임라인/표시
    def _step(self, d):
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    def _on_slider(self, _):
        t = self.slider.value() / self.fps
        cs = round(t * 100)
        m, cs = divmod(cs, 6000)
        self.lbl_time.setText(f"{m:02d}:{cs/100:05.2f}")
        self._slider_timer.start()

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
        if self.cap is None:
            return
        f = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        if not ok:
            return
        b = self._current_auto_ball()
        if b is not None:
            cv2.circle(frame, (int(b[0]), int(b[1])), 28, (0, 220, 0), 6)
        for kf, kx, ky in self.keyframes:
            if abs(kf - f) <= 1.0 * self.fps:
                p = (int(kx), int(ky))
                cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS, 60, 8)
        self.pane.set_frame(frame)

    # ------------------------------------------------------------ 키프레임
    def _pane_clicked(self, fx, fy):
        if self.cap is None:
            return
        f = self.slider.value()
        x, y = fx * self.pano_w, fy * self.pano_h
        # 같은 시각(±0.5s) 클릭은 교체
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        self.keyframes.append([f, round(x, 1), round(y, 1)])
        self.keyframes.sort()
        self._save_keyframes()
        self._refresh_kf_list()
        self._show_frame()
        self.log(f"[ptz] 키프레임 {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _refresh_kf_list(self):
        self.kf_list.clear()
        for kf, kx, ky in self.keyframes:
            t = kf / self.fps
            self.kf_list.addItem(f"{int(t//60):02d}:{t%60:04.1f}  ({kx:.0f}, {ky:.0f})")

    def _goto_kf(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.keyframes):
            self.slider.setValue(int(self.keyframes[row][0]))

    def _delete_kf(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.keyframes):
            del self.keyframes[row]
            self._save_keyframes()
            self._refresh_kf_list()
            self._show_frame()

    def _save_keyframes(self):
        if self.pano_path is not None:
            self._kf_path().write_text(json.dumps(self.keyframes))

    def _load_keyframes(self):
        self.keyframes = []
        p = self._kf_path()
        if p.exists():
            try:
                self.keyframes = [list(k) for k in json.loads(p.read_text())]
                self.log(f"[ptz] 키프레임 {len(self.keyframes)}개 불러옴")
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 키프레임 파일 무시: {e}")
        self._refresh_kf_list()

    # ------------------------------------------------------------ 내보내기
    def _update_export_enabled(self):
        self.btn_export.setEnabled(self.analysis is not None)

    def _start_render(self):
        if self.analysis is None or self.pano_path is None:
            return
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
                            codec, self.spin_crf.value(), wide=wide)
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
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 {remain:.0f}분)")

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
