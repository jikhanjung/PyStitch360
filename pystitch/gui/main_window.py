"""PyStitch360 메인 윈도우.

탭 구성:
  1. 동기화   — 좌/우 영상 나란히 표시, 챕터 통합 타임라인, 오디오 자동/수동 오프셋
  2. 정합     — 자동 정합 + pitch/roll/yaw 미세조정 + 파노라마 미리보기
  3. 내보내기 — 구간/코덱/CRF/해상도, 진행률
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..core.chapters import ChapteredVideo, find_chapters
from ..core.lens import LensProfile, builtin_profiles
from .widgets import FramePane
from .workers import AlignWorker, ExportWorker, PreviewWorker, SyncWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyStitch360")
        self.resize(1500, 900)

        # 상태
        self.video_l: ChapteredVideo | None = None
        self.video_r: ChapteredVideo | None = None
        self.files_l: list[Path] = []
        self.files_r: list[Path] = []
        self.lens: LensProfile | None = None
        self.alignment = None
        self.cur_imgs = (None, None)     # 현재 표시 중인 (L, R) 프레임
        # 워커 참조는 용도별로 분리 (실행 중 재할당 시 QThread 파괴 → 크래시)
        self._sync_worker = None
        self._align_worker = None
        self._preview_worker = None
        self._export_worker = None

        tabs = QTabWidget()
        tabs.addTab(self._build_sync_tab(), "1. 영상·동기화")
        tabs.addTab(self._build_align_tab(), "2. 정합·미리보기")
        tabs.addTab(self._build_export_tab(), "3. 내보내기")

        self.log_box = QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumBlockCount(500)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(tabs)
        split.addWidget(self.log_box)
        split.setSizes([760, 140])
        self.setCentralWidget(split)

        self._load_profiles()

    # ------------------------------------------------------------ 공통

    def log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def _load_profiles(self):
        self.profiles = builtin_profiles()
        self.profile_combo.clear()
        self.profile_combo.addItems(list(self.profiles))
        if self.profiles:
            self.lens = LensProfile.load(next(iter(self.profiles.values())))

    # ------------------------------------------------------------ 탭 1: 동기화

    def _build_sync_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # 상단: 파일 열기 + 렌즈 프로파일
        top = QHBoxLayout()
        self.btn_open_l = QPushButton("좌측 영상 열기...")
        self.btn_open_r = QPushButton("우측 영상 열기...")
        self.btn_open_l.clicked.connect(lambda: self._open_video("L"))
        self.btn_open_r.clicked.connect(lambda: self._open_video("R"))
        self.lbl_files = QLabel("영상을 열어주세요 (첫 챕터 GOPR*.MP4 선택 시 챕터 자동 연결)")
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        top.addWidget(self.btn_open_l)
        top.addWidget(self.btn_open_r)
        top.addWidget(self.lbl_files, 1)
        top.addWidget(QLabel("렌즈:"))
        top.addWidget(self.profile_combo)
        v.addLayout(top)

        # 중앙: 듀얼 뷰어
        panes = QHBoxLayout()
        self.pane_l = FramePane("좌측 영상")
        self.pane_r = FramePane("우측 영상")
        panes.addWidget(self.pane_l)
        panes.addWidget(self.pane_r)
        v.addLayout(panes, 1)

        # 타임라인
        tl = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_time = QLabel("--:--.--")
        self.lbl_time.setMinimumWidth(110)
        tl.addWidget(self.slider, 1)
        tl.addWidget(self.lbl_time)
        v.addLayout(tl)
        self._slider_timer = QTimer(singleShot=True, interval=120)
        self._slider_timer.timeout.connect(self._show_current_frames)

        # 프레임 이동 + 오프셋
        ctl = QHBoxLayout()
        for text, d in [("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)]:
            b = QPushButton(text)
            b.setMaximumWidth(48)
            b.clicked.connect(lambda _, dd=d: self._step(dd))
            ctl.addWidget(b)
        ctl.addSpacing(24)
        ctl.addWidget(QLabel("동기화 오프셋 (초, R−L):"))
        self.spin_offset = QDoubleSpinBox(decimals=3, minimum=-30.0, maximum=30.0,
                                          singleStep=0.033)
        self.spin_offset.valueChanged.connect(lambda _: self._show_current_frames())
        ctl.addWidget(self.spin_offset)
        self.btn_autosync = QPushButton("오디오 자동 동기화")
        self.btn_autosync.clicked.connect(self._auto_sync)
        ctl.addWidget(self.btn_autosync)
        ctl.addStretch(1)
        v.addLayout(ctl)
        return w

    def _open_video(self, side: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"{'좌측' if side == 'L' else '우측'} 영상 (첫 챕터)", "",
            "GoPro 영상 (*.MP4 *.mp4)")
        if not path:
            return
        try:
            chapters = find_chapters(path)
            vid = ChapteredVideo(chapters)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "열기 실패", str(e))
            return
        if side == "L":
            self.video_l, self.files_l = vid, chapters
        else:
            self.video_r, self.files_r = vid, chapters
        names = " + ".join(p.name for p in chapters)
        self.log(f"[open] {side}: {names} ({vid.total_frames}프레임, {vid.duration/60:.1f}분)")
        self._update_file_label()
        if self.video_l and self.video_r:
            self.slider.setEnabled(True)
            self.slider.setRange(0, self.video_l.total_frames - 1)
            self._show_current_frames()

    def _update_file_label(self):
        l = self.files_l[0].name if self.files_l else "—"
        r = self.files_r[0].name if self.files_r else "—"
        nl = f" (+{len(self.files_l)-1}챕터)" if len(self.files_l) > 1 else ""
        nr = f" (+{len(self.files_r)-1}챕터)" if len(self.files_r) > 1 else ""
        self.lbl_files.setText(f"L: {l}{nl}   R: {r}{nr}")

    def _on_profile_changed(self, name):
        if name in getattr(self, "profiles", {}):
            self.lens = LensProfile.load(self.profiles[name])
            self.log(f"[lens] {name}")

    def _on_slider(self, _):
        self._slider_timer.start()   # 디바운스: 드래그 중 과도한 4K 디코딩 방지
        if self.video_l:
            t = self.slider.value() / self.video_l.fps
            cs = round(t * 100)                      # 표시 단위(1/100초)로 먼저 반올림
            m, cs = divmod(cs, 6000)                 # "04:60.00" 방지
            self.lbl_time.setText(f"{m:02d}:{cs/100:05.2f}")

    def _step(self, d: int):
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    def _show_current_frames(self):
        if not (self.video_l and self.video_r):
            return
        fl = self.slider.value()
        fr = int(round(fl + self.spin_offset.value() * self.video_l.fps))
        ok_l, img_l = self.video_l.read_at(fl)
        ok_r, img_r = self.video_r.read_at(fr)
        self.cur_imgs = (img_l if ok_l else None, img_r if ok_r else None)
        self.pane_l.set_frame(self.cur_imgs[0])
        self.pane_r.set_frame(self.cur_imgs[1])

    def _auto_sync(self):
        if not (self.files_l and self.files_r):
            return
        start = self.slider.value() / self.video_l.fps if self.video_l else 0.0
        self.btn_autosync.setEnabled(False)
        self.log(f"[sync] 오디오 상관 분석 중 (t={start:.0f}s 부터 90초)...")
        self._sync_worker = SyncWorker(self.files_l[0], self.files_r[0], start)
        self._sync_worker.done.connect(self._sync_done)
        self._sync_worker.failed.connect(self._worker_failed)
        self._sync_worker.start()

    def _sync_done(self, offset, conf):
        self.btn_autosync.setEnabled(True)
        self.spin_offset.setValue(offset)
        fps = self.video_l.fps if self.video_l else 29.97
        self.log(f"[sync] 오프셋 {offset:+.3f}s ({offset*fps:+.1f}프레임), 신뢰도 {conf:.1f}"
                 + (" — 낮음, 수동 확인 필요" if conf < 4 else ""))

    def _worker_failed(self, msg):
        self.btn_autosync.setEnabled(True)
        self.btn_align.setEnabled(True)
        self.log(f"[오류] {msg}")

    # ------------------------------------------------------------ 탭 2: 정합

    def _build_align_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        top = QHBoxLayout()
        self.btn_align = QPushButton("현재 프레임으로 자동 정합")
        self.btn_align.clicked.connect(self._run_align)
        self.lbl_align = QLabel("정합 없음")
        top.addWidget(self.btn_align)
        top.addWidget(self.lbl_align, 1)
        v.addLayout(top)

        self.pano_pane = FramePane("파노라마 미리보기")
        v.addWidget(self.pano_pane, 1)

        adj = QGroupBox("미세조정 (자동 추정값에 더해짐, 도 단위)")
        g = QGridLayout(adj)
        self.spin_user = {}
        for col, (key, label) in enumerate([("pitch", "수평(pitch)"),
                                            ("roll", "기울기(roll)"),
                                            ("yaw", "센터링(yaw)")]):
            g.addWidget(QLabel(label), 0, col * 2)
            sp = QDoubleSpinBox(decimals=1, minimum=-15.0, maximum=15.0, singleStep=0.1)
            sp.valueChanged.connect(lambda _: self._preview_debounced())
            self.spin_user[key] = sp
            g.addWidget(sp, 0, col * 2 + 1)
        g.addWidget(QLabel("심 페더(px)"), 0, 6)
        self.spin_feather = QSpinBox(minimum=2, maximum=400, value=40)
        self.spin_feather.valueChanged.connect(lambda _: self._preview_debounced())
        g.addWidget(self.spin_feather, 0, 7)
        v.addWidget(adj)

        self._preview_timer = QTimer(singleShot=True, interval=400)
        self._preview_timer.timeout.connect(self._render_preview)
        return w

    def _run_align(self):
        img_l, img_r = self.cur_imgs
        if img_l is None or img_r is None:
            QMessageBox.information(self, "정합", "먼저 1번 탭에서 영상을 열고 프레임을 선택하세요.")
            return
        if self.lens is None:
            QMessageBox.warning(self, "정합", "렌즈 프로파일이 없습니다.")
            return
        self.btn_align.setEnabled(False)
        self.lbl_align.setText("정합 중...")
        self._align_worker = AlignWorker(img_l.copy(), img_r.copy(), self.lens)
        self._align_worker.log.connect(self.log)
        self._align_worker.done.connect(self._align_done)
        self._align_worker.failed.connect(self._align_failed)
        self._align_worker.start()

    def _align_done(self, alignment):
        self.btn_align.setEnabled(True)
        self.alignment = alignment
        a = alignment
        self.lbl_align.setText(
            f"인라이어 {a.n_inliers}/{a.n_matches}, 잔차 {a.residual_deg:.2f}°, "
            f"상대회전 {a.yaw_split_deg:.1f}°, 자동 pitch {a.pitch_auto*57.3:+.1f}° "
            f"roll {a.roll_auto*57.3:+.1f}°")
        self._render_preview()

    def _align_failed(self, msg):
        self.btn_align.setEnabled(True)
        self.lbl_align.setText("정합 실패")
        self.log(f"[오류] {msg}")

    def _preview_debounced(self):
        if self.alignment is not None:
            self._preview_timer.start()

    def _render_preview(self):
        img_l, img_r = self.cur_imgs
        if self.alignment is None or img_l is None or img_r is None:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_timer.start()   # 이전 렌더 끝난 뒤 재시도
            return
        self._preview_worker = PreviewWorker(
            self.lens, self.alignment, img_l.copy(), img_r.copy(),
            self.spin_user["pitch"].value(), self.spin_user["roll"].value(),
            self.spin_user["yaw"].value(), self.spin_feather.value())
        self._preview_worker.log.connect(self.log)
        self._preview_worker.done.connect(self.pano_pane.set_frame)
        self._preview_worker.failed.connect(lambda m: self.log(f"[오류] {m}"))
        self._preview_worker.start()

    # ------------------------------------------------------------ 탭 3: 내보내기

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGridLayout()
        g.addWidget(QLabel("시작 (초)"), 0, 0)
        self.spin_start = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1e6)
        g.addWidget(self.spin_start, 0, 1)
        g.addWidget(QLabel("끝 (초)"), 0, 2)
        self.spin_end = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1e6, value=60.0)
        g.addWidget(self.spin_end, 0, 3)
        g.addWidget(QLabel("코덱"), 1, 0)
        self.combo_codec = QComboBox()
        self.combo_codec.addItems(["libx264 (H.264)", "libx265 (HEVC)"])
        g.addWidget(self.combo_codec, 1, 1)
        g.addWidget(QLabel("CRF"), 1, 2)
        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=19)
        g.addWidget(self.spin_crf, 1, 3)
        g.addWidget(QLabel("해상도"), 2, 0)
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100% (~5900px)", "50% (~2950px)"])
        g.addWidget(self.combo_scale, 2, 1)
        v.addLayout(g)

        h = QHBoxLayout()
        self.btn_export = QPushButton("내보내기 시작...")
        self.btn_export.clicked.connect(self._start_export)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_export)
        h.addWidget(self.btn_export)
        h.addWidget(self.btn_cancel)
        h.addStretch(1)
        v.addLayout(h)

        self.progress = QProgressBar()
        v.addWidget(self.progress)
        v.addStretch(1)
        return w

    def _start_export(self):
        if self.alignment is None:
            QMessageBox.information(self, "내보내기", "먼저 2번 탭에서 정합을 실행하세요.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "출력 파일", "panorama.mp4",
                                             "MP4 (*.mp4)")
        if not out:
            return
        codec = "libx264" if self.combo_codec.currentIndex() == 0 else "libx265"
        scale = 1.0 if self.combo_scale.currentIndex() == 0 else 0.5
        self._export_worker = ExportWorker(
            self.lens, self.alignment,
            [str(p) for p in self.files_l], [str(p) for p in self.files_r],
            self.spin_offset.value(), self.spin_start.value(), self.spin_end.value(),
            out,
            self.spin_user["pitch"].value(), self.spin_user["roll"].value(),
            self.spin_user["yaw"].value(),
            codec=codec, crf=self.spin_crf.value(), scale=scale,
            feather_px=self.spin_feather.value())
        self._export_worker.log.connect(self.log)
        self._export_worker.progress.connect(self._export_progress)
        self._export_worker.finished_ok.connect(self._export_done)
        self._export_worker.failed.connect(self._export_failed)
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._export_worker.start()

    def _export_progress(self, done, total, fps):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        remain = (total - done) / fps if fps > 0 else 0
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain/60:.0f}분)")

    def _export_done(self, path):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log(f"[export] 저장: {path}")
        QMessageBox.information(self, "내보내기", f"완료: {path}")

    def _export_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log(f"[export] 실패: {msg}")

    def _cancel_export(self):
        if self._export_worker:
            self._export_worker.cancel()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
