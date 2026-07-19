"""백그라운드 워커 스레드 (동기화, 정합, 미리보기, 내보내기)."""
from __future__ import annotations

import time

from PyQt6.QtCore import QThread, pyqtSignal

from ..core.align import Alignment, estimate_alignment
from ..core.chapters import ChapteredVideo
from ..core.export import export_pano
from ..core.lens import LensProfile
from ..core.render import Renderer
from ..core.sync import estimate_offset


class SyncWorker(QThread):
    done = pyqtSignal(float, float)      # offset_sec, confidence
    failed = pyqtSignal(str)

    def __init__(self, left_file, right_file, start, duration=90.0):
        super().__init__()
        self.args = (str(left_file), str(right_file), start, duration)

    def run(self):
        try:
            offset, conf = estimate_offset(*self.args)
            self.done.emit(offset, conf)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class GpmfWorker(QThread):
    """자이로 기반 충격 이벤트 탐지 (좌측 카메라 기준)."""

    done = pyqtSignal(list)              # list[GyroEvent]
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, files, durations):
        super().__init__()
        self.files = [str(f) for f in files]
        self.durations = durations

    def run(self):
        try:
            from ..core.gpmf import detect_bump_events
            events = detect_bump_events(self.files, self.durations,
                                        log=lambda s: self.log.emit(s))
            self.done.emit(events)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class AlignWorker(QThread):
    done = pyqtSignal(object)            # Alignment
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, img_l, img_r, lens: LensProfile, reuse_level=None):
        super().__init__()
        self.img_l, self.img_r, self.lens = img_l, img_r, lens
        self.reuse_level = reuse_level

    def run(self):
        try:
            a = estimate_alignment(self.img_l, self.img_r, self.lens,
                                   log=lambda s: self.log.emit(s),
                                   reuse_level=self.reuse_level)
            self.done.emit(a)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PreviewWorker(QThread):
    """미리보기 렌더러 재구성 + 1프레임 렌더 (슬라이더 조정 시 디바운스 후 호출)."""

    done = pyqtSignal(object)            # pano ndarray
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, lens, alignment: Alignment, img_l, img_r,
                 pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                 feather_px=40, scale=0.25, persp_k=0.0, persp_m=1.0,
                 el0=None, el1=None):
        super().__init__()
        self.lens, self.a = lens, alignment
        self.img_l, self.img_r = img_l, img_r
        self.user = (pitch_user, roll_user, yaw_user)
        self.feather_px, self.scale = feather_px, scale
        self.persp = (persp_k, persp_m)
        self.el = (el0, el1)

    def run(self):
        try:
            R_wl, R_wr = self.a.rotations(self.user[0], self.user[1])
            yaw0, yaw1 = self.a.window(self.user[2])
            el0 = self.el[0] if self.el[0] is not None else self.a.el0
            el1 = self.el[1] if self.el[1] is not None else self.a.el1
            r = Renderer(self.lens, R_wl, R_wr, yaw0, yaw1, el0, el1,
                         scale=self.scale, feather_px=self.feather_px,
                         persp_k=self.persp[0], persp_m=self.persp[1])
            r.set_gains_from(self.img_l, self.img_r)
            r.refine_seam(self.img_l, self.img_r, log=lambda s: self.log.emit(s))
            self.done.emit(r.render(self.img_l, self.img_r))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportWorker(QThread):
    progress = pyqtSignal(int, int, float)   # done, total, fps
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, lens, segments, left_files, right_files,
                 offset_sec, start_sec, end_sec, out_path,
                 pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                 codec="libx264", crf=19, scale=1.0, feather_px=40,
                 ptz=False, persp_k=0.0, persp_m=1.0, el0=None, el1=None):
        super().__init__()
        self.lens = lens
        # segments: [{"start_sec": float, "alignment": Alignment}, ...] 오름차순
        # (하위호환: Alignment 단일 객체도 허용)
        if isinstance(segments, Alignment):
            segments = [{"start_sec": 0.0, "alignment": segments}]
        self.segments = sorted(segments, key=lambda s: s["start_sec"])
        self.left_files, self.right_files = left_files, right_files
        self.offset, self.t_start, self.t_end = offset_sec, start_sec, end_sec
        self.out_path = str(out_path)
        self.user = (pitch_user, roll_user, yaw_user)
        self.codec, self.crf, self.scale = codec, crf, scale
        self.feather_px = feather_px
        self.ptz = ptz
        self.persp = (persp_k, persp_m)
        self.el = (el0, el1)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            out = export_pano(
                self.lens, self.segments, self.left_files, self.right_files,
                self.offset, self.t_start, self.t_end, self.out_path,
                pitch_user=self.user[0], roll_user=self.user[1],
                yaw_user=self.user[2],
                codec=self.codec, crf=self.crf, scale=self.scale,
                feather_px=self.feather_px, ptz=self.ptz,
                persp_k=self.persp[0], persp_m=self.persp[1],
                el0=self.el[0], el1=self.el[1],
                progress=self.progress.emit,
                log=self.log.emit,
                cancel=lambda: self._cancel)
            self.finished_ok.emit(out)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PlaybackWorker(QThread):
    """정합된 파노라마 동영상 재생 (미리보기 해상도).

    렌더러는 시작 시 1회만 구성(게인·심 보정 포함)하고, 자체
    ChapteredVideo 인스턴스로 순차 디코딩 — GUI 쪽 디코더와 충돌 없음.
    display_fps 로 프레임을 건너뛰며 실시간 속도에 맞춘다.
    """

    frame_ready = pyqtSignal(object, int)     # pano ndarray, 절대 프레임 번호
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, lens, alignment, left_files, right_files, offset_sec,
                 start_frame, pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                 feather_px=40, scale=0.25, display_fps=10.0,
                 persp_k=0.0, persp_m=1.0, el0=None, el1=None):
        super().__init__()
        self.lens, self.a = lens, alignment
        self.left_files = [str(f) for f in left_files]
        self.right_files = [str(f) for f in right_files]
        self.offset, self.start_frame = offset_sec, start_frame
        self.user = (pitch_user, roll_user, yaw_user)
        self.feather_px, self.scale = feather_px, scale
        self.display_fps = display_fps
        self.persp = (persp_k, persp_m)
        self.el = (el0, el1)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        vid_l = vid_r = None
        try:
            vid_l = ChapteredVideo(self.left_files)
            vid_r = ChapteredVideo(self.right_files)
            fps = vid_l.fps
            step = max(1, int(round(fps / self.display_fps)))
            f = self.start_frame
            ok_l, img_l = vid_l.read_at(f)
            ok_r, img_r = vid_r.read_at(int(round(f + self.offset * fps)))
            if not (ok_l and ok_r):
                raise RuntimeError("재생 시작 프레임 읽기 실패")
            R_wl, R_wr = self.a.rotations(self.user[0], self.user[1])
            yaw0, yaw1 = self.a.window(self.user[2])
            el0 = self.el[0] if self.el[0] is not None else self.a.el0
            el1 = self.el[1] if self.el[1] is not None else self.a.el1
            rend = Renderer(self.lens, R_wl, R_wr, yaw0, yaw1, el0, el1,
                            scale=self.scale, feather_px=self.feather_px,
                            persp_k=self.persp[0], persp_m=self.persp[1])
            rend.set_gains_from(img_l, img_r)
            rend.refine_seam(img_l, img_r, log=lambda s: self.log.emit(s))
            t0 = time.perf_counter()
            played = 0
            while not self._stop:
                self.frame_ready.emit(rend.render(img_l, img_r), f)
                # 실시간 페이스: 다음 표시 시각까지 대기 (렌더가 느리면 자연 감속)
                played += 1
                lag = played * step / fps - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
                for k in range(step):
                    if k < step - 1:   # 건너뛰는 프레임은 색변환 없이 grab만
                        ok_l, ok_r = vid_l.grab(), vid_r.grab()
                    else:
                        ok_l, img_l = vid_l.read()
                        ok_r, img_r = vid_r.read()
                f += step
                if not (ok_l and ok_r):
                    break
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
        finally:
            for v in (vid_l, vid_r):
                if v is not None:
                    v.release()
