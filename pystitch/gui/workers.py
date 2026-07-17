"""백그라운드 워커 스레드 (동기화, 정합, 미리보기, 내보내기)."""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..core.align import Alignment, estimate_alignment
from ..core.chapters import ChapteredVideo
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


class AlignWorker(QThread):
    done = pyqtSignal(object)            # Alignment
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, img_l, img_r, lens: LensProfile):
        super().__init__()
        self.img_l, self.img_r, self.lens = img_l, img_r, lens

    def run(self):
        try:
            a = estimate_alignment(self.img_l, self.img_r, self.lens,
                                   log=lambda s: self.log.emit(s))
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
                 feather_px=40, scale=0.25):
        super().__init__()
        self.lens, self.a = lens, alignment
        self.img_l, self.img_r = img_l, img_r
        self.user = (pitch_user, roll_user, yaw_user)
        self.feather_px, self.scale = feather_px, scale

    def run(self):
        try:
            R_wl, R_wr = self.a.rotations(self.user[0], self.user[1])
            yaw0, yaw1 = self.a.window(self.user[2])
            r = Renderer(self.lens, R_wl, R_wr, yaw0, yaw1, self.a.el0, self.a.el1,
                         scale=self.scale, feather_px=self.feather_px)
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

    def __init__(self, lens, alignment: Alignment, left_files, right_files,
                 offset_sec, start_sec, end_sec, out_path,
                 pitch_user=0.0, roll_user=0.0, yaw_user=0.0,
                 codec="libx264", crf=19, scale=1.0, feather_px=40):
        super().__init__()
        self.lens, self.a = lens, alignment
        self.left_files, self.right_files = left_files, right_files
        self.offset, self.start, self.end = offset_sec, start_sec, end_sec
        self.out_path = str(out_path)
        self.user = (pitch_user, roll_user, yaw_user)
        self.codec, self.crf, self.scale = codec, crf, scale
        self.feather_px = feather_px
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def _run(self):
        vid_l = ChapteredVideo(self.left_files)
        vid_r = ChapteredVideo(self.right_files)
        fps = vid_l.fps
        f_start = int(round(self.start * fps))
        f_end = min(int(round(self.end * fps)), vid_l.total_frames)
        r_start = int(round((self.start + self.offset) * fps))
        total = max(0, f_end - f_start)
        if total == 0:
            raise RuntimeError("내보낼 구간이 비어 있음")

        self.log.emit("정합 렌더러 준비 중...")
        ok_l, img_l = vid_l.read_at(f_start)
        ok_r, img_r = vid_r.read_at(r_start)
        if not (ok_l and ok_r):
            raise RuntimeError("시작 프레임 읽기 실패")
        R_wl, R_wr = self.a.rotations(self.user[0], self.user[1])
        yaw0, yaw1 = self.a.window(self.user[2])
        rend = Renderer(self.lens, R_wl, R_wr, yaw0, yaw1, self.a.el0, self.a.el1,
                        scale=self.scale, feather_px=self.feather_px)
        rend.set_gains_from(img_l, img_r)
        rend.refine_seam(img_l, img_r, log=lambda s: self.log.emit(s))

        # 오디오: 좌측 챕터 체인을 concat demuxer 로 연결
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            for f in self.left_files:
                tf.write(f"file '{Path(f).as_posix()}'\n")
            concat_list = tf.name

        duration = total / fps
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{rend.out_w}x{rend.out_h}", "-r", f"{fps}", "-i", "-",
               "-f", "concat", "-safe", "0", "-ss", f"{self.start}",
               "-t", f"{duration}", "-i", concat_list,
               "-map", "0:v", "-map", "1:a?",
               "-c:v", self.codec, "-preset", "fast", "-crf", str(self.crf),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
        if self.codec == "libx265":
            cmd += ["-tag:v", "hvc1"]
        cmd.append(self.out_path)
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        vid_l.seek_frame(f_start)
        vid_r.seek_frame(r_start)
        t0 = time.perf_counter()
        done = 0
        try:
            for i in range(total):
                if self._cancel:
                    self.log.emit("사용자 취소")
                    break
                ok_l, img_l = vid_l.read()
                ok_r, img_r = vid_r.read()
                if not (ok_l and ok_r):
                    self.log.emit(f"입력 종료 (프레임 {i})")
                    break
                frame = rend.render(img_l, img_r)
                enc.stdin.write(frame.tobytes())
                done += 1
                if done % 30 == 0:
                    self.progress.emit(done, total, done / (time.perf_counter() - t0))
        finally:
            enc.stdin.close()
            enc.wait()
            vid_l.release()
            vid_r.release()
            Path(concat_list).unlink(missing_ok=True)

        if self._cancel:
            self.failed.emit("취소됨")
        else:
            el = time.perf_counter() - t0
            self.log.emit(f"완료: {done} 프레임 / {el:.0f}s = {done/max(el,1e-9):.2f} fps")
            self.finished_ok.emit(self.out_path)
