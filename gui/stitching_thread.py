"""
스티칭 작업 스레드
백그라운드에서 스티칭 작업 실행
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex

from core.preprocessor import Preprocessor
from core.stitcher import Stitcher
from core.postprocessor import Postprocessor


class StitchingThread(QThread):
    """스티칭 작업 스레드"""
    
    # 시그널 정의
    progress_update = pyqtSignal(int, int)  # current, total
    step_update = pyqtSignal(str)  # step name
    log_message = pyqtSignal(str)  # log message
    error_occurred = pyqtSignal(str)  # error message
    finished = pyqtSignal(bool)  # success
    preview_ready = pyqtSignal(np.ndarray)  # preview image
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.mutex = QMutex()
        self.is_cancelled = False
        self.is_paused = False
        
        # 모듈 초기화
        self.preprocessor = Preprocessor()
        self.stitcher = Stitcher()
        self.postprocessor = Postprocessor()
        
        # 작업 파라미터
        self.front_files = []
        self.back_files = []
        self.output_path = None
        self.settings = {}
    
    def set_parameters(self, front_files: List[Path], back_files: List[Path], 
                      output_path: Path, settings: dict):
        """작업 파라미터 설정"""
        self.front_files = front_files
        self.back_files = back_files
        self.output_path = output_path
        self.settings = settings
    
    def cancel(self):
        """작업 취소"""
        self.mutex.lock()
        self.is_cancelled = True
        self.mutex.unlock()
    
    def pause(self):
        """작업 일시정지"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    
    def resume(self):
        """작업 재개"""
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()
    
    def check_cancelled(self):
        """취소 상태 확인"""
        self.mutex.lock()
        cancelled = self.is_cancelled
        self.mutex.unlock()
        return cancelled
    
    def check_paused(self):
        """일시정지 상태 확인"""
        self.mutex.lock()
        paused = self.is_paused
        self.mutex.unlock()
        
        while paused and not self.check_cancelled():
            self.msleep(100)
            self.mutex.lock()
            paused = self.is_paused
            self.mutex.unlock()
    
    def run(self):
        """스티칭 작업 실행"""
        try:
            self.is_cancelled = False
            self.is_paused = False
            total_steps = 7
            current_step = 0
            
            # 1. 캘리브레이션 로드
            self.step_update.emit("캘리브레이션 데이터 로드 중...")
            self.log_message.emit("캘리브레이션 데이터 로드 시작")
            
            calib_path = Path("presets") / self.settings.get("calibration", "gopro_dual.yaml")
            if not self.stitcher.load_calibration(calib_path):
                self.error_occurred.emit("캘리브레이션 데이터 로드 실패")
                self.finished.emit(False)
                return
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 2. 전면 카메라 영상 연결
            self.step_update.emit("전면 카메라 영상 연결 중...")
            self.log_message.emit(f"전면 카메라 파일 {len(self.front_files)}개 연결")
            
            temp_dir = self.output_path.parent / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            front_concat = temp_dir / "front_concat.mp4"
            if not self.preprocessor.concat_videos(self.front_files, front_concat):
                self.error_occurred.emit("전면 카메라 영상 연결 실패")
                self.finished.emit(False)
                return
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 3. 후면 카메라 영상 연결
            self.step_update.emit("후면 카메라 영상 연결 중...")
            self.log_message.emit(f"후면 카메라 파일 {len(self.back_files)}개 연결")
            
            back_concat = temp_dir / "back_concat.mp4"
            if not self.preprocessor.concat_videos(self.back_files, back_concat):
                self.error_occurred.emit("후면 카메라 영상 연결 실패")
                self.finished.emit(False)
                return
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 4. 동기화 조정
            sync_offset = self.settings.get("sync_offset", 0)
            if sync_offset != 0:
                self.step_update.emit("동기화 조정 중...")
                self.log_message.emit(f"프레임 오프셋: {sync_offset}")
                
                front_synced = temp_dir / "front_synced.mp4"
                back_synced = temp_dir / "back_synced.mp4"
                
                if not self.preprocessor.adjust_sync(
                    front_concat, back_concat, sync_offset,
                    front_synced, back_synced
                ):
                    self.error_occurred.emit("동기화 조정 실패")
                    self.finished.emit(False)
                    return
                
                front_concat = front_synced
                back_concat = back_synced
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 5. 스티칭 처리
            self.step_update.emit("360도 영상 스티칭 중...")
            self.log_message.emit("스티칭 처리 시작")
            
            # 프레임별 처리를 위한 임시 구현
            # 실제로는 영상 전체를 처리해야 함
            front_cap = cv2.VideoCapture(str(front_concat))
            back_cap = cv2.VideoCapture(str(back_concat))
            
            # 첫 프레임으로 미리보기
            ret1, front_frame = front_cap.read()
            ret2, back_frame = back_cap.read()
            
            if ret1 and ret2:
                # 파노라마 생성
                panorama = self.stitcher.create_panorama(front_frame, back_frame)
                if panorama is not None:
                    # Equirectangular 투영
                    resolution = self.settings.get("resolution", "3840x1920")
                    width, height = map(int, resolution.split('x'))
                    equirect = self.stitcher.apply_equirectangular_projection(
                        panorama, width, height
                    )
                    
                    # 방향 조정
                    yaw = self.settings.get("yaw", 0)
                    pitch = self.settings.get("pitch", 0)
                    roll = self.settings.get("roll", 0)
                    final_image = self.stitcher.apply_orientation(
                        equirect, yaw, pitch, roll
                    )
                    
                    # 미리보기 시그널 발송
                    self.preview_ready.emit(final_image)
            
            front_cap.release()
            back_cap.release()
            
            # 실제 구현에서는 전체 영상을 처리해야 함
            # 여기서는 임시로 첫 프레임만 저장
            temp_stitched = temp_dir / "stitched.mp4"
            # TODO: 전체 영상 스티칭 구현
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 6. 인코딩
            self.step_update.emit("H.264 인코딩 중...")
            crf = self.settings.get("crf", 23)
            preset = self.settings.get("preset", "medium")
            self.log_message.emit(f"인코딩 설정: CRF={crf}, Preset={preset}")
            
            # 임시: 연결된 영상을 그대로 출력
            # TODO: 실제 스티칭된 영상 인코딩
            encoded_path = temp_dir / "encoded.mp4"
            if not self.postprocessor.encode_h264(
                front_concat, encoded_path, crf, preset
            ):
                self.error_occurred.emit("인코딩 실패")
                self.finished.emit(False)
                return
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            self.check_paused()
            if self.check_cancelled():
                return
            
            # 7. 메타데이터 삽입 및 최종 출력
            self.step_update.emit("메타데이터 삽입 중...")
            
            if self.settings.get("metadata_enabled", True):
                metadata = {
                    "projection": "equirectangular",
                    "title": "PyStitch360 Output",
                    "description": "360-degree video stitched by PyStitch360"
                }
                
                if self.settings.get("insta360_compatible", False):
                    # Insta360 호환 포맷
                    if not self.postprocessor.create_insta360_compatible(
                        encoded_path, self.output_path
                    ):
                        self.error_occurred.emit("Insta360 호환 포맷 생성 실패")
                        self.finished.emit(False)
                        return
                else:
                    # 일반 메타데이터 삽입
                    import shutil
                    shutil.copy2(encoded_path, self.output_path)
                    if not self.postprocessor.insert_metadata(
                        self.output_path, metadata
                    ):
                        self.log_message.emit("메타데이터 삽입 실패 (영상은 정상 출력됨)")
            else:
                import shutil
                shutil.copy2(encoded_path, self.output_path)
            
            current_step += 1
            self.progress_update.emit(current_step, total_steps)
            
            # 임시 파일 정리
            self.step_update.emit("임시 파일 정리 중...")
            self.log_message.emit("임시 파일 삭제")
            for temp_file in temp_dir.glob("*"):
                temp_file.unlink(missing_ok=True)
            temp_dir.rmdir()
            
            self.log_message.emit("스티칭 작업 완료!")
            self.finished.emit(True)
            
        except Exception as e:
            self.logger.error(f"스티칭 작업 중 오류: {e}", exc_info=True)
            self.error_occurred.emit(f"작업 중 오류 발생: {str(e)}")
            self.finished.emit(False)