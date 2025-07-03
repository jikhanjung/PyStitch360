"""
전처리 모듈
GoPro 파일 인식, concat, 동기화 처리
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import ffmpeg


class Preprocessor:
    """영상 전처리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_gopro_files(self, directory: Path) -> Tuple[List[Path], List[Path]]:
        """
        GoPro 파일 자동 감지
        
        Args:
            directory: 검색할 디렉터리
            
        Returns:
            Tuple[front_files, back_files]: 전면/후면 카메라 파일 리스트
        """
        self.logger.info(f"GoPro 파일 검색: {directory}")
        
        if not directory.exists():
            self.logger.error(f"디렉터리가 존재하지 않음: {directory}")
            return [], []
        
        front_files = []
        back_files = []
        
        # GoPro 파일 패턴: GOPR[숫자].MP4, GP[숫자][숫자].MP4
        for file_path in directory.glob("*.MP4"):
            file_name = file_path.name
            
            # GOPR로 시작하는 파일들 (메인 파일)
            if re.match(r'GOPR\d+\.MP4', file_name):
                # 파일명에서 숫자 추출하여 분류
                number = re.search(r'GOPR(\d+)\.MP4', file_name).group(1)
                if int(number) % 2 == 0:  # 짝수는 전면
                    front_files.append(file_path)
                else:  # 홀수는 후면
                    back_files.append(file_path)
            
            # GP로 시작하는 연속 파일들
            elif re.match(r'GP\d+\d+\.MP4', file_name):
                # 첫 번째 숫자로 분류
                first_digit = file_name[2]
                if int(first_digit) % 2 == 0:
                    front_files.append(file_path)
                else:
                    back_files.append(file_path)
        
        # 파일명 순으로 정렬
        front_files.sort()
        back_files.sort()
        
        self.logger.info(f"감지된 파일: 전면 {len(front_files)}개, 후면 {len(back_files)}개")
        return front_files, back_files
    
    def concat_videos(self, file_list: List[Path], output_path: Path) -> bool:
        """
        FFmpeg를 사용한 영상 연결 (demuxer 방식)
        
        Args:
            file_list: 연결할 파일 리스트
            output_path: 출력 파일 경로
            
        Returns:
            성공 여부
        """
        if not file_list:
            self.logger.error("연결할 파일이 없음")
            return False
        
        if len(file_list) == 1:
            # 파일이 하나면 복사만
            try:
                import shutil
                shutil.copy2(file_list[0], output_path)
                self.logger.info(f"단일 파일 복사: {file_list[0]} -> {output_path}")
                return True
            except Exception as e:
                self.logger.error(f"파일 복사 실패: {e}")
                return False
        
        self.logger.info(f"영상 연결: {len(file_list)}개 파일 -> {output_path}")
        
        try:
            # concat list 파일 생성
            concat_file = output_path.parent / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for file_path in file_list:
                    f.write(f"file '{file_path.absolute()}'\n")
            
            # FFmpeg concat 실행
            cmd = [
                'ffmpeg', '-y',  # 덮어쓰기
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',  # 재인코딩 없이 복사
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 임시 파일 삭제
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self.logger.info("영상 연결 성공")
                return True
            else:
                self.logger.error(f"FFmpeg 오류: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"영상 연결 실패: {e}")
            return False
    
    def adjust_sync(self, video1_path: Path, video2_path: Path, 
                   offset_frames: int, output1_path: Path, output2_path: Path) -> bool:
        """
        동기화 조정
        
        Args:
            video1_path: 첫 번째 영상
            video2_path: 두 번째 영상
            offset_frames: 프레임 오프셋 (양수면 video2를 늦춤)
            output1_path: 조정된 첫 번째 영상 출력 경로
            output2_path: 조정된 두 번째 영상 출력 경로
            
        Returns:
            성공 여부
        """
        self.logger.info(f"동기화 조정: {offset_frames} 프레임 오프셋")
        
        try:
            # 영상 정보 가져오기
            probe1 = ffmpeg.probe(str(video1_path))
            video_stream1 = next(s for s in probe1['streams'] if s['codec_type'] == 'video')
            fps = eval(video_stream1['r_frame_rate'])  # 프레임레이트
            
            # 프레임을 시간으로 변환
            offset_seconds = offset_frames / fps
            
            if offset_frames > 0:
                # video2를 늦춤 (앞부분 잘라냄)
                ffmpeg.input(str(video1_path)).output(str(output1_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
                ffmpeg.input(str(video2_path), ss=offset_seconds).output(str(output2_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
            elif offset_frames < 0:
                # video1을 늦춤 (앞부분 잘라냄)
                ffmpeg.input(str(video1_path), ss=abs(offset_seconds)).output(str(output1_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
                ffmpeg.input(str(video2_path)).output(str(output2_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
            else:
                # 오프셋이 0이면 단순 복사
                ffmpeg.input(str(video1_path)).output(str(output1_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
                ffmpeg.input(str(video2_path)).output(str(output2_path), vcodec='copy', acodec='copy').run(overwrite_output=True, quiet=True)
            
            self.logger.info("동기화 조정 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"동기화 조정 실패: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> Optional[dict]:
        """
        영상 정보 가져오기
        
        Args:
            video_path: 영상 파일 경로
            
        Returns:
            영상 정보 딕셔너리
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            info = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'duration': float(video_stream['duration']),
                'codec': video_stream['codec_name']
            }
            
            self.logger.info(f"영상 정보: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"영상 정보 가져오기 실패: {e}")
            return None