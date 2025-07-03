"""
전처리 모듈 테스트
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from core.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    """Preprocessor 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.preprocessor = Preprocessor()
        self.test_dir = Path(tempfile.mkdtemp())
        
        # 테스트용 더미 파일 생성
        self.create_test_files()
    
    def tearDown(self):
        """테스트 정리"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """테스트용 GoPro 파일 생성"""
        # 전면 카메라 파일 (짝수)
        (self.test_dir / "GOPR0001.MP4").touch()
        (self.test_dir / "GP010001.MP4").touch()
        (self.test_dir / "GP020001.MP4").touch()
        
        # 후면 카메라 파일 (홀수)
        (self.test_dir / "GOPR0002.MP4").touch()
        (self.test_dir / "GP010002.MP4").touch()
        (self.test_dir / "GP020002.MP4").touch()
        
        # 관련 없는 파일
        (self.test_dir / "random_file.mp4").touch()
        (self.test_dir / "document.txt").touch()
    
    def test_detect_gopro_files(self):
        """GoPro 파일 감지 테스트"""
        front_files, back_files = self.preprocessor.detect_gopro_files(self.test_dir)
        
        # 전면 카메라 파일 확인
        self.assertEqual(len(front_files), 3)
        front_names = [f.name for f in front_files]
        self.assertIn("GOPR0001.MP4", front_names)
        self.assertIn("GP010001.MP4", front_names)
        self.assertIn("GP020001.MP4", front_names)
        
        # 후면 카메라 파일 확인
        self.assertEqual(len(back_files), 3)
        back_names = [f.name for f in back_files]
        self.assertIn("GOPR0002.MP4", back_names)
        self.assertIn("GP010002.MP4", back_names)
        self.assertIn("GP020002.MP4", back_names)
    
    def test_detect_gopro_files_empty_directory(self):
        """빈 디렉터리 테스트"""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        front_files, back_files = self.preprocessor.detect_gopro_files(empty_dir)
        
        self.assertEqual(len(front_files), 0)
        self.assertEqual(len(back_files), 0)
    
    def test_detect_gopro_files_nonexistent_directory(self):
        """존재하지 않는 디렉터리 테스트"""
        nonexistent_dir = self.test_dir / "nonexistent"
        
        front_files, back_files = self.preprocessor.detect_gopro_files(nonexistent_dir)
        
        self.assertEqual(len(front_files), 0)
        self.assertEqual(len(back_files), 0)
    
    def test_get_video_info(self):
        """영상 정보 가져오기 테스트 (실제 영상 파일 없이)"""
        # 실제 영상 파일이 없으므로 None 반환 예상
        fake_video = self.test_dir / "GOPR0001.MP4"
        info = self.preprocessor.get_video_info(fake_video)
        
        self.assertIsNone(info)


if __name__ == '__main__':
    unittest.main()