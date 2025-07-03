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
        # 좌측 카메라 파일 (짝수)
        (self.test_dir / "GOPR0001.MP4").touch()
        (self.test_dir / "GP010001.MP4").touch()
        (self.test_dir / "GP020001.MP4").touch()
        
        # 우측 카메라 파일 (홀수)
        (self.test_dir / "GOPR0002.MP4").touch()
        (self.test_dir / "GP010002.MP4").touch()
        (self.test_dir / "GP020002.MP4").touch()
        
        # 관련 없는 파일
        (self.test_dir / "random_file.mp4").touch()
        (self.test_dir / "document.txt").touch()
    
    def test_detect_gopro_files(self):
        """GoPro 파일 감지 테스트"""
        left_files, right_files = self.preprocessor.detect_gopro_files(self.test_dir)
        
        # 좌측 카메라 파일 확인
        self.assertEqual(len(left_files), 3)
        left_names = [f.name for f in left_files]
        self.assertIn("GOPR0001.MP4", left_names)
        self.assertIn("GP010001.MP4", left_names)
        self.assertIn("GP020001.MP4", left_names)
        
        # 우측 카메라 파일 확인
        self.assertEqual(len(right_files), 3)
        right_names = [f.name for f in right_files]
        self.assertIn("GOPR0002.MP4", right_names)
        self.assertIn("GP010002.MP4", right_names)
        self.assertIn("GP020002.MP4", right_names)
    
    def test_detect_gopro_files_empty_directory(self):
        """빈 디렉터리 테스트"""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        left_files, right_files = self.preprocessor.detect_gopro_files(empty_dir)
        
        self.assertEqual(len(left_files), 0)
        self.assertEqual(len(right_files), 0)
    
    def test_detect_gopro_files_nonexistent_directory(self):
        """존재하지 않는 디렉터리 테스트"""
        nonexistent_dir = self.test_dir / "nonexistent"
        
        left_files, right_files = self.preprocessor.detect_gopro_files(nonexistent_dir)
        
        self.assertEqual(len(left_files), 0)
        self.assertEqual(len(right_files), 0)
    
    def test_get_video_info(self):
        """영상 정보 가져오기 테스트 (실제 영상 파일 없이)"""
        # 실제 영상 파일이 없으므로 None 반환 예상
        fake_video = self.test_dir / "GOPR0001.MP4"
        info = self.preprocessor.get_video_info(fake_video)
        
        self.assertIsNone(info)


if __name__ == '__main__':
    unittest.main()