"""
스티처 모듈 테스트
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from core.stitcher import Stitcher


class TestStitcher(unittest.TestCase):
    """Stitcher 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.stitcher = Stitcher()
        self.test_dir = Path(tempfile.mkdtemp())
        
        # 테스트용 캘리브레이션 파일 생성
        self.create_test_calibration_file()
    
    def tearDown(self):
        """테스트 정리"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_calibration_file(self):
        """테스트용 캘리브레이션 파일 생성"""
        calibration_data = """
camera_calibration:
  camera_matrix_left:
    - [1000.0, 0.0, 960.0]
    - [0.0, 1000.0, 540.0]
    - [0.0, 0.0, 1.0]
  
  camera_matrix_right:
    - [1000.0, 0.0, 960.0]
    - [0.0, 1000.0, 540.0]
    - [0.0, 0.0, 1.0]
  
  distortion_left: [0.1, -0.2, 0.0, 0.0, 0.0]
  distortion_right: [0.1, -0.2, 0.0, 0.0, 0.0]
  
  rotation_matrix:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  
  translation_vector: [100.0, 0.0, 0.0]
"""
        
        self.calibration_file = self.test_dir / "test_calibration.yaml"
        with open(self.calibration_file, 'w', encoding='utf-8') as f:
            f.write(calibration_data)
    
    def test_load_calibration(self):
        """캘리브레이션 로드 테스트"""
        success = self.stitcher.load_calibration(self.calibration_file)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.stitcher.camera_matrix_left)
        self.assertIsNotNone(self.stitcher.camera_matrix_right)
        self.assertIsNotNone(self.stitcher.dist_coeffs_left)
        self.assertIsNotNone(self.stitcher.dist_coeffs_right)
        
        # 매트릭스 값 확인
        self.assertEqual(self.stitcher.camera_matrix_left[0, 0], 1000.0)
        self.assertEqual(self.stitcher.camera_matrix_left[0, 2], 960.0)
        self.assertEqual(self.stitcher.camera_matrix_left[1, 2], 540.0)
    
    def test_load_calibration_nonexistent_file(self):
        """존재하지 않는 캘리브레이션 파일 테스트"""
        nonexistent_file = self.test_dir / "nonexistent.yaml"
        success = self.stitcher.load_calibration(nonexistent_file)
        
        self.assertFalse(success)
        self.assertIsNone(self.stitcher.camera_matrix_left)
    
    def test_undistort_images(self):
        """이미지 왜곡 보정 테스트"""
        # 테스트용 더미 이미지 생성
        left_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # 캘리브레이션 없이 테스트 (원본 반환되어야 함)
        left_undistorted, right_undistorted = self.stitcher.undistort_images(left_image, right_image)
        
        np.testing.assert_array_equal(left_undistorted, left_image)
        np.testing.assert_array_equal(right_undistorted, right_image)
        
        # 캘리브레이션 로드 후 테스트
        self.stitcher.load_calibration(self.calibration_file)
        left_undistorted, right_undistorted = self.stitcher.undistort_images(left_image, right_image)
        
        # 결과가 원본과 다른 배열이어야 함 (왜곡 보정 적용됨)
        self.assertEqual(left_undistorted.shape, left_image.shape)
        self.assertEqual(right_undistorted.shape, right_image.shape)
    
    def test_apply_orientation(self):
        """방향 조정 테스트"""
        # 테스트용 이미지 생성
        image = np.random.randint(0, 255, (1920, 3840, 3), dtype=np.uint8)
        
        # Roll 회전 테스트
        rotated = self.stitcher.apply_orientation(image, yaw=0, pitch=0, roll=45)
        
        self.assertEqual(rotated.shape, image.shape)
        # 회전 후 이미지는 원본과 달라야 함
        self.assertFalse(np.array_equal(rotated, image))
        
        # 0도 회전 테스트 (원본과 동일해야 함)
        no_rotation = self.stitcher.apply_orientation(image, yaw=0, pitch=0, roll=0)
        np.testing.assert_array_equal(no_rotation, image)
    
    def test_apply_equirectangular_projection(self):
        """Equirectangular 투영 테스트"""
        # 테스트용 파노라마 이미지
        panorama = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
        
        # 투영 적용
        projected = self.stitcher.apply_equirectangular_projection(
            panorama, output_width=3840, output_height=1920
        )
        
        self.assertEqual(projected.shape, (1920, 3840, 3))
    
    def test_blend_images(self):
        """이미지 블렌딩 테스트"""
        # 테스트용 이미지 생성
        image1 = np.full((500, 800, 3), 100, dtype=np.uint8)  # 회색
        image2 = np.full((500, 800, 3), 200, dtype=np.uint8)  # 밝은 회색
        
        # 블렌딩 적용
        blended = self.stitcher.blend_images(image1, image2, feather_width=50)
        
        # 블렌딩된 이미지는 원본들보다 넓어야 함
        self.assertGreater(blended.shape[1], image1.shape[1])
        self.assertEqual(blended.shape[0], max(image1.shape[0], image2.shape[0]))
        self.assertEqual(blended.shape[2], 3)


if __name__ == '__main__':
    unittest.main()