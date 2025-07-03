"""
스티칭 모듈
OpenCV 기반 360도 영상 스티칭 엔진
"""

import logging
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class Stitcher:
    """360도 영상 스티칭 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calibration_data = None
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.rotation_matrix = None
        self.translation_vector = None
        
    def load_calibration(self, calibration_path: Path) -> bool:
        """
        카메라 캘리브레이션 데이터 로드
        
        Args:
            calibration_path: 캘리브레이션 파일 경로
            
        Returns:
            성공 여부
        """
        self.logger.info(f"캘리브레이션 데이터 로드: {calibration_path}")
        
        try:
            with open(calibration_path, 'r', encoding='utf-8') as f:
                self.calibration_data = yaml.safe_load(f)
            
            # 캘리브레이션 데이터 파싱
            calib = self.calibration_data['camera_calibration']
            
            self.camera_matrix_left = np.array(calib['camera_matrix_left'], dtype=np.float32)
            self.camera_matrix_right = np.array(calib['camera_matrix_right'], dtype=np.float32)
            self.dist_coeffs_left = np.array(calib['distortion_left'], dtype=np.float32)
            self.dist_coeffs_right = np.array(calib['distortion_right'], dtype=np.float32)
            self.rotation_matrix = np.array(calib['rotation_matrix'], dtype=np.float32)
            self.translation_vector = np.array(calib['translation_vector'], dtype=np.float32)
            
            self.logger.info("캘리브레이션 데이터 로드 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"캘리브레이션 데이터 로드 실패: {e}")
            return False
    
    def undistort_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지 왜곡 보정
        
        Args:
            left_image: 왼쪽 카메라 이미지
            right_image: 오른쪽 카메라 이미지
            
        Returns:
            보정된 이미지들
        """
        if self.camera_matrix_left is None:
            self.logger.warning("캘리브레이션 데이터가 없음. 원본 이미지 반환")
            return left_image, right_image
        
        # 왜곡 보정
        left_undistorted = cv2.undistort(left_image, self.camera_matrix_left, self.dist_coeffs_left)
        right_undistorted = cv2.undistort(right_image, self.camera_matrix_right, self.dist_coeffs_right)
        
        return left_undistorted, right_undistorted
    
    def create_panorama(self, left_image: np.ndarray, right_image: np.ndarray) -> Optional[np.ndarray]:
        """
        두 이미지로 파노라마 생성
        
        Args:
            left_image: 왼쪽 카메라 이미지
            right_image: 오른쪽 카메라 이미지
            
        Returns:
            스티칭된 파노라마 이미지
        """
        self.logger.info("파노라마 이미지 생성")
        
        try:
            # 1. 왜곡 보정
            left_undistorted, right_undistorted = self.undistort_images(left_image, right_image)
            
            # 2. 특징점 검출 및 매칭
            left_warped, right_warped = self.apply_cylindrical_projection(left_undistorted, right_undistorted)
            
            # 3. 이미지 스티칭
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            status, panorama = stitcher.stitch([left_warped, right_warped])
            
            if status == cv2.Stitcher_OK:
                self.logger.info("파노라마 생성 성공")
                return panorama
            else:
                self.logger.error(f"스티칭 실패: status={status}")
                return None
                
        except Exception as e:
            self.logger.error(f"파노라마 생성 실패: {e}")
            return None
    
    def apply_cylindrical_projection(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        원통 투영 적용 (360도 이미지 준비)
        
        Args:
            left_image: 왼쪽 이미지
            right_image: 오른쪽 이미지
            
        Returns:
            투영된 이미지들
        """
        h, w = left_image.shape[:2]
        
        # 초점거리 추정 (카메라 매트릭스에서)
        if self.camera_matrix_left is not None:
            focal_length = self.camera_matrix_left[0, 0]
        else:
            focal_length = w  # 기본값
        
        # 원통 투영 매핑
        def cylindrical_warp(img, focal):
            h, w = img.shape[:2]
            cylinder = np.zeros_like(img)
            
            # 변환 매트릭스 생성
            for y in range(h):
                for x in range(w):
                    # 정규화 좌표
                    x_norm = (x - w/2) / focal
                    y_norm = (y - h/2) / focal
                    
                    # 원통 좌표로 변환
                    x_cyl = int(focal * np.arctan(x_norm) + w/2)
                    y_cyl = int(focal * y_norm / np.sqrt(1 + x_norm**2) + h/2)
                    
                    if 0 <= x_cyl < w and 0 <= y_cyl < h:
                        cylinder[y, x] = img[y_cyl, x_cyl]
            
            return cylinder
        
        left_warped = cylindrical_warp(left_image, focal_length)
        right_warped = cylindrical_warp(right_image, focal_length)
        
        return left_warped, right_warped
    
    def apply_equirectangular_projection(self, panorama: np.ndarray, 
                                       output_width: int = 3840, output_height: int = 1920) -> np.ndarray:
        """
        Equirectangular 투영 적용
        
        Args:
            panorama: 파노라마 이미지
            output_width: 출력 너비
            output_height: 출력 높이
            
        Returns:
            투영된 이미지
        """
        self.logger.info("Equirectangular 투영 적용")
        
        try:
            h, w = panorama.shape[:2]
            
            # Equirectangular 변환 매핑
            map_x = np.zeros((output_height, output_width), dtype=np.float32)
            map_y = np.zeros((output_height, output_width), dtype=np.float32)
            
            for y in range(output_height):
                for x in range(output_width):
                    # 정규화 좌표
                    longitude = (x / output_width) * 2 * np.pi - np.pi  # -π to π
                    latitude = (y / output_height) * np.pi - np.pi / 2  # -π/2 to π/2
                    
                    # 원본 이미지 좌표로 변환
                    src_x = (longitude + np.pi) / (2 * np.pi) * w
                    src_y = (latitude + np.pi / 2) / np.pi * h
                    
                    map_x[y, x] = src_x
                    map_y[y, x] = src_y
            
            # 리매핑 수행
            equirectangular = cv2.remap(panorama, map_x, map_y, cv2.INTER_LINEAR)
            
            self.logger.info("Equirectangular 투영 완료")
            return equirectangular
            
        except Exception as e:
            self.logger.error(f"Equirectangular 투영 실패: {e}")
            return panorama
    
    def blend_images(self, image1: np.ndarray, image2: np.ndarray, 
                    feather_width: int = 50) -> np.ndarray:
        """
        이미지 블렌딩 (Feather blending)
        
        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            feather_width: 페더링 폭
            
        Returns:
            블렌딩된 이미지
        """
        self.logger.info(f"이미지 블렌딩: feather_width={feather_width}")
        
        try:
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            
            # 두 이미지의 크기를 맞춤
            h = max(h1, h2)
            w = w1 + w2
            
            # 결과 이미지 초기화
            result = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 첫 번째 이미지 배치
            result[:h1, :w1] = image1
            
            # 겹치는 영역 계산
            overlap_start = w1 - feather_width
            overlap_end = w1
            
            # 두 번째 이미지의 시작 위치
            offset_x = overlap_start
            
            # 겹치지 않는 부분 배치
            result[:h2, overlap_end:overlap_end+w2-feather_width] = image2[:, feather_width:]
            
            # 페더링 블렌딩
            for x in range(feather_width):
                alpha = x / feather_width  # 0에서 1로 변화
                
                if overlap_start + x < w and x < w2:
                    # 가중 평균으로 블렌딩
                    result[:min(h1, h2), overlap_start + x] = (
                        (1 - alpha) * image1[:min(h1, h2), overlap_start + x] +
                        alpha * image2[:min(h1, h2), x]
                    ).astype(np.uint8)
            
            self.logger.info("이미지 블렌딩 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"이미지 블렌딩 실패: {e}")
            return image1
    
    def apply_orientation(self, image: np.ndarray, yaw: float = 0, pitch: float = 0, roll: float = 0) -> np.ndarray:
        """
        방향 조정 적용
        
        Args:
            image: 입력 이미지
            yaw: 좌우 회전 (도)
            pitch: 상하 회전 (도)
            roll: 기울기 회전 (도)
            
        Returns:
            방향 조정된 이미지
        """
        self.logger.info(f"방향 조정: yaw={yaw}, pitch={pitch}, roll={roll}")
        
        try:
            h, w = image.shape[:2]
            
            # 회전 매트릭스 생성
            center = (w // 2, h // 2)
            
            # Roll (Z축 회전)만 우선 구현
            rotation_matrix = cv2.getRotationMatrix2D(center, roll, 1.0)
            
            # 회전 적용
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            
            # Yaw, Pitch는 equirectangular 좌표계에서 처리
            # (추후 더 정교한 구현 필요)
            
            return rotated
            
        except Exception as e:
            self.logger.error(f"방향 조정 실패: {e}")
            return image