"""
프로젝트 관리자 모듈 테스트
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from core.project_manager import ProjectManager


class TestProjectManager(unittest.TestCase):
    """ProjectManager 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.project_manager = ProjectManager()
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_project_file = self.test_dir / "test_project.pys360"
    
    def tearDown(self):
        """테스트 정리"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_new_project(self):
        """새 프로젝트 생성 테스트"""
        project_data = self.project_manager.create_new_project("테스트 프로젝트")
        
        # 필수 섹션 확인
        required_sections = [
            "project_info", "input_files", "preprocessing",
            "stitching", "orientation", "postprocessing", "output"
        ]
        
        for section in required_sections:
            self.assertIn(section, project_data)
        
        # 프로젝트 정보 확인
        self.assertEqual(project_data["project_info"]["name"], "테스트 프로젝트")
        self.assertEqual(project_data["project_info"]["version"], "1.0.0")
    
    def test_save_and_load_project(self):
        """프로젝트 저장 및 로드 테스트"""
        # 새 프로젝트 생성
        original_data = self.project_manager.create_new_project("테스트 프로젝트")
        
        # 일부 설정 변경
        original_data["stitching"]["feather_width"] = 100
        original_data["orientation"]["yaw"] = 45.0
        
        # 저장
        success = self.project_manager.save_project(self.test_project_file, original_data)
        self.assertTrue(success)
        self.assertTrue(self.test_project_file.exists())
        
        # 새 인스턴스로 로드
        new_manager = ProjectManager()
        loaded_data = new_manager.load_project(self.test_project_file)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["project_info"]["name"], "테스트 프로젝트")
        self.assertEqual(loaded_data["stitching"]["feather_width"], 100)
        self.assertEqual(loaded_data["orientation"]["yaw"], 45.0)
    
    def test_save_as_template(self):
        """템플릿 저장 테스트"""
        # 프로젝트 생성 및 입력 파일 설정
        project_data = self.project_manager.create_new_project("원본 프로젝트")
        project_data["input_files"]["front_camera"] = ["/path/to/front.mp4"]
        project_data["input_files"]["back_camera"] = ["/path/to/back.mp4"]
        project_data["output"]["path"] = "/path/to/output.mp4"
        
        self.project_manager.project_data = project_data
        
        # 템플릿으로 저장
        template_file = self.test_dir / "template.pys360"
        success = self.project_manager.save_as_template(template_file, "테스트 템플릿")
        
        self.assertTrue(success)
        self.assertTrue(template_file.exists())
        
        # 템플릿 로드 및 확인
        with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        # 템플릿에는 입력 파일과 출력 경로가 비어있어야 함
        self.assertEqual(template_data["input_files"]["front_camera"], [])
        self.assertEqual(template_data["input_files"]["back_camera"], [])
        self.assertEqual(template_data["output"]["path"], "")
        self.assertTrue(template_data["project_info"]["template"])
        self.assertEqual(template_data["project_info"]["name"], "테스트 템플릿")
    
    def test_validate_project_data(self):
        """프로젝트 데이터 검증 테스트"""
        # 유효한 데이터
        valid_data = self.project_manager.create_new_project()
        self.assertTrue(self.project_manager._validate_project_data(valid_data))
        
        # 필수 섹션 누락된 데이터
        invalid_data = {"project_info": {}}
        self.assertFalse(self.project_manager._validate_project_data(invalid_data))
    
    def test_relative_path_conversion(self):
        """상대 경로 변환 테스트"""
        base_path = Path("/home/user/projects")
        
        # 테스트 데이터
        data = {
            "input_files": {
                "front_camera": ["/home/user/projects/videos/front.mp4"],
                "back_camera": ["/home/user/projects/videos/back.mp4"]
            },
            "output": {
                "path": "/home/user/projects/output/result.mp4"
            }
        }
        
        # 상대 경로로 변환
        relative_data = self.project_manager._convert_to_relative_paths(data, base_path)
        
        self.assertEqual(relative_data["input_files"]["front_camera"][0], "videos/front.mp4")
        self.assertEqual(relative_data["input_files"]["back_camera"][0], "videos/back.mp4")
        self.assertEqual(relative_data["output"]["path"], "output/result.mp4")
        
        # 다시 절대 경로로 변환
        absolute_data = self.project_manager._convert_to_absolute_paths(relative_data, base_path)
        
        self.assertTrue(absolute_data["input_files"]["front_camera"][0].endswith("videos/front.mp4"))
        self.assertTrue(absolute_data["input_files"]["back_camera"][0].endswith("videos/back.mp4"))
        self.assertTrue(absolute_data["output"]["path"].endswith("output/result.mp4"))
    
    def test_load_nonexistent_project(self):
        """존재하지 않는 프로젝트 로드 테스트"""
        nonexistent_file = self.test_dir / "nonexistent.pys360"
        result = self.project_manager.load_project(nonexistent_file)
        
        self.assertIsNone(result)
    
    def test_get_current_project_name(self):
        """현재 프로젝트 이름 가져오기 테스트"""
        # 기본값
        self.assertEqual(self.project_manager.get_current_project_name(), "새 프로젝트")
        
        # 프로젝트 생성 후
        self.project_manager.create_new_project("테스트 프로젝트")
        self.assertEqual(self.project_manager.get_current_project_name(), "테스트 프로젝트")


if __name__ == '__main__':
    unittest.main()