"""
프로젝트 관리 모듈
프로젝트 설정 저장/불러오기
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ProjectManager:
    """프로젝트 설정 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_project_path = None
        self.project_data = {}
    
    def create_new_project(self, project_name: str = "새 프로젝트") -> Dict[str, Any]:
        """새 프로젝트 생성"""
        self.project_data = {
            "project_info": {
                "name": project_name,
                "created_date": datetime.now().isoformat(),
                "modified_date": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "input_files": {
                "front_camera": [],
                "back_camera": []
            },
            "preprocessing": {
                "sync_offset_frames": 0,
                "concat_method": "demuxer"
            },
            "stitching": {
                "calibration_preset": "gopro_dual.yaml",
                "blend_type": "Linear",
                "feather_width": 50,
                "projection_type": "Equirectangular",
                "output_resolution": "3840x1920"
            },
            "orientation": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            },
            "postprocessing": {
                "encoding": {
                    "codec": "H.264 (libx264)",
                    "crf": 23,
                    "preset": "medium"
                },
                "metadata": {
                    "enabled": True,
                    "projection": "equirectangular",
                    "insta360_compatible": False
                }
            },
            "output": {
                "path": "",
                "format": "mp4"
            }
        }
        
        self.current_project_path = None
        self.logger.info(f"새 프로젝트 생성: {project_name}")
        return self.project_data
    
    def save_project(self, file_path: Path, project_data: Dict[str, Any]) -> bool:
        """프로젝트 저장"""
        try:
            # 수정 시간 업데이트
            project_data["project_info"]["modified_date"] = datetime.now().isoformat()
            
            # 상대 경로로 변환 (파일 경로만)
            project_data_copy = self._convert_to_relative_paths(project_data, file_path.parent)
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data_copy, f, ensure_ascii=False, indent=2)
            
            self.current_project_path = file_path
            self.project_data = project_data
            self.logger.info(f"프로젝트 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 저장 실패: {e}")
            return False
    
    def load_project(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """프로젝트 불러오기"""
        try:
            if not file_path.exists():
                self.logger.error(f"프로젝트 파일이 존재하지 않음: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            
            # 절대 경로로 변환
            project_data = self._convert_to_absolute_paths(project_data, file_path.parent)
            
            # 프로젝트 데이터 검증
            if not self._validate_project_data(project_data):
                self.logger.error("프로젝트 데이터 검증 실패")
                return None
            
            self.current_project_path = file_path
            self.project_data = project_data
            self.logger.info(f"프로젝트 불러오기 완료: {file_path}")
            return project_data
            
        except Exception as e:
            self.logger.error(f"프로젝트 불러오기 실패: {e}")
            return None
    
    def save_as_template(self, file_path: Path, template_name: str) -> bool:
        """템플릿으로 저장 (입력 파일 제외)"""
        try:
            template_data = self.project_data.copy()
            template_data["project_info"]["name"] = template_name
            template_data["project_info"]["template"] = True
            template_data["input_files"] = {"front_camera": [], "back_camera": []}
            template_data["output"]["path"] = ""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"템플릿 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"템플릿 저장 실패: {e}")
            return False
    
    def get_recent_projects(self, max_count: int = 5) -> list:
        """최근 프로젝트 목록 가져오기"""
        settings_file = Path.home() / ".pystitch360" / "recent_projects.json"
        
        if not settings_file.exists():
            return []
        
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                recent_projects = json.load(f)
            
            # 존재하는 파일만 필터링
            valid_projects = []
            for project in recent_projects[:max_count]:
                if Path(project["path"]).exists():
                    valid_projects.append(project)
            
            return valid_projects
            
        except Exception as e:
            self.logger.error(f"최근 프로젝트 목록 로드 실패: {e}")
            return []
    
    def add_to_recent_projects(self, project_path: Path, project_name: str):
        """최근 프로젝트에 추가"""
        settings_dir = Path.home() / ".pystitch360"
        settings_dir.mkdir(exist_ok=True)
        settings_file = settings_dir / "recent_projects.json"
        
        try:
            # 기존 목록 로드
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    recent_projects = json.load(f)
            else:
                recent_projects = []
            
            # 중복 제거
            recent_projects = [p for p in recent_projects if p["path"] != str(project_path)]
            
            # 새 프로젝트 추가 (맨 앞에)
            new_project = {
                "name": project_name,
                "path": str(project_path),
                "last_opened": datetime.now().isoformat()
            }
            recent_projects.insert(0, new_project)
            
            # 최대 10개까지만 유지
            recent_projects = recent_projects[:10]
            
            # 저장
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(recent_projects, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"최근 프로젝트 추가 실패: {e}")
    
    def _convert_to_relative_paths(self, data: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """파일 경로를 상대 경로로 변환"""
        data_copy = json.loads(json.dumps(data))  # 깊은 복사
        
        # 입력 파일 경로 변환
        if "input_files" in data_copy:
            for camera_type in ["front_camera", "back_camera"]:
                if camera_type in data_copy["input_files"]:
                    relative_paths = []
                    for file_path in data_copy["input_files"][camera_type]:
                        try:
                            abs_path = Path(file_path)
                            rel_path = abs_path.relative_to(base_path)
                            relative_paths.append(str(rel_path))
                        except ValueError:
                            # 상대 경로로 변환할 수 없는 경우 절대 경로 유지
                            relative_paths.append(str(file_path))
                    data_copy["input_files"][camera_type] = relative_paths
        
        # 출력 경로 변환
        if "output" in data_copy and "path" in data_copy["output"]:
            if data_copy["output"]["path"]:
                try:
                    abs_path = Path(data_copy["output"]["path"])
                    rel_path = abs_path.relative_to(base_path)
                    data_copy["output"]["path"] = str(rel_path)
                except ValueError:
                    pass  # 변환할 수 없으면 그대로 유지
        
        return data_copy
    
    def _convert_to_absolute_paths(self, data: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """상대 경로를 절대 경로로 변환"""
        # 입력 파일 경로 변환
        if "input_files" in data:
            for camera_type in ["front_camera", "back_camera"]:
                if camera_type in data["input_files"]:
                    absolute_paths = []
                    for file_path in data["input_files"][camera_type]:
                        path = Path(file_path)
                        if not path.is_absolute():
                            abs_path = base_path / path
                            absolute_paths.append(str(abs_path.resolve()))
                        else:
                            absolute_paths.append(str(file_path))
                    data["input_files"][camera_type] = absolute_paths
        
        # 출력 경로 변환
        if "output" in data and "path" in data["output"]:
            if data["output"]["path"]:
                path = Path(data["output"]["path"])
                if not path.is_absolute():
                    abs_path = base_path / path
                    data["output"]["path"] = str(abs_path.resolve())
        
        return data
    
    def _validate_project_data(self, data: Dict[str, Any]) -> bool:
        """프로젝트 데이터 검증"""
        required_sections = [
            "project_info", "input_files", "preprocessing",
            "stitching", "orientation", "postprocessing", "output"
        ]
        
        for section in required_sections:
            if section not in data:
                self.logger.error(f"필수 섹션 누락: {section}")
                return False
        
        return True
    
    def get_current_project_name(self) -> str:
        """현재 프로젝트 이름 반환"""
        if self.project_data and "project_info" in self.project_data:
            return self.project_data["project_info"].get("name", "새 프로젝트")
        return "새 프로젝트"
    
    def is_project_modified(self, current_settings: Dict[str, Any]) -> bool:
        """프로젝트가 수정되었는지 확인"""
        if not self.project_data:
            return True
        
        # 주요 설정 비교 (파일 경로 제외)
        current_core = {
            "preprocessing": current_settings.get("preprocessing", {}),
            "stitching": current_settings.get("stitching", {}),
            "orientation": current_settings.get("orientation", {}),
            "postprocessing": current_settings.get("postprocessing", {})
        }
        
        saved_core = {
            "preprocessing": self.project_data.get("preprocessing", {}),
            "stitching": self.project_data.get("stitching", {}),
            "orientation": self.project_data.get("orientation", {}),
            "postprocessing": self.project_data.get("postprocessing", {})
        }
        
        return current_core != saved_core