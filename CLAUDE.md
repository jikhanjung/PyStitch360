# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

PyStitch360는 전처리부터 최종 출력까지 전체 워크플로우를 자동화하는 포괄적인 360도 영상 스티칭 툴입니다. 듀얼 GoPro 영상을 자동 스티칭, 방향 조정, 인코딩을 통해 처리합니다.

**핵심 파이프라인:** 입력 처리 → 스티칭 → 방향 조정 → 후처리 → 출력

## 아키텍처

### 제안된 모듈 구조
- `main.py`: 애플리케이션 진입점
- `gui/`: PyQt6 기반 GUI 컴포넌트
  - `stitcher_window.py`: 메인 애플리케이션 윈도우
- `core/`: 핵심 처리 모듈
  - `preprocessor.py`: 입력 처리 및 ffmpeg concat 작업
  - `stitcher.py`: equirectangular 투영 기반 OpenCV 스티칭 엔진
  - `postprocessor.py`: 인코딩 및 메타데이터 삽입
- `presets/`: 카메라 캘리브레이션 데이터 (JSON/YAML)
- `projects/`: 프로젝트 설정 파일

### 주요 기술 스택
- **GUI 프레임워크**: PyQt6
- **영상 처리**: ffmpeg-python, OpenCV
- **스티칭**: equirectangular 투영 기반 OpenCV 스티칭 API
- **인코딩**: H.264 (libx264) 기반 ffmpeg
- **설정**: JSON 프로젝트 파일

## 개발 명령어

새 프로젝트이므로 표준 Python 개발 명령어를 사용합니다:

```bash
# 의존성 설치 (requirements.txt 존재 시)
pip install -r requirements.txt

# 애플리케이션 실행
python main.py

# 테스트 실행 (테스트 프레임워크 설정 시)
python -m pytest

# 코드 포맷팅 (black 사용 시)
black .

# 코드 린팅 (flake8 사용 시)
flake8 .
```

## 구현해야 할 주요 기능

### 입력 처리
- GoPro 파일 자동 인식 (`GOPR*.MP4`, `GP01*.MP4`)
- demuxer 방식 ffmpeg concat
- 프레임 오프셋을 통한 수동 동기화 조정 슬라이더

### 스티칭 엔진
- 고정된 카메라 캘리브레이션 파라미터
- Equirectangular 투영 워핑
- 설정 가능한 블렌딩 옵션 (Linear blending, Feather width)

### 방향 제어
- PyQt6 기반 실시간 미리보기
- 마우스 기반 Yaw/Pitch/Roll 조정
- 워핑 파이프라인과 설정 통합

### 출력 처리
- 설정 가능한 CRF 및 프리셋으로 H.264 인코딩
- Equirectangular 투영 메타데이터 삽입
- Insta360 Studio 포맷 호환성

### 프로젝트 관리
- JSON 기반 설정 지속성
- 다양한 영상에서 재사용 가능한 파라미터 세트

## 중요한 구현 참고사항

- 워핑 및 블렌딩을 위해 OpenCV의 상세 스티칭 API 사용
- OpenCV (`cv2.imshow`) 또는 OpenGL을 사용한 실시간 미리보기 구현
- 카메라 캘리브레이션 데이터를 JSON/YAML 형식으로 저장
- 360도 영상 플레이어와의 메타데이터 호환성 보장
- 향후 확장을 위한 모듈형 아키텍처 설계 (optical flow, 멀티카메라 지원)

## 향후 확장 기능
- Optical Flow 기반 적응형 심 감지
- AI 기반 안정화
- 멀티카메라 (2개 이상) 지원
- 오디오 기반 자동 동기화