# PyStitch360

GoPro 듀얼 카메라로 촬영한 360도 영상을 위한 통합 스티칭 툴

## 개요

PyStitch360는 360도 영상 제작 워크플로우를 완전 자동화하는 GUI 애플리케이션입니다. 기존의 복잡하고 비효율적인 작업 과정을 하나의 프로그램으로 통합하여 전처리부터 최종 출력까지 원스톱으로 처리할 수 있습니다.

**기존 워크플로우 문제점:**
```
ffmpeg (concat) → VideoStitch Studio (MPEG2만 지원) → ffmpeg (x264 인코딩) → Insta360 Studio (wide 포맷 변환)
```
→ 여러 프로그램 사용, 수작업 많음, 비효율적

**PyStitch360 솔루션:**
```
입력 처리 → 스티칭 → 방향 조정 → 후처리 → 출력
```
→ 하나의 프로그램으로 통합, 자동화, 효율적

## 주요 기능

### 🎥 입력 처리
- GoPro 파일 자동 인식 (`GOPR*.MP4`, `GP01*.MP4`)
- FFmpeg 기반 영상 결합
- 수동 동기화 조정 슬라이더

### 🔄 스티칭 엔진
- OpenCV 기반 equirectangular 투영
- 고정 카메라 캘리브레이션
- 설정 가능한 블렌딩 옵션

### 🎛️ 방향 제어
- PyQt6 기반 실시간 미리보기
- 마우스로 Yaw/Pitch/Roll 조정
- 직관적인 GUI 인터페이스

### 💾 출력 처리
- H.264 인코딩 (설정 가능한 CRF/프리셋)
- 360도 메타데이터 자동 삽입
- Insta360 Studio 호환 포맷

### 📁 프로젝트 관리
- JSON 기반 설정 저장
- 재사용 가능한 파라미터 세트

## 기술 스택

- **GUI**: PyQt6
- **영상 처리**: OpenCV, ffmpeg-python
- **스티칭**: OpenCV stitching API
- **인코딩**: FFmpeg (libx264)

## 설치 및 실행

### 사전 요구사항
FFmpeg 바이너리가 시스템에 설치되어 있어야 합니다:

```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

### 애플리케이션 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python main.py
```

## 향후 계획

- Optical Flow 기반 적응형 심 감지
- AI 기반 영상 안정화
- 멀티카메라 (2개 이상) 지원
- 오디오 기반 자동 동기화

## 라이선스

MIT License