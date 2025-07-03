# PyStitch360: 통합 스티칭 툴 요구사항 정리

## 📌 목적
GoPro 두 대로 촬영한 360도 영상을 **전처리 → 스티칭 → 후처리 → 출력**까지 하나의 프로그램에서 자동화 처리할 수 있는 통합 GUI 툴 개발.

기존 워크플로우:  
`ffmpeg (concat)` → `VideoStitch Studio (MPEG2만 지원)` → `ffmpeg (x264 인코딩)` → `Insta360 Studio (wide 포맷 변환)`  
→ **비효율적이며 수작업이 많아 통합 툴 필요**

---

## 🧭 전체 워크플로우

### 1. 📥 입력 처리 (전처리)
- GoPro 파일 자동 인식 (`GOPR*.MP4`, `GP01*.MP4`)
- `ffmpeg` concat (demuxer 방식)
- 수동 동기화 조절 슬라이더 (프레임 오프셋 조정)
- 향후 오디오 기반 자동 동기화 기능 확장

### 2. 🔄 스티칭 (중앙 모듈)
- 고정된 camera calibration 값 이용 (json/yaml)
- equirectangular 투영 기반 warping
- 블렌딩 옵션:
  - Linear blending
  - Feather width 조정
- Optical Flow 기반 보정은 차후 옵션

### 3. 🎛️ Orientation 조정
- PyQt 기반 GUI에서 실시간 미리보기
- 마우스로 Yaw / Pitch / Roll 조절
- orientation 설정값이 warping 및 출력에 반영됨

### 4. 🧪 후처리 및 인코딩
- ffmpeg 기반 H.264 인코딩:
  - `libx264`, `crf`, `preset` 선택 가능
- 메타데이터 삽입 (projection=equirectangular)
- Insta360 Studio 호환 포맷 설정 옵션 포함

### 5. 💾 프로젝트 설정 저장
- `.json` 형태로 전체 파이프라인 설정 저장
- 영상만 바꿔도 동일 파라미터로 반복 사용 가능

---

## 🧱 기술 스택 제안

| 구성요소     | 기술 스택                       |
|--------------|----------------------------------|
| GUI          | PyQt5 / PySide6                  |
| 영상 입력/출력 | ffmpeg-python, OpenCV            |
| 미리보기     | OpenCV (`cv2.imshow`) or OpenGL |
| Warping      | OpenCV stitching (detail API)    |
| Blending     | OpenCV Feather / Multiband       |
| Metadata     | ffmpeg or exiftool               |

---

## 🔄 향후 확장 가능 기능
- Optical Flow 기반 adaptive seam
- AI 기반 stabilization (e.g. DeepStab)
- Insta360 형식 자동 preset
- 멀티카메라 (>2) 지원

---

## 📁 예시 파일 구조
```
PyStitch360/
├── main.py
├── gui/
│   └── stitcher_window.py
├── core/
│   ├── preprocessor.py
│   ├── stitcher.py
│   └── postprocessor.py
├── presets/
│   └── gopro_dual.yaml
├── projects/
│   └── sample_project.json
└── README.md
```

---

## 📌 요약
PyStitch360는 영상 전문가 또는 360도 콘텐츠 제작자에게 필요한 end-to-end 스티칭 도구로서, 현재의 불편한 작업 흐름을 완전 자동화함으로써 **효율성, 품질, 재사용성**을 동시에 개선한다.

