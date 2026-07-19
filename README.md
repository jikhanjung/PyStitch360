# PyStitch360

듀얼 GoPro로 촬영한 축구 경기를 초광폭 원통(cylindrical) 파노라마 영상으로
스티칭하는 GUI 툴. 경기장 전체가 한 화면에 들어오는 ~5,900px 폭 파노라마와,
공을 자동 추적하는 가상 PTZ 1080p 출력을 지원한다.

```
[좌 GoPro 챕터들]──┐   오디오 자동 동기화     원통 워핑(remap 캐싱)
                   ├─→ 렌즈 프로파일 보정 ─→ 자동 정합(SIFT+RANSAC) ─→ 하프라인 심
[우 GoPro 챕터들]──┘   자동 수평/센터링       게인 보정 + 심 Y 정렬     ↓
                                              ┌──────────────────────────┴───┐
                                              │ 파노라마 전체 (~5900×1680)    │
                                              │ 가상 PTZ 1080p (YOLO 공 추적) │
                                              └──────────────────────────────┘
```

## 특징

- **자동 정합**: 겹침 영역 SIFT 매칭 → 광선 기반 RANSAC 회전 추정 (잔차 ~0.13°)
- **자동 수평/센터링**: 먼 쪽 터치라인으로 pitch/roll, 하프라인으로 yaw 자동 보정
  (+ 사용자 미세조정 슬라이더)
- **하프라인 수직 심**: 심 좌측은 좌 카메라, 우측은 우 카메라만 사용.
  심 세로 어긋남은 템플릿 매칭 실측으로 국소 보정 (rms 3.7→0.6px)
- **GoPro 챕터 통합**: GOPR→GP01→GP02... 분할 파일을 하나의 타임라인으로
- **오디오 자동 동기화**: 상호상관으로 좌/우 시작 시차 추정
- **세그먼트**: 촬영 중 카메라가 틀어지면(바람/공) 그 지점부터 재정합.
  GPMF 자이로에서 충격 이벤트 후보를 자동 탐지
- **내보내기**: H.264/HEVC (NVENC 자동 감지), 오디오 포함, 진행률/취소
- **가상 PTZ**: YOLOv8 공/선수 감지 + 스무딩으로 1080p 크롭이 경기를 따라감

## 설치

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics            # 선택: 가상 PTZ 기능

# ffmpeg 바이너리 필요 (PATH)
#   Windows: winget install Gyan.FFmpeg / macOS: brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
```

## 사용

```bash
python main.py
```

1. **영상·동기화 탭**: 좌/우 영상의 첫 챕터(GOPR*.MP4) 선택 → 챕터 자동 연결.
   "오디오 자동 동기화" 클릭 (또는 오프셋 수동 입력).
2. **정합·미리보기 탭**: 경기 중 조용한 프레임에서 "자동 정합" 클릭.
   미리보기를 보며 pitch/roll/yaw 미세조정. 카메라가 중간에 틀어진 세션이면
   "자이로에서 충격 이벤트 탐지" → 후보 시점으로 이동해 재정합(새 세그먼트).
3. **내보내기 탭**: 구간·코덱·출력 형식(파노라마/가상 PTZ) 선택 후 시작.
4. **가상 PTZ 탭**: 분석 → 검수 → 이벤트/하이라이트 → 클립·리포트.
   전체 절차는 [docs/ptz_workflow.md](docs/ptz_workflow.md) 참고.

프로젝트(파일 목록, 동기화, 정합, 보정값)는 파일 메뉴에서 JSON 으로 저장/복원.

## 촬영 가이드 (GoPro HERO5 Black 기준)

- 4K 16:9 30fps **Wide** 모드, **EIS(영상 안정화) 끔** — 필수
- 두 카메라를 한 바에 **최대한 밀착** (광축 간격이 시차 이중상을 결정)
- 하프라인 부근, 터치라인에서 3~5m, 가능한 한 높게
- 두 카메라의 노출/화이트밸런스 설정 통일 권장

## 렌즈 프로파일

`presets/lens_profiles/` 의 Gyroflow 형식(OpenCV fisheye) JSON.
기본 내장: HERO5 Black 4K 16:9 Wide. 다른 모델/모드는
[Gyroflow lens profile DB](https://github.com/gyroflow/lens_profiles) 에서
받아 추가하면 된다.

## 저장소 구조

```
pystitch/core/    스티칭 코어 (렌즈, 기하, 정합, 렌더, 챕터, 동기화, GPMF, PTZ)
pystitch/gui/     PyQt6 GUI (메인 윈도우, 워커 스레드)
presets/          렌즈 프로파일, YOLO 가중치
prototype/        검증용 스탠드얼론 스크립트 (개발 기록)
devlog/           작업 기록 (YYYYMMDD_NN_title.md)
legacy/           구 360° equirect 구현 (참고용, 미사용)
```

## 라이선스

MIT License
