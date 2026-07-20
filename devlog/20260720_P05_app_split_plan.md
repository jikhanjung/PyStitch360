# P05: 실행파일 분리 — 스티칭 앱 / 경기 분석 앱

- 날짜: 2026-07-20
- 종류: 계획
- 관련: [P03](../devlog/20260719_P03_highlights_roadmap_plan.md) (분석 기능 확장으로
  두 성격의 독립성이 뚜렷해짐)

## 결론 (ideation 합의)

**저장소·core 는 하나, 실행 진입점은 둘.** 완전 분리(별도 프로젝트)는
공유 core(encoders·audio·field·chapters, 파인튜닝 A/B 는 정합 기하 +
분석 양쪽 필요)가 드리프트하므로 하지 않는다.

## 근거

- 분석 쪽 상태는 이미 전부 영상 사이드카(.analysis/.ptz/.events/
  .whistle.json)에 있음 — 프로젝트 JSON 과의 결합은 "마지막 파노라마
  경로" 하나. 결합점은 pano.mp4 한 방향 핸드오프뿐.
- 사용 패턴 상반: 스티칭 = 경기당 1회 배치, 분석 = 반복 인터랙티브
  검수. 사용자도 다를 수 있음 (촬영 담당 / 분석 담당).
- 의존성·패키징: 분석은 torch/ultralytics/easyocr (GB 급), 스티칭은
  ffmpeg/NVENC 만 — 분리하면 스티칭 exe 가 가벼워짐 (Windows 패키징
  장기 항목과 직결).

## 이름 (확정)

- 스티칭 앱: **PitchStitch** (`pitchstitch.py`)
- 경기 분석 앱: **PitchWatch** (`pitchwatch.py`)
- -er 형(PitchStitcher/PitchWatcher)도 검토했으나 짧은 형의 라임
  (-tch 운율 형제)을 우선 — 제품화 단계에서 재논의 여지만 남김.
- 참고: PitchView 는 기존 제품(pitchview.app) 존재로 제외 확인.

## 작업 항목

### 1. 런처 분리

- `pitchstitch.py` — 탭 1~3(영상·동기화/정합/내보내기)만 있는 윈도우.
- `pitchwatch.py` — PtzTab 을 자체 MainWindow 로 승격:
  - 자체 메뉴바 (현 분석 메뉴 + 파일 메뉴: 파노라마 열기)
  - **최근 파일 = 최근 pano.mp4 목록** (프로젝트 아님)
  - 자체 하단 없이 로그 탭만 (이미 구현)
- `main.py` — 통합 실행 유지 (전환기 호환).
- PtzTab 의존 정리: log_fn 주입은 이미 됨. main_window 참조가 남아
  있는지 점검 (recent dir 콜백 등 — 이미 주입식).

### 2. 핸드오프 UX

- PitchStitch 내보내기 완료 대화에 "PitchWatch 에서 열기" 버튼 —
  `pitchwatch.py <pano.mp4>` 프로세스 실행 (CLI 인자로 경로).
- PitchWatch 는 argv[1] 파노라마 자동 열기.

### 3. 브랜딩 적용

- 창 제목·About 에 PitchStitch/PitchWatch 반영. 저장소 이름 변경
  여부는 별도 결정 (당분간 PyStitch360 유지 가능).
- QSettings 는 당분간 현행 조직/앱 키 유지 (설정 유실 방지) —
  앱별 분리는 필요해질 때 마이그레이션과 함께.

### 4. 패키징 (P05 범위 밖, 준비만)

- PyInstaller spec 2개 (stitch: torch 제외 / match: 포함) 구조 검토.
- 실제 패키징은 기존 장기 보류 항목과 병행 결정.

## 완료 기준

- 두 런처가 각각 단독 실행 (스티칭 런처는 torch 미설치 환경에서도 기동).
- PitchWatch: 파노라마 열기→검수→하이라이트→내보내기 전 과정이 통합
  실행과 동일 동작 + CLI 경로 인자 동작.
- 핸드오프 버튼으로 PitchStitch→PitchWatch 이어짐.
- docs/ptz_workflow.md 진입점 갱신.

## 비고

- 스크립트(scripts/)는 이미 제3의 헤드리스 표면 — 변경 없음.
- QSettings 키는 현행 유지가 안전 (두 앱이 같은 키를 읽어도 충돌하는
  키 없음 — 창 크기류만 앱별 분리 검토).
