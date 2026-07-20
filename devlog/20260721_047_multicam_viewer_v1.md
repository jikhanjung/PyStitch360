# 047: 멀티캠 뷰어 v1 — match.json + PiP/분할/전환 (P07-1)

- 날짜: 2026-07-21
- 종류: 구현
- 관련: [P07](20260721_P07_pitchwatch_multicam_ui.md) 1단계,
  [046](20260721_046_p06_sync_realdata.md) (동기화 실데이터)

## core/match.py

`<경기>.match.json` v1 = 하프별 {primary 파노라마, alts: [{video,
clock}]}. 시계 모델은 per-video sync 의 사본 — 파일 하나로 열린다.
경로는 저장 시 match.json 상대(디렉터리 이동 대비), 로드 시 상대→절대
→크로스플랫폼(/mnt/x ↔ X:) 순 복원. `match_from_sync_sidecars` 로
기존 .events.json "sync" 에서 구성. 합성 테스트 4개.

## gui/multicam.py — "alt 목록 + 배치 전략"

- **AltDecodeWorker**: 카메라별 요청 슬롯, 슬롯당 최신 요청만 처리 —
  4K 랜덤 시크(수백 ms)가 UI 를 안 막고 밀린 요청은 버린다. v2 다중
  페인(그리드)이 워커 수정 없이 카메라 수만큼 request 가능한 구조.
- **AltPane**: 읽기 전용 FramePane. PiP 모드에선 드래그 이동 +
  우하단 리사이즈, 지오메트리는 QSettings.
- **MulticamViewer**: 모드(pip/split/swap)·활성 카메라·재생 중 스로틀
  (~5fps). 전환 모드는 메인 페인에 alt 원본을 그린다.
- **MatchBuildDialog + SyncRunWorker**: 하프 추가(기존 sync 자동 인식)
  → 앵글 추가(호각 추출부터 필요 시 그 자리에서) → 저장.

## PtzTab/PitchWatch 통합 (최소 접점)

- pane 을 가로 QSplitter 로 감싸 분할 모드 수용 (기존 세로 3행 유지).
- `open_match(doc, half)` — primary 는 기존 open_path 그대로. 단독
  파노라마를 열면 경기 컨텍스트 자동 해제.
- 오버레이 카메라/모드 바, 숫자키 1..9 는 Space 와 같은 "커서가 영상/
  타임라인 위" 관례 (eventFilter 확장 — 명단 입력 등과 충돌 없음).
- `_redraw` 인터셉트로 전환 모드 표시, 6개 페인 입력 핸들러에 swap
  가드 (읽기 전용 보장).
- TimelineView 에 커버리지 스트립 (눈금자 아래 청록, 카메라별 줄).
- PitchWatch 메뉴: 경기 열기/최근 경기/만들기/하프 전환, argv 로
  .match.json 직접 열기. closeEvent 에서 디코드 스레드 정리.

## 검증

- offscreen e2e 스모크: 합성 영상 2개(offset −1s) match 로 열어
  커버리지 스팬(−30f), PiP 표시·프레임 도착, 분할 재배치, 전환
  인터셉트 + 편집 차단, 컨텍스트 해제까지 단언. tests/ 119개 통과.
- 실경기(20241020, AX700) GUI 확인은 Windows 쪽 몫 — v1 완료 기준의
  나머지 (±0.5s 동일 장면, 재생 중 저주기 추종).

## 남은 것 (P07-1)

- 하이라이트 목록 앵글 뱃지 (1-4) — 대체 앵글 일괄 추출 자체는 기존
  기능이라 v1 뷰어와 독립.
- 커버리지 레인 클릭 → 카메라 전환 (지금은 시크만).
