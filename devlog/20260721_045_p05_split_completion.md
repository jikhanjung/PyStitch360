# 045: P05 마무리 — PitchStitch 를 진짜 torch-free 로

- 날짜: 2026-07-21
- 종류: 구현
- 관련: [P05](20260720_P05_app_split_plan.md)

## MainWindow(with_ptz=False)

기존 pitchstitch.py 는 MainWindow 전체(PtzTab 포함)를 만든 뒤 탭을
떼고 "분석" 메뉴를 텍스트 매칭으로 지우는 임시 구조였다. 이제
`with_ptz=False` 모드에서:

- PtzTab 을 **생성하지 않음** (모듈 import 자체가 없음),
- 분석 메뉴 미생성 (텍스트 매칭 제거),
- `_gather_project` 는 프로젝트에 있던 pano 참조를 보존
  (`_loaded_ptz_pano`) — PitchStitch 에서 저장해도 PTZ 상태가 지워지지
  않는다,
- `_new_project` 는 with_ptz 와 앱 이름을 새 창에 승계.

## ptz_available: import → find_spec

숨은 함정: `_build_export_tab` 이 생성 시점에 `ptz_available()` 을
호출(콤보 툴팁)하는데, 구현이 `import ultralytics` 라 **torch 전체가
창 생성에 로드**됐다 — pitchstitch 뿐 아니라 통합 앱 기동도 수 초
지연시키던 원인. `importlib.util.find_spec("ultralytics")` 로 변경 —
설치 여부만 보고 실제 import 는 Detector 등 사용 시점으로 미룬다.

## 검증

- 스모크(offscreen): with_ptz=False 창 생성 후
  `sys.modules` 에 torch/ultralytics/easyocr/ptz_tab **부재 단언** —
  "torch 미설치 환경 기동" 완료 기준보다 강한 검증.
- 통합 모드 4탭 + 분석 메뉴, PitchWatchWindow 생성 정상.
- tests/ 114개 통과. 단 `test_project_resolves_cross_platform_paths` 는
  저장소 개명(PyStitch360 → TouchlineAnalyst, 2026-07-21)으로
  하드코딩 경로가 깨져 있어 저장소 이름 독립으로 수정.

## packaging/ 초안 복원

커밋 320d375 메시지에 언급된 spec 초안이 실제론 저장소에 없었다
(untracked 유실 추정). pitchstitch.spec (torch 계열 excludes) /
pitchwatch.spec (지연 import 라 hiddenimports 필요, onedir) 초안을
packaging/ 에 추가 — 여전히 미검증, 실제 패키징은 보류 항목 그대로.
