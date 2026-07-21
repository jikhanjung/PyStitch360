# 개발 현황

*최신 사실은 devlog/ 가 기준이다 — 이 문서는 큰 그림 요약만 유지한다.*

## 현재 모습 (2026-07-21)

하나의 저장소, 세 실행 표면:

| 표면 | 진입점 | 역할 |
|---|---|---|
| **PitchStitch** | `pitchstitch.py` | 듀얼 GoPro 스티칭 (동기화·정합·내보내기, torch 불필요) |
| **PitchWatch** | `pitchwatch.py [pano.mp4 \| .match.json]` | 경기 분석 — 공/선수 검출·검수, PTZ 내보내기, 하이라이트, 멀티캠 |
| 헤드리스 | `main.py --headless <L> <R>`, `scripts/` | 무인 일괄 처리 (스티칭→분석→OCR) |

`main.py` 는 통합 실행(전환기 호환). 문서: docs/ptz_workflow.md (분석
워크플로우), docs/heuristics.md (도메인 휴리스틱 카탈로그).

## 기능군 상태

- **스티칭 코어**: 안정 — 자동 정합(잔차 ~0.1°), auto-level(프레임 풀링
  강건화), 드리프트 감지/재정합, NVENC 인코딩.
- **분석/검수**: 안정 — YOLO 검출(GPU), ByteTrack, 갭필, 역할/팀 분류,
  등번호 OCR, 이벤트(킥오프·호각)·하이라이트, 리포트.
- **헤드리스 일괄**: 가동 중 — F:/Pictures 대상 목록은
  devlog/20260720_P04_headless_targets.md 참조.
- **멀티캠 (P06/P07)**: 진행 중 — 호각 동기화 실경기 검증 완료(±0.2s),
  match.json + PiP/분할/전환 뷰어 + 앵글 타임라인 레인 출시.
  AX700 필드 정합(rotcam)은 수학 코어+실영상 프로브까지 — 기준
  캘리브레이션·필드 지형지물 자동 앵커가 다음 단계.

## 장기 보류

- Windows 패키징 (PyInstaller spec 초안만 — packaging/)
- 카메라 증설/업그레이드 (P06 확장 시나리오: 4대 리그, Ace Pro 2)

---
*마지막 업데이트: 2026-07-21*
