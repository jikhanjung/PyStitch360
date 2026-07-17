# 002: 새 GUI 골격 + pystitch 패키지 구축

- 날짜: 2026-07-17
- 종류: 작업 기록
- 관련: [P01 계획](20260717_P01_cylindrical_panorama_plan.md), [001 프로토타입](20260717_001_prototype_stitching_results.md)

## 구조

```
pystitch/
├── core/
│   ├── lens.py       # Gyroflow 렌즈 프로파일 로드, 내장 프리셋 목록
│   ├── geometry.py   # 광선/회전/원통 remap (프로토타입 검증 코드)
│   ├── align.py      # Alignment(자동값 보존 + 사용자 오프셋 즉시 적용),
│   │                 #   SIFT 정합, auto-level, 하프라인 센터링
│   ├── render.py     # Renderer(remap 캐싱, scale 지원), 심 가중치, 게인,
│   │                 #   refine_seam(심 Y 어긋남 국소 보정)
│   ├── chapters.py   # GoPro 챕터 그룹핑(GOPR→GP01...), ChapteredVideo
│   └── sync.py       # 오디오 상호상관 오프셋 (numpy FFT)
├── gui/
│   ├── main_window.py  # 탭 3개: 영상·동기화 / 정합·미리보기 / 내보내기
│   ├── workers.py      # Sync/Align/Preview/Export QThread 워커
│   └── widgets.py      # FramePane (종횡비 유지 프레임 표시)
main.py                 # 진입점
```

## 심 Y 어긋남 보정 (refine_seam)

사용자 지적: 하프라인 기준 상하가 약간 어긋남. 원인은 시차 잔재라 전역 회전으로
제거 불가. 해결: 심 밴드(±90px)에서 L 템플릿을 R 에서 템플릿 매칭 → 행별 dy 실측
→ 2차 다항 피팅 → R remap 을 심 주변(테이퍼 360px)에서만 세로 이동. 적용 후
재측정해서 악화 시 롤백(부호 안전장치). 실측: **rms 3.7px → 0.6px** (풀해상도).

## GUI 동작 확인 (헤드리스 스모크 테스트)

QT_QPA_PLATFORM=offscreen 으로 실제 39.5분 영상(챕터 5개) 로드 → 자동 정합 →
미리보기 렌더까지 검증. 스크린샷 확인 완료.

- 듀얼 뷰어: 챕터 통합 타임라인(GOPR0395+GP01~04 = 71,065프레임), 프레임 스텝,
  오프셋 스핀박스(R−L 초), 오디오 자동 동기화 버튼
- 정합 탭: 자동 정합 버튼 → 미리보기(1/4 해상도, ~0.6s), pitch/roll/yaw/페더
  미세조정(400ms 디바운스)
- 내보내기 탭: 구간, H.264/HEVC, CRF, 100%/50% 해상도, 진행률/취소.
  오디오는 좌측 챕터 체인을 ffmpeg concat demuxer 로 연결.

## 잡은 버그

- QThread 참조를 용도별로 분리하지 않으면 (align 완료 콜백에서 self._worker 를
  preview 워커로 재할당) 실행 중 스레드가 GC 되어 미리보기가 죽음.
  → _sync/_align/_preview/_export 워커 참조 분리 + 실행 중 재진입 가드.
- 시간 라벨 "04:60.00" 반올림 버그 → 표시 단위로 먼저 반올림 후 분/초 분리.

## 다음 단계

- 프로젝트 저장/불러오기 (JSON: 파일 목록, 오프셋, 정합, 사용자 보정)
- 내보내기 성능 (스레드 파이프라이닝, NVENC 옵션)
- 세그먼트(충격 이벤트) 관리 UI
- 가상 PTZ (PitchStitch YOLO 추적 이식)
