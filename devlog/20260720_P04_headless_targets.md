# 헤드리스 작업 대상 목록 (F:/Pictures, 2026-07-20 조사)

`python main.py --headless <L> <R>` 대상. 경로는 WSL 기준
`/mnt/f/Pictures/...`. 짝 안의 영상(챕터 체인) 매칭은 크기 기반 자동.

## 처리 대상 (10쌍)

| # | Left | Right | 규모 | 비고 |
|---|------|-------|------|------|
| 1 | 20241013_GoPro5_1_1 | 20241013_GoPro5_2_1 | 영상 2개, 챕터 7, ~30GB | **L/R 방향 미확인** (카메라 1/2 표기) |
| 2 | 20241013_GoPro5_1_2 | 20241013_GoPro5_2_2 | 영상 1개, 챕터 10, ~41GB | 같은 날 두 번째 경기, L/R 방향 미확인 |
| 3 | 20241020_GoPro5_Match1_Left | 20241020_GoPro5_Match1_Right | L 영상 4 / R 영상 3 | 영상 수 불일치 — 짝 없는 1개는 자동 제외됨 |
| 4 | 20241020_GoPro5_Match2_Left | 20241020_GoPro5_Match2_Right | L 영상 3 / R 영상 2, 챕터 13 | 〃 |
| 5 | 20250420_GoPro5_11am_L | 20250420_GoPro5_11am_R | 영상 1개, 챕터 8 (~36GB/쪽) | 디렉터리 크기 차이는 부속 파일 탓, 체인은 대칭 |
| 6 | 20250427_GoPro_Left | 20250427_GoPro_Right | 영상 3개, 챕터 29/30, ~145GB/쪽 | 대용량 |
| 7 | 20251026_GoPro5 | "20251026_GoPro5 2" | 영상 1개, 챕터 25, 99GB/쪽 | **L/R 방향 미확인**, 경로에 공백 — 따옴표 필요 |
| 8 | 20251102_GoProLeft | 20251102_GoProRight | 영상 2개, 챕터 41, 161GB/쪽 | 대용량 |
| 9 | 20260621_GoPro5_Left | 20260621_GoPro5_Right | L 영상 2/챕터 45, R 영상 3/챕터 52 | 영상 수 불일치 |
| 10 | 20260712_GoPro5_L | 20260712_GoPro5_R | 영상 3(워밍업 포함)/챕터 4, 19GB/쪽 | 스모크 검증 완료 — 전체 길이 실행은 미완 |

카메라 시리얼(GOPR5xxx/GOPR0xxx)이 날짜마다 좌우가 바뀌므로
(20241020 은 Left=5xxx, 20250420 은 L=0xxx) 시리얼로 방향 추정 불가 —
1·2·7번은 실행 전 방향 확인 필요. 방향이 뒤집혀도 스티칭 자체는 되지만
심 위치·하프라인 검출이 흐트러진다.

## 제외

- `20250823_GoPro5` — 단일 카메라 (짝 없음, 1영상 33챕터 254GB)
- `20250427_GoPro_Morning`, `20251026_GoPro`, `20251102_GoPro` — 편집
  산출물 모음 (.mov/.ptvb/합성 mp4), GoPro 원본 아님
- `20241013_AX700`, `20241020_AX700_*` 등 — 타 카메라

## 실행 예

```bash
cd /mnt/d/projects/PyStitch360
python main.py --headless /mnt/f/Pictures/20250427_GoPro_Left /mnt/f/Pictures/20250427_GoPro_Right
# 출력: /mnt/f/Pictures/20250427_GoPro/ (이름 공통부분 자동)
```

7번은 공백 경로 주의:

```bash
python main.py --headless /mnt/f/Pictures/20251026_GoPro5 "/mnt/f/Pictures/20251026_GoPro5 2"
# 공통부분이 입력 L 과 동명 → 자동으로 20251026_GoPro5_pano/ 로 회피됨
```
