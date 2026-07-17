"""원근비 조절: 근경(하단) 압축 + 원경(수평선 근처) 확대 리맵.

터치라인 삼각대 촬영 특성상 가까운 지면이 화면 대부분을 차지하고 반대편
선수는 작게 보인다. 두 변형을 합성해 근/원경 배율 차이를 줄인다.

1. 수직 비선형 리맵 (강도 k, 0=없음):
   수평선 아래 정규화 좌표 u에 대해 소스 행 f(u) = (1-k)u + k*u^2.
   수평선 직하 세로 배율 1/(1-k), 최하단 압축 (1+k). 내용 손실 없음.
2. 키스톤/사영 기울임 (최상단 배율 m, 1=없음):
   행 스케일 w(y) = c*(y-H)+1 (c = (1-1/m)/H). 상단으로 갈수록 가로·세로
   모두 확대. 상단 좌우가 (1 - 1/m) 비율만큼 잘려나간다.

두 변형은 행 단위 스케일이라 하나의 cv2.remap 맵으로 합성된다 (보간 1회).
수평선 행은 렌더 파이프라인에서는 elevation=0 행(el0/(el0-el1)*H)을 쓰면
정확하고, 완성된 파노라마에는 픽셀 값으로 지정한다.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_perspective_maps(w: int, h: int, horizon: float,
                           k: float = 0.3, m: float = 1.3):
    """출력→소스 remap 맵 (map_x, map_y) 생성. k=0, m=1이면 항등."""
    if not 0.0 <= k < 1.0:
        raise ValueError(f"k는 [0,1) 범위여야 함: {k}")
    if m < 1.0:
        raise ValueError(f"m은 1 이상이어야 함: {m}")

    yb = h - 1.0  # 최하단 행 (고정점)
    yo = np.arange(h, dtype=np.float64)
    # 1) 키스톤 역변환: 출력 행 -> (수직 보정된 중간 이미지의) 행 + 행 스케일
    if m > 1.0:
        c = (1.0 - 1.0 / m) / yb
        y_mid = yo * (c * yb - 1.0) / (yo * c - 1.0)
        ws = c * (y_mid - yb) + 1.0  # 행별 가로 스케일 (상단 1/m .. 하단 1)
    else:
        y_mid, ws = yo, np.ones(h)
    # 2) 수직 리맵 역변환: 중간 행 -> 소스 행 (수평선 아래만)
    ys = y_mid.copy()
    below = y_mid > horizon
    u = (y_mid[below] - horizon) / (yb - horizon)
    ys[below] = horizon + (yb - horizon) * ((1.0 - k) * u + k * u * u)

    cx = (w - 1) / 2.0
    xo = np.arange(w, dtype=np.float64)
    map_x = ((xo[None, :] - cx) * ws[:, None] + cx).astype(np.float32)
    map_y = np.repeat(ys[:, None], w, axis=1).astype(np.float32)
    return map_x, map_y


class PerspectiveWarp:
    """맵을 캐싱하는 프레임 단위 적용기. k=0, m=1이면 no-op."""

    def __init__(self, w: int, h: int, horizon: float,
                 k: float = 0.3, m: float = 1.3):
        self.identity = k == 0.0 and m == 1.0
        if not self.identity:
            self.map_x, self.map_y = build_perspective_maps(w, h, horizon, k, m)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.identity:
            return frame
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
