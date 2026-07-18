"""오디오 상호상관 기반 좌/우 카메라 동기화 오프셋 추정."""
from __future__ import annotations

import subprocess
from .encoders import ffmpeg_bin
import tempfile
import wave
from pathlib import Path

import numpy as np


def _extract_audio(video: str, start: float, duration: float, sr: int) -> np.ndarray:
    """ffmpeg 로 모노 wav 추출 → float 배열."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        subprocess.run(
            [ffmpeg_bin(), "-y", "-v", "error", "-ss", str(start), "-i", video,
             "-t", str(duration), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", path],
            check=True,
        )
        with wave.open(path, "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float64)
    finally:
        Path(path).unlink(missing_ok=True)


def estimate_offset(left_video: str, right_video: str, start: float = 0.0,
                    duration: float = 120.0, sr: int = 8000):
    """동기화 오프셋 추정.

    반환 (offset_sec, confidence):
      offset_sec = 같은 사건의 R 타임스탬프 − L 타임스탬프.
      즉 L 의 t 초 프레임과 R 의 t + offset 초 프레임이 같은 순간.
      confidence = 상관 피크 / 차순위 피크 비율 (>4 정도면 신뢰).
    """
    a_l = _extract_audio(left_video, start, duration, sr)
    a_r = _extract_audio(right_video, start, duration, sr)
    # 간단한 고역 강조 (저주파 바람 소리 억제)
    a_l = np.diff(a_l)
    a_r = np.diff(a_r)

    n = len(a_l) + len(a_r) - 1
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a_l, nfft) * np.conj(np.fft.rfft(a_r, nfft)), nfft)
    # 순환 상관: 인덱스 k 는 lag k (k > nfft/2 는 음수 lag k-nfft).
    # lag k 의 의미: a_l[t] ≈ a_r[t-k]
    c = np.abs(corr)
    peak = int(np.argmax(c))
    lag = peak if peak <= nfft // 2 else peak - nfft
    peak_val = c[peak]
    mask = np.ones_like(c, dtype=bool)
    mask[max(0, peak - sr) : peak + sr] = False
    if peak - sr < 0:  # 순환 배열 반대편 끝도 피크 주변
        mask[nfft + (peak - sr):] = False
    noise = c[mask].max() if mask.any() else peak_val
    confidence = float(peak_val / noise) if noise > 0 else float("inf")

    # a_l[t] ≈ a_r[t - k] → L 의 사건이 R 파일에서 k 샘플 이른 위치 →
    # 같은 사건의 R 시각 = L 시각 - k/sr → offset = -k/sr
    offset_sec = -lag / sr
    return offset_sec, confidence
