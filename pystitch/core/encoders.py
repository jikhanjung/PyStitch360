"""사용 가능한 ffmpeg 비디오 인코더 감지."""
from __future__ import annotations

import subprocess
from functools import lru_cache

# 표시 이름 → (인코더, 추가 인자 빌더)
CANDIDATES = {
    "libx264 (H.264, CPU)": "libx264",
    "libx265 (HEVC, CPU)": "libx265",
    "h264_nvenc (H.264, NVIDIA GPU)": "h264_nvenc",
    "hevc_nvenc (HEVC, NVIDIA GPU)": "hevc_nvenc",
}


@lru_cache(maxsize=1)
def available_encoders() -> dict[str, str]:
    """실제 사용 가능한 인코더만 (표시 이름 → ffmpeg 인코더 이름)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        out = ""
    result = {}
    for label, enc in CANDIDATES.items():
        if f" {enc} " in out:
            result[label] = enc
    if not result:
        result = {"libx264 (H.264, CPU)": "libx264"}
    return result


def encoder_args(encoder: str, crf: int) -> list[str]:
    """인코더별 품질/프리셋 인자."""
    if encoder.endswith("_nvenc"):
        # NVENC 는 CRF 대신 CQ. p4 = 중간 프리셋
        args = ["-c:v", encoder, "-preset", "p4", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0"]
    else:
        args = ["-c:v", encoder, "-preset", "fast", "-crf", str(crf)]
    if encoder in ("libx265", "hevc_nvenc"):
        args += ["-tag:v", "hvc1"]
    return args
