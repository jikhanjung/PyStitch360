"""렌즈 프로파일 (Gyroflow 형식, OpenCV fisheye 모델)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROFILE_DIR = Path(__file__).resolve().parents[2] / "presets" / "lens_profiles"


@dataclass
class LensProfile:
    name: str
    K: np.ndarray       # 3x3 camera matrix
    D: np.ndarray       # fisheye distortion k1..k4
    width: int
    height: int

    @classmethod
    def load(cls, path: str | Path) -> "LensProfile":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        if not p.get("use_opencv_fisheye"):
            raise ValueError(f"OpenCV fisheye 프로파일만 지원: {path}")
        return cls(
            name=p.get("name", Path(path).stem),
            K=np.array(p["fisheye_params"]["camera_matrix"], dtype=np.float64),
            D=np.array(p["fisheye_params"]["distortion_coeffs"], dtype=np.float64),
            width=p["calib_dimension"]["w"],
            height=p["calib_dimension"]["h"],
        )

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])


def builtin_profiles() -> dict[str, Path]:
    """presets/lens_profiles/ 의 내장 프로파일 목록 (이름 → 경로)."""
    if not PROFILE_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(PROFILE_DIR.glob("*.json"))}
