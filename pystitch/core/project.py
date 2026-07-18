"""프로젝트 파일 (JSON) 저장/불러오기.

형식 (version 1):
{
  "version": 1,
  "left_files": [...], "right_files": [...],     # 절대경로
  "left_names": [...], "right_names": [...],     # 프로젝트 파일 기준 상대(이동 대비)
  "offset_sec": 0.068,
  "lens_profile": "GoPro_HERO5_Black_Wide_4K_16x9",
  "segments": [ { "start_sec": 0.0, "alignment": {...} }, ... ],
  "user": { "pitch": 0, "roll": 0, "yaw": 0, "feather_px": 40 },
  "export": { "start": 0, "end": 60, "codec": "libx264", "crf": 19, "scale": 1.0 }
}
세그먼트는 시작 시각 오름차순. 각 alignment 는 Alignment 직렬화.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .align import EL0_RAD, EL1_RAD, Alignment

VERSION = 1


def alignment_to_dict(a: Alignment) -> dict:
    return {
        "Rh": np.asarray(a.Rh).tolist(),
        "yaw_split_deg": a.yaw_split_deg,
        "pitch_auto": a.pitch_auto,
        "roll_auto": a.roll_auto,
        "yaw_auto": a.yaw_auto,
        "n_matches": a.n_matches,
        "n_inliers": a.n_inliers,
        "residual_deg": a.residual_deg,
        "el0": a.el0,
        "el1": a.el1,
    }


def alignment_from_dict(d: dict) -> Alignment:
    return Alignment(
        Rh=np.array(d["Rh"], dtype=np.float64),
        yaw_split_deg=float(d["yaw_split_deg"]),
        pitch_auto=float(d["pitch_auto"]),
        roll_auto=float(d["roll_auto"]),
        yaw_auto=float(d["yaw_auto"]),
        n_matches=int(d.get("n_matches", 0)),
        n_inliers=int(d.get("n_inliers", 0)),
        residual_deg=float(d.get("residual_deg", 0.0)),
        el0=float(d.get("el0", EL0_RAD)),
        el1=float(d.get("el1", EL1_RAD)),
    )


def _cross_platform_candidates(p: str) -> list[str]:
    """WSL(/mnt/x/...) ↔ Windows(X:/...) 경로 후보 — 같은 프로젝트 파일을
    양쪽 환경에서 열 수 있게 한다 (내보내기는 Windows 네이티브 NVENC 등)."""
    out = []
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
    if m:
        out.append(f"{m.group(1).upper()}:/{m.group(2)}")
    m = re.match(r"^([a-zA-Z]):[\\/](.*)$", p)
    if m:
        out.append(f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/"))
    return out


def save_project(path: str | Path, data: dict):
    path = Path(path)
    base = path.parent
    out = dict(data)
    out["version"] = VERSION
    for side in ("left", "right"):
        files = [str(Path(f)) for f in data.get(f"{side}_files", [])]
        out[f"{side}_files"] = files
        names = []
        for f in files:
            try:
                names.append(str(Path(f).relative_to(base)))
            except ValueError:
                names.append(Path(f).name)
        out[f"{side}_names"] = names
    out["segments"] = [
        {"start_sec": s["start_sec"],
         "align_sec": s.get("align_sec", s["start_sec"]),   # 정합 추정 프레임 시각
         "alignment": alignment_to_dict(s["alignment"])}
        for s in data.get("segments", [])
    ]
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def load_project(path: str | Path) -> dict:
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("version", 1) > VERSION:
        raise ValueError(f"지원하지 않는 프로젝트 버전: {d.get('version')}")
    base = path.parent
    for side in ("left", "right"):
        resolved = []
        files = d.get(f"{side}_files", [])
        names = d.get(f"{side}_names", [])
        for i, f in enumerate(files):
            p = Path(f)
            if not p.exists() and i < len(names) and (base / names[i]).exists():
                p = base / names[i]   # 프로젝트 폴더 기준 상대경로로 복구
            if not p.exists():
                for cand in _cross_platform_candidates(str(f)):
                    if Path(cand).exists():   # WSL ↔ Windows 드라이브 경로 변환
                        p = Path(cand)
                        break
            resolved.append(str(p))
        d[f"{side}_files"] = resolved
    d["segments"] = [
        {"start_sec": float(s["start_sec"]),
         "align_sec": float(s.get("align_sec", s["start_sec"])),
         "alignment": alignment_from_dict(s["alignment"])}
        for s in d.get("segments", [])
    ]
    d["segments"].sort(key=lambda s: s["start_sec"])
    return d
