"""멀티캠 경기 컨테이너 (P07-1) — P06 데이터 모델의 v1 부분집합.

`<경기>.match.json` = 하프별 {주 파노라마 + alt 영상들 + 시계 모델}.
시계 모델은 per-video 동기화(.events.json "sync", scripts/sync_cams.py)
결과의 사본 — match.json 하나만 있으면 열린다. 공간 정합·융합 트랙은
v2 (P06-3~5 이후).

시계 규약 (sync_multi 와 동일): t_primary = offset + drift · t_alt.
"""
from __future__ import annotations

import json
from pathlib import Path

from .events import load_events_doc
from .project import _cross_platform_candidates

MATCH_VERSION = 1
MATCH_SUFFIX = ".match.json"


# ------------------------------------------------------------------ 시계
def to_alt_time(clock: dict, t_primary: float) -> float:
    """primary 시각 → alt 시각."""
    return (t_primary - clock["offset"]) / clock.get("drift", 1.0)


def to_primary_time(clock: dict, t_alt: float) -> float:
    """alt 시각 → primary 시각."""
    return clock["offset"] + clock.get("drift", 1.0) * t_alt


def alt_coverage(clock: dict, alt_dur_s: float) -> tuple[float, float]:
    """alt 전체 [0, dur] 가 덮는 primary 시간 구간 (t0, t1)."""
    return (to_primary_time(clock, 0.0), to_primary_time(clock, alt_dur_s))


# ------------------------------------------------------------------ 경로
def _resolve(p: str, base: Path) -> str:
    """멤버 경로 복원: match.json 상대 → 절대 → 크로스플랫폼 후보."""
    cand = [str(base / p)] if not Path(p).is_absolute() else []
    cand.append(p)
    for c in list(cand):
        cand += _cross_platform_candidates(c)
    for c in cand:
        if Path(c).exists():
            return str(Path(c))
    return p                                   # 못 찾으면 원문 유지 (경고는 호출부)


def _portable(p: str, base: Path) -> str:
    """저장용: match.json 옆이면 상대 경로로 (디렉터리 통째 이동 대비)."""
    try:
        return str(Path(p).relative_to(base))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ 문서
def load_match(path: str | Path) -> dict:
    """match.json 로드 + 멤버 경로 복원. 형식 오류는 ValueError."""
    path = Path(path)
    d = json.loads(path.read_text())
    if d.get("version", 0) > MATCH_VERSION:
        raise ValueError(f"match.json v{d['version']} — 지원은 v{MATCH_VERSION} 까지")
    if not d.get("halves"):
        raise ValueError("halves 없음")
    base = path.parent
    for h in d["halves"]:
        if "primary" not in h:
            raise ValueError(f"하프 {h.get('label', '?')}: primary 없음")
        h["primary"] = _resolve(h["primary"], base)
        for a in h.get("alts", []):
            if "video" not in a or "clock" not in a \
                    or "offset" not in a["clock"]:
                raise ValueError("alt 항목에 video/clock.offset 필요")
            a["video"] = _resolve(a["video"], base)
            a["clock"].setdefault("drift", 1.0)
    return d


def save_match(path: str | Path, doc: dict) -> Path:
    """저장 — 멤버 경로는 가능하면 match.json 상대로."""
    path = Path(path)
    base = path.parent
    out = {"version": MATCH_VERSION, "title": doc.get("title", path.stem),
           "halves": []}
    for h in doc["halves"]:
        oh = {"label": h.get("label", ""),
              "primary": _portable(h["primary"], base),
              "alts": [{"video": _portable(a["video"], base),
                        "clock": {"offset": round(float(a["clock"]["offset"]), 4),
                                  "drift": float(a["clock"].get("drift", 1.0))},
                        **({"stage": a["stage"]} if "stage" in a else {})}
                       for a in h.get("alts", [])]}
        out["halves"].append(oh)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return path


def match_from_sync_sidecars(primaries: list[str | Path],
                             labels: list[str] | None = None,
                             title: str = "") -> dict:
    """primary 파노라마들의 .events.json "sync" 로 v1 문서 구성.

    sync_cams.py 가 남긴 {"other", "offset", "drift", "stage"} 를 alt
    하나로 채운다 (sync 없는 하프는 alts=[]). 여러 alt 는 GUI 에서 추가.
    """
    labels = labels or (["전반", "후반"] if len(primaries) == 2
                        else [f"{i + 1}" for i in range(len(primaries))])
    halves = []
    for p, lab in zip(primaries, labels):
        alts = []
        sync = load_events_doc(p).get("sync")
        if sync and Path(sync.get("other", "")).exists():
            alts.append({"video": sync["other"],
                         "clock": {"offset": sync["offset"],
                                   "drift": sync.get("drift", 1.0)},
                         "stage": sync.get("stage", "whistle")})
        halves.append({"label": lab, "primary": str(p), "alts": alts})
    return {"version": MATCH_VERSION, "title": title, "halves": halves}
