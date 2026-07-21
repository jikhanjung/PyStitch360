"""core/ocr.py OnlineCropCache (P09-1) — 온라인 크롭 수집 의미론."""
import numpy as np

from pystitch.core.ocr import OnlineCropCache


def _frame():
    return np.full((1000, 2000, 3), 90, np.uint8)


def test_gates_and_budget():
    c = OnlineCropCache(fps=30.0, min_h=90.0, per_track=3, spacing_s=2.0)
    f = _frame()
    # 높이 게이트 미달 + tid 없음 → 무시
    c.hook(f, 0, 0, [[100, 500, 40, 80, 5], [200, 500, 40, 120, -1]])
    assert c.n == 0
    # 유효 크롭 축적, 같은 tid 는 2s 간격 제약
    for i in range(10):
        c.hook(f, i, i * 30, [[300, 500, 50, 100 + i, 7]])   # 1s 간격
    assert c.n <= 5                       # 2s 간격 → 절반만
    # per_track 상한: 높이 상위만 유지
    for i in range(10, 30):
        c.hook(f, i, i * 90, [[300, 500, 50, 100 + i, 7]])   # 3s 간격
    assert sum(len(v) for v in c._by_tid.values()) == 3
    assert min(e[0] for e in c._by_tid[7]) >= 120


def test_picked_merges_reps_and_roles():
    c = OnlineCropCache(fps=30.0, min_h=90.0, per_track=2, spacing_s=2.0)
    f = _frame()
    c.hook(f, 0, 0, [[300, 500, 50, 150, 7], [800, 500, 50, 140, 8]])
    c.hook(f, 1, 200, [[300, 500, 50, 160, 7]])
    # tid 8 은 심판(역할 5) → 제외, 7·9 는 같은 rep 로 병합
    c.hook(f, 2, 400, [[300, 500, 50, 155, 9]])
    picked = c.picked(role_of=lambda t: 5 if t == 8 else 0,
                      rep_of=lambda t: 7)
    assert len(picked) == 2               # rep 7 로 병합, per_track=2
    assert all(rep == 7 for _si, _b, rep in picked)
