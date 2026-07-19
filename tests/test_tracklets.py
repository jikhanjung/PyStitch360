"""core/tracklets.py — 병합 링크 제안·union 병합 맵 (P03-3)."""
import numpy as np

from pystitch.core.tracklets import (
    merge_map, suggest_links, tracklet_summaries,
)

RED, BLUE = (200.0, 10.0, 120.0), (-180.0, -30.0, 100.0)


def _s(t0, t1, p0, p1, feat, n=50):
    return {"t0": t0, "t1": t1, "p0": p0, "p1": p1, "feat": feat, "n": n}


def _summ():
    return {
        1: _s(0, 10, (0, 0), (10, 0), RED),
        2: _s(11.5, 20, (13, 0), (30, 0), RED),      # Δt=1.5s, 3m — 이음
        3: _s(21, 30, (31, 0), (50, 0), RED),        # Δt=1s, 1m — 이음
        4: _s(11.5, 19, (13.5, 0), (25, 5), BLUE),   # 색 다름 → 기각
        5: _s(11.5, 19, (60, 20), (70, 20), RED),    # 거리 초과 → 기각
        6: _s(35, 40, (52, 0), (60, 0), RED),        # Δt=5s → 기각
    }


def test_chain_and_gates():
    summ = _summ()
    roles = {t: 0 for t in summ}
    pairs = {(a, b) for a, b, _ in suggest_links(summ, roles)}
    assert (1, 2) in pairs and (2, 3) in pairs
    assert all(4 not in p and 5 not in p and 6 not in p for p in pairs)


def test_role_gate():
    summ = _summ()
    roles = {t: 0 for t in summ}
    roles[2] = 1
    pairs = {(a, b) for a, b, _ in suggest_links(summ, roles)}
    assert (1, 2) not in pairs


def test_number_gate():
    """등번호가 다른 트랙릿은 병합 후보에서 제외 (한쪽만 있으면 허용)."""
    summ = _summ()
    roles = {t: 0 for t in summ}
    pairs = {(a, b) for a, b, _ in
             suggest_links(summ, roles, nums={1: "7", 2: "10"})}
    assert (1, 2) not in pairs and (2, 3) in pairs   # 3은 번호 없음 → 허용
    pairs2 = {(a, b) for a, b, _ in
              suggest_links(summ, roles, nums={1: "7", 2: "7"})}
    assert (1, 2) in pairs2                           # 같은 번호는 병합


def test_single_successor():
    summ = _summ()
    summ[7] = _s(11.6, 19.5, (12, 1), (29, 1), RED)  # 2와 경합
    links = suggest_links(summ, {t: 0 for t in summ})
    assert len([b for a, b, _ in links if a == 1]) == 1


def test_merge_map_union():
    n_det = {1: 50, 2: 200, 3: 30, 9: 10}
    m = merge_map([(1, 2), (2, 3)], n_det)
    assert m == {1: 2, 3: 2}                          # 대표 = 검출 최다
    m2 = merge_map([(3, 9)] + list(m.items()), n_det)  # 빠진 조각 추가
    assert m2 == {1: 2, 3: 2, 9: 2}


def test_summaries(calib):
    ana = {"fps": 30.0, "frames": list(range(0, 300, 3)),
           "players": [[[2800 + si * 4, 1200, 40, 120, 11, 60, 200, 180]]
                       for si in range(100)]}
    sm = tracklet_summaries(ana, calib)
    assert 11 in sm and sm[11]["n"] == 100
    assert abs(sm[11]["t1"] - 9.9) < 0.2
    assert np.isfinite(sm[11]["p0"]).all() and np.isfinite(sm[11]["p1"]).all()
