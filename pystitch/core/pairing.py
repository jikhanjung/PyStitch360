"""좌/우 카메라 디렉터리의 GoPro 챕터 체인 짝 맞추기 (헤드리스 모드).

같은 경기를 동시에 찍은 좌/우 영상은 녹화 길이가 거의 같으므로 챕터
체인의 총 파일 크기가 비슷하다 — 총 크기 상대차를 비용으로 그리디
매칭한다. 챕터 수 차이는 비용에 가산 (녹화 시작 시차 허용, 동수 선호).
"""
from __future__ import annotations

from pathlib import Path

from .chapters import group_directory


def chain_size(chain: list[Path]) -> int:
    return sum(p.stat().st_size for p in chain)


def pair_directories(left_dir, right_dir, max_rel_diff=0.25):
    """반환 (pairs, unmatched_l, unmatched_r).

    pairs = [(left_chain, right_chain, cost), ...] 좌측 파일명 순.
    cost = 총 크기 상대차 + 0.05×|챕터 수 차| — max_rel_diff 초과 조합은
    짝으로 보지 않는다.
    """
    gl = group_directory(left_dir)
    gr = group_directory(right_dir)
    if not gl:
        raise RuntimeError(f"GoPro 영상 없음 (GOPR*.MP4): {left_dir}")
    if not gr:
        raise RuntimeError(f"GoPro 영상 없음 (GOPR*.MP4): {right_dir}")
    cands = []
    sizes_r = [chain_size(c) for c in gr]
    for i, cl in enumerate(gl):
        sl = chain_size(cl)
        for j, cr in enumerate(gr):
            sr = sizes_r[j]
            cost = abs(sl - sr) / max(sl, sr, 1)
            cost += 0.05 * abs(len(cl) - len(cr))
            if cost <= max_rel_diff:
                cands.append((cost, i, j))
    cands.sort()
    used_l: set[int] = set()
    used_r: set[int] = set()
    pairs = []
    for cost, i, j in cands:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append((gl[i], gr[j], cost))
    pairs.sort(key=lambda t: t[0][0].name)
    unmatched_l = [c for i, c in enumerate(gl) if i not in used_l]
    unmatched_r = [c for j, c in enumerate(gr) if j not in used_r]
    return pairs, unmatched_l, unmatched_r
