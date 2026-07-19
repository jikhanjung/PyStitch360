"""트랙릿 병합 (ReID 라이트) — P03-3.

같은 사람이 검출 끊김으로 여러 트랙릿에 갈라진 것을 시공간 연속성 +
유니폼 색 + 역할 일치로 잇는다. 결과는 병합 맵 {tid: 대표 tid} 로
.ptz.json "merges" 에 저장되는 **비파괴 편집** — 분석 원본과 역할
데이터는 불변이라 그룹 해체·멤버 분리·나중 추가가 언제든 가능하다
(팀 분류 오류는 먼저 다 고칠 필요 없이 병합 후 그룹 단위로 수정).
"""
from __future__ import annotations

import numpy as np

from .field import pano_to_field


def tracklet_summaries(analysis, calib, edge_k=5, min_det=3):
    """트랙릿별 병합 판단 요약 {tid: {t0, t1, p0, p1, feat, n}}.

    t0/t1 = 첫/끝 검출 시각, p0/p1 = 첫/끝 edge_k 개 필드 좌표 중앙값
    (유한값만), feat = classify_teams 와 같은 s-가중 색 벡터
    (s·cos h, s·sin h, v) 의 중앙값. 검출 min_det 미만 트랙릿과 필드
    좌표가 없는 트랙릿은 제외 (병합 판단 불가).
    """
    fps = analysis["fps"]
    frames = np.asarray(analysis["frames"])
    ts: dict[int, list] = {}
    pts: dict[int, list] = {}
    feats: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        rows = [p for p in prow if len(p) >= 5 and p[4] >= 0]
        if not rows:
            continue
        t = float(frames[si] / fps)
        fxy = pano_to_field(calib, [(p[0], p[1] + p[3] / 2.0) for p in rows])
        for (gx, gy), p in zip(fxy, rows):
            tid = int(p[4])
            ts.setdefault(tid, []).append(t)
            if np.isfinite(gx):
                pts.setdefault(tid, []).append((t, float(gx), float(gy)))
            if len(p) >= 8:
                a = p[5] / 90.0 * np.pi
                feats.setdefault(tid, []).append(
                    (p[6] * np.cos(a), p[6] * np.sin(a), p[7]))
    out = {}
    for tid, tt in ts.items():
        if len(tt) < min_det or len(pts.get(tid, ())) < 2 \
                or tid not in feats:
            continue
        P = pts[tid]
        out[tid] = {
            "t0": tt[0], "t1": tt[-1], "n": len(tt),
            "p0": tuple(np.median([q[1:] for q in P[:edge_k]], axis=0)),
            "p1": tuple(np.median([q[1:] for q in P[-edge_k:]], axis=0)),
            "feat": tuple(np.median(np.asarray(feats[tid]), axis=0)),
        }
    return out


def suggest_links(summ, roles, max_gap_s=3.0, speed_mps=8.0, base_m=2.0,
                  color_thr=60.0, overlap_tol_s=0.3):
    """병합 링크 제안 [(a, b, cost), ...] — a 끝 → b 시작 체인.

    조건 (P03-3): (a) 시간 겹침 없음 (overlap_tol_s 허용 — 트래커
    경계 지터), (b) Δt < max_gap_s 이고 필드 거리 < Δt×speed + base,
    (c) 색 거리 < color_thr, (d) 유효 역할 일치. 그리디(비용 순)로
    트랙릿당 후속/선행 각 1개만 — 체인 병합.
    """
    ids = sorted(summ, key=lambda t: summ[t]["t0"])
    cands = []
    for i, a in enumerate(ids):
        A = summ[a]
        for b in ids[i + 1:]:
            B = summ[b]
            dt = B["t0"] - A["t1"]
            if dt > max_gap_s:
                break                # ids 는 t0 순 — 이후 b 는 전부 초과
            if dt < -overlap_tol_s:
                continue
            reach = speed_mps * max(dt, 0.3) + base_m
            d = float(np.hypot(A["p1"][0] - B["p0"][0],
                               A["p1"][1] - B["p0"][1]))
            if d > reach:
                continue
            c = float(np.linalg.norm(np.subtract(A["feat"], B["feat"])))
            if c > color_thr:
                continue
            if roles.get(a, 2) != roles.get(b, 2):
                continue
            cands.append((a, b, d / reach + c / color_thr))
    cands.sort(key=lambda x: x[2])
    succ, pred = set(), set()
    links = []
    for a, b, cost in cands:
        if a in succ or b in pred:
            continue
        succ.add(a)
        pred.add(b)
        links.append((a, b, cost))
    return links


def merge_map(links, n_det):
    """링크 [(a, b)] → {tid: 대표 tid} (연결 성분, 대표 = 검출 최다).

    union 이라 기존 병합(수동 포함)과 새 제안을 합쳐 넣으면 나중에
    빠진 조각을 그룹에 추가하는 것도 같은 경로로 처리된다.
    """
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in links:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list] = {}
    for t in parent:
        groups.setdefault(find(t), []).append(t)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda t: (n_det.get(t, 0), -t))
        for t in members:
            if t != rep:
                out[t] = rep
    return out
