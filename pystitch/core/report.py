"""선수 히트맵/활동량 리포트 (P03-4).

병합 대표 tid 기준으로 트랙릿을 합쳐 팀/선수별 점유 히트맵과
이동거리·속도 통계를 내고, PNG 묶음 + 요약 markdown 을 출력한다.

- 히트맵은 등방(px/m 가로=세로) — 경기장이 입력 실측 비율 그대로.
- 거리/속도는 0.5s 창 평균 위치로 리샘플해 계산 — 검출 지터(±0.3m)가
  랜덤워크로 거리를 부풀리는 것을 막고, 속도 상한(11 m/s) 초과 스텝은
  트래커 점프로 보고 제외한다.
- cv2 폰트 제약으로 PNG 안 텍스트는 ASCII, 한글은 markdown 에만.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .field import field_outline, pano_to_field

SPRINT_MPS = 11.0                     # 이 이상은 트래커 점프로 간주


def player_field_tracks(analysis, calib, merges=None, t_range=None):
    """대표 tid별 필드 궤적 {rep: [(t, gx, gy), ...]} (시각 순).

    merges({tid: 대표})가 있으면 멤버 검출을 대표로 합친다.
    t_range=(t0, t1)초 지정 시 그 구간 샘플만 — 한 파노라마에 여러
    경기가 담긴 영상(pano_5316: 앞 31분 타 경기)에서 경기 구간 한정.
    """
    merges = merges or {}
    fps = analysis["fps"]
    frames = np.asarray(analysis["frames"])
    out: dict[int, list] = {}
    for si, prow in enumerate(analysis["players"]):
        rows = [p for p in prow if len(p) >= 5 and p[4] >= 0]
        if not rows:
            continue
        t = float(frames[si] / fps)
        if t_range is not None and not (t_range[0] <= t <= t_range[1]):
            continue
        fxy = pano_to_field(calib, [(p[0], p[1] + p[3] / 2.0) for p in rows])
        for (gx, gy), p in zip(fxy, rows):
            if np.isfinite(gx):
                rep = merges.get(int(p[4]), int(p[4]))
                out.setdefault(rep, []).append((t, float(gx), float(gy)))
    for tr in out.values():
        tr.sort()
    return out


def movement_stats(track, bin_s=0.5, max_gap_s=2.0, max_speed=SPRINT_MPS):
    """궤적 → {time_s, dist_m, avg_mps, max_mps, n}.

    bin_s 창 평균 위치로 리샘플 후 인접 창 간 거리를 누적. 창 간격이
    max_gap_s 를 넘으면 (검출 공백) 잇지 않고, 속도가 max_speed 를
    넘는 스텝은 점프로 제외한다. avg 는 활동 시간 기준.
    """
    if len(track) < 2:
        return {"time_s": 0.0, "dist_m": 0.0, "avg_mps": 0.0,
                "max_mps": 0.0, "n": len(track)}
    t = np.array([p[0] for p in track])
    xy = np.array([[p[1], p[2]] for p in track])
    b = np.floor(t / bin_s).astype(np.int64)
    ub, inv = np.unique(b, return_inverse=True)
    bt = np.zeros(len(ub))
    bx = np.zeros((len(ub), 2))
    cnt = np.bincount(inv).astype(float)
    np.add.at(bt, inv, t)
    np.add.at(bx, inv, xy)
    bt /= cnt
    bx /= cnt[:, None]
    dist = active = 0.0
    vmax = 0.0
    for i in range(1, len(ub)):
        dt = bt[i] - bt[i - 1]
        if not 0.0 < dt <= max_gap_s:
            continue
        d = float(np.hypot(*(bx[i] - bx[i - 1])))
        v = d / dt
        if v > max_speed:
            continue
        dist += d
        active += dt
        vmax = max(vmax, v)
    return {"time_s": float(t[-1] - t[0]), "dist_m": round(dist, 1),
            "avg_mps": round(dist / active, 2) if active > 0 else 0.0,
            "max_mps": round(vmax, 2), "n": len(track)}


def heatmap_grid(points, length, width, cell_m=1.0):
    """필드 점 목록 → 점유 히트맵 (H, W) float — 등방 셀, 경기장 안만."""
    gw = max(2, int(round(length / cell_m)))
    gh = max(2, int(round(width / cell_m)))
    grid = np.zeros((gh, gw), np.float32)
    if len(points):
        P = np.asarray(points, np.float64)
        cx = np.clip(((P[:, 0] + length / 2) / cell_m).astype(int), 0, gw - 1)
        cy = np.clip(((width / 2 - P[:, 1]) / cell_m).astype(int), 0, gh - 1)
        np.add.at(grid, (cy, cx), 1.0)
    return grid


def render_heatmap(grid, length, width, px_per_m=8.0, title="",
                   alpha_max=0.8):
    """히트맵 BGR 이미지: 잔디 배경 + TURBO 컬러맵 + 흰 경기장 선."""
    mx = 3.0                                    # 바깥 여백 (m)
    W = int(round((length + 2 * mx) * px_per_m)) & ~1
    H = int(round((width + 2 * mx) * px_per_m)) & ~1
    img = np.full((H, W, 3), (26, 52, 30), np.uint8)

    def px(X, Y):
        return (int(round((X + length / 2 + mx) * px_per_m)),
                int(round((width / 2 + mx - Y) * px_per_m)))

    big = cv2.resize(grid, (int(length * px_per_m), int(width * px_per_m)),
                     interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=1.5 * px_per_m)
    big = np.maximum(big, 0.0)
    peak = float(big.max())
    if peak > 0:
        norm = big / peak
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        x0, y0 = px(-length / 2, width / 2)
        roi = img[y0:y0 + big.shape[0], x0:x0 + big.shape[1]]
        a = (norm * alpha_max)[..., None].astype(np.float32)
        roi[:] = (heat * a + roi * (1.0 - a)).astype(np.uint8)
    lw = max(1, int(px_per_m / 5))
    for line in field_outline(length, width):
        pts = np.array([px(X, Y) for X, Y in line], np.int32)
        cv2.polylines(img, [pts], False, (235, 235, 235), lw,
                      cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (int(4 * px_per_m), int(2.2 * px_per_m)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.06 * px_per_m,
                    (255, 255, 255), max(1, int(px_per_m / 5)), cv2.LINE_AA)
    return img


def generate_report(analysis, calib, roles_of, out_dir, merges=None,
                    team_names=("Team1", "Team2"), min_det=150, top_n=30,
                    t_range=None, log=print):
    """팀/선수 히트맵 PNG + players.md 요약 → {dir, files, rows}.

    roles_of = {tid: 유효 역할} (병합 대표 기준으로 해석된 값).
    선수 히트맵은 팀 역할(0/1/3/4)·검출 min_det 이상·상위 top_n 만.
    t_range=(t0, t1)초 — 지정 시 그 구간만 집계 (players.md 에 명시).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L, W = calib["length"], calib["width"]
    tracks = player_field_tracks(analysis, calib, merges=merges,
                                 t_range=t_range)
    names = [n if n.isascii() and n.strip() else f"T{i + 1}"
             for i, n in enumerate(team_names)]
    files = []
    # 팀 히트맵 (GK 포함)
    for team, roles in ((0, (0, 3)), (1, (1, 4))):
        pts = [q[1:] for rep, tr in tracks.items()
               if roles_of.get(rep, 2) in roles for q in tr]
        img = render_heatmap(heatmap_grid(pts, L, W), L, W,
                             title=f"{names[team]}  ({len(pts)} det)")
        p = out / f"team{team + 1}_heatmap.png"
        cv2.imwrite(str(p), img)
        files.append(p)
        log(f"[report] {p.name}: 검출 {len(pts)}개")
    # 선수별 (병합 대표, 검출 수 순)
    players = sorted(
        (rep for rep, tr in tracks.items()
         if roles_of.get(rep, 2) in (0, 1, 3, 4) and len(tr) >= min_det),
        key=lambda r: -len(tracks[r]))[:top_n]
    rows = []
    n_merged = sum(1 for m in (merges or {}).values())
    for rep in players:
        tr = tracks[rep]
        st = movement_stats(tr)
        role = roles_of.get(rep, 2)
        k = 1 + sum(1 for r in (merges or {}).values() if r == rep)
        tag = f"#{rep}" + (f"(+{k - 1})" if k > 1 else "")
        team = names[0 if role in (0, 3) else 1]
        gk = " GK" if role in (3, 4) else ""
        img = render_heatmap(
            heatmap_grid([q[1:] for q in tr], L, W), L, W,
            title=f"{tag}  {team}{gk}  {st['dist_m']/1000:.2f}km "
                  f"avg {st['avg_mps']:.1f} top {st['max_mps']:.1f} m/s")
        p = out / f"player_{rep}.png"
        cv2.imwrite(str(p), img)
        files.append(p)
        rows.append({"tid": rep, "k": k, "role": role, **st})
    # 요약 markdown
    rng = ("전체 구간" if t_range is None else
           f"{t_range[0] / 60:.1f}~{t_range[1] / 60:.1f}분 (IN/OUT 마커)")
    md = [f"# 선수 활동량 리포트", "",
          f"- 집계 구간: {rng}",
          f"- 트랙릿 병합: {n_merged}개 조각 병합됨",
          f"- 선수 기준: 검출 {min_det}회 이상, 상위 {top_n}명", "",
          "| 선수 | 팀 | 관측 | 검출 | 이동거리 | 평균속도 | 최고속도 |",
          "|---|---|---|---|---|---|---|"]
    role_name = {0: team_names[0], 1: team_names[1],
                 3: f"{team_names[0]} GK", 4: f"{team_names[1]} GK"}
    for r in rows:
        plus = f" (+{r['k'] - 1})" if r["k"] > 1 else ""
        md.append(
            f"| #{r['tid']}{plus} "
            f"| {role_name.get(r['role'], '?')} "
            f"| {r['time_s']/60:.1f}분 | {r['n']} "
            f"| {r['dist_m']/1000:.2f} km | {r['avg_mps']:.1f} m/s "
            f"| {r['max_mps']:.1f} m/s |")
    mdp = out / "players.md"
    mdp.write_text("\n".join(md) + "\n", encoding="utf-8")
    files.append(mdp)
    log(f"[report] 선수 {len(rows)}명 → {out}")
    return {"dir": str(out), "files": [str(f) for f in files], "rows": rows}
