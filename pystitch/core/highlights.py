"""하이라이트 후보: 이벤트 소스 융합 → 공유용 구간 제안 (P03-1).

각 소스(킥오프·선심 기 신호·롱 휘슬·공중볼→박스·공 속도 급증·사용자
이벤트)를 규칙 구간 (t0, t1, 종류, 가중치)로 펼치고, 겹치거나 가까운
(< merge_gap_s) 구간을 병합해 가중 합으로 점수를 매긴다. 사용자
이벤트가 최상 가중 — 수동 마크(골 등)가 자동 신호에 묻히지 않는다.

결과는 .events.json "highlights" 리스트로 저장되는 비파괴 문서:
  {t0, t1, kinds, label, score, state}   (state: cand | accept | reject)
재생성해도 기존 수락/제외 상태는 구간 겹침 매칭으로 이어받는다.
"""
from __future__ import annotations

import numpy as np

#: 규칙 가중치 — 병합 구간 점수 = 기여 규칙 가중 합.
WEIGHTS = {"user": 5.0, "kickoff": 3.0, "foul": 2.5, "offside": 2.5,
           "whistle": 2.0, "air": 2.0, "speed": 1.5}
KIND_LABEL = {"kickoff": "킥오프", "foul": "파울", "offside": "오프사이드",
              "whistle": "휘슬", "air": "공중볼", "speed": "강슛"}


def ball_speed_events(t, gxy, min_speed=20.0, max_speed=45.0,
                      max_dt=0.6, hold=2, merge_gap_s=6.0):
    """수락 공 필드 궤적 → 속도 급증(강슛/롱패스) 시각 [(t, peak), ...].

    연속 유한 샘플 간 속도가 min_speed 이상으로 hold 스텝 지속되고,
    구간 전체의 순변위 속도도 그에 준해야(왕복 지터 배제) 이벤트.
    max_speed 초과는 검출 점프(오인식)로 보고 무시. merge_gap_s 내
    이벤트는 하나로 (최고 속도).
    """
    t = np.asarray(t, dtype=np.float64)
    g = np.asarray(gxy, dtype=np.float64)
    ev = []
    run = []                      # 연속 초과 스텝 [(t, v, i0, i1)]

    def flush():
        if len(run) >= hold:
            ia, ib = run[0][2], run[-1][3]
            net = float(np.hypot(*(g[ib] - g[ia])) / max(t[ib] - t[ia], 1e-6))
            if net >= min_speed * 0.6:    # 한 샘플 튐(왕복)은 순변위가 작다
                ev.append((run[0][0], max(v for _, v, _, _ in run)))
        run.clear()

    prev = None
    for i in range(len(t)):
        if not np.isfinite(g[i, 0]):
            flush()
            prev = None
            continue
        if prev is not None:
            dt = t[i] - t[prev]
            if 1e-6 < dt <= max_dt:
                v = float(np.hypot(*(g[i] - g[prev])) / dt)
                if min_speed <= v <= max_speed:
                    run.append((float(t[prev]), v, prev, i))
                else:
                    flush()
            else:
                flush()
        prev = i
    flush()
    merged = []
    for t0_, v in ev:
        if merged and t0_ - merged[-1][0] < merge_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], v))
        else:
            merged.append((t0_, v))
    return merged


def airborne_box_events(segments, length, width,
                        margin_x=25.0, margin_y=4.0):
    """공중볼 구간 중 착지점이 골문 쪽인 것 → 슛/크로스 후보.

    segments: .events.json "airborne" 의 [(i0, i1, fit)]
    (fit: p0, v, t0, T — detect_airborne_segments 결과).
    착지점 |x| 가 경기장 끝 margin_x m 안이고 |y| 가 터치라인 근방이면
    채택. 반환: [(발사 t, 착지 t), ...].
    """
    out = []
    for _i0, _i1, fit in segments or []:
        p0, v = fit.get("p0"), fit.get("v")
        t0, T = float(fit.get("t0", 0.0)), float(fit.get("T", 0.0))
        if not p0 or not v or T <= 0:
            continue
        lx, ly = p0[0] + v[0] * T, p0[1] + v[1] * T
        if abs(lx) >= length / 2.0 - margin_x \
                and abs(ly) <= width / 2.0 + margin_y:
            out.append((t0, t0 + T))
    return out


def build_highlights(duration_s, kickoffs=(), whistles=(), signals=(),
                     air_events=(), speed_events=(), user_events=(),
                     merge_gap_s=3.0, min_db=20.0, long_whistle_s=0.8,
                     weights=None):
    """이벤트 소스 → 병합·점수화된 하이라이트 후보 리스트 (t0 순).

    규칙 구간 (P03 계획 표):
      킥오프 [t−10, t+20], 기 신호 [호각−8, +7], 롱 휘슬(비킥오프)
      [t−8, +5], 공중볼→박스 [발사−4, 착지+6], 속도 급증 [t−3, +5],
      사용자 이벤트 [t−8, +8].
    킥오프 ±5s 의 호각은 킥오프 규칙이 이미 커버하므로 휘슬 규칙에서
    제외. 라벨은 최고 가중 기여 규칙 (사용자 라벨 우선).
    """
    W = dict(WEIGHTS)
    W.update(weights or {})
    raw = []                      # (t0, t1, kind, weight, label)

    def add(kind, a, b, label=None):
        a, b = max(0.0, a), min(float(duration_s), b)
        if b - a >= 1.0:
            raw.append((a, b, kind, W.get(kind, 1.0),
                        label or KIND_LABEL.get(kind, kind)))

    ko_ts = [float(k["t"]) for k in kickoffs]
    for tk in ko_ts:
        add("kickoff", tk - 10.0, tk + 20.0)
    for s in signals:
        kind = (s.get("near") or {}).get("signal")
        if kind in ("foul", "offside"):
            tw = float(s.get("whistle_t", 0.0))
            add(kind, tw - 8.0, tw + 7.0)
    for w0, w1, db in whistles:
        if db < min_db or (w1 - w0) < long_whistle_s:
            continue
        if any(abs(w0 - tk) < 5.0 for tk in ko_ts):
            continue
        add("whistle", w0 - 8.0, w1 + 5.0)
    for ta, tb in air_events:
        add("air", ta - 4.0, tb + 6.0)
    for ts, _peak in speed_events:
        add("speed", ts - 3.0, ts + 5.0)
    for tu, label in user_events:
        add("user", float(tu) - 8.0, float(tu) + 8.0, label=str(label))

    raw.sort(key=lambda r: r[0])
    out = []
    for a, b, kind, w, label in raw:
        if out and a - out[-1]["t1"] < merge_gap_s:
            h = out[-1]
            h["t1"] = max(h["t1"], b)
            h["score"] += w
            if kind not in h["kinds"]:
                h["kinds"].append(kind)
            if w > h["_w"]:
                h["_w"], h["label"] = w, label
        else:
            out.append({"t0": a, "t1": b, "kinds": [kind], "label": label,
                        "score": w, "_w": w, "state": "cand"})
    for h in out:
        h.pop("_w")
        h["t0"], h["t1"] = round(h["t0"], 2), round(h["t1"], 2)
        h["score"] = round(h["score"], 2)
    return out


def carry_states(new, old, min_frac=0.5):
    """재생성 후보에 기존 수락/제외 상태를 승계 (비파괴 재생성).

    기존 항목과의 겹침이 짧은 쪽 길이의 min_frac 이상이면 같은 장면 —
    state 를 이어받고, 수락된 항목은 사용자가 조정한 경계도 유지한다.
    새 후보와 안 겹치는 기존 수락/제외 항목은 그대로 보존.
    """
    used = set()
    out = []
    for h in new:
        best, bi = 0.0, None
        for i, o in enumerate(old or []):
            if i in used:
                continue
            ov = min(h["t1"], o["t1"]) - max(h["t0"], o["t0"])
            span = min(h["t1"] - h["t0"], o["t1"] - o["t0"])
            frac = ov / span if span > 0 else 0.0
            if frac > best:
                best, bi = frac, i
        if bi is not None and best >= min_frac:
            used.add(bi)
            o = old[bi]
            h["state"] = o.get("state", "cand")
            if h["state"] == "accept":     # 검수 완료 경계 존중
                h["t0"], h["t1"] = o["t0"], o["t1"]
        out.append(h)
    for i, o in enumerate(old or []):
        if i not in used and o.get("state") in ("accept", "reject"):
            out.append(dict(o))
    out.sort(key=lambda h: h["t0"])
    return out
