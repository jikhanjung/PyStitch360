"""두 카메라 영상 동기화 (P06-1 실데이터 검증, Windows 실행).

전제: 두 영상 모두 scripts/whistle.py 를 먼저 실행 (.whistle.json).
공 궤적 정밀화는 두 영상 모두 분석(.analysis.json) + 경기장
캘리브레이션(.ptz.json field_points)이 있을 때만 (파노라마↔파노라마;
AX700 은 P06-3 필드 정합 후 가능).

결과는 A 영상의 .events.json "sync" 에 저장:
  {"other": B경로, "offset", "drift", "transform", "stage", ...}

사용법: python scripts/sync_cams.py <A.mp4> <B.mp4>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pystitch.core.audio import load_whistle_track  # noqa: E402
from pystitch.core.events import save_events  # noqa: E402
from pystitch.core.field import fit_field_calibration, pano_to_field  # noqa: E402
from pystitch.core.sync_multi import (  # noqa: E402
    refine_clock_by_ball, sync_by_whistles,
)


def _ball_track(pano: Path):
    """분석+캘리브레이션이 있으면 (t, 필드 xy) — 없으면 None."""
    ap = pano.with_suffix(".analysis.json")
    sp = pano.with_suffix(".ptz.json")
    if not ap.exists() or not sp.exists():
        return None
    ana = json.loads(ap.read_text())
    doc = json.loads(sp.read_text())
    if not doc.get("field_points"):
        return None
    calib = fit_field_calibration(
        doc["field_points"], ana["pano_w"], ana["pano_h"],
        length=doc.get("field_size", [105, 68])[0],
        width=doc.get("field_size", [105, 68])[1],
        line_points=doc.get("line_points"))
    if calib is None:
        return None
    t = np.asarray(ana["frames"], float) / ana["fps"]
    xy = np.full((len(t), 2), np.nan)
    for i, b in enumerate(ana["balls"]):
        if b is not None:
            g = pano_to_field(calib, [[b[0], b[1]]])[0]
            if np.isfinite(g[0]):
                xy[i] = g
    return t, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--min-db", type=float, default=15.0)
    args = ap.parse_args()
    a, b = Path(args.video_a), Path(args.video_b)
    _, ev_a = load_whistle_track(a)
    _, ev_b = load_whistle_track(b)
    if not ev_a or not ev_b:
        sys.exit("호각 트랙 없음 — 두 영상 모두 scripts/whistle.py 먼저")
    coarse = sync_by_whistles(ev_a, ev_b, min_db=args.min_db)
    if coarse is None:
        sys.exit("호각 매칭 실패 — min-db 를 낮추거나 이벤트 확인")
    ppm = (coarse["drift"] - 1.0) * 1e6
    print(f"[거친 동기화] 호각 {coarse['n']}쌍 매칭: "
          f"offset {coarse['offset']:+.3f}s, drift {ppm:+.1f}ppm, "
          f"잔차 rms {coarse['rms_s'] * 1000:.0f}ms (음속 지터 포함)")
    result = {"other": str(b), "stage": "whistle",
              "offset": round(coarse["offset"], 4),
              "drift": coarse["drift"], "n_whistles": coarse["n"],
              "rms_s": round(coarse["rms_s"], 3)}
    ta = _ball_track(a)
    tb = _ball_track(b)
    if ta is not None and tb is not None:
        r = refine_clock_by_ball(ta[0], ta[1], tb[0], tb[1], coarse)
        if r is not None:
            ck = r["clock"]
            print(f"[정밀 동기화] 공 궤적 {r['n_overlap']}샘플 겹침: "
                  f"offset {ck['offset']:+.4f}s "
                  f"(δ {r['delta'] * 1000:+.0f}ms), "
                  f"대칭 {r['transform_name']}, "
                  f"궤적 거리 중앙값 {r['rms_m']:.2f}m")
            result.update({"stage": "ball",
                           "offset": round(ck["offset"], 4),
                           "drift": ck["drift"],
                           "transform": r["transform_name"],
                           "ball_rms_m": round(r["rms_m"], 2)})
        else:
            print("[정밀 동기화] 공 궤적 겹침 부족 — 호각 결과 유지")
    else:
        print("[정밀 동기화] 생략 (양쪽 분석+캘리브레이션 필요 — "
              "AX700 은 P06-3 이후)")
    save_events(a, sync=result)
    print(f"저장: {a.with_suffix('.events.json')} \"sync\"")


if __name__ == "__main__":
    main()
