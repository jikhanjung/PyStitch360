"""오디오 분석: 호각(휘슬) 트랙 추출 — 이벤트 엔진의 청각 신호.

심판 호각은 3.2~4.8kHz 협대역 순음(피 휘슬은 비브라토 포함)이라
밴드 에너지 비율 + 협대역성(피크 우세)만으로 야외 소음에서 분리된다.
전체 타임라인을 연속 점수 트랙으로 저장하고(<video>.whistle.json),
이벤트(구간)는 트랙 위에서 히스테리시스로 파생 — 트랙 원본을 보관해
향후 다른 이벤트 작업에 재사용한다.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .encoders import ffmpeg_bin

SR = 16000
N_FFT = 1024
HOP = 512                      # 32ms — 트랙 해상도
BAND = (3200.0, 4800.0)        # 호각 대역
REF_LO = (500.0, 2800.0)       # 비교 대역 (아래)
REF_HI = (5500.0, 7500.0)      # 비교 대역 (위)


def extract_audio(video_path, sr=SR):
    """ffmpeg 로 모노 float32 파형 추출."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def whistle_track(samples, sr=SR, n_fft=N_FFT, hop=HOP):
    """전체 타임라인 호각 트랙: {hop_s, band_db, tonal}.

    band_db: 호각 대역(3.2~4.8kHz) 절대 에너지 (dB) — 원본 보관용.
    tonal: 대역 내 피크 우세(최대 빈 / 평균 빈) — 협대역성.
    실측(devlog 021): 야외 녹음은 관중·바람이 광대역을 지배해 대역 '비율'
    은 변별력이 없고, 대역 에너지의 시간축 프로미넌스(롤링 중앙값 대비)
    가 호각을 +24~31dB 로 분리한다 — 이벤트 파생은 whistle_events 에서.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < n_fft:
        return {"hop_s": hop / sr, "band_db": np.array([]),
                "tonal": np.array([])}
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    b_band = (freqs >= BAND[0]) & (freqs < BAND[1])
    win = np.hanning(n_fft)
    band_db, tonal = [], []
    chunk = 1 << 20                       # 메모리 제한: ~1M 샘플씩
    start = 0
    while start + n_fft <= len(x):
        stop = min(start + chunk, len(x))
        n = 1 + (stop - start - n_fft) // hop
        if n <= 0:
            break
        idx = np.arange(n_fft)[None] + hop * np.arange(n)[:, None] + start
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        p_band = spec[:, b_band]
        band_db.append(10.0 * np.log10(p_band.sum(axis=1) + 1e-12))
        tonal.append(p_band.max(axis=1)
                     / np.maximum(p_band.mean(axis=1), 1e-15))
        start += n * hop
    return {"hop_s": hop / sr,
            "band_db": np.concatenate(band_db) if band_db else np.array([]),
            "tonal": np.concatenate(tonal) if tonal else np.array([])}


def whistle_prominence(track, baseline_s=4.0):
    """대역 에너지의 롤링 중앙값 대비 상승분 (dB) — 호각 점수."""
    bd = np.asarray(track["band_db"], dtype=np.float64)
    if len(bd) == 0:
        return bd
    k = max(3, int(round(baseline_s / float(track["hop_s"]))) | 1)
    from numpy.lib.stride_tricks import sliding_window_view
    med = np.median(sliding_window_view(
        np.pad(bd, k // 2, mode="edge"), k)[:len(bd)], axis=1)
    return bd - med


def whistle_events(track, hi_db=15.0, lo_db=8.0, min_tonal=6.0,
                   min_dur=0.1, merge_gap=0.25, baseline_s=4.0):
    """트랙 → 호각 구간 [(t0, t1, peak_db), ...] (프로미넌스 히스테리시스).

    프로미넌스가 hi 를 넘는 곳에서 시작, lo 아래로 떨어질 때까지 확장.
    구간 피크 프레임의 협대역성(tonal)이 낮으면(광대역 충격음) 기각.
    가까운 구간(merge_gap)은 병합 — 이중 호각(삑-삑)도 한 구간.
    """
    r = whistle_prominence(track, baseline_s)
    tn = np.asarray(track["tonal"], dtype=np.float64)
    hop_s = float(track["hop_s"])
    if len(r) == 0:
        return []
    active = r >= hi_db
    weak = r >= lo_db
    events = []
    i = 0
    while i < len(r):
        if not active[i]:
            i += 1
            continue
        j0 = i
        while j0 > 0 and weak[j0 - 1]:
            j0 -= 1
        j1 = i
        while j1 + 1 < len(r) and weak[j1 + 1]:
            j1 += 1
        events.append([j0, j1])
        i = j1 + 1
    # 병합
    merged = []
    for e in events:
        if merged and (e[0] - merged[-1][1]) * hop_s <= merge_gap:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    out = []
    for j0, j1 in merged:
        if (j1 - j0 + 1) * hop_s < min_dur:
            continue
        pk = j0 + int(np.argmax(r[j0:j1 + 1]))
        if tn[pk] < min_tonal:               # 광대역 충격음(박수 등) 기각
            continue
        out.append((round(j0 * hop_s, 3), round((j1 + 1) * hop_s, 3),
                    round(float(r[pk]), 1)))
    return out


def whistle_json_path(video_path) -> Path:
    return Path(video_path).with_suffix(".whistle.json")


def save_whistle_track(video_path, track, events=None):
    """트랙 전체를 압축 저장 (0.1 단위 정수 양자화) + 파생 이벤트."""
    doc = {"hop_s": track["hop_s"],
           "band_db_x10": np.round(np.asarray(track["band_db"]) * 10)
           .astype(int).tolist(),
           "tonal_x10": np.round(np.asarray(track["tonal"]) * 10)
           .astype(int).tolist(),
           "band": list(BAND),
           "events": events if events is not None else []}
    p = whistle_json_path(video_path)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(p)
    return p


def load_whistle_track(video_path):
    """저장된 트랙 로드 (없으면 None). 반환: (track, events)."""
    p = whistle_json_path(video_path)
    if not p.exists():
        return None, []
    doc = json.loads(p.read_text())
    if "band_db_x10" not in doc:          # 구형(ratio 기반) — 재추출 필요
        return None, []
    track = {"hop_s": float(doc["hop_s"]),
             "band_db": np.asarray(doc["band_db_x10"], float) / 10.0,
             "tonal": np.asarray(doc["tonal_x10"], float) / 10.0}
    return track, [tuple(e) for e in doc.get("events", [])]
