"""GoPro GPMF 텔레메트리 파싱 — 자이로 기반 충격(방향 변동) 이벤트 감지.

HERO5 Black 이후 모델은 mp4 안에 gpmd 데이터 트랙으로 자이로/가속도를 기록한다.
ffmpeg 으로 트랙을 추출한 뒤 KLV(FourCC-type-size-repeat) 구조를 파싱한다.

이벤트 감지 용도로는 밀리초 정밀도가 필요 없으므로, 샘플 시각은
"샘플 인덱스 → 영상 길이" 선형 매핑으로 근사한다 (±1초 수준).
"""
from __future__ import annotations

import struct
import subprocess
from .encoders import ffmpeg_bin
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _gpmd_stream_index(video: str) -> int | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_tag_string", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and parts[1] == "gpmd":
            return int(parts[0])
    return None


def extract_gpmf(video: str) -> bytes:
    idx = _gpmd_stream_index(video)
    if idx is None:
        raise ValueError(f"GPMF(gpmd) 트랙 없음: {video}")
    r = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(video),
         "-map", f"0:{idx}", "-c", "copy", "-f", "data", "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"GPMF 추출 실패: {r.stderr.decode(errors='replace')[:200]}")
    return r.stdout


def _iter_klv(data: bytes, offset: int = 0, end: int | None = None):
    """(fourcc, type, struct_size, repeat, payload) 반복. 컨테이너는 재귀 탐색용."""
    end = len(data) if end is None else end
    while offset + 8 <= end:
        fourcc = data[offset : offset + 4].decode("latin-1")
        typ = data[offset + 4]
        ssz = data[offset + 5]
        rep = struct.unpack(">H", data[offset + 6 : offset + 8])[0]
        size = ssz * rep
        payload = data[offset + 8 : offset + 8 + size]
        yield fourcc, typ, ssz, rep, payload, offset + 8
        offset += 8 + ((size + 3) & ~3)  # 4바이트 정렬


def _collect_streams(data: bytes, fourcc_wanted: str):
    """전체 KLV 트리에서 (SCAL, wanted 데이터) 쌍들을 수집."""
    results = []

    def walk(off, end, scal):
        local_scal = scal
        for fourcc, typ, ssz, rep, payload, poff in _iter_klv(data, off, end):
            if typ == 0:  # 컨테이너 (nested)
                walk(poff, poff + ssz * rep, local_scal)
            elif fourcc == "SCAL":
                if typ == ord("s") and ssz == 2:
                    local_scal = float(struct.unpack(">h", payload[:2])[0])
                elif typ == ord("l") and ssz == 4:
                    local_scal = float(struct.unpack(">l", payload[:4])[0])
            elif fourcc == fourcc_wanted and typ == ord("s") and ssz == 6:
                arr = np.frombuffer(payload, dtype=">i2").reshape(rep, 3).astype(np.float64)
                results.append((local_scal, arr))

    walk(0, len(data), 1.0)
    return results


def read_gyro(video: str) -> np.ndarray:
    """영상 파일에서 자이로 (N,3) rad/s 배열을 읽는다."""
    data = extract_gpmf(video)
    chunks = _collect_streams(data, "GYRO")
    if not chunks:
        raise ValueError(f"GYRO 스트림 없음: {video}")
    return np.vstack([arr / (scal if scal else 1.0) for scal, arr in chunks])


@dataclass
class GyroEvent:
    time_sec: float        # 챕터 체인 기준 절대 시각
    peak_rad_s: float      # 피크 각속도
    net_angle_deg: float   # 이벤트 창(±2s)에서의 순 회전량 — 클수록 방향이 바뀐 것
    persistent: bool       # 방향 변경으로 보이는지 (일시 흔들림이면 False)


def detect_bump_events(chapter_files: list[str], durations: list[float],
                       thresh_rad_s: float = 1.0, min_gap_sec: float = 5.0,
                       persist_deg: float = 0.5, log=print) -> list[GyroEvent]:
    """챕터 체인 전체에서 자이로 스파이크 이벤트를 찾는다.

    thresh_rad_s: 이벤트 판정 각속도 (정지 삼각대 잡음은 ≪0.1 rad/s)
    persist_deg: 이벤트 전후 순 회전량이 이보다 크면 '방향 변경'으로 표시
    """
    events: list[GyroEvent] = []
    t_base = 0.0
    for f, dur in zip(chapter_files, durations):
        try:
            gyro = read_gyro(f)
        except ValueError as e:
            log(f"[gpmf] {Path(f).name}: {e}")
            t_base += dur
            continue
        n = len(gyro)
        sr = n / dur  # 근사 샘플레이트 (~400Hz)
        mag = np.linalg.norm(gyro, axis=1)
        above = np.flatnonzero(mag > thresh_rad_s)
        log(f"[gpmf] {Path(f).name}: {n}샘플 ({sr:.0f}Hz), "
            f"잡음중앙값 {np.median(mag):.3f} rad/s, 임계 초과 {len(above)}샘플")
        # 스파이크를 min_gap 간격으로 그룹핑
        last_end = -1e9
        for i in above:
            t = i / sr
            if t - last_end < min_gap_sec:
                last_end = t
                continue
            last_end = t
            w0, w1 = max(0, int((t - 2) * sr)), min(n, int((t + 2) * sr))
            net = np.linalg.norm(gyro[w0:w1].sum(axis=0)) / sr  # ∫ω dt (rad)
            net_deg = float(np.rad2deg(net))
            peak = float(mag[w0:w1].max())
            events.append(GyroEvent(
                time_sec=t_base + t, peak_rad_s=peak,
                net_angle_deg=net_deg, persistent=net_deg > persist_deg))
        t_base += dur
    return events
