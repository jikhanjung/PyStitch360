"""가상 PTZ: 파노라마에서 공/선수를 추적해 16:9 크롭 창을 부드럽게 이동.

PitchStitch/PitchAnalysis.py 의 가중평균 + 슬라이딩 윈도우 스무딩 아이디어를
재구현. YOLO(ultralytics) 는 선택 의존성 — 없으면 PTZ 기능만 비활성화된다.

  pip install ultralytics   (CPU torch 로 충분)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# COCO 클래스
_CLS_PERSON = 0
_CLS_BALL = 32  # sports ball

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "presets" / "yolov8n.pt"


def ptz_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


class Detector:
    """YOLO 래퍼. detect() → (N,4) [cx, cy, weight, is_ball]."""

    def __init__(self, weights: str | Path | None = None, imgsz: int = 960):
        from ultralytics import YOLO
        w = str(weights or _DEFAULT_WEIGHTS)
        self.model = YOLO(w)
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf,
                                 classes=[_CLS_PERSON, _CLS_BALL],
                                 verbose=False)[0]
        out = []
        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            is_ball = cls == _CLS_BALL
            # 공은 압도적 가중치 — 공이 보이면 공을 따라간다
            weight = c * (30.0 if is_ball else 1.0)
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, weight, float(is_ball)))
        return np.array(out) if out else np.zeros((0, 4))


class PTZSmoother:
    """감지 결과의 가중평균 목표점을 EMA + 속도 제한으로 부드럽게 추종."""

    def __init__(self, alpha: float = 0.08, max_speed_px: float = 25.0):
        self.alpha = alpha            # EMA 계수 (프레임당)
        self.max_speed = max_speed_px  # 프레임당 최대 이동 픽셀
        self.pos: np.ndarray | None = None

    def update(self, detections: np.ndarray, fallback: tuple[float, float]):
        """detections: (N,4) [cx, cy, weight, is_ball]. 반환: 부드러운 (x, y)."""
        if len(detections):
            w = detections[:, 2]
            target = (detections[:, :2] * w[:, None]).sum(axis=0) / w.sum()
        elif self.pos is not None:
            target = self.pos          # 감지 없음 → 제자리 유지
        else:
            target = np.array(fallback, dtype=np.float64)
        if self.pos is None:
            self.pos = np.array(target, dtype=np.float64)
        else:
            step = self.alpha * (np.asarray(target) - self.pos)
            n = np.linalg.norm(step)
            if n > self.max_speed:
                step *= self.max_speed / n
            self.pos = self.pos + step
        return float(self.pos[0]), float(self.pos[1])


class VirtualPTZ:
    """파노라마 프레임 → 공 추적 16:9 크롭 (기본 1920x1080).

    감지는 detect_every 프레임마다 축소본에서 수행하고,
    크롭 중심은 매 프레임 스무더로 갱신한다.
    """

    def __init__(self, pano_w: int, pano_h: int, out_w: int = 1920, out_h: int = 1080,
                 detect_every: int = 3, detect_width: int = 2944,
                 weights: str | Path | None = None):
        if pano_w < out_w or pano_h < out_h:
            raise ValueError(
                f"파노라마({pano_w}x{pano_h})가 크롭({out_w}x{out_h})보다 작음 — "
                "내보내기 해상도를 100% 로 하세요")
        self.pano_w, self.pano_h = pano_w, pano_h
        self.out_w, self.out_h = out_w, out_h
        self.detect_every = detect_every
        self.det_scale = detect_width / pano_w
        # imgsz 를 축소본 크기와 일치시켜 YOLO 내부 재축소로 인한
        # 실효 해상도 손실(공 소실의 주범)을 막는다
        self.detector = Detector(weights, imgsz=detect_width)
        self.smoother = PTZSmoother()
        self._i = 0
        self._last_det = np.zeros((0, 4))

    def process(self, pano: np.ndarray) -> np.ndarray:
        if self._i % self.detect_every == 0:
            small = cv2.resize(pano, (int(self.pano_w * self.det_scale),
                                      int(self.pano_h * self.det_scale)))
            det = self.detector.detect(small)
            if len(det):
                det[:, :2] /= self.det_scale
            self._last_det = det
        self._i += 1

        cx, cy = self.smoother.update(self._last_det,
                                      fallback=(self.pano_w / 2, self.pano_h * 0.45))
        x0 = int(round(cx - self.out_w / 2))
        y0 = int(round(cy - self.out_h / 2))
        x0 = max(0, min(x0, self.pano_w - self.out_w))
        y0 = max(0, min(y0, self.pano_h - self.out_h))
        return np.ascontiguousarray(
            pano[y0 : y0 + self.out_h, x0 : x0 + self.out_w])


# ================================================================ 2패스 PTZ
# 오프라인 내보내기용: 1패스에서 전체 검출 궤적을 모으고(전역 스무딩이
# 가능해짐), 2패스에서 크롭·인코딩한다. 실측(devlog 007): 검출 해상도를
# 2944px 로 올리면 경기장 안 공 검출률 0% → 52% (conf>=0.25).

def detect_raw(model, frame, det_w=2944, conf=0.2):
    """(N,6) [cx, cy, conf, is_ball, w, h] — 파노라마 원본 좌표계."""
    scale = det_w / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    r = model.predict(small, imgsz=det_w, conf=conf,
                      classes=[_CLS_PERSON, _CLS_BALL], verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(((x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale,
                    float(b.conf[0]), float(int(b.cls[0]) == _CLS_BALL),
                    (x2 - x1) / scale, (y2 - y1) / scale))
    return np.array(out) if out else np.zeros((0, 6))


def analyze_video(path, detect_every=3, det_w=2944, field_top_frac=0.26,
                  weights=None, far_boost=True, far_band_frac=0.58,
                  cancel=None, log=print):
    """1패스: 프레임 샘플마다 공/선수 검출. 반환 dict 는 JSON 직렬화 가능.

    field_top_frac 위(원경 트랙·관중석)의 공/선수는 장외로 버린다.
    far_boost: 원경 밴드(field_top~far_band_frac)를 원본 해상도 타일 2장으로
    잘라 공 전용 추가 검출 — 전체 프레임은 ~50% 축소라 원경 공(~10px)이
    뭉개지는 문제를 보완한다. 트래커 상태 보호를 위해 별도 모델 인스턴스 사용.
    """
    from ultralytics import YOLO
    model = YOLO(str(weights or _DEFAULT_WEIGHTS))
    model_far = YOLO(str(weights or _DEFAULT_WEIGHTS)) if far_boost else None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"열 수 없음: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    field_top = field_top_frac * pano_h
    frames_idx, balls, players = [], [], []
    ball_cands = []
    import time
    t0 = time.perf_counter()
    i = 0
    while True:
        if cancel is not None and cancel():
            cap.release()
            log("[analyze] 사용자 취소")
            return None
        ok = cap.grab()
        if not ok:
            break
        if i % detect_every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            scale = det_w / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            r = model.track(small, imgsz=det_w, conf=0.2,
                            classes=[_CLS_PERSON, _CLS_BALL],
                            persist=True, verbose=False)[0]
            far_balls = []
            if model_far is not None:
                # 정사각형 타일: 학습 크기(~640) 분포에 맞고 네이티브 해상도 유지
                y0b, y1b = int(field_top), int(pano_h * far_band_frac)
                strip = frame[y0b:y1b]
                th = strip.shape[0]
                step = max(1, int(th * 0.85))          # 15% 겹침
                offs = list(range(0, max(strip.shape[1] - th, 1), step))
                offs.append(strip.shape[1] - th)       # 우측 끝 보장
                tiles = [strip[:, x:x + th] for x in offs]
                imgsz_t = (th + 31) // 32 * 32
                results = model_far.predict(tiles, imgsz=imgsz_t, conf=0.15,
                                            classes=[_CLS_BALL], verbose=False)
                for x_off, r2 in zip(offs, results):
                    for b2 in r2.boxes:
                        bx1, by1, bx2, by2 = b2.xyxy[0].tolist()
                        far_balls.append([
                            round((bx1 + bx2) / 2 + x_off, 1),
                            round((by1 + by2) / 2 + y0b, 1),
                            round(float(b2.conf[0]), 3),
                            round(bx2 - bx1, 1), round(by2 - by1, 1)])
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            bcands = []                            # 이 샘플의 공 후보들
            prow = []
            for b_ in r.boxes:
                x1, y1, x2, y2 = b_.xyxy[0].tolist()
                cxs, cys = (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale
                conf = float(b_.conf[0])
                if cys < field_top:
                    continue                       # 장외 (원경 트랙·관중석)
                if int(b_.cls[0]) == _CLS_BALL:
                    bcands.append([round(cxs, 1), round(cys, 1), round(conf, 3),
                                   round((x2 - x1) / scale, 1),
                                   round((y2 - y1) / scale, 1)])
                    continue
                tid = int(b_.id[0]) if b_.id is not None else -1
                # 유니폼 색: 박스 상반신(위 절반, 가로 중앙 60%) HSV 평균
                tx1 = int(x1 + (x2 - x1) * 0.2)
                tx2 = int(x2 - (x2 - x1) * 0.2)
                torso = hsv[int(y1):int((y1 + y2) / 2), max(tx1, 0):max(tx2, 1)]
                hm, sm, vm = (torso.reshape(-1, 3).mean(axis=0).tolist()
                              if torso.size else (0.0, 0.0, 0.0))
                prow.append([round(cxs, 1), round(cys, 1),
                             round((x2 - x1) / scale, 1), round((y2 - y1) / scale, 1),
                             tid, round(hm, 1), round(sm, 1), round(vm, 1)])
            # 원경 네이티브 검출 병합 → 근접 중복 제거 후 상위 3개 저장.
            # 후보 다수 보존: 미끼가 conf 를 이겨도 진짜 공이 살아남아
            # 트랙 연결 단계에서 별도 트랙으로 경쟁할 수 있다.
            bcands += far_balls
            bcands.sort(key=lambda b: -b[2])
            kept = []
            for b in bcands:
                if all(np.hypot(b[0] - k[0], b[1] - k[1]) > 30 for k in kept):
                    kept.append(b)
                if len(kept) == 3:
                    break
            frames_idx.append(i)
            balls.append(kept[0] if kept else None)
            ball_cands.append(kept)
            players.append(prow)
            if len(frames_idx) % 300 == 0:
                el = time.perf_counter() - t0
                log(f"[analyze] {i}/{total} ({i/max(el,1e-9):.1f}fps)")
        i += 1
    cap.release()
    return {"video": str(path), "total_frames": i, "fps": fps,
            "players_fmt": "cxcywh_id_hsv",   # 구캐시(2/4열)와 소비부 호환
            "far_boost": bool(far_boost),
            "pano_w": pano_w, "pano_h": pano_h,
            "detect_every": detect_every, "det_w": det_w,
            "field_top_frac": field_top_frac,
            "frames": frames_idx, "balls": balls, "ball_cands": ball_cands,
            "players": players}


def classify_teams(analysis, k=3):
    """트랙릿(선수 ID)별 유니폼 색으로 팀 분류.

    분석의 선수 행이 [cx,cy,w,h,id,h,s,v] 형식일 때만 동작. 특징은
    채도로 가중한 색상 벡터 (s·cos h, s·sin h, v) 의 트랙릿 중앙값.
    farthest-point 초기화 k-means (k=3) → 검출 수 기준 상위 2개 군집이
    팀 0/1, 나머지는 2(심판·GK 등). 반환: {track_id: 팀번호}.
    """
    feats: dict[int, list] = {}
    for prow in analysis["players"]:
        for p in prow:
            if len(p) >= 8 and p[4] >= 0:
                h, s_, v = p[5], p[6], p[7]
                a = h / 90.0 * np.pi          # OpenCV H 는 0~180
                feats.setdefault(int(p[4]), []).append(
                    (s_ * np.cos(a), s_ * np.sin(a), v))
    if not feats:
        return {}
    ids = sorted(feats)
    X = np.array([np.median(feats[t], axis=0) for t in ids])
    n_det = np.array([len(feats[t]) for t in ids], float)
    k = min(k, len(ids))
    # farthest-point 초기화 (결정적)
    centers = [X[int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]]
    while len(centers) < k:
        d = np.min([np.linalg.norm(X - c, axis=1) for c in centers], axis=0)
        centers.append(X[int(np.argmax(d))])
    C = np.array(centers)
    for _ in range(20):
        lab = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    # 군집 크기(검출 수 합) 순으로 팀 번호 재배열: 0,1 = 양 팀, 2 = 기타
    sizes = [n_det[lab == j].sum() for j in range(k)]
    order = np.argsort(sizes)[::-1]
    remap = {int(order[r]): min(r, 2) for r in range(k)}
    return {t: remap[int(l)] for t, l in zip(ids, lab)}


def ground_positions(players_row, pano_w, pano_h, cam_height=4.0,
                     el_top_deg=10.0, el_bottom_deg=-38.0):
    """선수 발 위치(박스 하단 중앙)를 카메라 기준 지면 좌표(m)로 투영.

    원통 파노라마: 열 ↔ yaw 선형, 행 ↔ tan(elevation) 선형. 초점거리는
    f = H / (tan el_top − tan el_bottom) 으로 역산되므로 파라미터는 카메라
    높이뿐. 반환 X = 터치라인 방향(우+), Y = 경기장 안쪽 깊이(m).
    수평선 근처(el > −0.5°)는 기하적으로 불안정하므로 제외.

    입력 행 형식: [cx, cy, w, h, (id), (h,s,v)] — 4열 이상이면 동작.
    반환: [(X, Y, track_id, 원소 인덱스), ...]
    """
    t_top, t_bot = np.tan(np.deg2rad(el_top_deg)), np.tan(np.deg2rad(el_bottom_deg))
    f = pano_h / (t_top - t_bot)
    yaw_span = pano_w / f
    out = []
    for j, p in enumerate(players_row):
        if len(p) < 4:
            continue
        foot_x, foot_y = p[0], p[1] + p[3] / 2.0
        phi = (foot_x / max(pano_w - 1, 1) - 0.5) * yaw_span
        t = t_top + (foot_y / max(pano_h - 1, 1)) * (t_bot - t_top)
        if t > -0.0087:                  # el > -0.5도 — 수평선 근처 제외
            continue
        d = cam_height / (-t)
        tid = int(p[4]) if len(p) >= 5 else -1
        out.append((d * np.sin(phi), d * np.cos(phi), tid, j))
    return out


def same_spot_spans(linked, f0, f1, radius=60.0, static_r80=40.0):
    """기준 트랙(f0~f1 구간)과 '같은 자리'의 정적 트랙 구간 전부 반환.

    낙엽·마킹 같은 오브젝트는 한 위치에서 시간대만 다른 여러 트랙으로
    쪼개져 반복 검출된다. 기준 트랙의 중앙 위치에서 radius 이내이고
    자체 요동(r80)이 static_r80 이하인 트랙을 모두 모아, 한 번의 무시로
    일괄 처리할 수 있게 한다. 반환: [(시작, 끝) 프레임, ...] (기준 포함).
    """
    idx, tracks = linked["idx"], linked["tracks"]
    ref = [t for t in tracks if idx[t["i"][0]] <= f1 and f0 <= idx[t["i"][-1]]]
    if not ref:
        return []
    ref_med = np.median(np.vstack([t["pts"] for t in ref]), axis=0)
    out = []
    for t in tracks:
        med = np.median(t["pts"], axis=0)
        if np.hypot(*(med - ref_med)) > radius:
            continue
        r80 = (float(np.percentile(np.hypot(*(t["pts"] - med).T), 80))
               if len(t["i"]) > 1 else 0.0)
        if r80 <= static_r80:
            out.append((int(idx[t["i"][0]]), int(idx[t["i"][-1]]),
                        round(float(med[0]), 1), round(float(med[1]), 1)))
    return sorted(out)


def export_training_labels(analysis, keyframes=None, ignore_ranges=None,
                           linked=None):
    """사용자 마킹을 커스텀 공 검출 모델 학습 라벨로 변환.

    - 무시 구간 안의 공 검출 → "not_ball" (하드 네거티브 — 낙엽 등)
    - 수락 트랙의 공 검출   → "ball" (자동 양성, 약한 라벨)
    - 사용자 키프레임       → "ball_manual" (사람이 확인한 양성, 박스 없음)

    반환: [{"frame", "x", "y", "w", "h", "conf", "label"}, ...]
    원본 분석은 수정하지 않는다 (마킹은 비파괴).
    """
    idx, _, spans = accept_ball_tracks(analysis, ignore_ranges=ignore_ranges,
                                       linked=linked, log=None)
    ig = [(r[0], r[1]) for r in (ignore_ranges or [])]
    out = []
    for i, b in enumerate(analysis["balls"]):
        if b is None:
            continue
        f = int(idx[i])
        w = float(b[3]) if len(b) >= 5 else 0.0
        h = float(b[4]) if len(b) >= 5 else 0.0
        rec = {"frame": f, "x": float(b[0]), "y": float(b[1]),
               "w": w, "h": h, "conf": float(b[2])}
        if any(lo <= f <= hi for lo, hi in ig):
            rec["label"] = "not_ball"
        elif any(f0 <= f <= f1 for f0, f1 in spans):
            rec["label"] = "ball"
        else:
            continue                      # 자동 기각(불확실) — 라벨로 안 씀
        out.append(rec)
    for kf, kx, ky in (keyframes or []):
        out.append({"frame": int(kf), "x": float(kx), "y": float(ky),
                    "w": 0.0, "h": 0.0, "conf": 1.0, "label": "ball_manual"})
    out.sort(key=lambda r: r["frame"])
    return out


def _gaussian1d(x, sigma):
    """NaN 없는 1D 배열의 가우시안 스무딩 (경계는 edge 패딩, zero-phase)."""
    if sigma <= 0:
        return x.copy()
    n = int(4 * sigma) | 1
    t = np.arange(n) - n // 2
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(n // 2, x[0]), x, np.full(n // 2, x[-1])])
    return np.convolve(pad, k, "valid")


def _ball_candidates(analysis, ball_conf):
    """샘플별 공 후보 [(x, y, conf), ...] — 신형 ball_cands 우선, 구형은 balls."""
    bc = analysis.get("ball_cands")
    if bc is not None:
        return [[(p[0], p[1], p[2]) for p in row if p[2] >= ball_conf]
                for row in bc]
    return [[(b[0], b[1], b[2])] if b is not None and b[2] >= ball_conf else []
            for b in analysis["balls"]]


def link_ball_tracks(analysis, ball_conf=0.25, max_jump_per_frame=120.0):
    """공 후보 트랙 연결 (느린 단계 — 분석에만 의존하므로 캐시 가능).

    샘플당 후보가 여러 개면 conf 순으로 각자 가장 가까운 트랙에 배정
    (트랙당 샘플당 1개). 남는 후보는 새 트랙 — 미끼와 진짜 공이 같은
    시간대에 별도 트랙으로 공존하고, 수락/무시가 트랙 단위로 고른다.
    반환 dict 를 accept_ball_tracks(linked=...) 에 넘기면 수락 단계만
    다시 돌면 된다 (즉각 반응).
    """
    fps = analysis["fps"]
    idx = np.array(analysis["frames"])
    n = len(idx)
    cands = _ball_candidates(analysis, ball_conf)

    assoc_gap = 2.5 * fps               # 이 이상 끊기면 새 트랙
    tracks: list[dict] = []             # {"i": [샘플], "x": [], "y": []}
    for i in range(n):
        used = set()
        for x, y, c in sorted(cands[i], key=lambda p: -p[2]):
            best, best_d = None, None
            for ti, t in enumerate(tracks):
                if ti in used:
                    continue
                gap = idx[i] - idx[t["i"][-1]]
                if not 0 < gap <= assoc_gap:
                    continue
                # 최근 5점 평균 기준 — 검출 노이즈로 트랙이 갈라지지 않게.
                # +150px 은 빠른 이동 시 평균 지연 보상. 거리 상한 700px:
                # 긴 공백 뒤 원거리 오브젝트가 흡수되는 것을 차단.
                rx = sum(t["x"][-5:]) / len(t["x"][-5:])
                ry = sum(t["y"][-5:]) / len(t["y"][-5:])
                d = float(np.hypot(x - rx, y - ry))
                if (d <= min(max_jump_per_frame * gap, 700) + 150
                        and (best_d is None or d < best_d)):
                    best, best_d = ti, d
            if best is None:
                tracks.append({"i": [i], "x": [x], "y": [y]})
                used.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["i"].append(i)
                t["x"].append(x)
                t["y"].append(y)
                used.add(best)
    for t in tracks:
        t["i"] = np.array(t["i"])
        t["pts"] = np.column_stack([t["x"], t["y"]]).astype(float)
        del t["x"], t["y"]

    # 샘플별 선수 통계 프리컴퓨트 (공 부재 시 목표/줌 계산용 — 캐시 대상)
    p_cnt = np.zeros(n, int)
    p_tx = np.full(n, np.nan)
    p_ty = np.full(n, np.nan)
    p_span = np.zeros(n)
    for i in range(n):
        pl = np.asarray(analysis["players"][i], dtype=float)
        pl = pl[:, :2] if pl.ndim == 2 else pl.reshape(-1, 2)
        p_cnt[i] = len(pl)
        if len(pl) >= 3:
            med = np.median(pl[:, 0])
            keep = pl[np.abs(pl[:, 0] - med) < analysis.get("pano_w", 5906) * 0.25]
            p_tx[i], p_ty[i] = keep[:, 0].mean(), keep[:, 1].mean()
            p_span[i] = np.percentile(pl[:, 0], 90) - np.percentile(pl[:, 0], 10)
    return {"idx": idx, "tracks": tracks, "fps": fps,
            "p_cnt": p_cnt, "p_tx": p_tx, "p_ty": p_ty, "p_span": p_span}


def _track_iso_frac(analysis, t, iso_px):
    """트랙 점별 최근접 선수 거리 > iso_px 비율 (선수 없으면 0=근접 취급)."""
    far = 0
    for k, i in enumerate(t["i"]):
        prow = analysis["players"][int(i)]
        if not prow:
            continue
        pl = np.asarray(prow, dtype=float)[:, :2]
        if np.hypot(*(pl - t["pts"][k]).T).min() > iso_px:
            far += 1
    return far / max(len(t["i"]), 1)


def accept_ball_tracks(analysis, ball_conf=0.25, max_jump_per_frame=120.0,
                       decoy_static_px=30.0, decoy_iso_px=700.0,
                       decoy_win_sec=3.0, ignore_ranges=None, linked=None,
                       spot_radius=60.0, log=None):
    """공 트랙 수락/기각 (트랙 단위 오검출 처리).

    트랙 통계(정지/고립/중복)로 통째로 기각 — 정지 낙엽, 장외에서 움직이는
    다른 공, 순간 오검출. ignore_ranges 와 겹치는 트랙은 사용자 지정
    오인식 — 무조건 기각. linked 에 link_ball_tracks 결과를 주면 연결
    단계를 건너뛴다 (GUI 즉각 반응용).

    반환: (idx, ball(n,2), spans) — spans 는 수락 트랙의 (시작,끝) 프레임.
    """
    if linked is None:
        linked = link_ball_tracks(analysis, ball_conf, max_jump_per_frame)
    idx, tracks, fps = linked["idx"], linked["tracks"], linked["fps"]
    n = len(idx)

    def _track_stats(t):
        pts = t["pts"]
        # 정지 판정은 중앙값 기준 80% 반경 — 흡수된 이탈 샘플에 강건
        med = np.median(pts, axis=0)
        r80 = float(np.percentile(np.hypot(*(pts - med).T), 80))
        dur = (idx[t["i"][-1]] - idx[t["i"][0]]) / fps
        if "iso_frac" not in t:
            t["iso_frac"] = _track_iso_frac(analysis, t, decoy_iso_px)
        return r80, dur, t["iso_frac"]

    def _ignored(t):
        # 항목: (f0, f1) = 시간만 (구형), (f0, f1, x, y) = 시간+자리.
        # 다중 후보에서는 같은 시간대에 미끼/진짜 공 트랙이 공존하므로
        # 위치가 있으면 그 자리(150px)의 트랙만 기각한다.
        if not ignore_ranges:
            return False
        f0, f1 = idx[t["i"][0]], idx[t["i"][-1]]
        med = None
        for rng in ignore_ranges:
            lo, hi = rng[0], rng[1]
            if not (f0 <= hi and lo <= f1):
                continue
            if len(rng) >= 4:
                if med is None:
                    med = np.median(t["pts"], axis=0)
                if np.hypot(med[0] - rng[2], med[1] - rng[3]) > 150:
                    continue
            return True
        return False

    accepted: list[dict] = []
    covered = np.zeros(n, bool)
    rej = {"static": 0, "isolated": 0, "overlap": 0, "user": 0}
    # 점수: 길이 + 선수 근접 가점 (경기 공은 선수 곁에 오래 머문다)
    order = sorted(tracks, key=lambda t: -(len(t["i"]) * (2.0 - _track_stats(t)[2])))
    for t in order:
        if _ignored(t):
            rej["user"] += 1            # 사용자 지정 오인식 구간
            continue
        r80, dur, iso_frac = _track_stats(t)
        if r80 <= decoy_static_px and iso_frac > 0.5 and dur >= decoy_win_sec:
            rej["static"] += 1          # 정지 + 고립 (낙엽·마킹)
            continue
        # 실측: 진짜 공 트랙의 고립 비율은 0.00~0.04, 미끼는 0.8 안팎 —
        # 5초 이상 지속 트랙이 절반 넘게 고립돼 있으면 경기 공이 아니다
        if dur >= 5.0 and iso_frac > 0.5:
            rej["isolated"] += 1        # 선수와 무관한 물체 (낙엽·장외 공)
            continue
        lo, hi = t["i"][0], t["i"][-1]
        if covered[lo:hi + 1].mean() > 0.5:
            rej["overlap"] += 1         # 이미 수락된 트랙과 시간 중복 (경합 오검출)
            continue
        accepted.append(t)
        covered[lo:hi + 1] = True

    # 무시된 트랙들의 위치(스팟): 수락 트랙에 흡수된 같은 자리 샘플도
    # 제거해, 크롭이 오인식 지점으로 끌려가는 것을 막는다 (시간+공간 무시)
    spots = [np.median(t["pts"], axis=0) for t in tracks
             if _ignored(t) and len(t["i"]) >= 2]
    ball = np.full((n, 2), np.nan)
    dropped_spot = 0
    for t in accepted:                  # 점수 순 — 먼저 수락된 트랙이 우선
        for k, i in enumerate(t["i"]):
            if not np.isnan(ball[i, 0]):
                continue
            if spots and min(float(np.hypot(*(t["pts"][k] - sp)))
                             for sp in spots) <= spot_radius:
                dropped_spot += 1
                continue
            ball[i] = t["pts"][k]
    if log and dropped_spot:
        log(f"[plan] 무시 지점 근처 샘플 {dropped_spot}개 추가 제거")
    if log and (tracks or accepted):
        log(f"[plan] 공 트랙 {len(tracks)}개 → 수락 {len(accepted)}개 "
            f"(기각: 정지미끼 {rej['static']}, 장외고립 {rej['isolated']}, "
            f"중복 {rej['overlap']}, 사용자무시 {rej['user']})")
    spans = sorted((int(idx[t["i"][0]]), int(idx[t["i"][-1]])) for t in accepted)
    return idx, ball, spans


def build_plan(analysis, pano_w, pano_h, out_w=1920, out_h=1080,
               ball_conf=0.25, max_jump_per_frame=120.0, gap_fill_sec=2.0,
               sigma_slow=1.2, sigma_fast=0.35, fast_err_px=400.0,
               zoom_margin=260.0, top_margin=160, near_widen=1.6,
               far_zoom=1.0,
               decoy_static_px=30.0, decoy_iso_px=700.0, decoy_win_sec=3.0,
               keyframes=None, kf_suppress_sec=1.5, kf_bridge_sec=8.0,
               wide=False, ignore_ranges=None, linked=None, log=print):
    """검출 궤적 → 프레임별 (cx, cy, crop_w) 계획.

    - 공: conf 게이팅 + 점프 게이팅, gap_fill_sec 까지 선형 보간.
    - 정적 미끼 필터: decoy_win_sec 창 내내 decoy_static_px 안에 정지해 있고
      모든 선수로부터 decoy_iso_px 이상 고립된 '공'은 오검출(낙엽·마킹 등)로
      기각. 실제 공은 경기 중 장시간 정지+고립 상태가 없다 (코너킥 준비 등
      짧은 예외는 줌아웃으로 처리돼 해가 없음).
    - keyframes: [(frame_idx, x, y), ...] 사용자 지정 공 위치. 자동 검출보다
      우선한다 — 키프레임 ±kf_suppress_sec 의 자동 샘플은 버리고, 인접
      키프레임끼리는 kf_bridge_sec 까지 직접 보간으로 잇는다.
    - 공 없는 구간: 선수 중앙값 주변(트림 평균) 목표 + 줌아웃(선수 분포 커버).
    - 스무딩: 기본 sigma_slow(초). 잔차가 fast_err_px 를 넘는 구간(공이 빠르게
      이동)만 sigma_fast 추종으로 크로스페이드 — "평상시 최대한 부드럽게".
    - top_margin: 파노라마 상단 검은 스티칭 경계를 크롭에 넣지 않기 위한
      상단 여백(px). 최대 줌아웃 높이도 이만큼 줄어든다.
    - near_widen: 공이 화면 아래(근경)일수록 크롭을 넓힘 — 가까운 선수는
      크게 보이므로 타이트한 줌인이 불필요. 원경 1.0배 → 최하단 near_widen배.
    - far_zoom: 원경 공에 대한 추가 줌인 배율 (1.0=없음). 크롭 폭이
      out_w/far_zoom 까지 줄어들며 출력 시 업스케일된다 (원경은 원본
      디테일이 작아 체감 손실 미미).
    - wide=True: 감상용 와이드 모드 — 크롭 폭을 항상 최대(세로 꽉 채움)로
      고정하고 가로만 완만하게 팬. out_w/out_h 를 21:9 등으로 주고
      sigma_slow 를 크게(권장 3.0) 주면 방송 와이드샷처럼 움직인다.
    """
    fps = analysis["fps"]
    total = analysis["total_frames"]
    idx = np.array(analysis["frames"])
    n = len(idx)

    # --- 0~1. 공 트랙 수락 (accept_ball_tracks 참조) --------------------
    if linked is None:
        linked = link_ball_tracks(analysis, ball_conf, max_jump_per_frame)
    _, ball, _ = accept_ball_tracks(
        analysis, ball_conf=ball_conf, max_jump_per_frame=max_jump_per_frame,
        decoy_static_px=decoy_static_px, decoy_iso_px=decoy_iso_px,
        decoy_win_sec=decoy_win_sec, ignore_ranges=ignore_ranges,
        linked=linked, log=log)

    # --- 1.5 사용자 키프레임 병합 (자동보다 우선) -----------------------
    kf_idx = []
    if keyframes:
        sup = kf_suppress_sec * fps
        for kf_f, kx, ky in sorted(keyframes):
            near = np.abs(idx - kf_f) <= sup
            ball[near] = np.nan          # 키프레임 주변 자동 샘플 무효화
        for kf_f, kx, ky in sorted(keyframes):
            i = int(np.argmin(np.abs(idx - kf_f)))
            ball[i] = (kx, ky)
            kf_idx.append(i)

    # --- 2. 갭 보간 (짧은 가림/미검출) ----------------------------------
    known = ~np.isnan(ball[:, 0])
    gap_max = gap_fill_sec * fps
    filled = ball.copy()
    ki = np.where(known)[0]
    kf_set = set(kf_idx)
    for a, b_ in zip(ki[:-1], ki[1:]):
        gap = idx[b_] - idx[a]
        # 인접 키프레임 쌍은 더 긴 간격도 직접 보간으로 잇는다
        limit = kf_bridge_sec * fps if (a in kf_set and b_ in kf_set) else gap_max
        if 0 < gap <= limit:
            for col in (0, 1):
                filled[a:b_ + 1, col] = np.interp(idx[a:b_ + 1], [idx[a], idx[b_]],
                                                  [ball[a, col], ball[b_, col]])
    known = ~np.isnan(filled[:, 0])

    # --- 3. 목표점 + 줌 ------------------------------------------------
    max_crop_w = int(min(pano_w, (pano_h - top_margin) * out_w / out_h))
    tx = np.empty(n)
    ty = np.empty(n)
    zw = np.full(n, float(out_w))
    prev = (pano_w / 2, pano_h * 0.55)
    field_top = analysis.get("field_top_frac", 0.26) * pano_h
    p_cnt, p_tx = linked["p_cnt"], linked["p_tx"]
    p_ty, p_span = linked["p_ty"], linked["p_span"]
    for i in range(n):                      # 프리컴퓨트 덕에 경량 루프
        if known[i]:
            tx[i], ty[i] = filled[i]
            # 원경 out_w/far_zoom(추가 줌인) → 최하단 out_w*near_widen 선형 보간
            depth_t = min(max((ty[i] - field_top) / max(pano_h - field_top, 1), 0.0), 1.0)
            zw_far = out_w / max(far_zoom, 1.0)
            zw[i] = min(zw_far + (out_w * near_widen - zw_far) * depth_t, max_crop_w)
        elif p_cnt[i] >= 3:
            tx[i], ty[i] = p_tx[i], p_ty[i]
            zw[i] = min(max(p_span[i] + 2 * zoom_margin, out_w), max_crop_w)
        else:
            tx[i], ty[i] = prev
            zw[i] = max_crop_w   # 정보 부족 — 최대한 넓게
        prev = (tx[i], ty[i])

    # --- 4. 프레임별 업샘플 + 적응 스무딩 -------------------------------
    if wide:
        zw[:] = max_crop_w              # 항상 최대 폭 (줌 변동 없음)
    fr = np.arange(total)
    fx = np.interp(fr, idx, tx)
    fy = np.interp(fr, idx, ty)
    fz = np.interp(fr, idx, zw)
    slow = _gaussian1d(fx, sigma_slow * fps)
    fast = _gaussian1d(fx, sigma_fast * fps)
    err = np.abs(fx - slow)
    w = 1.0 / (1.0 + np.exp(-(err - fast_err_px) / (fast_err_px * 0.25)))
    w = _gaussian1d(w, 0.5 * fps)
    cx = (1 - w) * slow + w * fast
    cy = _gaussian1d(fy, 2.0 * fps)
    cw = _gaussian1d(fz, 2.0 * fps)
    # 키프레임 앵커: 스무딩이 뭉갠 위치를 국소 보정해 클릭 시점에는
    # 카메라가 정확히 그 지점을 보도록 보장 (가우시안 창으로 부드럽게)
    if keyframes:
        anchor_sigma = 0.8 * fps
        for kf_f, kx, ky in sorted(keyframes):
            j = int(np.clip(kf_f, 0, total - 1))
            g = np.exp(-0.5 * ((fr - j) / anchor_sigma) ** 2)
            cx = cx + (kx - cx[j]) * g
            cy = cy + (ky - cy[j]) * g
    if log:
        log(f"[plan] 공 궤적 {known.mean():.0%} (보간 포함), "
            f"빠른 추종 구간 {(w > 0.5).mean():.0%}, "
            f"줌아웃(>1.05x) {(cw > out_w * 1.05).mean():.0%}")
    return {"cx": cx, "cy": cy, "crop_w": cw, "top_margin": top_margin}


def render_plan(pano_path, out_path, plan, out_w=1920, out_h=1080,
                codec="libx264", crf=20, log=print, progress=None,
                cancel=None):
    """2패스: 계획대로 크롭(필요 시 줌아웃 다운스케일)해 인코딩. fps 반환."""
    import subprocess
    import time

    from .encoders import encoder_args, ffmpeg_bin
    cap = cv2.VideoCapture(str(pano_path))
    pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cmd = ([ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-",
            "-i", str(pano_path), "-map", "0:v", "-map", "1:a?"]
           + encoder_args(codec, crf)
           + ["-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out_path)])
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cx, cy, cw = plan["cx"], plan["cy"], plan["crop_w"]
    top_margin = int(plan.get("top_margin", 0))
    t0 = time.perf_counter()
    done = 0
    try:
        for i in range(len(cx)):
            if cancel is not None and cancel():
                log("[render] 사용자 취소")
                break
            ok, frame = cap.read()
            if not ok:
                break
            w = int(round(min(cw[i], pano_w, pano_h * out_w / out_h))) & ~1
            h = int(round(w * out_h / out_w)) & ~1
            x0 = int(round(cx[i] - w / 2))
            y0 = int(round(cy[i] - h / 2))
            x0 = max(0, min(x0, pano_w - w))
            y0 = max(min(top_margin, pano_h - h), min(y0, pano_h - h))
            crop = frame[y0:y0 + h, x0:x0 + w]
            if w != out_w:
                interp = cv2.INTER_AREA if w > out_w else cv2.INTER_CUBIC
                crop = cv2.resize(crop, (out_w, out_h), interpolation=interp)
            enc.stdin.write(np.ascontiguousarray(crop).tobytes())
            done += 1
            if done % 90 == 0 and progress is not None:
                progress(done, len(cx), done / (time.perf_counter() - t0))
            if done % 900 == 0:
                el = time.perf_counter() - t0
                log(f"[render] {done}/{len(cx)} @ {done/el:.2f}fps")
    finally:
        enc.stdin.close()
        enc.wait()
        cap.release()
    return done / max(time.perf_counter() - t0, 1e-9)
