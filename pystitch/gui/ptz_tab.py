"""4번 탭: 가상 PTZ — 자동 공 검출을 기본으로 깔고 클릭으로 키프레임 교정.

워크플로우: 완성 파노라마 열기 → 자동 분석(캐시) → 타임라인을 훑으며
공이 아닌 곳을 보는 구간에서 화면 클릭(=키프레임) → PTZ 내보내기.
키프레임은 <파노라마>.ptz_keyframes.json 에 자동 저장된다.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QRect, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QCheckBox
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar,
    QPushButton, QScrollBar, QSlider, QSpinBox, QTabWidget, QVBoxLayout,
    QWidget,
)

from ..core.encoders import available_encoders
from ..core.field import (
    LANDMARKS, detect_sideline_points, field_outline, field_to_pano,
    fit_field_calibration, landmark_positions, pano_to_field,
)
from ..core.ptz import (
    accept_ball_tracks, analyze_video, build_plan, build_radar_data,
    classify_teams, gapfill_analysis, gapfill_targets, ground_positions,
    link_ball_tracks, propagate_seed, ptz_available, render_plan,
    same_spot_spans, tracklet_colors,
)
from .widgets import FramePane


# BGR — 인덱스 = 역할 번호 (core.ptz ROLE_*): 팀1, 팀2, 기타, 팀1 GK, 팀2 GK, 심판
TEAM_COLORS = [(60, 60, 230), (230, 140, 40), (160, 160, 160),
               (180, 105, 255), (230, 230, 0), (60, 200, 230)]
ROLE_NAMES = ["팀1", "팀2", "기타", "팀1 GK", "팀2 GK", "심판"]
ROLE_TAGS = {3: "GK1", 4: "GK2", 5: "REF"}
# 미리보기 오버레이용 랜드마크 약칭 (cv2 폰트는 한글 불가)
LANDMARK_TAGS = {"corner_far_l": "FL", "corner_far_r": "FR",
                 "corner_near_l": "NL", "corner_near_r": "NR",
                 "half_far": "HF", "half_near": "HN",
                 "circle_far": "CF", "circle_near": "CN",
                 "sideline_near_l": "SL", "sideline_near_r": "SR",
                 "pen_l_far": "PLF", "pen_l_near": "PLN",
                 "pen_r_far": "PRF", "pen_r_near": "PRN",
                 "pen_l_box_far": "BLF", "pen_l_box_near": "BLN",
                 "pen_r_box_far": "BRF", "pen_r_box_near": "BRN",
                 "center_near": "CM",
                 "circle_l": "CL", "circle_r": "CR"}


def _boost_bgr(bgr, s_gain=1.35, v_gain=1.55, v_floor=190):
    """그림자 보정: 유니폼 대표색을 실제 옷 색에 가깝게 밝고 진하게.

    상반신 샘플은 그림자가 섞여 실제 유니폼보다 어둡게 나온다 —
    표시용으로만 명도(V)·채도(S)를 끌어올린다.
    """
    h, s, v = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0, 0]
    s = min(int(s * s_gain), 255)
    v = min(max(int(v * v_gain), v_floor), 255)
    b, g, r = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(b), int(g), int(r))


def _hsv_hex(hsv):
    """OpenCV HSV(0~180, 0~255, 0~255) → '#rrggbb' (그림자 보정 포함)."""
    px = np.uint8([[[int(hsv[0]) % 180, int(min(hsv[1], 255)),
                     int(min(hsv[2], 255))]]])
    b, g, r = _boost_bgr(tuple(int(v) for v in
                               cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]))
    return f"#{r:02x}{g:02x}{b:02x}"


class RadarView(QWidget):
    """탑다운 레이더: 카메라 기준 지면 좌표(m)의 선수(팀 색)·공 표시."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(180)
        self.setFixedWidth(int(180 * 110 / 75))   # 기본(카메라 기준) 범위 비율
        self.points: list = []      # (X, Y, team)
        self.ball = None            # (X, Y) or None
        self.field = None           # {length, width, cam} — 캘리브레이션 후
        self.palette: dict = {}     # {역할: BGR} 유니폼 대표색 (없으면 기본)

    def set_data(self, points, ball=None):
        self.points, self.ball = list(points), ball
        self.update()

    def set_field(self, field):
        """경기장 절대 좌표 모드 on/off — 켜지면 실측 경기장을 그린다.

        위젯 폭을 표시 범위의 실제 가로세로 비율에 맞춰 고정 —
        경기장 사각형이 입력 크기(예: 100×62m) 비율 그대로 보인다.
        """
        if field != self.field:
            self.field = field
            h = self.height()
            if field:
                hl, hw = field["length"] / 2, field["width"] / 2
                camy = field["cam"][1]
                span_x = (hl + 6) * 2
                span_y = (hw + 6) - (min(camy, -hw) - 4)
            else:
                span_x, span_y = 110.0, 75.0
            self.setFixedWidth(int(round(h * span_x / span_y)))
            self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 60, 34))
        W, H = self.width(), self.height()
        if self.field:
            hl, hw = self.field["length"] / 2, self.field["width"] / 2
            camx, camy = self.field["cam"]
            xmin, xmax = -hl - 6, hl + 6
            ymin, ymax = min(camy, -hw) - 4, hw + 6
        else:
            # 캘리브레이션 전: 카메라 기준 X -55..55m, Y 0..75m
            xmin, xmax, ymin, ymax = -55, 55, 0, 75

        def px(X, Y):
            return (int((X - xmin) / (xmax - xmin) * (W - 1)),
                    int((1 - (Y - ymin) / (ymax - ymin)) * (H - 1)))

        if self.field:
            p.setPen(QColor(255, 255, 255, 150))
            for (a, b), (c, d) in [((-hl, -hw), (hl, -hw)),
                                   ((hl, -hw), (hl, hw)),
                                   ((hl, hw), (-hl, hw)),
                                   ((-hl, hw), (-hl, -hw)),
                                   ((0, -hw), (0, hw))]:
                p.drawLine(*px(a, b), *px(c, d))
            rx = int(9.15 / (xmax - xmin) * (W - 1))
            ry = int(9.15 / (ymax - ymin) * (H - 1))
            cx0, cy0 = px(0, 0)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx0 - rx, cy0 - ry, 2 * rx, 2 * ry)
            p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setPen(QColor(255, 255, 255, 45))
            for gx in range(-50, 51, 10):
                p.drawLine(*px(gx, 0), *px(gx, 75))
            for gy in range(0, 76, 10):
                p.drawLine(*px(-55, gy), *px(55, gy))
            p.setPen(QColor(255, 255, 255, 140))
            p.drawLine(*px(0, 0), *px(0, 75))        # 하프라인 방향
        for X, Y, team in self.points:
            b, g, r = self.palette.get(
                team, TEAM_COLORS[min(max(team, 0), len(TEAM_COLORS) - 1)])
            p.setBrush(QColor(r, g, b))
            p.setPen(Qt.PenStyle.NoPen)
            x, y = px(X, Y)
            p.drawEllipse(x - 4, y - 4, 8, 8)
        if self.ball is not None:
            x, y = px(*self.ball)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QColor(0, 0, 0, 180))            # 외곽선 — 배경 대비
            p.drawEllipse(x - 6, y - 6, 12, 12)
            p.setBrush(QColor(255, 140, 0))          # 공 = 주황
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(x - 5, y - 5, 10, 10)
        p.setPen(QColor(255, 255, 255, 160))
        p.drawText(6, 14, "레이더 (카메라 기준, 10m 격자)")
        p.end()


class TimelineView(QWidget):
    """NLE 스타일 멀티트랙 타임라인.

    레인: 크롭/키프레임, 공(수락=초록·무시=빨강·승격=자홍), 팀1, 팀2, 기타.
    선수 트랙릿·공 트랙·키프레임이 가로 바/박스로 그려지고 클릭으로
    선택(pick 시그널)할 수 있다. Ctrl+휠 = 커서 기준 확대축소, 휠 = 팬,
    빈 곳 드래그 = 팬, 빈 곳 클릭 = 시크. 빨간 세로선 = 현재 위치.
    """

    seek = pyqtSignal(int)
    pick = pyqtSignal(str, int)      # ("kf"|"ball"|"ignore"|"player", 키)
    range_menu = pyqtSignal(int, object)   # 우클릭: (프레임, 전역 좌표)
    view_changed = pyqtSignal(float, float, int)   # (t0, 보이는 폭, total)

    RULER = 16
    LANE_H = 20                  # 기본 레인 높이 (개별 조절 가능)
    GUTTER = 64
    LANES = ["크롭/KF", "공", "뜬 공", "팀1", "팀2", "기타", "호각", "이벤트",
             "하이라이트"]
    WHISTLE_MIN_DB = 20.0        # 이 피크 이상만 '확실한 호각'으로 표시

    def __init__(self):
        super().__init__()
        self.lanes = list(self.LANES)     # 인스턴스 사본 (팀 이름 반영)
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_timeline_lanes", None)
        try:
            self.lane_h = [max(10, min(120, int(v))) for v in saved]
            assert len(self.lane_h) <= len(self.lanes)
            # 레인이 추가된 구버전 저장값은 기본 높이로 채움
            self.lane_h += [self.LANE_H] * (len(self.lanes) - len(self.lane_h))
        except Exception:  # noqa: BLE001
            self.lane_h = [self.LANE_H] * len(self.lanes)
        self._apply_height()
        self.setMouseTracking(True)
        self.setToolTip("레이블 쪽에서 레인 경계 드래그 = 높이 조절 "
                        "(Shift+드래그 = 전체 비례)")
        self.total = 0
        self.fps = 30.0
        self.pos = 0
        self.spans: list = []
        self.ignores: list = []
        self.kfs: list = []
        self.promotes: list = []
        self._players: list = []     # [(tid, f0, f1, role, 서브행), ...]
        self.t0 = 0.0                # 보이는 시작 프레임
        self.ppf = 0.0               # 픽셀/프레임 (0 = 전체 맞춤)
        self.selected = None         # (종류, 키)
        self.mark_in = None          # 내보내기 시작 프레임 (None=미지정)
        self.mark_out = None         # 내보내기 끝 프레임
        self._whistle = None         # (hop_s, 프로미넌스 배열, 이벤트)
        self.events = []             # [(frame, label, kind)] kind: auto|user
        self.airborne = []           # [(f0, f1, apex_z)] 공중 구간
        self.highlights = []         # [(f0, f1, state, label)] 하이라이트 구간
        self.pauses = []             # [(f0, f1)] 경기 중단 (시계 정지) 구간
        self._press = None
        self._resize = None          # 레인 경계 드래그 상태

    def set_lane_names(self, team1, team2):
        self.lanes[3], self.lanes[4] = team1, team2
        self.update()

    def set_events(self, events):
        self.events = list(events)
        self.update()

    def set_airborne(self, segs):
        """공중 구간 [(f0, f1, apex_z)] — '뜬 공' 레인."""
        self.airborne = list(segs)
        self.update()

    def set_highlights(self, hls):
        """하이라이트 구간 [(f0, f1, state, label)] — 이벤트 레인 바."""
        self.highlights = list(hls)
        self.update()

    def set_pauses(self, pauses):
        """경기 중단 구간 [(f0, f1)] — 회색 세로 밴드 (시계 정지)."""
        self.pauses = list(pauses)
        self.update()

    def _emit_view(self):
        vis = (self.width() - self.GUTTER) / max(self._eff_ppf(), 1e-9)
        self.view_changed.emit(float(self.t0), float(vis), int(self.total))

    def set_view_start(self, f):
        """외부 스크롤바 → 보이는 시작 프레임 이동."""
        if abs(float(f) - self.t0) >= 1.0:
            self.t0 = float(f)
            self._clamp_view()
            self.update()

    def set_whistle(self, hop_s, prom, events):
        self._whistle = (float(hop_s), np.asarray(prom, dtype=np.float32),
                         list(events))
        self.update()

    def set_range(self, f0, f1):
        if (f0, f1) != (self.mark_in, self.mark_out):
            self.mark_in, self.mark_out = f0, f1
            self.update()

    def contextMenuEvent(self, ev):
        if self.total <= 1:
            return
        x = int(ev.pos().x())
        f = int(min(max(self._f(max(x, self.GUTTER)), 0), self.total - 1))
        self.range_menu.emit(f, ev.globalPos())

    # --------------------------------------------------------------- 데이터
    def set_data(self, total, spans, ignores, kfs, promotes=None):
        self.total = max(1, int(total))
        self.spans, self.ignores = list(spans), list(ignores)
        self.kfs = list(kfs)
        if promotes is not None:
            self.promotes = list(promotes)
        self._clamp_view()
        self.update()
        self._emit_view()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp_view()
        self._emit_view()

    def set_players(self, players):
        """{tid: (f0, f1, role)} → 레인별 서브행 배치 (겹침 회피 3행)."""
        by_lane: dict[int, list] = {}
        for tid, (f0, f1, role) in players.items():
            by_lane.setdefault(self._lane_of_role(role), []).append(
                (f0, f1, tid, role))
        out = []
        for lane, items in by_lane.items():
            ends = []                          # 서브행별 마지막 끝 프레임
            for f0, f1, tid, role in sorted(items):
                for si, e in enumerate(ends):
                    if e <= f0:
                        ends[si] = f1
                        break
                else:
                    si = len(ends) % 3
                    if len(ends) < 3:
                        ends.append(f1)
                    else:
                        ends[si] = max(ends[si], f1)
                out.append((tid, f0, f1, role, si))
        self._players = out
        self.update()

    def set_pos(self, f):
        if f != self.pos:
            self.pos = f
            self.update()

    def set_selection(self, kind, key):
        sel = (kind, key) if kind is not None else None
        if sel != self.selected:
            self.selected = sel
            self.update()

    @staticmethod
    def _lane_of_role(role):
        if role in (0, 3):
            return 3
        if role in (1, 4):
            return 4
        return 5

    # --------------------------------------------------------------- 좌표계
    def _eff_ppf(self):
        fit = (self.width() - self.GUTTER) / max(self.total, 1)
        return max(self.ppf, fit) if self.ppf > 0 else fit

    def _clamp_view(self):
        ppf = self._eff_ppf()
        vis = (self.width() - self.GUTTER) / max(ppf, 1e-9)
        self.t0 = min(max(self.t0, 0.0), max(0.0, self.total - vis))

    def _x(self, f):
        return int(self.GUTTER + (f - self.t0) * self._eff_ppf())

    def _f(self, x):
        return (x - self.GUTTER) / self._eff_ppf() + self.t0

    def _apply_height(self):
        self.setFixedHeight(self.RULER + sum(self.lane_h))

    def _lane_rect(self, lane):
        return self.RULER + sum(self.lane_h[:lane]), self.lane_h[lane]

    def _lane_at(self, y):
        acc = self.RULER
        for i, h in enumerate(self.lane_h):
            if acc <= y < acc + h:
                return i
            acc += h
        return -1

    def _boundary_at(self, y, tol=4):
        """y 가 레인 경계(하단선) 근처면 그 레인 인덱스."""
        acc = self.RULER
        for i, h in enumerate(self.lane_h):
            acc += h
            if abs(y - acc) <= tol:
                return i
        return None

    def _apply_lane_resize(self, dy):
        """진행 중인 경계 드래그 적용 (Shift = 전체 비례)."""
        i, orig = self._resize["lane"], self._resize["orig"]
        if self._resize["all"]:
            total0 = sum(orig)
            factor = max(0.3, (total0 + dy * len(orig)) / total0)
            self.lane_h = [max(10, min(120, int(round(h * factor))))
                           for h in orig]
        else:
            self.lane_h = list(orig)
            self.lane_h[i] = max(10, min(120, orig[i] + dy))
        self._apply_height()
        self.update()

    # --------------------------------------------------------------- 그리기
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(34, 34, 38))
        W = self.width()
        # 레인 배경/라벨
        for i, name in enumerate(self.lanes):
            y, lh = self._lane_rect(i)
            if i % 2 == 0:
                p.fillRect(self.GUTTER, y, W - self.GUTTER, lh,
                           QColor(255, 255, 255, 8))
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRect(4, y, self.GUTTER - 8, lh),
                       Qt.AlignmentFlag.AlignVCenter, name)
            p.setPen(QColor(255, 255, 255, 25))
            p.drawLine(0, y + lh, W, y + lh)
        # 눈금자: 60px 이상 간격이 되는 단위 선택
        if self.total > 1:
            ppf = self._eff_ppf()
            for step_s in (1, 2, 5, 10, 30, 60, 120, 300, 600):
                if step_s * self.fps * ppf >= 60:
                    break
            f = int(self.t0 // (step_s * self.fps)) * step_s * self.fps
            p.setPen(QColor(180, 180, 180))
            while f <= self.t0 + (W - self.GUTTER) / ppf:
                x = self._x(f)
                if x >= self.GUTTER:
                    p.drawLine(x, 0, x, self.RULER - 4)
                    t = f / self.fps
                    p.drawText(x + 3, self.RULER - 4,
                               f"{int(t//60):02d}:{int(t%60):02d}")
                f += step_s * self.fps
        # 레인 0: 키프레임 (3요소 = 위치, 4요소 = 크롭 박스 폭 포함)
        y, lh = self._lane_rect(0)
        for i, k in enumerate(self.kfs):
            x = self._x(k[0])
            crop = len(k) > 3
            c = QColor(255, 130, 0) if crop else QColor(255, 190, 0)
            r = (x - 4, y + 3, 9, lh - 6)
            p.fillRect(*r, c)
            if self.selected == ("kf", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        # 레인 1: 공 — 수락 트랙, 무시, 승격
        y, lh = self._lane_rect(1)
        for i, (f0, f1) in enumerate(self.spans):
            r = (self._x(f0), y + 4, max(2, self._x(f1) - self._x(f0)), lh - 8)
            p.fillRect(*r, QColor(70, 200, 90))
            if self.selected == ("ball", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        for i, rg in enumerate(self.ignores):
            r = (self._x(rg[0]), y + 4,
                 max(2, self._x(rg[1]) - self._x(rg[0])), lh - 8)
            p.fillRect(*r, QColor(220, 70, 60))
            if self.selected == ("ignore", i):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        for pr in self.promotes:
            x = self._x(pr[0])
            p.fillRect(x - 1, y + 2, 3, lh - 4, QColor(200, 0, 255))
        # 선수 레인: 팀/역할 색 바 (서브행 3개)
        for tid, f0, f1, role, si in self._players:
            y, lh = self._lane_rect(self._lane_of_role(role))
            bh = (lh - 4) // 3
            ry = y + 2 + si * bh
            b, g, rr = TEAM_COLORS[min(role, len(TEAM_COLORS) - 1)]
            r = (self._x(f0), ry, max(2, self._x(f1) - self._x(f0)), bh - 1)
            p.fillRect(*r, QColor(rr, g, b))
            if self.selected == ("player", tid):
                p.setPen(QColor(255, 255, 255))
                p.drawRect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)
        # 뜬 공 레인: 공중 구간 바 (하늘색, 정점 높이 라벨)
        if self.airborne and self.total > 1:
            y, lh = self._lane_rect(2)
            c = QColor(120, 200, 255)
            for f0, f1, apex in self.airborne:
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                p.fillRect(max(x0_, self.GUTTER), y + 4,
                           max(3, x1_ - max(x0_, self.GUTTER)), lh - 8, c)
                if x1_ - x0_ > 44:
                    p.setPen(QColor(20, 40, 60))
                    p.drawText(QRect(x0_ + 3, y, x1_ - x0_ - 4, lh),
                               Qt.AlignmentFlag.AlignVCenter,
                               f"{apex:.1f}m")
        # 호각 레인: 확실한 이벤트(피크 ≥ WHISTLE_MIN_DB)만 불연속 마커로
        # — 연속 바는 노이즈처럼 보여 제거 (원본 트랙은 파일에 보관)
        if self._whistle is not None and self.total > 1:
            _, _, events = self._whistle
            y, lh = self._lane_rect(6)
            for t0_, t1_, db in events:
                if db < self.WHISTLE_MIN_DB:
                    continue
                x_ = self._x(t0_ * self.fps)
                if not (self.GUTTER - 4 <= x_ <= W):
                    continue
                w_ = max(3, self._x(t1_ * self.fps) - x_)
                c = QColor(255, 150, 40)
                p.fillRect(x_, y + 3, w_, lh - 6, c)
                p.setPen(c)
                # 긴 호각(킥오프·종료 후보)은 위 눈금까지 강조
                if t1_ - t0_ >= 0.8:
                    p.drawLine(x_, y + 1, x_ + w_, y + 1)
        # 하이라이트 레인: 구간 바 (후보=호박, 수락=초록, 제외=회색)
        if self.highlights and self.total > 1:
            y, lh = self._lane_rect(8)
            for i, (f0, f1, state, label) in enumerate(self.highlights):
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                c = {"accept": QColor(110, 210, 70, 200),
                     "reject": QColor(130, 130, 130, 80)}.get(
                    state, QColor(255, 176, 32, 180))
                r = (max(x0_, self.GUTTER), y + 3,
                     max(3, x1_ - max(x0_, self.GUTTER)), lh - 6)
                p.fillRect(*r, c)
                if self.selected == ("hl", i):
                    p.setPen(QColor(255, 255, 255))
                    p.drawRect(r[0], r[1], r[2] - 1, r[3] - 1)
                if x1_ - x0_ > 44 and state != "reject":
                    p.setPen(QColor(35, 28, 8))
                    p.drawText(QRect(r[0] + 3, y, x1_ - r[0] - 4, lh),
                               Qt.AlignmentFlag.AlignVCenter, label)
        # 이벤트 레인: 자동(킥오프)=초록, 사용자=시안 마커 + 라벨
        if self.events and self.total > 1:
            y, lh = self._lane_rect(7)
            for i, (f_, label, kind) in enumerate(self.events):
                x_ = self._x(f_)
                if not (self.GUTTER - 4 <= x_ <= W + 4):
                    continue
                c = (QColor(90, 220, 120) if kind == "auto"
                     else QColor(80, 200, 230))
                p.fillRect(x_ - 1, y + 2, 3, lh - 4, c)
                p.setPen(c)
                if self.selected == ("event", i):
                    p.drawRect(x_ - 4, y + 1, 8, lh - 2)
                p.drawText(QRect(x_ + 5, y, 140, lh),
                           Qt.AlignmentFlag.AlignVCenter, label)
        # 경기 중단 구간: 회색 세로 밴드 (시계 정지 — hydration break 등)
        if self.pauses and self.total > 1:
            for f0, f1 in self.pauses:
                x0_, x1_ = self._x(f0), self._x(f1)
                if x1_ < self.GUTTER or x0_ > W:
                    continue
                p.fillRect(max(x0_, self.GUTTER), self.RULER,
                           max(2, x1_ - max(x0_, self.GUTTER)),
                           self.height() - self.RULER,
                           QColor(150, 150, 160, 45))
                p.setPen(QColor(170, 170, 180))
                p.drawText(max(x0_, self.GUTTER) + 3, self.RULER + 11, "II")
        # 내보내기 구간: 바깥은 어둡게, IN/OUT 브래킷 표시
        if self.mark_in is not None or self.mark_out is not None:
            fi = 0 if self.mark_in is None else self.mark_in
            fo = self.total - 1 if self.mark_out is None else self.mark_out
            xa, xb = self._x(fi), self._x(fo)
            dim = QColor(0, 0, 0, 110)
            if xa > self.GUTTER:
                p.fillRect(self.GUTTER, self.RULER,
                           xa - self.GUTTER, self.height(), dim)
            if xb < W:
                p.fillRect(xb, self.RULER, W - xb, self.height(), dim)
            p.setPen(QColor(80, 230, 120))
            p.drawLine(xa, 0, xa, self.height())
            p.drawText(xa + 3, self.height() - 4, "IN")
            p.setPen(QColor(240, 120, 80))
            p.drawLine(xb, 0, xb, self.height())
            p.drawText(xb - 26, self.height() - 4, "OUT")
        # 거터 마스크 + 플레이헤드
        p.fillRect(0, self.RULER, self.GUTTER, self.height(), QColor(34, 34, 38))
        for i, name in enumerate(self.lanes):     # 라벨 다시 (마스크 위)
            y, lh = self._lane_rect(i)
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRect(4, y, self.GUTTER - 8, lh),
                       Qt.AlignmentFlag.AlignVCenter, name)
        x = self._x(self.pos)
        if x >= self.GUTTER:
            p.setPen(QColor(255, 60, 60))
            p.drawLine(x, 0, x, self.height())
        p.end()

    # --------------------------------------------------------------- 조작
    def _hit(self, x, y):
        f = self._f(x)
        lane = self._lane_at(y)
        if lane == 0:
            best, bd = None, 8.0
            for i, k in enumerate(self.kfs):
                d = abs(self._x(k[0]) - x)
                if d < bd:
                    best, bd = ("kf", i), d
            return best
        if lane == 1:
            for i, rg in enumerate(self.ignores):
                if rg[0] <= f <= rg[1]:
                    return ("ignore", i)
            for i, (f0, f1) in enumerate(self.spans):
                if f0 <= f <= f1:
                    return ("ball", i)
            return None
        if lane in (3, 4, 5):
            ly, lh = self._lane_rect(int(lane))
            bh = (lh - 4) // 3
            for tid, f0, f1, role, si in self._players:
                if self._lane_of_role(role) != lane:
                    continue
                ry = ly + 2 + si * bh
                if f0 <= f <= f1 and ry - 1 <= y <= ry + bh:
                    return ("player", tid)
        if lane == 7:
            best, bd = None, 10.0
            for i, (f_, label, kind) in enumerate(self.events):
                d = abs(self._x(f_) - x)
                if d < bd:
                    best, bd = ("event", i), d
            return best
        if lane == 8:
            for i, (f0, f1, state, label) in enumerate(self.highlights):
                if f0 <= f <= f1:
                    return ("hl", i)
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        x, y = int(ev.position().x()), int(ev.position().y())
        if y >= self.height() - 4:          # 하단 가장자리 = 전체 비례 리사이즈
            self._resize = {"lane": len(self.lane_h) - 1, "y": y,
                            "orig": list(self.lane_h), "all": True}
            return
        if x < self.GUTTER:                 # 레이블 영역: 경계 드래그 = 높이
            b = self._boundary_at(y)
            if b is not None:
                self._resize = {"lane": b, "y": y, "orig": list(self.lane_h),
                                "all": bool(ev.modifiers()
                                            & Qt.KeyboardModifier.ShiftModifier)}
                return
        if self.total <= 1:
            return
        self._press = {"x": x, "t0": self.t0, "moved": False,
                       "hit": self._hit(x, y) if y >= self.RULER else None}

    def mouseMoveEvent(self, ev):
        x, y = int(ev.position().x()), int(ev.position().y())
        if self._resize is not None:
            self._apply_lane_resize(y - self._resize["y"])
            return
        if self._press is None:
            # 호버 커서: 하단 가장자리(전체) / 레이블 영역 경계(개별)
            on_edge = (y >= self.height() - 4
                       or (x < self.GUTTER
                           and self._boundary_at(y) is not None))
            self.setCursor(Qt.CursorShape.SizeVerCursor if on_edge
                           else Qt.CursorShape.ArrowCursor)
            return
        dx = x - self._press["x"]
        if abs(dx) > 3:
            self._press["moved"] = True
        if self._press["moved"] and self._press["hit"] is None:
            self.t0 = self._press["t0"] - dx / self._eff_ppf()   # 팬
            self._clamp_view()
            self.update()
            self._emit_view()

    def mouseReleaseEvent(self, ev):
        if self._resize is not None:
            self._resize = None
            QSettings("PyStitch360", "PyStitch360").setValue(
                "ptz_timeline_lanes", [int(v) for v in self.lane_h])
            return
        pr, self._press = self._press, None
        if pr is None or pr["moved"]:
            return
        if pr["hit"] is not None:
            self.selected = pr["hit"]
            self.update()
            self.pick.emit(*pr["hit"])
        elif ev.position().x() >= self.GUTTER:
            self.seek.emit(int(min(max(self._f(int(ev.position().x())), 0),
                                   self.total - 1)))

    def wheelEvent(self, ev):
        if self.total <= 1:
            return
        delta = ev.angleDelta().y() / 120.0
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            x = int(ev.position().x())
            f_at = self._f(max(x, self.GUTTER))
            fit = (self.width() - self.GUTTER) / self.total
            self.ppf = min(max(self._eff_ppf() * (1.25 ** delta), fit), 30.0)
            self.t0 = f_at - (max(x, self.GUTTER) - self.GUTTER) / self.ppf
        else:
            self.t0 -= delta * 80 / self._eff_ppf()
        self._clamp_view()
        self.update()
        self._emit_view()
        ev.accept()


class AnalyzeWorker(QThread):
    done = pyqtSignal(dict)
    progress = pyqtSignal(int, int, float)   # done_frames, total, fps
    log = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, weights=None, checkpoint_path=None):
        super().__init__()
        self.video_path = video_path
        self.weights = weights
        self.checkpoint_path = checkpoint_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            d = analyze_video(self.video_path, weights=self.weights,
                              cancel=lambda: self._cancel,
                              progress=lambda i, t, f: self.progress.emit(i, t, f),
                              checkpoint_path=self.checkpoint_path,
                              log=lambda s: self.log.emit(s))
            if d is None:
                self.failed.emit("취소됨")
            else:
                self.done.emit(d)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class LinkWorker(QThread):
    """트랙 연결(느린 단계) 백그라운드 계산 — 분석에만 의존하므로 1회면 됨."""

    done = pyqtSignal(dict)

    def __init__(self, analysis):
        super().__init__()
        self.analysis = analysis

    def run(self):
        linked = link_ball_tracks(self.analysis)
        linked["teams"] = classify_teams(self.analysis)
        self.done.emit(linked)


class GapfillWorker(QThread):
    """갭필 2차 패스: 트랙 갭 보간 위치의 저문턱 타일 검출 (공+선수)."""

    progress = pyqtSignal(int, int, float)
    done = pyqtSignal(dict, int, int)     # analysis, n_ball, n_person
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pano_path, analysis, targets, weights=None):
        super().__init__()
        self.args = (pano_path, analysis, targets, weights)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, analysis, targets, weights = self.args
        try:
            nb, np_ = gapfill_analysis(
                pano, analysis, targets, weights=weights,
                progress=lambda i, t, f: self.progress.emit(i, t, f),
                cancel=lambda: self._cancel,
                log=lambda s: self.log.emit(s))
            self.done.emit(analysis, nb, np_)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class AirborneWorker(QThread):
    """공중볼 스캔 (correct_ball_track) — UI 프리즈 방지용 백그라운드."""

    done = pyqtSignal(int, dict, list, str)  # gen, 캐시, 구간(직렬화), 로그

    def __init__(self, gen, frames, fps, acc, calib, segments=None):
        super().__init__()
        self.args = (gen, frames, fps, acc, calib, segments)

    def run(self):
        gen, frames, fps, acc, calib, segments = self.args
        try:
            from ..core.airborne import correct_ball_track
            from ..core.field import pano_to_field
            fin = np.isfinite(acc[:, 0])
            if fin.sum() < 10:
                self.done.emit(gen, {}, [], "")
                return
            g = np.full((len(acc), 2), np.nan)
            g[fin] = pano_to_field(calib, acc[fin])
            t = frames / fps
            corr, z, segs = correct_ball_track(
                t, g, (calib["ex"], calib["ey"]), calib["h"],
                segments=segments)
            cache = {}
            for i0, i1, fit in segs:
                for si in range(i0, i1 + 1):
                    cache[si] = (float(corr[si, 0]), float(corr[si, 1]),
                                 float(z[si]))
            msg = ""
            if segs:
                cached = " (캐시 재사용)" if segments is not None else ""
                msg = (f"[air] 공중 구간 {len(segs)}개 (정점 최대 "
                       f"{max(f['apex_z'] for _, _, f in segs):.1f}m)"
                       + cached)
            self.done.emit(gen, cache,
                           [[int(i0), int(i1), f] for i0, i1, f in segs],
                           msg)
        except Exception as e:  # noqa: BLE001
            self.done.emit(gen, {}, [], f"[air] 공중볼 보정 실패: {e}")


class SeedWorker(QThread):
    """시드 전파: 수동 공/선수 인식을 앞뒤 샘플로 확장 (propagate_seed)."""

    done = pyqtSignal(str, list, object)   # kind, matches, ctx(tid 등)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pano_path, analysis, f0, x0, y0, kind,
                 weights=None, ctx=None):
        super().__init__()
        self.args = (pano_path, analysis, f0, x0, y0, kind, weights, ctx)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pano, analysis, f0, x0, y0, kind, weights, ctx = self.args
        try:
            m = propagate_seed(pano, analysis, f0, x0, y0, kind=kind,
                               weights=weights,
                               cancel=lambda: self._cancel,
                               log=lambda s: self.log.emit(s))
            self.done.emit(kind, m, ctx)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PlanWorker(QThread):
    """크롭 계획 미리보기 재계산 (키프레임/무시 변경 시 백그라운드)."""

    done = pyqtSignal(dict, tuple)   # plan, (out_w, out_h)

    def __init__(self, analysis, keyframes, ignores, wide, linked=None,
                 far_zoom=1.0, promotes=None):
        super().__init__()
        self.args = (analysis, keyframes, ignores, wide, linked, far_zoom,
                     promotes or [])

    def run(self):
        analysis, kfs, ignores, wide, linked, far_zoom, promotes = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes, linked=linked,
                              far_zoom=far_zoom,
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0, log=None)
            self.done.emit(plan, (out_w, out_h))
        except Exception:  # noqa: BLE001
            pass


class ProxyWorker(QThread):
    """스크럽 프록시 생성: 1920px, 키프레임 0.5s 간격 → 즉시 탐색용."""

    progress = pyqtSignal(float)         # 0.0 ~ 1.0
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, proxy_path, duration_sec=0.0):
        super().__init__()
        self.pano_path, self.proxy_path = str(pano_path), str(proxy_path)
        self.duration = duration_sec

    def run(self):
        import subprocess

        from ..core.encoders import available_encoders, ffmpeg_bin
        encs = available_encoders().values()
        if "h264_nvenc" in encs:
            venc = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr",
                    "-cq", "27", "-b:v", "0"]
        else:
            venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]
        tmp = self.proxy_path + ".part.mp4"
        cmd = ([ffmpeg_bin(), "-y", "-v", "error", "-nostats",
                "-i", self.pano_path, "-vf", "scale=1920:-2"] + venc
               + ["-g", "15", "-an", "-progress", "pipe:1", tmp])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            tail = []
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms=") and self.duration > 0:
                    try:                     # ffmpeg 의 out_time_ms 는 마이크로초
                        t = int(line.split("=", 1)[1]) / 1e6
                        self.progress.emit(min(t / self.duration, 1.0))
                    except ValueError:
                        pass
                elif line and "=" not in line:
                    tail.append(line)        # 진행 키가 아닌 줄 = 오류 메시지 후보
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(" / ".join(tail[-3:]) or "ffmpeg 실패")
            Path(tmp).rename(self.proxy_path)
            self.finished_ok.emit(self.proxy_path)
        except Exception as e:  # noqa: BLE001
            Path(tmp).unlink(missing_ok=True)
            self.failed.emit(str(e))


class PtzPlayWorker(QThread):
    """파노라마 순차 재생 (자체 디코더) — 오버레이는 GUI 쪽 _redraw 가 담당."""

    frame_ready = pyqtSignal(object, int)

    def __init__(self, path, start_frame, fps, display_fps=15.0):
        super().__init__()
        self.path, self.start_frame = str(path), start_frame
        self.fps, self.display_fps = fps, display_fps
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import time
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            step = max(1, int(round(self.fps / self.display_fps)))
            f = self.start_frame
            t0 = time.perf_counter()
            shown = 0
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_ready.emit(frame, f)
                shown += 1
                lag = shown * step / self.fps - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
                for _ in range(step - 1):
                    cap.grab()
                f += step
        finally:
            cap.release()


class PtzRenderWorker(QThread):
    progress = pyqtSignal(int, int, float)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_path, analysis, keyframes, codec, crf,
                 wide=False, ignores=None, far_zoom=1.0, promotes=None,
                 radar=None, start=0, end=None, clock=None):
        super().__init__()
        self.args = (pano_path, out_path, analysis, keyframes, codec, crf, wide,
                     ignores or [], far_zoom, promotes or [], radar, start, end,
                     clock)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        (pano, out, analysis, kfs, codec, crf, wide, ignores, far_zoom,
         promotes, radar, start, end, clock) = self.args
        try:
            out_w, out_h = (2560, 1080) if wide else (1920, 1080)
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              wide=wide, ignore_ranges=ignores,
                              force_ranges=promotes,
                              far_zoom=far_zoom,
                              sigma_slow=3.0 if wide else 1.2,
                              fast_err_px=800.0 if wide else 400.0,
                              log=lambda s: self.log.emit(s))
            render_plan(pano, out, plan, out_w=out_w, out_h=out_h,
                        codec=codec, crf=crf,
                        log=lambda s: self.log.emit(s),
                        progress=lambda d, t, f: self.progress.emit(d, t, f),
                        cancel=lambda: self._cancel, radar=radar,
                        start=start, end=end, clock=clock)
            if self._cancel:
                self.failed.emit("취소됨")
            else:
                self.finished_ok.emit(str(out))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportDialog(QDialog):
    """PTZ 내보내기 설정 대화창 — 구간(IN/OUT 마커 기준)·모드·코덱·미니맵."""

    def __init__(self, parent, total, fps, export_range, mode_idx,
                 encoders, crf, radar_on, default_dir, default_stem,
                 clock_on=None):
        super().__init__(parent)
        self.setWindowTitle("PTZ 내보내기")
        self.fps = fps
        self.total = total
        self.export_range = export_range      # (f0, f1) 정규화됨 or None
        self.default_dir = default_dir
        self.default_stem = default_stem
        form = QFormLayout(self)

        self.combo_range = QComboBox()
        f0, f1 = export_range if export_range else (0, total)
        dur = (f1 - f0) / fps
        if export_range:
            self.combo_range.addItem(
                f"마커 구간  {self._hms(f0/fps)} ~ {self._hms(f1/fps)} "
                f"(길이 {self._hms(dur)}, {f1-f0}프레임)")
        self.combo_range.addItem(
            f"전체  0:00:00 ~ {self._hms(total/fps)} ({total}프레임)")
        form.addRow("구간", self.combo_range)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.setCurrentIndex(mode_idx)
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("모드", self.combo_mode)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(list(encoders))
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_export_codec", "")
        labels = list(encoders)
        if saved in labels:                       # 직전 선택 기억
            self.combo_codec.setCurrentIndex(labels.index(saved))
        else:                                     # GPU 있으면 hevc_nvenc 기본
            for idx, lbl in enumerate(labels):
                if encoders[lbl] == "hevc_nvenc":
                    self.combo_codec.setCurrentIndex(idx)
                    break
        form.addRow("코덱", self.combo_codec)

        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=crf)
        form.addRow("CRF/CQ", self.spin_crf)

        self.check_radar = QCheckBox("우하단 반투명 탑다운 미니맵 (선수·공)")
        self.check_radar.setChecked(radar_on)
        form.addRow("미니맵", self.check_radar)

        self.check_clock = QCheckBox("좌상단 경기 시계 (분:초 누적, "
                                     "골1/골2 이벤트로 스코어)")
        if clock_on is None:
            self.check_clock.setEnabled(False)
            self.check_clock.setToolTip(
                "분석 메뉴 \"경기 정보\"에서 킥오프 앵커를 지정하세요")
        else:
            self.check_clock.setChecked(bool(clock_on))
        form.addRow("경기 시계", self.check_clock)

        path_row = QHBoxLayout()
        self.edit_path = QLineEdit(self._default_path())
        path_row.addWidget(self.edit_path, 1)
        b = QPushButton("찾아보기...")
        b.clicked.connect(self._browse)
        path_row.addWidget(b)
        form.addRow("출력 파일", path_row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("내보내기")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(560, self.sizeHint().height())

    @staticmethod
    def _hms(sec):
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{s:04.1f}"

    def _default_path(self):
        wide = self.combo_mode.currentIndex() == 1
        part = ""
        if self.export_range and self.combo_range.currentIndex() == 0:
            part = f"_{int(self.export_range[0]/self.fps)}s"
        return str(Path(self.default_dir)
                   / f"{self.default_stem}{part}"
                     f"{'_wide' if wide else '_ptz'}.mp4")

    def _mode_changed(self, _):
        self.edit_path.setText(self._default_path())

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "PTZ 출력 파일",
                                           self.edit_path.text(),
                                           "MP4 (*.mp4)")
        if p:
            self.edit_path.setText(p)

    def config(self):
        """선택 결과: {start, end, wide, codec_name, crf, radar, path}."""
        use_marks = self.export_range and self.combo_range.currentIndex() == 0
        f0, f1 = (self.export_range if use_marks else (0, self.total))
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_export_codec", self.combo_codec.currentText())
        return {"start": int(f0), "end": int(f1),
                "wide": self.combo_mode.currentIndex() == 1,
                "codec_name": self.combo_codec.currentText(),
                "crf": self.spin_crf.value(),
                "radar": self.check_radar.isChecked(),
                "clock": self.check_clock.isChecked(),
                "path": self.edit_path.text().strip()}


class MatchInfoDialog(QDialog):
    """경기 정보 — 시계 앵커(킥오프)·전/후반·하프 길이·중단 구간 요약.

    시계는 축구 관례의 분:초 누적 표기 (후반 30분 하프면 30:00부터,
    연장 포함 90/120분도 분 단위로 계속 커진다).
    """

    def __init__(self, parent, fps, total, kickoffs, info, cur_frame):
        super().__init__(parent)
        self.setWindowTitle("경기 정보")
        info = info or {}
        self.fps = fps
        form = QFormLayout(self)

        self.combo_half = QComboBox()
        self.combo_half.addItems(["전반", "후반"])
        self.combo_half.setCurrentIndex(1 if info.get("half", 1) == 2 else 0)
        form.addRow("이 영상의 하프", self.combo_half)

        self.spin_len = QDoubleSpinBox(minimum=5.0, maximum=90.0,
                                       value=float(info.get("half_len_min",
                                                            45.0)))
        self.spin_len.setSuffix(" 분")
        self.spin_len.setDecimals(0)
        form.addRow("하프 길이", self.spin_len)

        self.combo_anchor = QComboBox()
        self._anchor_frames: list = []
        for k in kickoffs:
            self.combo_anchor.addItem(
                f"킥오프 {ExportDialog._hms(k['t'])} (신뢰도 {k['score']:.2f})")
            self._anchor_frames.append(int(k["t"] * fps))
        self.combo_anchor.addItem(
            f"현재 프레임 {ExportDialog._hms(cur_frame / fps)}")
        self._anchor_frames.append(int(cur_frame))
        self.combo_anchor.addItem("지정 안 함 (시계 사용 불가)")
        self._anchor_frames.append(None)
        saved = info.get("anchor_f")
        if saved is not None:
            best = min(range(len(self._anchor_frames) - 1),
                       key=lambda i: abs(self._anchor_frames[i] - saved),
                       default=None)
            if best is not None \
                    and abs(self._anchor_frames[best] - saved) > 2 * fps:
                # 저장된 앵커가 목록에 없음 — 그대로 보존하는 항목 추가
                self.combo_anchor.insertItem(
                    0, f"저장된 앵커 {ExportDialog._hms(saved / fps)}")
                self._anchor_frames.insert(0, int(saved))
                best = 0
            self.combo_anchor.setCurrentIndex(best if best is not None else 0)
        else:
            self.combo_anchor.setCurrentIndex(len(self._anchor_frames) - 1)
        form.addRow("킥오프 앵커 (시계 0점)", self.combo_anchor)

        self.check_cum = QCheckBox("후반 시계에 전반 시간 누적 "
                                   "(하프 길이만큼 더해 30:00/45:00부터)")
        self.check_cum.setChecked(bool(info.get("cumulative", True)))
        form.addRow("", self.check_cum)

        n_pause = len(info.get("pauses") or [])
        form.addRow("중단 구간", QLabel(
            f"{n_pause}개 — 타임라인 우클릭 \"IN/OUT → 경기 중단\"으로 "
            "추가/삭제 (중단 동안 시계 정지)"))
        form.addRow("스코어", QLabel(
            "사용자 이벤트 라벨 \"골1\"/\"골2\" = 팀1/팀2 득점으로 집계 "
            "→ 시계 옆에 표시"))

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._pauses = [list(p) for p in info.get("pauses") or []]

    def config(self) -> dict:
        return {"half": self.combo_half.currentIndex() + 1,
                "half_len_min": float(self.spin_len.value()),
                "anchor_f": self._anchor_frames[
                    self.combo_anchor.currentIndex()],
                "cumulative": self.check_cum.isChecked(),
                "pauses": self._pauses}


class HighlightBatchWorker(QThread):
    """수락 하이라이트 구간들을 개별 클립으로 순차 렌더 (계획 1회 재사용)."""

    progress = pyqtSignal(int, int, float)   # (전체 누적, 전체 프레임, fps)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(int, str)       # (완료 개수, 출력 폴더)
    failed = pyqtSignal(str)

    def __init__(self, pano_path, out_dir, stem, analysis, keyframes, codec,
                 crf, ignores=None, far_zoom=1.0, promotes=None, radar=None,
                 segments=(), clock=None):
        super().__init__()
        self.args = (pano_path, out_dir, stem, analysis, keyframes, codec,
                     crf, ignores or [], far_zoom, promotes or [], radar,
                     list(segments), clock)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        (pano, out_dir, stem, analysis, kfs, codec, crf, ignores, far_zoom,
         promotes, radar, segs, clock) = self.args
        try:
            out_w, out_h = 1920, 1080
            plan = build_plan(analysis, analysis["pano_w"], analysis["pano_h"],
                              out_w=out_w, out_h=out_h, keyframes=kfs,
                              ignore_ranges=ignores, force_ranges=promotes,
                              far_zoom=far_zoom,
                              log=lambda s: self.log.emit(s))
            total = sum(e - s for s, e, _ in segs)
            base = 0
            for k, (s, e, name) in enumerate(segs, 1):
                if self._cancel:
                    break
                out = Path(out_dir) / f"{stem}_h{k:02d}_{name}.mp4"
                self.log.emit(f"[hl] {k}/{len(segs)} 렌더: {out.name}")
                render_plan(pano, out, plan, out_w=out_w, out_h=out_h,
                            codec=codec, crf=crf,
                            log=lambda s: self.log.emit(s),
                            progress=lambda d, _t, f, b=base:
                                self.progress.emit(b + d, total, f),
                            cancel=lambda: self._cancel, radar=radar,
                            start=s, end=e, clock=clock)
                base += e - s
            if self._cancel:
                self.failed.emit("취소됨")
            else:
                self.finished_ok.emit(len(segs), str(out_dir))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class HighlightExportDialog(QDialog):
    """하이라이트 일괄 내보내기 — 구간 체크 목록·출력 폴더·코덱/CRF/미니맵."""

    def __init__(self, parent, highlights, fps, encoders, crf, radar_on,
                 default_dir, clock_on=None):
        super().__init__(parent)
        self.setWindowTitle("하이라이트 일괄 내보내기")
        self.fps = fps
        form = QFormLayout(self)

        self.list = QListWidget()
        any_accept = any(h.get("state") == "accept" for h in highlights)
        for h in highlights:
            dur = h["t1"] - h["t0"]
            it = QListWidgetItem(
                f"{ExportDialog._hms(h['t0'])} ~ {ExportDialog._hms(h['t1'])}"
                f"  ({dur:.0f}초)  {h.get('label', '')}"
                f"  점수 {h.get('score', 0):.1f}"
                + ("  [수락]" if h.get("state") == "accept" else ""))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # 수락 항목이 하나라도 있으면 수락만 체크, 없으면 전부 체크
            it.setCheckState(Qt.CheckState.Checked
                             if h.get("state") == "accept" or not any_accept
                             else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.setMinimumHeight(180)
        form.addRow("구간", self.list)

        dir_row = QHBoxLayout()
        self.edit_dir = QLineEdit(default_dir)
        dir_row.addWidget(self.edit_dir, 1)
        b = QPushButton("찾아보기...")
        b.clicked.connect(self._browse)
        dir_row.addWidget(b)
        form.addRow("출력 폴더", dir_row)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(list(encoders))
        saved = QSettings("PyStitch360", "PyStitch360").value(
            "ptz_export_codec", "")
        labels = list(encoders)
        if saved in labels:
            self.combo_codec.setCurrentIndex(labels.index(saved))
        else:
            for idx, lbl in enumerate(labels):
                if encoders[lbl] == "hevc_nvenc":
                    self.combo_codec.setCurrentIndex(idx)
                    break
        form.addRow("코덱", self.combo_codec)

        self.spin_crf = QSpinBox(minimum=10, maximum=35, value=crf)
        form.addRow("CRF/CQ", self.spin_crf)
        self.check_radar = QCheckBox("우하단 반투명 탑다운 미니맵 (선수·공)")
        self.check_radar.setChecked(radar_on)
        form.addRow("미니맵", self.check_radar)
        self.check_clock = QCheckBox("좌상단 경기 시계 (분:초 누적)")
        if clock_on is None:
            self.check_clock.setEnabled(False)
            self.check_clock.setToolTip(
                "분석 메뉴 \"경기 정보\"에서 킥오프 앵커를 지정하세요")
        else:
            self.check_clock.setChecked(bool(clock_on))
        form.addRow("경기 시계", self.check_clock)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("일괄 내보내기")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(640, self.sizeHint().height())

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "출력 폴더",
                                             self.edit_dir.text())
        if d:
            self.edit_dir.setText(d)

    def config(self):
        """선택 결과: {indices, dir, codec_name, crf, radar}."""
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_export_codec", self.combo_codec.currentText())
        idx = [i for i in range(self.list.count())
               if self.list.item(i).checkState() == Qt.CheckState.Checked]
        return {"indices": idx, "dir": self.edit_dir.text().strip(),
                "codec_name": self.combo_codec.currentText(),
                "crf": self.spin_crf.value(),
                "radar": self.check_radar.isChecked(),
                "clock": self.check_clock.isChecked()}


class PtzTab(QWidget):
    """가상 PTZ 탭. log_fn 은 메인 윈도우 로그 박스."""

    def __init__(self, log_fn, video_dir_fn=None, remember_dir_fn=None):
        super().__init__()
        self.log = log_fn
        self._video_dir = video_dir_fn or (lambda: "")
        self._remember_dir = remember_dir_fn or (lambda p: None)
        self.pano_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.total = 0
        self.pano_w = self.pano_h = 0
        self.analysis: dict | None = None
        self.keyframes: list[list] = []   # [frame, x, y]
        self.ignores: list[list] = []     # [f0, f1] 사용자 지정 오인식 구간
        self.promotes: list[list] = []    # [f, x, y] 회색 공 → 트랙 강제 수락
        self.export_range: list = [None, None]   # 내보내기 IN/OUT 프레임
        self.team_names = ["팀1", "팀2"]  # 사용자 입력 팀 이름
        self.user_events: list[list] = []  # [[frame, label]] 사용자 이벤트
        self.highlights: list[dict] = []   # .events.json "highlights" 문서
        self.match_info: dict | None = None  # 경기 정보 (하프·앵커·중단)
        self.roles: dict[int, int] = {}   # {track_id: 역할} 사용자 지정 (GK 등)
        self.merges: dict[int, int] = {}  # {tid: 대표 tid} 트랙릿 병합 (비파괴)
        self.field_points: dict[str, list] = {}   # {랜드마크키: [x, y]}
        self.line_points: list[list] = []  # 흰 선 검출 샘플 [x, y] (사이드라인)
        self.field_size = [105.0, 68.0]   # 경기장 길이×폭 (m)
        self._field_calib = None          # fit_field_calibration 결과
        self.extra_players: dict[int, list] = {}  # {샘플si: [[cx,cy,w,h,id]]}
        self._next_extra_id = 900001      # 수동 검출 ID (분석 ID와 분리)
        self._adhoc = None                # 주변 재검출용 YOLO 캐시
        self._native_cap = None           # 프록시 표시 중 원본 프레임용
        self.track_spans: list = []
        self._pcache_id = None            # 선수 요약 캐시 기준 분석 객체 id
        self._pspans: dict = {}           # {tid: [f0, f1, 검출수]}
        self._pcolors: dict = {}          # {tid: (h, s, v)} 유니폼 대표색
        self._accepted_ball = None        # accept_ball_tracks 의 샘플별 수락 공
        self._air: dict[int, tuple] = {}  # {si: (X, Y, z)} 공중볼 보정 캐시
        self._hover = None                # 커서가 가리키는 오브젝트
        self._hover_key = None            # hover 변경 감지 키
        self._plan_box = None             # 현재 프레임에 그려진 크롭 박스 (x0,y0,w,h)
        self._box_hover = None            # 크롭 박스 hover 존: ("corner",i)|("border",None)
        self._box_edit = None             # 진행 중인 박스 드래그 상태
        self._box_commit = None           # 커밋 직후 낙관적 박스 (f, cx, cy, w)
        self._lm_drag = None              # 드래그 중인 랜드마크 키
        self._analyze_worker = None
        self._gapfill_worker = None
        self._seed_worker = None
        self._air_worker = None
        self._air_gen = 0
        self._render_worker = None
        self._plan_worker = None
        self._link_worker = None
        self._linked = None
        self._teams = {}
        self._role_colors: dict[int, tuple] = {}   # 역할별 유니폼 대표색(BGR)
        self.kit_colors: dict[int, list] = {}      # 사용자 지정 표시 색(BGR)
        self._play_worker = None
        self._playing = False
        self._audio = None                # QMediaPlayer (지연 초기화, False=불가)
        self._audio_out = None
        self._proxy_worker = None
        self.disp_path = None
        self.disp_scale = 1.0
        self.plan = None
        self.plan_out = (1920, 1080)
        self._build_ui()
        self._plan_timer = QTimer(singleShot=True, interval=150)
        self._plan_timer.timeout.connect(self._run_plan)
        self._save_timer = QTimer(singleShot=True, interval=1500)
        self._save_timer.timeout.connect(self._write_sidecar)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_open = QPushButton("파노라마 영상 열기...")
        self.btn_open.clicked.connect(self._open_pano)
        self.lbl_file = QLabel("완성된 파노라마 mp4 를 열어주세요")
        self.btn_analyze = QPushButton("자동 공/선수 분석")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_analyze)
        self.btn_proxy = QPushButton("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setToolTip("1920px·조밀 키프레임 사본 — 타임라인 클릭 즉시 표시")
        self.btn_proxy.clicked.connect(self._make_proxy)
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        top.addWidget(QLabel("모델"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n (기본·빠름)", "yolov8s", "yolov8m",
                                   "yolo11n", "yolo11s", "yolo11m (정확)",
                                   "사용자 .pt 파일..."])
        self.combo_model.setToolTip(
            "공/선수 검출 모델. GPU 추론이라 큰 모델도 부담 적음 —\n"
            "정확도가 아쉬우면 yolo11s/m 권장. 이름 모델은 최초 1회 자동 다운로드.")
        saved_model = QSettings("PyStitch360", "PyStitch360").value("ptz_model", 0)
        self.combo_model.setCurrentIndex(int(saved_model))
        self.combo_model.currentIndexChanged.connect(self._model_changed)
        top.addWidget(self.combo_model)
        top.addWidget(self.btn_proxy)
        top.addWidget(self.btn_analyze)
        v.addLayout(top)

        # 영상은 전체 폭 사용, 목록·레이더는 하단 스트립으로
        self.pane = FramePane("클릭 = 오브젝트 조작 / 빈 곳 = 키프레임, 우클릭 = 메뉴",
                              interactive=True)
        self.pane.clicked.connect(self._pane_clicked)
        self.pane.context_requested.connect(self._pane_context)
        self.pane.hover.connect(self._pane_hover)
        self.pane.pressed.connect(self._pane_pressed)
        self.pane.drag_moved.connect(self._pane_dragged)
        self.pane.released.connect(self._pane_released)
        v.addWidget(self.pane, 1)

        tl = QHBoxLayout()
        # 컴팩트 트랜스포트: 2행 그리드 — 1행 재생/트랙 이동 + 시각,
        # 2행 스텝 버튼 6개. 타임라인 왼쪽에 좁게 붙는다.
        grid = QGridLayout()
        grid.setSpacing(2)

        def _tbtn(text, tip, slot, row, col, colspan=1):
            b = QPushButton(text)
            b.setFixedWidth(30 * colspan + 2 * (colspan - 1))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            grid.addWidget(b, row, col, 1, colspan)
            return b

        self.btn_play = _tbtn("▶", "재생/정지 (Space)", self._toggle_play, 0, 0)
        _tbtn("|◀", "이전 공 트랙으로", lambda: self._jump_track(-1), 0, 1)
        _tbtn("▶|", "다음 공 트랙으로", lambda: self._jump_track(1), 0, 2)
        # 빈 곳 클릭 모드: 공(수동 공 위치, 줌 자동) / KF(크롭 키프레임)
        self.btn_mode_ball = _tbtn("공", "빈 곳 클릭 = 수동 공 위치 (줌 자동)",
                                   lambda: self._set_click_mode(True), 0, 3)
        self.btn_mode_kf = _tbtn("KF", "빈 곳 클릭 = 크롭 키프레임 "
                                 "(현재 계획 폭 고정)",
                                 lambda: self._set_click_mode(False), 0, 4)
        for b in (self.btn_mode_ball, self.btn_mode_kf):
            b.setCheckable(True)
        self.btn_mode_ball.setChecked(True)
        self.btn_mute = _tbtn("🔊", "재생 소리 켬/끔",
                              self._toggle_mute, 0, 5)
        self.btn_mute.setCheckable(True)
        self.btn_mute.setChecked(
            QSettings("PyStitch360", "PyStitch360")
            .value("ptz_muted", "false") == "true")
        if self.btn_mute.isChecked():
            self.btn_mute.setText("🔇")
        for col, (text, tip, d) in enumerate(
                [("≪", "-10초", -300), ("<", "-1초", -30), ("‹", "-1프레임", -1),
                 ("›", "+1프레임", 1), (">", "+1초", 30), ("≫", "+10초", 300)]):
            _tbtn(text, tip, lambda _, dd=d: self._step(dd), 1, col)
        self.lbl_time = QLabel("-:--:--.- / -:--:--")
        self.lbl_time.setToolTip("현재 위치 / 전체 길이")
        grid.addWidget(self.lbl_time, 2, 0, 1, 6,
                       Qt.AlignmentFlag.AlignCenter)
        grid.setRowStretch(3, 1)          # 남는 세로 공간은 아래로
        tl.addLayout(grid)
        # 멀티트랙 타임라인은 슬라이더 바로 위, 같은 폭
        self.trackbar = TimelineView()
        self.trackbar.seek.connect(lambda f: self.slider.setValue(f))
        self.trackbar.pick.connect(self._timeline_pick)
        self.trackbar.range_menu.connect(self._timeline_menu)
        # I/O = 내보내기 구간 시작/끝 (NLE 관례)
        QShortcut(QKeySequence(Qt.Key.Key_I), self,
                  activated=lambda: self._set_export_mark("in",
                                                          self.slider.value()))
        QShortcut(QKeySequence(Qt.Key.Key_O), self,
                  activated=lambda: self._set_export_mark("out",
                                                          self.slider.value()))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._stop_play)
        self.slider.valueChanged.connect(self._on_slider)
        bar_col = QVBoxLayout()
        bar_col.setSpacing(1)
        bar_col.addWidget(self.trackbar)
        # 줌인 시 가로 스크롤바 (전체 보기에선 숨김)
        self.tl_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.tl_scroll.setMaximumHeight(12)
        self.tl_scroll.hide()
        self.tl_scroll.valueChanged.connect(
            lambda v: self.trackbar.set_view_start(v))
        self.trackbar.view_changed.connect(self._tl_view_changed)
        bar_col.addWidget(self.tl_scroll)
        bar_col.addWidget(self.slider)
        tl.addLayout(bar_col, 1)
        v.addLayout(tl)
        self._slider_timer = QTimer(singleShot=True, interval=120)
        self._slider_timer.timeout.connect(self._show_frame)

        # 하단 스트립: [공 | 선수] 탭 + 레이더 (로그 위 공간 활용)
        strip = QHBoxLayout()
        w_ball = QWidget()
        ball_strip = QHBoxLayout(w_ball)
        ball_strip.setContentsMargins(0, 2, 0, 0)
        col_ball = QVBoxLayout()
        col_ball.addWidget(QLabel("공 — 자동 트랙 + 수동 지정 (↑↓=이동, →/Del=오인식으로)"))
        self.track_list = QListWidget()
        self.track_list.setMaximumHeight(150)
        # 클릭·키보드 화살표 선택 모두에서 이동 (currentRowChanged 는 둘 다 발생)
        self.track_list.currentRowChanged.connect(lambda _: self._goto_track())
        for key in (Qt.Key.Key_Right, Qt.Key.Key_Delete):   # → 또는 Del = 오인식
            QShortcut(QKeySequence(key), self.track_list,
                      activated=lambda: self._ignore_selected_track(
                          advance=True),   # 키보드 검수: 다음 항목으로 이동
                      context=Qt.ShortcutContext.WidgetShortcut)
        col_ball.addWidget(self.track_list, 1)
        ball_strip.addLayout(col_ball, 2)

        # 두 목록 사이 이동 버튼: >> = 오인식으로, << = 복원
        col_move = QVBoxLayout()
        col_move.addStretch(1)
        self.btn_ignore = QPushButton("≫")
        self.btn_ignore.setMaximumWidth(44)
        self.btn_ignore.setToolTip("선택한 공 트랙을 오인식으로 (→/Del)")
        self.btn_ignore.clicked.connect(self._to_ignore)
        col_move.addWidget(self.btn_ignore)
        self.btn_restore = QPushButton("≪")
        self.btn_restore.setMaximumWidth(44)
        self.btn_restore.setToolTip("선택한 오인식을 복원 (←)")
        self.btn_restore.clicked.connect(self._delete_kf)
        col_move.addWidget(self.btn_restore)
        col_move.addStretch(1)
        ball_strip.addLayout(col_move)

        col_ig = QVBoxLayout()
        col_ig.addWidget(QLabel("오인식 — 공 아님 (더블클릭=이동, ←=복원)"))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(150)
        self.kf_list.itemDoubleClicked.connect(self._goto_kf)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.kf_list,
                  activated=self._delete_kf,
                  context=Qt.ShortcutContext.WidgetShortcut)
        col_ig.addWidget(self.kf_list, 1)
        ball_strip.addLayout(col_ig, 2)

        # 선수 탭: 트랙릿 목록(유니폼 색 스와치) + 역할 지정 + 팀 색 범례
        w_players = QWidget()
        pl = QVBoxLayout(w_players)
        pl.setContentsMargins(0, 2, 0, 0)
        # 유니폼 색 범례: 역할별 스와치 버튼(클릭=컬러피커) + 인원수
        legend = QHBoxLayout()
        legend.setSpacing(4)
        self.lbl_team_colors = QLabel("팀 색: 분석 후 표시")
        legend.addWidget(self.lbl_team_colors)
        self._kit_btns, self._kit_lbls = {}, {}
        for r in (0, 1, 3, 4, 5):
            b = QPushButton()
            b.setFixedSize(26, 18)
            b.setToolTip(f"{self._role_name(r)} 표시 색 — 클릭: 색 선택, "
                         "우클릭: 측정색으로 되돌리기")
            b.clicked.connect(lambda _=False, rr=r: self._pick_kit_color(rr))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda _pos, rr=r: self._reset_kit_color(rr))
            lb = QLabel("")
            self._kit_btns[r], self._kit_lbls[r] = b, lb
            b.hide()
            legend.addWidget(b)
            legend.addWidget(lb)
        legend.addStretch(1)
        pl.addLayout(legend)
        self.player_list = QListWidget()
        self.player_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.player_list.currentRowChanged.connect(
            lambda _: self._goto_player())
        self.player_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.player_list.customContextMenuRequested.connect(self._player_menu)
        pl.addWidget(self.player_list, 1)
        rl = QHBoxLayout()
        rl.addWidget(QLabel("선택한 트랙릿 →"))
        self._role_btns = {}
        for r in (3, 4, 5, 0, 1):
            b = QPushButton(self._role_name(r))
            b.setMaximumWidth(90)
            b.clicked.connect(lambda _, rr=r: self._assign_selected_role(rr))
            self._role_btns[r] = b
            rl.addWidget(b)
        b = QPushButton("팀 이름...")
        b.setMaximumWidth(72)
        b.clicked.connect(self._edit_team_names)
        rl.addWidget(b)
        b = QPushButton("자동")
        b.setMaximumWidth(56)
        b.setToolTip("사용자 지정을 지우고 색 기반 자동 분류로")
        b.clicked.connect(lambda: self._assign_selected_role(None))
        rl.addWidget(b)
        rl.addStretch(1)
        pl.addLayout(rl)

        # 경기장 탭: 랜드마크 자동 매칭 찍기 + 캘리브레이션 상태
        w_field = QWidget()
        fl = QVBoxLayout(w_field)
        fl.setContentsMargins(0, 2, 0, 0)
        self.lbl_field_status = QLabel(
            "목록 순서대로(최외곽 선부터) 찍는 걸 권장 — 클릭=자동 매칭, "
            "우클릭=수동 지정, 드래그=이동. 외곽 6~7점이면 외곽선이 잡힙니다.")
        fl.addWidget(self.lbl_field_status)
        self.field_list = QListWidget()
        fl.addWidget(self.field_list, 1)
        fr = QHBoxLayout()
        self.btn_field_pick = QPushButton("랜드마크 찍기")
        self.btn_field_pick.setCheckable(True)
        self.btn_field_pick.setToolTip(
            "켜면 미리보기 클릭이 랜드마크 지정이 됩니다 (휴리스틱 자동 매칭)")
        self.btn_field_pick.toggled.connect(self._field_pick_toggled)
        fr.addWidget(self.btn_field_pick)
        b = QPushButton("선택 지우기")
        b.clicked.connect(self._field_clear_selected)
        fr.addWidget(b)
        b = QPushButton("모두 지우기")
        b.setToolTip("찍은 랜드마크 전체 삭제")
        b.clicked.connect(self._field_clear_all)
        fr.addWidget(b)
        b = QPushButton("흰 선 정밀화")
        b.setToolTip("예측 사이드라인 주변의 흰 픽셀을 검출해 "
                     "가까운 사이드라인을 실측 선에 맞춤")
        b.clicked.connect(self._refine_sideline)
        fr.addWidget(b)
        b = QPushButton("흰 선 취소")
        b.setToolTip("흰 선 정밀화 샘플을 지우고 랜드마크만으로 재피팅")
        b.clicked.connect(self._clear_line_points)
        fr.addWidget(b)
        fr.addWidget(QLabel("경기장(m)"))
        self.spin_field_len = QDoubleSpinBox()
        self.spin_field_len.setRange(80.0, 130.0)
        self.spin_field_len.setValue(105.0)
        self.spin_field_len.setSuffix(" L")
        self.spin_field_w = QDoubleSpinBox()
        self.spin_field_w.setRange(40.0, 90.0)
        self.spin_field_w.setValue(68.0)
        self.spin_field_w.setSuffix(" W")
        for sp in (self.spin_field_len, self.spin_field_w):
            sp.valueChanged.connect(self._field_size_changed)
            fr.addWidget(sp)
        fr.addStretch(1)
        fl.addLayout(fr)

        self.tabs_review = QTabWidget()
        self.tabs_review.addTab(w_ball, "공")
        self.tabs_review.addTab(w_players, "선수")
        self.tabs_review.addTab(w_field, "경기장")
        self.tabs_review.setMaximumHeight(210)
        self.tabs_review.currentChanged.connect(self._review_tab_changed)
        strip.addWidget(self.tabs_review, 5)

        col_radar = QVBoxLayout()
        self.radar = RadarView()
        col_radar.addWidget(self.radar)
        self.check_players = QCheckBox("선수 표시 (팀 색)")
        self.check_players.setChecked(True)
        self.check_players.toggled.connect(lambda _: self._redraw())
        col_radar.addWidget(self.check_players)
        strip.addLayout(col_radar, 1)
        v.addLayout(strip)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("출력"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["공 추적 PTZ (1920×1080)",
                                  "와이드 감상 (2560×1080, 완만한 팬)"])
        self.combo_mode.currentIndexChanged.connect(lambda _: self._plan_dirty())
        bottom.addWidget(self.combo_mode)
        lbl_fz = QLabel("원경 줌")
        lbl_fz.setToolTip("공이 반대편(원경)에 있을 때 추가 줌인 배율 (1.0=없음)")
        bottom.addWidget(lbl_fz)
        self.spin_far_zoom = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=1.5,
                                            singleStep=0.05)
        self.spin_far_zoom.setValue(float(QSettings("PyStitch360", "PyStitch360")
                                          .value("ptz_far_zoom", 1.0)))
        self.spin_far_zoom.valueChanged.connect(self._far_zoom_changed)
        bottom.addWidget(self.spin_far_zoom)
        self.encoders = available_encoders()
        self.btn_export = QPushButton("PTZ 내보내기...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._start_render)
        bottom.addWidget(self.btn_export)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_render)
        bottom.addWidget(self.btn_cancel)
        bottom.addStretch(1)
        v.addLayout(bottom)
        self.progress = QProgressBar()
        v.addWidget(self.progress)

    # ------------------------------------------------------------ 파일/분석
    def _open_pano(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "완성 파노라마 영상", self._video_dir(), "영상 (*.mp4 *.MP4 *.mkv)")
        if path:
            self.open_path(path)

    def open_path(self, path: str, quiet: bool = False):
        """파노라마 열기 (프로젝트 복원 경로 포함). 분석/키프레임 사이드카 자동 로드."""
        if not Path(path).exists():
            from ..core.project import _cross_platform_candidates
            for cand in _cross_platform_candidates(path):
                if Path(cand).exists():
                    path = cand
                    break
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if not quiet:
                QMessageBox.warning(self, "열기 실패", path)
            else:
                self.log(f"[ptz] 파노라마 없음 — 건너뜀: {path}")
            return
        if self.cap is not None:
            self.cap.release()
        if self._native_cap is not None:
            self._native_cap.release()
            self._native_cap = None
        self.cap = cap
        self.pano_path = Path(path)
        self._remember_dir(str(self.pano_path.parent))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.trackbar.fps = self.fps
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.lbl_file.setText(f"{self.pano_path.name} — {self.pano_w}x{self.pano_h}, "
                              f"{self.total/self.fps/60:.1f}분")
        with self._busy(f"호각 트랙 읽기"):
            try:
                from ..core.audio import load_whistle_track, \
                    whistle_prominence
                tr, ev = load_whistle_track(self.pano_path)
                if tr is not None:
                    self.trackbar.set_whistle(tr["hop_s"],
                                              whistle_prominence(tr), ev)
                    self.log(f"[ptz] 호각 트랙 로드: 이벤트 {len(ev)}개")
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 호각 트랙 무시: {e}")
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(0)
        self._use_display_source()
        self.btn_analyze.setEnabled(ptz_available())
        if not ptz_available():
            self.btn_analyze.setToolTip("ultralytics 미설치 (pip install ultralytics)")
        self.analysis = None
        self._load_sidecar()
        with self._busy("목록/이벤트 표시 갱신"):
            self._apply_team_names()
            self._refresh_events()
            self._show_frame()
        self._update_export_enabled()

    def _refresh_events(self):
        """자동(킥오프)·사용자 이벤트 + 하이라이트(.events.json) → 타임라인."""
        items = []
        doc = {}
        try:
            from ..core.events import load_events_doc
            doc = load_events_doc(self.pano_path)
        except Exception as e:  # noqa: BLE001
            self.log(f"[ptz] 이벤트 파일 무시: {e}")
        for k in doc.get("kickoffs", []):
            items.append((int(k["t"] * self.fps),
                          "킥오프" + (" ●" if k.get("long_whistle")
                                     else ""), "auto"))
        for f, label in self.user_events:
            items.append((int(f), label, "user"))
        items.sort()
        self.trackbar.set_events(items)
        self.highlights = doc.get("highlights", [])
        self._refresh_highlight_lane()
        self.trackbar.set_pauses(
            (self.match_info or {}).get("pauses") or [])

    def _refresh_highlight_lane(self):
        self.trackbar.set_highlights(
            [(int(h["t0"] * self.fps), int(h["t1"] * self.fps),
              h.get("state", "cand"), h.get("label", ""))
             for h in self.highlights])

    def _save_highlights(self):
        try:
            from ..core.events import save_events
            save_events(self.pano_path, highlights=self.highlights)
        except Exception as e:  # noqa: BLE001
            self.log(f"[hl] 저장 실패: {e}")
        self._refresh_highlight_lane()

    def _proxy_path(self) -> Path:
        return self.pano_path.with_suffix(".scrub.mp4")

    def _use_display_source(self):
        """표시/재생용 소스 선택: 프록시가 있으면 프록시 (즉시 탐색)."""
        proxy = self._proxy_path()
        if proxy.exists():
            cap = cv2.VideoCapture(str(proxy))
            if cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                self.cap = cap
                self.disp_path = proxy
                pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.disp_scale = pw / max(self.pano_w, 1)
                self.btn_proxy.setText("프록시 사용 중")
                self.btn_proxy.setEnabled(False)
                self.log(f"[ptz] 스크럽 프록시 사용: {proxy.name} (즉시 탐색)")
                return
        self.disp_path = self.pano_path
        self.disp_scale = 1.0
        self.btn_proxy.setText("스크럽 프록시 생성")
        self.btn_proxy.setEnabled(True)

    def _make_proxy(self):
        if self.pano_path is None:
            return
        self.btn_proxy.setEnabled(False)
        self.btn_proxy.setText("프록시 생성 중...")
        self.log("[ptz] 스크럽 프록시 생성 중 (NVENC 가용 시 수 분)...")
        w = ProxyWorker(self.pano_path, self._proxy_path(),
                        duration_sec=self.total / max(self.fps, 1e-9))
        w.progress.connect(self._proxy_progress)
        w.finished_ok.connect(lambda _: (self.log("[ptz] 프록시 완료"),
                                         self._use_display_source(),
                                         self._show_frame()))
        w.failed.connect(lambda m: (self.log(f"[오류] 프록시: {m}"),
                                    self.btn_proxy.setText("스크럽 프록시 생성"),
                                    self.btn_proxy.setEnabled(True)))
        self._proxy_worker = w
        w.start()

    def _proxy_progress(self, p: float):
        self.btn_proxy.setText(f"프록시 생성 {p:.0%}")
        # 내보내기가 진행바를 쓰고 있지 않을 때만 진행바에도 표시
        if self._render_worker is None or not self._render_worker.isRunning():
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(p * 1000))
            self.progress.setFormat(f"프록시 생성 %p%")

    def _sidecar_path(self) -> Path:
        """사용자 에디트 사이드카: {keyframes, ignores, promotes}."""
        return self.pano_path.with_suffix(".ptz.json")

    def _analysis_path(self) -> Path:
        """순수 분석 결과 (분석 완료 시 1회 기록, 검수로는 불변)."""
        return self.pano_path.with_suffix(".analysis.json")

    def _kf_path(self) -> Path:                 # 구버전 (마이그레이션용)
        return self.pano_path.with_suffix(".ptz_keyframes.json")

    def _load_sidecar(self):
        """에디트(.ptz.json)와 순수 분석(.analysis.json)을 분리 로드.

        구 통합본(.ptz.json 안에 analysis 포함)은 두 파일로 쪼개 이전한다.
        검수 마킹은 비파괴라 통합본의 analysis 키가 곧 순수 원본이다.
        .analysis.json 이 사이드카보다 새로우면(외부 분석) 그쪽을 채택.
        """
        self.keyframes, self.ignores, self.promotes, self.analysis = \
            [], [], [], None
        self.roles = {}
        self.field_points = {}
        self.line_points = []
        self.extra_players = {}
        self.kit_colors = {}
        self.user_events = []
        self.match_info = None
        self.merges = {}
        sp = self._sidecar_path()
        doc = None
        if sp.exists():
            try:
                with self._busy("편집 사이드카 읽기 (.ptz.json)"):
                    doc = json.loads(sp.read_text())
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 사이드카 무시: {e}")
        if doc is not None:
            self.keyframes = [list(k) for k in doc.get("keyframes", [])]
            self.ignores = [list(r) for r in doc.get("ignores", [])]
            self.promotes = [list(p) for p in doc.get("promotes", [])]
            self.roles = {int(t): int(r)
                          for t, r in (doc.get("roles") or {}).items()}
            self.field_points = {k: [float(v[0]), float(v[1])]
                                 for k, v in
                                 (doc.get("field_points") or {}).items()}
            self.line_points = [list(p) for p in doc.get("line_points", [])]
            self.extra_players = {int(si): [list(p) for p in rows]
                                  for si, rows in
                                  (doc.get("extra_players") or {}).items()}
            self.kit_colors = {int(r): [int(v) for v in c] for r, c in
                               (doc.get("kit_colors") or {}).items()}
            er = doc.get("export_range")
            self.export_range = ([None if v is None else int(v) for v in er]
                                 if er else [None, None])
            r = self._norm_export_range()
            self.trackbar.set_range(*(r if r else (None, None)))
            tn = doc.get("team_names")
            if tn and len(tn) == 2:
                self.team_names = [str(tn[0]), str(tn[1])]
            self.user_events = [[int(e[0]), str(e[1])]
                                for e in doc.get("user_events", [])]
            self.match_info = doc.get("match_info")
            self.merges = {int(t): int(r) for t, r in
                           (doc.get("merges") or {}).items()}
            ids = [int(p[4]) for rows in self.extra_players.values()
                   for p in rows]
            self._next_extra_id = max(ids, default=900000) + 1
            fs = doc.get("field_size")
            if fs:
                self.field_size = [float(fs[0]), float(fs[1])]
                for sp, v in ((self.spin_field_len, fs[0]),
                              (self.spin_field_w, fs[1])):
                    sp.blockSignals(True)
                    sp.setValue(float(v))
                    sp.blockSignals(False)
            self.analysis = doc.get("analysis")   # 구 통합본 (이전 대상)
        migrated = self.analysis is not None
        ap = self._analysis_path()
        if ap.exists() and (self.analysis is None or
                            ap.stat().st_mtime > sp.stat().st_mtime):
            try:
                with self._busy("분석 읽기 (.analysis.json, 수 MB)"):
                    self.analysis = json.loads(ap.read_text())
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 분석 파일 무시: {e}")
        if doc is None and self._kf_path().exists():
            try:
                d = json.loads(self._kf_path().read_text())
                if isinstance(d, list):
                    self.keyframes = [list(k) for k in d]
                else:
                    self.keyframes = [list(k) for k in d.get("keyframes", [])]
                    self.ignores = [list(r) for r in d.get("ignores", [])]
                migrated = True
            except Exception as e:  # noqa: BLE001
                self.log(f"[ptz] 구형 키프레임 무시: {e}")
        # 분석 유효성 (프레임 수 허용 오차)
        if self.analysis is not None:
            if abs(self.analysis.get("total_frames", 0) - self.total) \
                    > max(60, self.total // 100):
                self.log("[ptz] 사이드카 분석이 다른 영상 기준 — 무시")
                self.analysis = None
        if migrated:
            self._write_analysis()
            self._write_sidecar()
            self.log("[ptz] 사이드카 분리: 분석(.analysis.json) / "
                     "에디트(.ptz.json)")
        if self.analysis is not None:
            self.log(f"[ptz] 분석 불러옴 (검출 샘플 "
                     f"{len(self.analysis['frames'])}개), 키프레임 "
                     f"{len(self.keyframes)}개, 무시 {len(self.ignores)}개")
            self._start_link()
        self._refresh_lists()
        self._refresh_team_label()
        self._refresh_player_list()
        self._refit_field()
        self._refresh_field_list()

    def _write_sidecar(self):
        """사용자 에디트만 저장 — 분석과 분리돼 있어 작고 즉시 쓴다."""
        if self.pano_path is None:
            return
        sp = self._sidecar_path()
        tmp = sp.with_suffix(".ptz.json.tmp")
        tmp.write_text(json.dumps({"keyframes": self.keyframes,
                                   "ignores": self.ignores,
                                   "promotes": self.promotes,
                                   "roles": self.roles,
                                   "field_points": self.field_points,
                                   "field_size": self.field_size,
                                   "line_points": self.line_points,
                                   "extra_players": self.extra_players,
                                   "kit_colors": self.kit_colors,
                                   "export_range": self.export_range,
                                   "team_names": self.team_names,
                                   "user_events": self.user_events,
                                   "match_info": self.match_info,
                                   "merges": {str(t): r for t, r in
                                              self.merges.items()}}))
        tmp.replace(sp)

    def _write_analysis(self):
        """순수 분석 결과 저장 — 분석 완료/이전 시에만, 검수로는 안 바뀜."""
        if self.pano_path is None or self.analysis is None:
            return
        with self._busy("분석 저장 (.analysis.json, 수 MB)"):
            ap = self._analysis_path()
            tmp = Path(str(ap) + ".tmp")
            tmp.write_text(json.dumps(self.analysis))
            tmp.replace(ap)

    _MODEL_NAMES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", None]

    def _model_changed(self, i):
        st = QSettings("PyStitch360", "PyStitch360")
        if self._MODEL_NAMES[i] is None:      # 사용자 .pt
            path, _ = QFileDialog.getOpenFileName(
                self, "YOLO 가중치 (.pt)", "", "PyTorch 가중치 (*.pt)")
            if path:
                st.setValue("ptz_model_custom", path)
            else:
                self.combo_model.setCurrentIndex(0)
                return
        st.setValue("ptz_model", i)

    def _model_weights(self):
        """선택된 모델의 가중치 경로/이름 (None=내장 기본 yolov8n)."""
        name = self._MODEL_NAMES[self.combo_model.currentIndex()]
        if name is None:
            custom = QSettings("PyStitch360", "PyStitch360").value("ptz_model_custom", "")
            return str(custom) if custom else None
        if name == "yolov8n.pt":
            return None                        # presets/yolov8n.pt (내장)
        local = Path(__file__).resolve().parents[2] / "presets" / name
        return str(local) if local.exists() else name   # 없으면 자동 다운로드

    def _run_analyze(self):
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            self._analyze_worker.cancel()          # 버튼이 취소 역할
            self.log("[ptz] 분석 취소 요청...")
            return
        if self.pano_path is None:
            return
        self.btn_analyze.setText("분석 취소")
        self.progress.setRange(0, 0)
        self.progress.setFormat("분석 중... (로그 참조)")
        weights = self._model_weights()
        self.log(f"[ptz] 자동 분석 시작 (모델: {weights or 'yolov8n(내장)'}) — "
                 "진행은 로그에 표시")
        ckpt = str(self.pano_path.with_suffix(".analysis.part.json"))
        w = AnalyzeWorker(str(self.pano_path), weights=weights,
                          checkpoint_path=ckpt)
        w.progress.connect(self._analyze_progress)
        w.log.connect(self.log)
        w.done.connect(self._analyze_done)
        w.failed.connect(self._analyze_failed)
        self._analyze_worker = w
        w.start()

    def _analyze_progress(self, done, total, fps):
        if self._render_worker is not None and self._render_worker.isRunning():
            return                              # 내보내기가 진행바 사용 중
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(
            f"분석 %p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _analyze_done(self, d):
        self.analysis = d
        self._write_analysis()
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("분석 완료")
        n_ball = sum(1 for b in d["balls"] if b is not None)
        self.log(f"[ptz] 분석 완료: 샘플 {len(d['frames'])}개, "
                 f"공 검출 {n_ball}개 ({n_ball/max(len(d['frames']),1):.0%}) → 캐시 저장")
        self._update_export_enabled()
        self._start_link()
        self._show_frame()

    def _analyze_failed(self, msg):
        self.btn_analyze.setText("자동 공/선수 분석")
        self.btn_analyze.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log(f"[오류] 분석: {msg}")

    # ------------------------------------------------------------ 타임라인/표시
    def _step(self, d):
        self._stop_play()
        if self.slider.isEnabled():
            self.slider.setValue(self.slider.value() + d)

    @staticmethod
    def _hms(sec, tenth=False):
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return (f"{int(h)}:{int(m):02d}:{s:04.1f}" if tenth
                else f"{int(h)}:{int(m):02d}:{int(s):02d}")

    @contextmanager
    def _busy(self, msg):
        """동기(UI 차단) 작업 래퍼: 시작 로그 + 웨이트 커서 + 완료 로그."""
        self.log(f"[작업] {msg}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()          # 커서·로그 즉시 반영
        t0 = time.perf_counter()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            el = time.perf_counter() - t0
            if el >= 0.5:                     # 짧은 작업은 완료 로그 생략
                self.log(f"[작업] {msg} 완료 ({el:.1f}s)")

    def _on_slider(self, _):
        # 재생 중 사용자 시크(워커 갱신이 아님) → 그 지점부터 재생 계속
        if self._playing and not getattr(self, "_play_sync", False):
            self._restart_play_at(self.slider.value())
            return
        self.trackbar.set_pos(self.slider.value())
        t = self.slider.value() / self.fps
        self.lbl_time.setText(f"{self._hms(t, tenth=True)} / "
                              f"{self._hms(self.total / self.fps)}")
        self._slider_timer.start()

    def _restart_play_at(self, f):
        """재생 중 시크: 이전 워커를 버리고 f 부터 재생 재시작."""
        w_old = self._play_worker
        if w_old is not None:
            try:
                w_old.frame_ready.disconnect()
                w_old.finished.disconnect()
            except TypeError:
                pass
            w_old.stop()                  # 남은 프레임 방출은 무시됨
        self._playing = False
        self._toggle_play()               # slider 값(=f)부터 새로 시작

    def _toggle_mute(self):
        muted = self.btn_mute.isChecked()
        self.btn_mute.setText("🔇" if muted else "🔊")
        QSettings("PyStitch360", "PyStitch360").setValue(
            "ptz_muted", "true" if muted else "false")
        if self._audio:
            self._audio_out.setMuted(muted)

    def _ensure_audio(self):
        """오디오 플레이어 지연 초기화 — 백엔드 없으면 조용히 무음."""
        if self._audio is not None:
            return self._audio or None
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            self._audio_out = QAudioOutput()
            self._audio_out.setMuted(self.btn_mute.isChecked())
            p = QMediaPlayer()
            p.setAudioOutput(self._audio_out)
            self._audio = p
        except Exception as e:  # noqa: BLE001
            self.log(f"[ptz] 오디오 재생 불가 (QtMultimedia): {e}")
            self._audio = False
        return self._audio or None

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
            return
        if self.cap is None:
            return
        w = PtzPlayWorker(self.disp_path or self.pano_path,
                          self.slider.value(), self.fps,
                          display_fps=30.0 if self.disp_scale < 1.0 else 15.0)
        w.frame_ready.connect(self._play_frame)
        w.finished.connect(self._play_finished)
        self._play_worker = w
        self._playing = True
        self.btn_play.setText("⏸")
        a = self._ensure_audio()
        if a is not None:
            from PyQt6.QtCore import QUrl
            url = QUrl.fromLocalFile(str(self.pano_path))
            if a.source() != url:
                a.setSource(url)      # 오디오는 항상 원본에서 (프록시 무관)
            a.setPosition(int(self.slider.value() / self.fps * 1000))
            a.play()
        w.start()

    def _stop_play(self):
        if self._play_worker is not None and self._play_worker.isRunning():
            self._play_worker.stop()
        if self._audio:
            self._audio.pause()

    def _play_frame(self, frame, f):
        self._cur_frame, self._cur_frame_idx = frame, f
        self._play_sync = True      # 워커 발 갱신 표시 — 사용자 시크와 구분
        try:
            self.slider.setValue(f)  # _show_frame 은 재생 중 가드로 무시됨
        finally:
            self._play_sync = False
        # 오디오 드리프트 보정 (±0.3s 넘으면 재동기)
        if self._audio and f % 150 < 5:
            want = int(f / self.fps * 1000)
            if abs(self._audio.position() - want) > 300:
                self._audio.setPosition(want)
        self._redraw()

    def _play_finished(self):
        self._playing = False
        self.btn_play.setText("▶")
        if self._audio:
            self._audio.pause()

    def _current_sample(self):
        """현재 시각 근처(±0.5s)의 분석 샘플 인덱스."""
        if self.analysis is None:
            return None
        idx = np.asarray(self.analysis["frames"])
        i = int(np.argmin(np.abs(idx - self.slider.value())))
        if abs(idx[i] - self.slider.value()) > 0.5 * self.fps:
            return None
        return i

    def _current_auto_ball(self):
        i = self._current_sample()
        return None if i is None else self.analysis["balls"][i]

    def _ball_in_ignore(self, f, bx, by):
        """(f, bx, by) 공이 어떤 무시 구간에 걸리는가 (자리 있으면 150px)."""
        return any(
            r[0] <= f <= r[1] and (len(r) < 4 or
                                   ((r[2] - bx) ** 2 + (r[3] - by) ** 2)
                                   ** 0.5 <= 150)
            for r in self.ignores)

    def _promote_near(self, f, bx, by):
        """(f, bx, by) 부근(±0.5s, 150px)에 걸린 승격 항목들 (토글/표시용)."""
        return [p for p in self.promotes
                if abs(p[0] - f) <= 0.5 * self.fps
                and (p[1] - bx) ** 2 + (p[2] - by) ** 2 <= 150 ** 2]

    # ---------------------------------------------------- 오브젝트 질의/조작
    def _candidates_at(self, si):
        """샘플 si 의 공 후보 [(x, y, conf), ...] (신형 ball_cands 우선)."""
        if self.analysis is None or si is None:
            return []
        bc = self.analysis.get("ball_cands")
        if bc is not None and si < len(bc):
            return [(float(p[0]), float(p[1]), float(p[2])) for p in bc[si]]
        b = self.analysis["balls"][si]
        return [(float(b[0]), float(b[1]), float(b[2]))] if b else []

    def _track_for(self, si, x, y):
        """샘플 si 에서 (x, y) 를 지나는 링크 트랙 (없으면 None)."""
        if self._linked is None or si is None:
            return None
        best, bestd = None, 80.0 ** 2
        for t in self._linked["tracks"]:
            hits = np.where(t["i"] == si)[0]
            if len(hits) == 0:
                continue
            k = int(hits[0])
            d = (t["pts"][k][0] - x) ** 2 + (t["pts"][k][1] - y) ** 2
            if d <= bestd:
                best, bestd = t, d
        return best

    def _track_span(self, t):
        idx = self._linked["idx"]
        return int(idx[t["i"][0]]), int(idx[t["i"][-1]])

    def _cand_state(self, f, si, x, y):
        """공 후보 상태: 'ignored' | 'promoted' | 'accepted' | 'rejected'."""
        if self._ball_in_ignore(f, x, y):
            return "ignored"
        if self._promote_near(f, x, y):
            return "promoted"
        ab = self._accepted_ball
        if (ab is not None and si is not None and si < len(ab)
                and not np.isnan(ab[si, 0])
                and (ab[si, 0] - x) ** 2 + (ab[si, 1] - y) ** 2 <= 60.0 ** 2):
            return "accepted"
        return "rejected"

    def _objects_at(self):
        """현재 프레임의 조작 가능한 오브젝트 (공 후보 + 근처 키프레임)."""
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        si = self._current_sample()
        objs = []
        for (x, y, conf) in self._candidates_at(si):
            objs.append({"kind": "ball", "x": x, "y": y, "conf": conf,
                         "state": self._cand_state(f, si, x, y), "si": si})
        for i, k in enumerate(self.keyframes):
            if abs(k[0] - f) <= 1.0 * self.fps:
                objs.append({"kind": "kf", "x": float(k[1]), "y": float(k[2]),
                             "i": i})
        return objs

    def _hit(self, x, y, r=80.0):
        """(x, y) 파노라마 좌표에서 반경 r 안 가장 가까운 오브젝트 (없으면 None)."""
        best, bestd = None, r * r
        for o in self._objects_at():
            d = (o["x"] - x) ** 2 + (o["y"] - y) ** 2
            if d <= bestd:
                best, bestd = o, d
        return best

    def _ignore_covers(self, entry):
        """entry(=[f0,f1,x,y]) 와 사실상 동일한 무시가 이미 있는가."""
        f0, f1 = entry[0], entry[1]
        for r in self.ignores:
            if r[0] <= f0 and f1 <= r[1] and (
                    len(r) < 4 or ((r[2] - entry[2]) ** 2
                                   + (r[3] - entry[3]) ** 2) ** 0.5 <= 150):
                return True
        return False

    def _mark_dirty_and_redraw(self):
        self._save_keyframes()
        self._recompute_tracks()
        self._redraw()

    def _promote_ball(self, f, si, x, y):
        """(x,y) 공을 경기 공으로 승격 + 같은 프레임 다른 후보 트랙 자동 무시."""
        if not self._promote_near(f, x, y):
            self.promotes.append([f, round(x, 1), round(y, 1)])
            self.promotes.sort()
        auto = 0
        for (cx, cy, cc) in self._candidates_at(si):
            if (cx - x) ** 2 + (cy - y) ** 2 <= 60.0 ** 2:
                continue                       # 승격 대상 자신
            t = self._track_for(si, cx, cy)
            if t is None:
                continue
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
            # 승격한 공을 함께 잡을 수 있는 무시는 건너뜀 (자기 트랙 보호)
            if (f0 <= f <= f1 and
                    ((med[0] - x) ** 2 + (med[1] - y) ** 2) ** 0.5 <= 150):
                continue
            if not self._ignore_covers(entry):
                self.ignores.append(entry)
                auto += 1
        self.ignores.sort()
        self._mark_dirty_and_redraw()
        extra = f", 경쟁 후보 {auto}개 자동 무시" if auto else ""
        self.log(f"[ptz] 경기 공 승격 {f/self.fps:.1f}s "
                 f"→ ({x:.0f}, {y:.0f}){extra}")

    def _unpromote_at(self, f, x, y):
        near = self._promote_near(f, x, y)
        if not near:
            return
        for p in near:
            self.promotes.remove(p)
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 승격 취소 {f/self.fps:.1f}s")

    def _ignore_track_at(self, f, si, x, y):
        """(x,y) 후보의 트랙을 오인식으로 무시 (링크 없으면 그 시점만)."""
        t = self._track_for(si, x, y)
        if t is not None:
            f0, f1 = self._track_span(t)
            med = np.median(t["pts"], axis=0)
            entry = [f0, f1, round(float(med[0]), 1), round(float(med[1]), 1)]
        else:
            entry = [int(f), int(f), round(x, 1), round(y, 1)]
        for p in self._promote_near(f, x, y):    # 무시가 승격보다 우선
            self.promotes.remove(p)
        if not self._ignore_covers(entry):
            self.ignores.append(entry)
            self.ignores.sort()
        self._mark_dirty_and_redraw()
        self.log(f"[ptz] 검출 무시 {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _restore_at(self, f, x, y):
        """(x,y) 위치에 걸린 무시 구간을 해제 (복원)."""
        keep, removed = [], 0
        for r in self.ignores:
            if (r[0] <= f <= r[1] and
                    (len(r) < 4 or ((r[2] - x) ** 2 + (r[3] - y) ** 2)
                     ** 0.5 <= 150)):
                removed += 1
            else:
                keep.append(r)
        if removed:
            self.ignores = keep
            self._mark_dirty_and_redraw()
            self.log(f"[ptz] 무시 해제 {f/self.fps:.1f}s ({removed}개 구간)")

    def _set_click_mode(self, ball: bool):
        self.btn_mode_ball.setChecked(ball)
        self.btn_mode_kf.setChecked(not ball)

    def _add_keyframe(self, f, x, y, width=None):
        """빈 곳 클릭 마크: 3요소 = 공(줌 자동), 4요소 = 크롭 키프레임."""
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        entry = [f, round(x, 1), round(y, 1)]
        if width is not None:
            entry.append(round(float(width), 1))
        self.keyframes.append(entry)
        self.keyframes.sort()
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()
        kind = "키프레임" if width is not None else "수동 공"
        self.log(f"[ptz] {kind} {f/self.fps:.1f}s → ({x:.0f}, {y:.0f})")

    def _toggle_kf_type(self, i):
        """공(3요소) ↔ 크롭 키프레임(4요소, 현재 계획 폭) 전환."""
        if not (0 <= i < len(self.keyframes)):
            return
        k = self.keyframes[i]
        if len(k) > 3:
            self.keyframes[i] = k[:3]
            self.log(f"[ptz] {k[0]/self.fps:.1f}s 키프레임 → 공 (줌 자동)")
        else:
            f = int(k[0])
            if self.plan is None or f >= len(self.plan["crop_w"]):
                self.log("[ptz] 계획이 아직 없어 폭을 정할 수 없음")
                return
            self.keyframes[i] = [k[0], k[1], k[2],
                                 round(float(self.plan["crop_w"][f]), 1)]
            self.log(f"[ptz] {k[0]/self.fps:.1f}s 공 → 키프레임 "
                     f"(폭 {self.keyframes[i][3]:.0f} 고정)")
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()

    def _delete_keyframe_idx(self, i):
        if 0 <= i < len(self.keyframes):
            k = self.keyframes[i]
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_kf_list()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes,
                                   promotes=self.promotes)
            self._plan_dirty()
            self._redraw()
            self.log(f"[ptz] 키프레임 삭제 {k[0]/self.fps:.1f}s")

    # ------------------------------------------------- 크롭 박스 드래그 편집
    def _disp_to_pano(self):
        """표시 픽셀 1개가 파노라마 몇 px 인가 (히트 톨러런스 환산)."""
        return self.pano_w / max(self.pane.displayed_width(), 1)

    def _box_hit(self, x, y):
        """크롭 박스 히트: ("corner", 0~3) | ("border", None) | None.

        모서리(핸들) = 리사이즈(줌), 테두리 밴드 = 이동. 내부는 공 클릭용으로
        비워 둔다 (공은 대개 박스 안에 있다).
        """
        if self._plan_box is None:
            return None
        x0, y0, w, h = self._plan_box
        k = self._disp_to_pano()
        ct = 18 * k                          # 모서리 히트 반경
        for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                       (x0, y0 + h), (x0 + w, y0 + h)]):
            if abs(x - hx) <= ct and abs(y - hy) <= ct:
                return ("corner", ci)
        bt = 10 * k                          # 테두리 밴드 폭
        on_v = (abs(x - x0) <= bt or abs(x - x0 - w) <= bt) \
            and y0 - bt <= y <= y0 + h + bt
        on_h = (abs(y - y0) <= bt or abs(y - y0 - h) <= bt) \
            and x0 - bt <= x <= x0 + w + bt
        if on_v or on_h:
            return ("border", None)
        return None

    def _landmark_at(self, x, y, r=60.0):
        """(x, y) 근처의 찍힌 랜드마크 키 (반경 r px, 없으면 None)."""
        best, bestd = None, r * r
        for k, p in self.field_points.items():
            d = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if d <= bestd:
                best, bestd = k, d
        return best

    def _pane_pressed(self, fx, fy):
        """좌버튼 프레스: 랜드마크(경기장 탭) 또는 크롭 박스 편집 시작."""
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        x, y = fx * self.pano_w, fy * self.pano_h
        if self._field_tab_active() or self.btn_field_pick.isChecked():
            self._lm_drag = self._landmark_at(x, y)   # 마커 잡으면 드래그 이동
            if self._lm_drag is not None:
                return
        if self.btn_field_pick.isChecked():   # 찍기 모드: 박스 편집 비활성
            return
        if self.plan is None or self.analysis is None:
            return
        if self._hit(x, y) is not None:      # 공/키프레임이 우선 (클릭 동작)
            return
        hit = self._box_hit(x, y)
        if hit is None or self._plan_box is None:
            return
        x0, y0, w, h = self._plan_box
        box = [x0 + w / 2, y0 + h / 2, float(w)]
        corners = [(x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h)]
        self._box_edit = {"mode": hit[0], "corner": hit[1],
                          "box": list(box), "orig": list(box),
                          # 리사이즈 앵커 = 잡은 코너의 대각 반대편 (고정점)
                          "anchor": corners[3 - hit[1]] if hit[0] == "corner"
                          else None,
                          "start": (x, y),
                          "frame": getattr(self, "_cur_frame_idx",
                                           self.slider.value())}

    def _pane_dragged(self, fx, fy):
        if self._lm_drag is not None:        # 랜드마크 이동 — 라이브 재피팅
            self.field_points[self._lm_drag] = \
                [round(fx * self.pano_w, 1), round(fy * self.pano_h, 1)]
            self._refit_field()
            self._refresh_field_list()
            self._redraw()
            return
        e = self._box_edit
        if e is None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        ow, oh = self.plan_out
        cx0, cy0, w0 = e["orig"]
        top = int(self.plan.get("top_margin", 0)) if self.plan else 0
        max_w = min(self.pano_w, (self.pano_h - top) * ow / oh)
        if e["mode"] == "border":            # 이동
            cx = cx0 + (x - e["start"][0])
            cy = cy0 + (y - e["start"][1])
            w = min(max(w0, ow / 6.0), max_w)
            h = w * oh / ow
            cx = min(max(cx, w / 2), self.pano_w - w / 2)
            cy = min(max(cy, h / 2), self.pano_h - h / 2)
        else:                                # 모서리 = 반대편 코너 고정 리사이즈
            ax, ay = e["anchor"]
            sx = -1 if e["corner"] in (0, 2) else 1   # 왼쪽 코너 = 앵커의 왼쪽으로
            sy = -1 if e["corner"] in (0, 1) else 1   # 위 코너 = 앵커의 위로
            w = max(abs(x - ax), abs(y - ay) * ow / oh)
            # 앵커를 고정한 채 파노라마 안에 들어가는 최대 폭
            max_w = min(max_w,
                        (self.pano_w - ax) if sx > 0 else ax,
                        ((self.pano_h - ay) if sy > 0 else ay) * ow / oh)
            w = min(max(w, ow / 6.0), max_w)
            h = w * oh / ow
            cx, cy = ax + sx * w / 2, ay + sy * h / 2
        e["box"] = [cx, cy, w]
        self._redraw()

    def _pane_released(self, fx, fy):
        if self._lm_drag is not None:
            key, self._lm_drag = self._lm_drag, None
            self._refit_field(log_result=True)
            self._save_keyframes()
            self._refresh_field_list()
            self._redraw()
            return
        e = self._box_edit
        if e is None:
            return
        self._box_edit = None
        cx, cy, w = e["box"]
        cx0, cy0, w0 = e["orig"]
        # 사실상 클릭(수 px 미만 이동)이면 커밋하지 않음
        if abs(cx - cx0) + abs(cy - cy0) + abs(w - w0) < 6 * self._disp_to_pano():
            self._redraw()
            return
        f = e["frame"]
        self.keyframes = [k for k in self.keyframes
                          if abs(k[0] - f) > 0.5 * self.fps]
        self.keyframes.append([f, round(cx, 1), round(cy, 1), round(w, 1)])
        self.keyframes.sort()
        # 새 계획이 올 때까지 이 위치를 그대로 그린다 — 이전 박스 깜박임 방지
        self._box_commit = (f, cx, cy, w)
        self._save_keyframes()
        self._refresh_kf_list()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] 크롭 키프레임 {f/self.fps:.1f}s → "
                 f"중심 ({cx:.0f}, {cy:.0f}), 폭 {w:.0f}px")

    def _show_frame(self):
        """현재 프레임 디코딩 + 오버레이. 디코딩 결과는 캐시된다."""
        if self.cap is None or self._playing:
            return
        f = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        if not ok:
            return
        self._cur_frame, self._cur_frame_idx = frame, f
        self._redraw()

    def _redraw(self):
        """오버레이만 다시 그림 — 키프레임/무시/계획 변경 시 디코딩 없이 즉시."""
        if getattr(self, "_cur_frame", None) is None:
            return
        frame = self._cur_frame.copy()
        f = self._cur_frame_idx
        sc = self.disp_scale                 # 프록시 표시 시 좌표 축소
        # 랜드마크 찍기 모드: 크롭 박스/공/키프레임/선수 오버레이 숨김 —
        # 경기장 선·마커만 보이게 해서 클릭 대상이 헷갈리지 않게 한다.
        picking = self.btn_field_pick.isChecked()
        # 계획된 크롭 창 (render_plan 과 동일한 클램프) — 결과 미리보기.
        # 드래그(이동)/모서리 핸들(줌)로 편집 가능 — 놓으면 줌 키프레임 커밋.
        self._plan_box = None
        if self.plan is not None and not picking and f < len(self.plan["cx"]):
            ow, oh = self.plan_out
            top = int(self.plan.get("top_margin", 0))
            commit = self._box_commit
            if self._box_edit is not None:      # 편집 중: 미확정 박스
                bcx, bcy, bw = self._box_edit["box"]
                w = int(round(bw)) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(bcx - w / 2))
                y0 = int(round(bcy - h / 2))
                box_color = (0, 255, 255)
            elif commit is not None and abs(f - commit[0]) <= 0.5 * self.fps:
                # 커밋 직후 (새 계획 계산 중): 놓은 자리 그대로 유지
                w = int(round(commit[3])) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(commit[1] - w / 2))
                y0 = int(round(commit[2] - h / 2))
                box_color = (255, 200, 0)
            else:
                w = int(round(min(self.plan["crop_w"][f], self.pano_w,
                                  self.pano_h * ow / oh))) & ~1
                h = int(round(w * oh / ow)) & ~1
                x0 = int(round(self.plan["cx"][f] - w / 2))
                y0 = int(round(self.plan["cy"][f] - h / 2))
                x0 = max(0, min(x0, self.pano_w - w))
                y0 = max(min(top, self.pano_h - h), min(y0, self.pano_h - h))
                box_color = (255, 200, 0)
            self._plan_box = (x0, y0, w, h)
            thick = max(2, int(6 * sc))
            if self._box_edit is None and self._box_hover == ("border", None):
                thick += max(2, int(4 * sc))    # 이동 가능 표시: 테두리 두껍게
            cv2.rectangle(frame, (int(x0 * sc), int(y0 * sc)),
                          (int((x0 + w) * sc), int((y0 + h) * sc)),
                          box_color, thick)
            hs = max(6, int(16 * sc))           # 모서리 핸들 (리사이즈 = 줌)
            for ci, (hx, hy) in enumerate([(x0, y0), (x0 + w, y0),
                                           (x0, y0 + h), (x0 + w, y0 + h)]):
                hov = (self._box_edit is None
                       and self._box_hover == ("corner", ci))
                r = hs + (max(3, int(8 * sc)) if hov else 0)
                cv2.rectangle(frame, (int(hx * sc) - r, int(hy * sc) - r),
                              (int(hx * sc) + r, int(hy * sc) + r),
                              (0, 255, 255) if hov else box_color, -1)
        # 공 후보 전부 표시 — 상태별 색: 초록=수락, 자홍=승격, 빨강X=무시,
        # 회색=자동 기각. 커서가 가리키는 오브젝트는 노란 링으로 하이라이트.
        si = self._current_sample()
        hv = self._hover_key[0] if self._hover_key else None

        def _is_hover(kind, ox, oy):
            return hv is not None and hv == (kind, round(ox), round(oy))

        rad = max(10, int(28 * sc))
        for (bx, by, conf) in ([] if picking else self._candidates_at(si)):
            p = (int(bx * sc), int(by * sc))
            st = self._cand_state(f, si, bx, by)
            if st == "ignored":
                cv2.circle(frame, p, rad, (60, 60, 200), 4)
                cv2.line(frame, (p[0] - 24, p[1] - 24), (p[0] + 24, p[1] + 24),
                         (60, 60, 200), 6)
                cv2.line(frame, (p[0] - 24, p[1] + 24), (p[0] + 24, p[1] - 24),
                         (60, 60, 200), 6)
                cv2.putText(frame, "IGNORED", (p[0] + 36, p[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 60, 200), 4)
            elif st == "promoted":
                cv2.circle(frame, p, rad, (200, 0, 255), 6)
            elif st == "accepted":
                cv2.circle(frame, p, rad, (0, 220, 0), 6)
            else:
                cv2.circle(frame, p, rad, (150, 150, 150), 4)
            if _is_hover("ball", bx, by):
                cv2.circle(frame, p, rad + max(6, int(12 * sc)), (0, 255, 255), 3)
        for k in ([] if picking else self.keyframes):
            if abs(k[0] - f) <= 1.0 * self.fps:
                kx, ky = float(k[1]), float(k[2])
                p = (int(kx * sc), int(ky * sc))
                th = max(2, int(6 * sc))
                if len(k) > 3:
                    # 크롭 키프레임: 동서남북 4방향 화살표 (이동+줌 앵커)
                    arm = max(12, int(34 * sc))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        cv2.arrowedLine(frame, p,
                                        (p[0] + dx * arm, p[1] + dy * arm),
                                        (0, 0, 255), th, tipLength=0.45)
                else:
                    # 수동 공: 주황 이중 링 (공 후보 원과 구별)
                    cv2.circle(frame, p, max(10, int(26 * sc)),
                               (0, 140, 255), th)
                    cv2.circle(frame, p, max(3, int(7 * sc)),
                               (0, 140, 255), -1)
                if _is_hover("kf", kx, ky):
                    cv2.circle(frame, p, max(16, int(40 * sc)), (0, 255, 255), 3)
        # 선수 박스(팀 색) + 레이더
        radar_pts = []
        if si is not None:
            prow = self._players_row(si)
            if self.check_players.isChecked() and not picking:
                for pp in prow:
                    if len(pp) < 4:
                        continue
                    team = self._role_of(int(pp[4])) if len(pp) >= 5 else 2
                    color = self._role_color(team)
                    x1 = int((pp[0] - pp[2] / 2) * sc)
                    y1 = int((pp[1] - pp[3] / 2) * sc)
                    x2 = int((pp[0] + pp[2] / 2) * sc)
                    y2 = int((pp[1] + pp[3] / 2) * sc)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                                  max(2, int(4 * sc)))
                    tag = ROLE_TAGS.get(team)
                    if tag:
                        cv2.putText(frame, tag, (x1, y1 - max(4, int(8 * sc))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    max(0.5, 1.1 * sc), color,
                                    max(1, int(3 * sc)))
            ball_g = None
            bb = self.analysis["balls"][si]
            if self._field_calib is not None:
                # 캘리브레이션 완료: 경기장 절대 좌표로 표시
                feet = [(pp[0], pp[1] + pp[3] / 2.0,
                         int(pp[4]) if len(pp) >= 5 else -1)
                        for pp in prow if len(pp) >= 4]
                if feet:
                    fc = pano_to_field(self._field_calib,
                                       [(a, b) for a, b, _ in feet])
                    for (gx, gy), (_, _, tid) in zip(fc, feet):
                        if np.isfinite(gx):
                            radar_pts.append((gx, gy, self._role_of(tid)))
                if si in self._air:                 # 공중볼: 보정 XY 사용
                    ball_g = self._air[si][:2]
                elif bb is not None:
                    g = pano_to_field(self._field_calib, [[bb[0], bb[1]]])[0]
                    if np.isfinite(g[0]):
                        ball_g = (float(g[0]), float(g[1]))
            else:
                for X, Y, tid, j in ground_positions(prow, self.pano_w,
                                                     self.pano_h):
                    radar_pts.append((X, Y, self._role_of(tid)))
                if bb is not None:
                    g = ground_positions([[bb[0], bb[1], 0.0, 0.0]],
                                         self.pano_w, self.pano_h)
                    ball_g = (g[0][0], g[0][1]) if g else None
            self.radar.set_data(radar_pts, ball_g)
        else:
            self.radar.set_data([], None)
        # 경기장 탭: 랜드마크 마커 + (캘리브레이션 후) 예상 경기장 선
        if self._field_tab_active() or self.btn_field_pick.isChecked():
            if self._field_calib is not None:
                for line in field_outline(*self.field_size):
                    q = field_to_pano(self._field_calib, line)
                    seg = []
                    for qx, qy in list(q) + [(np.nan, np.nan)]:
                        ok = (np.isfinite(qx)
                              and -0.2 * self.pano_w <= qx <= 1.2 * self.pano_w
                              and -0.2 * self.pano_h <= qy <= 1.2 * self.pano_h
                              and not (seg and abs(qx - seg[-1][0])
                                       > self.pano_w / 4))   # 랩어라운드 컷
                        if ok:
                            seg.append((qx, qy))
                            continue
                        if len(seg) >= 2:
                            arr = np.array([[int(a * sc), int(b * sc)]
                                            for a, b in seg], np.int32)
                            cv2.polylines(frame, [arr], False, (255, 255, 255),
                                          max(1, int(3 * sc)))
                        seg = [(qx, qy)] if np.isfinite(qx) else []
            for lx, ly in self.line_points:      # 흰 선 검출 샘플 (녹색 점)
                cv2.circle(frame, (int(lx * sc), int(ly * sc)),
                           max(2, int(5 * sc)), (0, 255, 0), -1)
            for k, pt in self.field_points.items():
                q = (int(pt[0] * sc), int(pt[1] * sc))
                cv2.drawMarker(frame, q, (255, 255, 0),
                               cv2.MARKER_TILTED_CROSS,
                               max(14, int(36 * sc)), max(2, int(6 * sc)))
                cv2.putText(frame, LANDMARK_TAGS[k],
                            (q[0] + 12, q[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, 1.0 * sc), (255, 255, 0),
                            max(1, int(3 * sc)))
        self.pane.set_frame(frame)

    # ------------------------------------------------------ 화면 오브젝트 조작
    def _pane_clicked(self, fx, fy):
        """좌클릭: 오브젝트 상태별 기본 동작 / 빈 곳은 키프레임 추가."""
        if self.cap is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        if (self._field_tab_active() or self.btn_field_pick.isChecked()) \
                and self._landmark_at(x, y) is not None:
            return              # 랜드마크 위 클릭 = 드래그 미수 — 무동작
        if self.btn_field_pick.isChecked():   # 랜드마크 찍기 모드가 우선
            key = self._match_landmark(x, y)
            if key is None:
                self.log("[field] 수평선 위 — 지면 지점을 클릭하세요")
            else:
                self._field_set_point(key, x, y)
            return
        o = self._hit(x, y)
        if o is None:
            if self._box_hit(x, y) is not None:
                return                      # 박스 테두리 클릭 = 드래그 미수 — 무동작
            # 빈 곳 = 모드에 따라 수동 공(기본) 또는 크롭 키프레임 추가
            w = None
            if self.btn_mode_kf.isChecked() and self.plan is not None \
                    and int(f) < len(self.plan["crop_w"]):
                w = float(self.plan["crop_w"][int(f)])
            self._add_keyframe(f, x, y, width=w)
            return
        if o["kind"] == "kf":               # 키프레임 = 삭제
            self._delete_keyframe_idx(o["i"])
            return
        st = o["state"]                     # 공 후보 = 상태별 기본 동작
        if st == "promoted":
            self._unpromote_at(f, o["x"], o["y"])
        elif st == "ignored":
            self._restore_at(f, o["x"], o["y"])
        else:                               # gray/green → 경기 공 승격
            self._promote_ball(f, o["si"], o["x"], o["y"])

    def _pane_context(self, fx, fy, gpos):
        """우클릭: 오브젝트별 전체 동작 메뉴."""
        if self.cap is None or self.analysis is None:
            return
        if self._playing:
            self._stop_play()
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        menu = QMenu(self)

        def _add_kf_here():
            w = (float(self.plan["crop_w"][int(f)])
                 if self.plan is not None and int(f) < len(self.plan["crop_w"])
                 else None)
            self._add_keyframe(f, x, y, width=w)

        if o is None:
            menu.addAction("여기 수동 공 추가",
                           lambda: self._add_keyframe(f, x, y))
            menu.addAction("여기 크롭 키프레임 추가", _add_kf_here)
        elif o["kind"] == "kf":
            is_kf = len(self.keyframes[o["i"]]) > 3
            if not is_kf:
                k = self.keyframes[o["i"]]
                menu.addAction("앞뒤로 추적 확장 (±4s)",
                               lambda: self._propagate(
                                   int(k[0]), float(k[1]), float(k[2]),
                                   "ball"))
            menu.addAction("공으로 전환 (줌 자동)" if is_kf
                           else "크롭 키프레임으로 전환 (현재 폭 고정)",
                           lambda: self._toggle_kf_type(o["i"]))
            menu.addAction("삭제",
                           lambda: self._delete_keyframe_idx(o["i"]))
        else:
            st, ox, oy, si = o["state"], o["x"], o["y"], o["si"]
            if st == "promoted":
                menu.addAction("승격 취소",
                               lambda: self._unpromote_at(f, ox, oy))
            else:
                menu.addAction("이 공을 경기 공으로 승격",
                               lambda: self._promote_ball(f, si, ox, oy))
            if st == "ignored":
                menu.addAction("무시 해제(복원)",
                               lambda: self._restore_at(f, ox, oy))
            else:
                menu.addAction("이 검출 무시(오인식)",
                               lambda: self._ignore_track_at(f, si, ox, oy))
            menu.addSeparator()
            menu.addAction("여기 수동 공 추가",
                           lambda: self._add_keyframe(f, x, y))
        tid = self._player_at(x, y)
        if tid is not None:
            menu.addSeparator()
            sub = menu.addMenu(
                f"선수 #{tid} ({self._role_name(self._role_of(tid))}) 역할 지정")
            for r in (3, 4, 5, 0, 1):
                sub.addAction(self._role_name(r),
                              lambda _=False, rr=r, t=tid: self._set_role(t, rr))
            if tid in self.roles:
                sub.addAction("자동 분류로 되돌리기",
                              lambda _=False, t=tid: self._set_role(t, None))
            if tid >= 900001:
                sub.addSeparator()
                row = next((p for si_ in self.extra_players
                            for p in self.extra_players[si_]
                            if p[4] == tid), None)
                if row is not None:
                    si_seed = next(si_ for si_ in self.extra_players
                                   if any(p[4] == tid for p in
                                          self.extra_players[si_]))
                    f_seed = int(self.analysis["frames"][si_seed])
                    sub.addAction("앞뒤로 추적 확장 (±4s)",
                                  lambda _=False, ff=f_seed, rr=row, t=tid:
                                  self._propagate(ff, rr[0], rr[1],
                                                  "person", ctx=t))
                sub.addAction("이 수동 검출 삭제",
                              lambda _=False, t=tid: self._delete_extra(t))
        if self.analysis is not None and self._current_sample() is not None:
            menu.addSeparator()
            menu.addAction("이 주변 사람 재검출",
                           lambda _=False: self._detect_here(f, x, y, gpos))
        if self._field_tab_active():
            menu.addSeparator()
            sub = menu.addMenu("이 위치를 랜드마크로 지정")
            for key, name, req in LANDMARKS:
                mark = "★ " if req else ""
                cur = " ✓" if key in self.field_points else ""
                sub.addAction(f"[{LANDMARK_TAGS[key]}] {mark}{name}{cur}",
                              lambda _=False, kk=key:
                              self._field_set_point(kk, x, y))
            near = self._landmark_at(x, y, r=100.0)
            if near is not None:
                near_name = next(n for k, n, _ in LANDMARKS if k == near)
                sub = menu.addMenu(f"마커 [{LANDMARK_TAGS[near]}] "
                                   f"{near_name} → 다른 랜드마크로 변경")
                for key, name, req in LANDMARKS:
                    if key == near:
                        continue
                    mark = "★ " if req else ""
                    cur = " ✓(교환)" if key in self.field_points else ""
                    sub.addAction(f"[{LANDMARK_TAGS[key]}] {mark}{name}{cur}",
                                  lambda _=False, o=near, n=key:
                                  self._field_reassign(o, n))
                menu.addAction(f"마커 [{LANDMARK_TAGS[near]}] 지정 해제",
                               lambda _=False, kk=near:
                               self._field_remove_point(kk))
        menu.exec(gpos)

    def _pane_hover(self, fx, fy):
        """커서 근처 오브젝트/크롭 박스를 하이라이트 (바뀔 때만 리드로우)."""
        if self.analysis is None or getattr(self, "_cur_frame", None) is None:
            return
        if self._box_edit is not None:
            return
        x, y = fx * self.pano_w, fy * self.pano_h
        o = self._hit(x, y)
        zone = None if o is not None else self._box_hit(x, y)
        key = ((None if o is None
                else (o["kind"], round(o["x"]), round(o["y"]))), zone)
        if key != self._hover_key:
            self._hover_key = key
            self._hover = o
            self._box_hover = zone
            if o is not None:
                cur = Qt.CursorShape.PointingHandCursor
            elif zone is None:
                cur = Qt.CursorShape.OpenHandCursor
            elif zone[0] == "border":
                cur = Qt.CursorShape.SizeAllCursor
            else:                            # 모서리: 대각 리사이즈 커서
                cur = (Qt.CursorShape.SizeFDiagCursor if zone[1] in (0, 3)
                       else Qt.CursorShape.SizeBDiagCursor)
            self.pane.setCursor(cur)
            self._redraw()

    def _refresh_lists(self):
        """위: 공(자동 트랙+수동 키프레임) / 아래: 오인식(무시 구간)."""
        # 위 목록 — 시간순 병합
        # 재구성 전 선택을 (종류, 시작 프레임) 으로 기억 — 승격/무시로
        # 목록이 다시 만들어져도 보던 항목을 따라간다 (점프 방지)
        old_top = getattr(self, "_top", [])
        cur = self.track_list.currentRow()
        sel_key = None
        if 0 <= cur < len(old_top):
            k, i = old_top[cur]
            if k == "kf" and i < len(self.keyframes):
                sel_key = ("kf", self.keyframes[i][0])
            elif k == "track":
                sel_key = ("track", None)   # 트랙 인덱스는 재계산됨 — 아래서 위치로
        self._top = ([("track", i) for i in range(len(self.track_spans))]
                     + [("kf", i) for i in range(len(self.keyframes))])
        self._top.sort(key=lambda e: (self.track_spans[e[1]][0] if e[0] == "track"
                                      else self.keyframes[e[1]][0]))
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for kind, i in self._top:
            if kind == "track":
                f0, f1 = self.track_spans[i]
                t0, t1 = f0 / self.fps, f1 / self.fps
                self.track_list.addItem(
                    f"{int(t0//60):02d}:{t0%60:04.1f} ~ "
                    f"{int(t1//60):02d}:{t1%60:04.1f}  ({t1-t0:.1f}s) 자동")
            else:
                k = self.keyframes[i]
                kf, kx, ky = k[0], k[1], k[2]
                t = kf / self.fps
                label = (f"◆ 키프레임 ({kx:.0f}, {ky:.0f}, 폭{k[3]:.0f})"
                         if len(k) > 3 else f"● 수동 공 ({kx:.0f}, {ky:.0f})")
                self.track_list.addItem(
                    f"{int(t//60):02d}:{t%60:04.1f}  {label}")
        # 선택 복원 (시그널 차단 상태 — 복원이 시크를 유발하지 않게)
        if sel_key is not None:
            pos = self.slider.value()
            for row, (k, i) in enumerate(self._top):
                if sel_key[0] == "kf" and k == "kf" \
                        and self.keyframes[i][0] == sel_key[1]:
                    self.track_list.setCurrentRow(row)
                    break
                if sel_key[0] == "track" and k == "track" \
                        and self.track_spans[i][0] <= pos <= self.track_spans[i][1]:
                    self.track_list.setCurrentRow(row)
                    break
        self.track_list.blockSignals(False)
        # 아래 목록 — 오인식만
        self.kf_list.clear()
        for r in self.ignores:
            f0, f1 = r[0], r[1]
            pos = f"  @({r[2]:.0f},{r[3]:.0f})" if len(r) >= 4 else ""
            self.kf_list.addItem(
                f"{int(f0/self.fps//60):02d}:{f0/self.fps%60:04.1f}~"
                f"{int(f1/self.fps//60):02d}:{f1/self.fps%60:04.1f}  공 아님{pos}")

    # 기존 호출부 호환 별칭
    _refresh_kf_list = _refresh_lists
    _refresh_track_list = _refresh_lists

    def _goto_kf(self):
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            self.slider.setValue(int(self.ignores[row][0]))

    def _delete_kf(self):
        """선택 복원 — 오인식 마킹을 철회해 트랙을 되살린다."""
        row = self.kf_list.currentRow()
        if 0 <= row < len(self.ignores):
            del self.ignores[row]
            self._save_keyframes()
            self._recompute_tracks()
            self._redraw()

    def _save_keyframes(self):
        """마킹 변경 저장 (1.5s 디바운스 — 분석 포함 파일이라 통으로 씀)."""
        self._save_timer.start()


    def _far_zoom_changed(self, v):
        QSettings("PyStitch360", "PyStitch360").setValue("ptz_far_zoom", v)
        self._plan_dirty()

    def _plan_dirty(self):
        if self.analysis is not None:
            self._plan_timer.start()

    def _run_plan(self):
        if self.analysis is None:
            return
        if self._plan_worker is not None and self._plan_worker.isRunning():
            self._plan_timer.start()      # 진행 중이면 잠시 뒤 재시도
            return
        w = PlanWorker(self.analysis, [tuple(k) for k in self.keyframes],
                       [tuple(r) for r in self.ignores],
                       self.combo_mode.currentIndex() == 1, linked=self._linked,
                       far_zoom=self.spin_far_zoom.value(),
                       promotes=[tuple(p) for p in self.promotes])
        w.done.connect(self._plan_done)
        self._plan_worker = w
        w.start()

    def _plan_done(self, plan, out):
        self.plan = plan
        self.plan_out = out
        self._box_commit = None          # 새 계획 도착 — 낙관적 박스 해제
        self._redraw()

    def _start_link(self):
        """트랙 연결(느린 단계)을 백그라운드로 1회 계산."""
        if self.analysis is None:
            return
        w = LinkWorker(self.analysis)
        w.done.connect(self._link_done)
        self._link_worker = w
        self.log("[ptz] 트랙 연결 계산 중... (완료 후 클릭 반응이 빨라짐)")
        w.start()

    def _airborne_key(self):
        """공중볼 캐시 무효화 키 — 수락 궤적에 영향 주는 입력의 해시."""
        import hashlib
        n_cands = sum(len(c) for c in (self.analysis.get("ball_cands")
                                       or []))
        payload = json.dumps([len(self.analysis["frames"]), n_cands,
                              self.ignores, self.promotes],
                             sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def _recompute_airborne(self):
        """공중볼 보정 재계산 — 파일 캐시(.events.json) 우선, 백그라운드."""
        self._air = {}
        if self.analysis is None or self._field_calib is None \
                or self._accepted_ball is None:
            return
        self._air_gen = getattr(self, "_air_gen", 0) + 1
        key = self._airborne_key()
        self._air_key = key
        segments = None
        try:                              # 저장된 구간이 유효하면 스캔 생략
            from ..core.events import events_json_path
            p = events_json_path(self.pano_path)
            if p.exists():
                doc = json.loads(p.read_text())
                air = doc.get("airborne")
                if air and air.get("key") == key:
                    segments = [(int(a), int(b), f)
                                for a, b, f in air["segments"]]
        except Exception as e:  # noqa: BLE001
            self.log(f"[air] 캐시 무시: {e}")
        w = AirborneWorker(self._air_gen,
                           np.asarray(self.analysis["frames"], dtype=float),
                           float(self.fps),
                           np.array(self._accepted_ball, copy=True),
                           dict(self._field_calib), segments=segments)
        w.done.connect(self._airborne_done)
        self._air_worker = w
        w.start()

    def _airborne_done(self, gen, cache, segments, msg):
        if gen != getattr(self, "_air_gen", 0):
            return                        # 그 사이 재계산됨 — 낡은 결과 폐기
        self._air = cache
        if msg:
            self.log(msg)
        if "캐시 재사용" not in msg:      # 새 스캔 결과만 저장
            try:
                from ..core.events import save_events
                save_events(self.pano_path,
                            airborne={"key": getattr(self, "_air_key", ""),
                                      "segments": segments})
            except Exception as e:  # noqa: BLE001
                self.log(f"[air] 캐시 저장 실패: {e}")
        # 타임라인 '뜬 공' 레인 갱신
        frames = np.asarray(self.analysis["frames"]) \
            if self.analysis is not None else None
        if frames is not None:
            self.trackbar.set_airborne(
                [(int(frames[i0]), int(frames[i1]),
                  float(f["apex_z"])) for i0, i1, f in segments])
        if cache:
            self._redraw()

    def _link_done(self, linked):
        self._linked = linked
        self._teams = linked.pop("teams", {}) or {}
        if self.roles and self.analysis is not None:
            self._teams = classify_teams(self.analysis, roles=self.roles)
        n_team = sum(1 for v in self._teams.values() if v < 2)
        self.log(f"[ptz] 트랙 연결 완료: {len(linked['tracks'])}개"
                 + (f", 팀 분류 선수 ID {n_team}개" if self._teams else
                    " (팀 분류: ID 포함 재분석 필요)"))
        self._refresh_team_label(log_colors=True)
        self._refresh_player_list()
        self._recompute_tracks()
        self._plan_dirty()

    # ------------------------------------------------------ 선수(역할) 검수
    def _player_cache(self):
        """트랙릿 요약({tid: [f0, f1, n]})·대표색 캐시 — 분석 객체당 1회."""
        if self.analysis is None:
            return {}, {}
        if self._pcache_id != id(self.analysis):
            spans: dict[int, list] = {}
            frames = self.analysis["frames"]
            for si, prow in enumerate(self.analysis["players"]):
                f = int(frames[si])
                for p in prow:
                    if len(p) >= 5 and p[4] >= 0:
                        e = spans.get(int(p[4]))
                        if e is None:
                            spans[int(p[4])] = [f, f, 1]
                        else:
                            e[1] = f
                            e[2] += 1
            for si, rows in self.extra_players.items():   # 수동 검출 포함
                if si < len(frames):
                    f = int(frames[si])
                    for p in rows:
                        spans[int(p[4])] = [f, f, 1]
            self._pspans = spans
            self._pcolors = tracklet_colors(self.analysis)
            self._pcache_id = id(self.analysis)
        return self._pspans, self._pcolors

    def _role_name(self, r):
        """역할 표시명 — 팀1/팀2 자리에 사용자 입력 팀 이름."""
        t1, t2 = self.team_names
        return {0: t1, 1: t2, 2: "기타",
                3: f"{t1} GK", 4: f"{t2} GK", 5: "심판"}.get(r, "기타")

    def _edit_team_names(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("팀 이름")
        form = QFormLayout(dlg)
        e1, e2 = QLineEdit(self.team_names[0]), QLineEdit(self.team_names[1])
        form.addRow("팀 1 (홈)", e1)
        form.addRow("팀 2 (원정)", e2)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec():
            return
        self.team_names = [e1.text().strip() or "팀1",
                           e2.text().strip() or "팀2"]
        self._save_keyframes()
        self._apply_team_names()

    def _apply_team_names(self):
        """팀 이름 변경을 역할 버튼·범례·목록·타임라인에 반영."""
        for r, b in getattr(self, "_role_btns", {}).items():
            b.setText(self._role_name(r))
        self.trackbar.set_lane_names(*self.team_names)
        self._refresh_team_label()
        self._refresh_player_list()

    def _rep(self, tid):
        """트랙릿의 병합 대표 tid (병합 없으면 자기 자신)."""
        return self.merges.get(int(tid), int(tid))

    def _role_of(self, tid):
        """유효 역할 — 병합 그룹은 대표 기준 (역할 데이터는 비파괴).

        대표의 역할이 우선이라 팀 분류가 틀린 조각도 그룹에 넣으면
        표시가 바로잡히고, 분리하면 원래 분류로 돌아간다.
        """
        rep = self.merges.get(tid, tid)
        if rep in self.roles:
            return self.roles[rep]
        if tid in self.roles:
            return self.roles[tid]
        return self._teams.get(rep, self._teams.get(tid, 2))

    def _role_color(self, role):
        """역할의 표시색(BGR) — 사용자 지정 > 측정 대표색 > 기본 팔레트."""
        if role in self.kit_colors:
            return tuple(self.kit_colors[role])
        return self._role_colors.get(
            role, TEAM_COLORS[min(max(role, 0), len(TEAM_COLORS) - 1)])

    def _refresh_team_label(self, log_colors=False):
        """팀/GK/심판별 대표 유니폼 색 범례 (분석 후 '이 팀이 이 색').

        스와치 버튼 색 = 표시색(사용자 지정 우선, 없으면 측정 대표색).
        """
        spans, cols = self._player_cache()
        for r in self._kit_btns:
            self._kit_btns[r].hide()
            self._kit_lbls[r].setText("")
        if not self._teams or not cols:
            self.lbl_team_colors.setText("팀 색: 분석 후 표시")
            return
        logs = []
        self._role_colors = {}
        shown = 0
        for r in (0, 1, 3, 4, 5):
            member = [t for t in cols
                      if self._role_of(t) == r and spans.get(t, [0, 0, 0])[2] >= 5]
            if not member:
                continue
            # 대표색: 검출 수 가중 없이 BGR 중앙값 (스와치 용도라 충분)
            bgr = np.median(np.array(
                [cv2.cvtColor(np.uint8([[[int(cols[t][0]) % 180,
                                          int(min(cols[t][1], 255)),
                                          int(min(cols[t][2], 255))]]]),
                              cv2.COLOR_HSV2BGR)[0, 0] for t in member]), axis=0)
            self._role_colors[r] = _boost_bgr(
                (int(bgr[0]), int(bgr[1]), int(bgr[2])))
            b_, g_, r_ = self._role_color(r)          # 사용자 지정 우선
            hexc = f"#{r_:02x}{g_:02x}{b_:02x}"
            self._kit_btns[r].setStyleSheet(
                f"background-color: {hexc}; border: 1px solid #666;")
            self._kit_btns[r].show()
            mark = "✎" if r in self.kit_colors else ""
            self._kit_lbls[r].setText(f"{self._role_name(r)}{mark} {len(member)}명")
            shown += 1
            logs.append(f"{self._role_name(r)} {hexc} ({len(member)}트랙릿)")
        self.lbl_team_colors.setText("유니폼 색:" if shown
                                     else "팀 색: 분석 후 표시")
        self.radar.palette = {r: self._role_color(r) for r in range(6)}
        if log_colors and logs:
            self.log("[ptz] 팀 색 분류: " + ", ".join(logs))

    def _pick_kit_color(self, role):
        """범례 스와치 클릭 → 컬러피커로 역할 표시 색 지정."""
        b, g, r = self._role_color(role)
        c = QColorDialog.getColor(QColor(r, g, b), self,
                                  f"{self._role_name(role)} 표시 색")
        if not c.isValid():
            return
        self.kit_colors[role] = [c.blue(), c.green(), c.red()]
        self._kit_colors_changed()
        self.log(f"[ptz] {self._role_name(role)} 표시 색 지정: {c.name()}")

    def _reset_kit_color(self, role):
        if self.kit_colors.pop(role, None) is not None:
            self._kit_colors_changed()
            self.log(f"[ptz] {self._role_name(role)} 표시 색 → 측정색으로 복귀")

    def _kit_colors_changed(self):
        self._save_keyframes()
        self._refresh_team_label()
        self._redraw()

    def _refresh_player_list(self):
        """선수 트랙릿 목록: 유니폼 색 스와치 + 역할 + 구간 (검출 수 순)."""
        spans, cols = self._player_cache()
        # 재구성 전 선택을 tid 로 기억해 복원 — 역할 지정 등으로 목록이
        # 다시 만들어질 때 선택이 리셋되면, 이후 포커스 이동이 첫 행을
        # 선택하며 엉뚱한 프레임으로 점프하는 문제가 있었다.
        old_rows = getattr(self, "_player_rows", [])
        cur = self.player_list.currentRow()
        prev_cur = old_rows[cur] if 0 <= cur < len(old_rows) else None
        prev_sel = {old_rows[i.row()] for i in self.player_list.selectedIndexes()
                    if i.row() < len(old_rows)}
        # 병합 그룹: 대표 아래 멤버를 들여쓰기로 (해제/분리를 위해 표시 유지)
        groups: dict[int, list] = {}
        for t in spans:
            groups.setdefault(self._rep(t), []).append(t)
        agg = {rep: [min(spans[t][0] for t in ms),
                     max(spans[t][1] for t in ms),
                     sum(spans[t][2] for t in ms)]
               for rep, ms in groups.items()}
        # GK 는 목록 하단 그룹으로 (그 안에선 검출 수 순)
        reps = sorted(groups, key=lambda t: (self._role_of(t) in (3, 4),
                                             -agg[t][2]))
        self._player_rows = []
        for rep in reps:
            self._player_rows.append(rep)
            self._player_rows += sorted(
                (t for t in groups[rep] if t != rep),
                key=lambda t: spans[t][0])
        self.player_list.blockSignals(True)
        self.player_list.clear()
        for tid in self._player_rows:
            rep = self._rep(tid)
            k = len(groups.get(tid, ()))
            f0, f1, n = agg[tid] if tid == rep else spans[tid]
            r = self._role_of(tid)
            t0, t1 = f0 / self.fps, f1 / self.fps
            mark = "● " if (tid in self.roles or rep in self.roles) else ""
            head = (f"  ↳ #{tid}" if tid != rep
                    else f"#{tid}" + (f" (+{k - 1})" if k > 1 else ""))
            it = QListWidgetItem(
                f"{head}  {mark}{self._role_name(r)}  "
                f"{int(t0//60):02d}:{t0%60:04.1f}~"
                f"{int(t1//60):02d}:{t1%60:04.1f}  ({n}회)")
            if tid in cols:
                px = QPixmap(14, 14)
                px.fill(QColor(_hsv_hex(cols[tid])))
                it.setIcon(QIcon(px))
            self.player_list.addItem(it)
        for row, tid in enumerate(self._player_rows):     # 선택 복원
            if tid in prev_sel:
                self.player_list.item(row).setSelected(True)
            if tid == prev_cur:
                self.player_list.setCurrentRow(row)
        self.player_list.blockSignals(False)
        self.trackbar.set_players(
            {t: (spans[t][0], spans[t][1], self._role_of(t)) for t in spans})

    def _goto_player(self):
        row = self.player_list.currentRow()
        rows = getattr(self, "_player_rows", [])
        if 0 <= row < len(rows):
            spans, _ = self._player_cache()
            self.slider.setValue(int(spans[rows[row]][0]))
            self.trackbar.set_selection("player", rows[row])

    def start_gapfill(self):
        """분석 메뉴: 갭필 2차 패스 — 트랙 갭을 저문턱 검출로 메꿈."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "갭필", "먼저 분석이 필요합니다.")
            return
        if self._gapfill_worker is not None and self._gapfill_worker.isRunning():
            self._gapfill_worker.cancel()
            self.log("[gapfill] 취소 요청")
            return
        targets = gapfill_targets(
            self.analysis,
            ignore_ranges=[tuple(r) for r in self.ignores],
            force_ranges=[tuple(p) for p in self.promotes],
            linked=self._linked)
        if not targets:
            QMessageBox.information(self, "갭필",
                                    "메꿀 갭이 없습니다 (≤4초 갭 기준).")
            return
        est = len(targets) * 0.17 / 60
        if QMessageBox.question(
                self, "갭필 2차 패스",
                f"트랙 갭 {len(targets)}지점을 저문턱 재검출합니다 "
                f"(예상 ~{est:.0f}분). 진행할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        w = GapfillWorker(str(self.pano_path), self.analysis, targets,
                          weights=self._model_weights())
        w.progress.connect(lambda i, t, f: (
            self.progress.setRange(0, t), self.progress.setValue(i),
            self.progress.setFormat(f"갭필 %p% ({f:.1f}/s)")))
        w.log.connect(self.log)
        w.done.connect(self._gapfill_done)
        w.failed.connect(lambda e: self.log(f"[gapfill] 실패: {e}"))
        self._gapfill_worker = w
        self.log(f"[gapfill] 시작: {len(targets)}지점")
        w.start()

    def _gapfill_done(self, analysis, nb, np_):
        self.analysis = analysis
        self._write_analysis()            # 주입 반영해 .analysis.json 갱신
        self._pcache_id = None            # 선수 캐시 무효화 (주입분 반영)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("갭필 완료")
        self.log(f"[gapfill] 완료: 공 {nb}개, 선수 {np_}명 주입 — 트랙 재연결")
        self._start_link()                # 링크·수락 재계산 → 목록/타임라인 갱신

    def detect_events(self):
        """분석 메뉴: 호각 × 대형 → 킥오프 검출 → .events.json + 타임라인."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "이벤트", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(self, "이벤트",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(대형 판정에 필드 좌표 사용).")
            return
        from ..core.audio import load_whistle_track
        from ..core.events import detect_kickoffs, formation_track, \
            save_events
        _, whistles = load_whistle_track(self.pano_path)
        if not whistles:
            QMessageBox.information(
                self, "이벤트",
                "호각 트랙이 없습니다 — scripts/whistle.py 를 먼저 "
                "실행하세요 (오디오 추출, ~20초).")
            return
        with self._busy("킥오프 검출 (대형 × 호각 융합)"):
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            tr = formation_track(self.analysis, teams, self._field_calib)
            ks = detect_kickoffs(tr, whistles)
            save_events(self.pano_path, ks)
        self._refresh_events()
        times = ", ".join(self._hms(t) for t, _, _ in ks) or "없음"
        self.log(f"[events] 킥오프 {len(ks)}개: {times}")

    def detect_highlights(self):
        """분석 메뉴: 이벤트 융합 → 하이라이트 후보 (.events.json, 비파괴)."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "하이라이트", "먼저 분석이 필요합니다.")
            return
        from ..core.audio import load_whistle_track
        from ..core.events import load_events_doc, save_events
        from ..core.highlights import (
            airborne_box_events, ball_speed_events, build_highlights,
            carry_states,
        )
        with self._busy("하이라이트 후보 생성 (이벤트 융합)"):
            doc = load_events_doc(self.pano_path)
            _, whistles = load_whistle_track(self.pano_path)
            air_ev, speed_ev = [], []
            if self._field_calib is not None:
                calib = self._field_calib
                air = doc.get("airborne") or {}
                air_ev = airborne_box_events(air.get("segments") or [],
                                             calib["length"], calib["width"])
                if self._accepted_ball is not None:
                    acc = np.asarray(self._accepted_ball, dtype=float)
                    fin = np.isfinite(acc[:, 0])
                    g = np.full((len(acc), 2), np.nan)
                    if fin.any():
                        g[fin] = pano_to_field(calib, acc[fin])
                    t = np.asarray(self.analysis["frames"], float) / self.fps
                    speed_ev = ball_speed_events(t, g)
            segs = build_highlights(
                self.total / self.fps,
                kickoffs=doc.get("kickoffs") or [],
                whistles=whistles or [],
                signals=doc.get("linesman_signals") or [],
                air_events=air_ev, speed_events=speed_ev,
                user_events=[(f / self.fps, lb) for f, lb in self.user_events])
            self.highlights = carry_states(segs, self.highlights)
            save_events(self.pano_path, highlights=self.highlights)
        self._refresh_highlight_lane()
        n_acc = sum(1 for h in self.highlights if h.get("state") == "accept")
        srcs = (f"킥오프 {len(doc.get('kickoffs') or [])}, 기 신호 "
                f"{sum(1 for s in doc.get('linesman_signals') or [] if (s.get('near') or {}).get('signal') in ('foul', 'offside'))}, "
                f"공중볼→박스 {len(air_ev)}, 속도 급증 {len(speed_ev)}, "
                f"사용자 {len(self.user_events)}")
        self.log(f"[hl] 하이라이트 후보 {len(self.highlights)}개 "
                 f"(수락 {n_acc}) — 소스: {srcs}")
        self.log("[hl] 이벤트 레인의 호박색 바 우클릭 → 수락/제외/경계 조정")
        if self._field_calib is None:
            self.log("[hl] 경기장 캘리브레이션 없음 — 공중볼/속도 규칙 생략됨")

    # ------------------------------------------------------ 하이라이트 검수
    def _add_manual_highlight(self, f=None):
        """수동 하이라이트 추가 — 자동이 놓친 장면 (빗나간 슛 등).

        f 기준 ±8초, f=None 이면 현재 IN/OUT 마커 구간. 수동 추가는
        검수가 끝난 것이므로 바로 수락 상태 — 재생성해도 보존된다.
        """
        if f is None:
            r = self._norm_export_range()
            if r is None:
                return
            t0, t1 = r[0] / self.fps, r[1] / self.fps
        else:
            t0 = max(0.0, f / self.fps - 8.0)
            t1 = min(self.total / self.fps, f / self.fps + 8.0)
        from PyQt6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getText(
            self, "하이라이트 추가",
            f"{self._hms(t0)} ~ {self._hms(t1)} 라벨 (예: 빗나간 슛):")
        if not ok or not label.strip():
            return
        self.highlights.append({"t0": round(t0, 2), "t1": round(t1, 2),
                                "kinds": ["user"], "label": label.strip(),
                                "score": 5.0, "state": "accept"})
        self.highlights.sort(key=lambda h: h["t0"])
        self._save_highlights()
        self.log(f"[hl] 수동 하이라이트 '{label.strip()}' "
                 f"{self._hms(t0)}~{self._hms(t1)} (수락 상태)")

    def _set_hl_state(self, i, state):
        if not 0 <= i < len(self.highlights):
            return
        h = self.highlights[i]
        h["state"] = state
        self._save_highlights()
        name = {"accept": "수락", "reject": "제외", "cand": "후보로 되돌림"}
        self.log(f"[hl] {name[state]}: {h.get('label', '')} "
                 f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def _hl_to_marks(self, i):
        """하이라이트 경계 → IN/OUT 마커 (미리보기·경계 조정용)."""
        h = self.highlights[i]
        self._set_export_mark("in", int(h["t0"] * self.fps))
        self._set_export_mark("out", int(h["t1"] * self.fps))
        self.slider.setValue(int(h["t0"] * self.fps))

    def _marks_to_hl(self, i):
        """현재 IN/OUT 마커 → 하이라이트 경계 (조정 커밋)."""
        r = self._norm_export_range()
        if r is None:
            return
        h = self.highlights[i]
        h["t0"], h["t1"] = round(r[0] / self.fps, 2), round(r[1] / self.fps, 2)
        self._save_highlights()
        self.log(f"[hl] 경계 갱신: {h.get('label', '')} "
                 f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def _del_highlight(self, i):
        if 0 <= i < len(self.highlights):
            h = self.highlights.pop(i)
            self._save_highlights()
            self.log(f"[hl] 삭제: {h.get('label', '')} "
                     f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")

    def export_highlights(self):
        """분석 메뉴: 검수된 하이라이트 구간들을 개별 클립으로 일괄 렌더."""
        if self.analysis is None or self.pano_path is None:
            QMessageBox.information(self, "하이라이트", "먼저 분석이 필요합니다.")
            return
        if self._render_worker is not None and self._render_worker.isRunning():
            QMessageBox.information(self, "하이라이트",
                                    "내보내기가 이미 진행 중입니다.")
            return
        cands = [h for h in self.highlights if h.get("state") != "reject"]
        if not cands:
            QMessageBox.information(
                self, "하이라이트",
                "내보낼 하이라이트가 없습니다 — 분석 메뉴에서 "
                "\"하이라이트 후보 생성\"을 먼저 실행하세요.")
            return
        self._stop_play()
        st = QSettings("PyStitch360", "PyStitch360")
        clock_avail = self._clock_config() is not None
        dlg = HighlightExportDialog(
            self, cands, self.fps, self.encoders,
            int(st.value("ptz_export_crf", 20)),
            st.value("ptz_export_radar", "true") == "true",
            str(self.pano_path.parent),
            clock_on=(st.value("ptz_export_clock", "true") == "true"
                      if clock_avail else None))
        if not dlg.exec():
            return
        cfg = dlg.config()
        if not cfg["indices"] or not cfg["dir"]:
            return
        st.setValue("ptz_export_crf", cfg["crf"])
        st.setValue("ptz_export_radar", "true" if cfg["radar"] else "false")
        if clock_avail:
            st.setValue("ptz_export_clock", "true" if cfg["clock"] else "false")
        import re
        segs = []
        for i in cfg["indices"]:
            h = cands[i]
            name = re.sub(r'[\\/:*?"<>|\s]+', "_",
                          h.get("label", "")).strip("_") or "장면"
            segs.append((int(h["t0"] * self.fps), int(h["t1"] * self.fps),
                         name))
        radar = None
        if cfg["radar"]:
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            radar = build_radar_data(
                self.analysis, teams, calib=self._field_calib,
                field_size=tuple(self.field_size),
                extra_players=self.extra_players,
                palette={r: self._role_color(r) for r in range(6)})
        dur = sum(e - s for s, e, _ in segs) / self.fps
        self.log(f"[hl] 일괄 내보내기 시작: {len(segs)}개 구간, "
                 f"총 {dur/60:.1f}분 → {cfg['dir']}")
        w = HighlightBatchWorker(
            str(self.pano_path), cfg["dir"], self.pano_path.stem,
            self.analysis, [tuple(k) for k in self.keyframes],
            self.encoders[cfg["codec_name"]], cfg["crf"],
            ignores=[tuple(r) for r in self.ignores],
            far_zoom=self.spin_far_zoom.value(),
            promotes=[tuple(p) for p in self.promotes],
            radar=radar, segments=segs,
            clock=self._clock_config() if cfg["clock"] else None)
        w.log.connect(self.log)
        w.progress.connect(self._render_progress)
        w.finished_ok.connect(self._batch_done)
        w.failed.connect(self._render_failed)
        self._render_worker = w
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("준비 중...")
        w.start()

    # ------------------------------------------------------ 경기 정보/시계
    def edit_match_info(self):
        """분석 메뉴: 경기 정보 (시계 앵커·하프·중단 구간) 입력."""
        if self.pano_path is None:
            QMessageBox.information(self, "경기 정보", "열린 파노라마가 없습니다.")
            return
        kicks = []
        try:
            from ..core.events import load_events
            kicks = load_events(self.pano_path)
        except Exception as e:  # noqa: BLE001
            self.log(f"[clock] 킥오프 목록 무시: {e}")
        dlg = MatchInfoDialog(self, self.fps, self.total, kicks,
                              self.match_info, int(self.slider.value()))
        if not dlg.exec():
            return
        self.match_info = dlg.config()
        self._save_keyframes()
        mi = self.match_info
        anchor = ("미지정" if mi["anchor_f"] is None
                  else self._hms(mi["anchor_f"] / self.fps))
        self.log(f"[clock] 경기 정보: {'후반' if mi['half'] == 2 else '전반'} "
                 f"{mi['half_len_min']:.0f}분, 앵커 {anchor}, "
                 f"중단 {len(mi.get('pauses') or [])}개")

    def _clock_config(self):
        """render_plan 용 시계 설정 — 앵커 미지정이면 None.

        표기는 분:초 누적 (후반 +하프길이, 연장 90/120분도 분이 계속
        커짐). cv2 폰트 제약으로 태그는 1H/2H, 비ASCII 팀 이름은 T1/T2.
        """
        mi = self.match_info or {}
        if mi.get("anchor_f") is None:
            return None
        half = 2 if int(mi.get("half", 1)) == 2 else 1
        base = (float(mi.get("half_len_min", 45.0)) * 60.0
                if half == 2 and mi.get("cumulative", True) else 0.0)
        goals = []
        for f, lb in self.user_events:
            lb = str(lb).replace(" ", "")
            if lb == "골1":
                goals.append([int(f), 1])
            elif lb == "골2":
                goals.append([int(f), 2])
        score = None
        if goals:
            names = [n.strip() if n.strip() and n.isascii() else f"T{i + 1}"
                     for i, n in enumerate(self.team_names)]
            score = (names[0], names[1], goals)
        return {"anchor_f": int(mi["anchor_f"]), "fps": self.fps,
                "base_s": base, "tag": "2H" if half == 2 else "1H",
                "pauses": [[int(a), int(b)]
                           for a, b in mi.get("pauses") or []],
                "score": score}

    def _add_pause_range(self):
        """IN/OUT 마커 구간 → 경기 중단 (시계 정지, hydration break 등)."""
        r = self._norm_export_range()
        if r is None:
            return
        mi = self.match_info or {"half": 1, "half_len_min": 45.0,
                                 "anchor_f": None, "cumulative": True,
                                 "pauses": []}
        mi.setdefault("pauses", []).append([int(r[0]), int(r[1])])
        mi["pauses"].sort()
        self.match_info = mi
        self._save_keyframes()
        self.trackbar.set_pauses(mi["pauses"])
        self.log(f"[clock] 경기 중단 구간 추가: "
                 f"{self._hms(r[0] / self.fps)}~{self._hms(r[1] / self.fps)}"
                 f" (시계 정지, 총 {len(mi['pauses'])}개)")

    def _del_pause(self, i):
        pauses = (self.match_info or {}).get("pauses") or []
        if 0 <= i < len(pauses):
            p0, p1 = pauses.pop(i)
            self._save_keyframes()
            self.trackbar.set_pauses(pauses)
            self.log(f"[clock] 경기 중단 구간 삭제: "
                     f"{self._hms(p0 / self.fps)}~{self._hms(p1 / self.fps)}")

    def _batch_done(self, n, out_dir):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("완료")
        self.log(f"[hl] 일괄 내보내기 완료: {n}개 클립 → {out_dir}")
        QMessageBox.information(self, "하이라이트",
                                f"{n}개 클립 저장 완료:\n{out_dir}")

    def reset_edits(self, scope="all"):
        """사용자 편집 무효화 → 순수 분석 원본 상태 (분석 메뉴에서 호출).

        분석(.analysis.json)은 검수로 안 바뀌므로 에디트만 지우면 원본이다.
        """
        names = {"ball": "공 트랙 편집 (키프레임·무시·승격)",
                 "roles": "선수 역할 지정",
                 "field": "경기장 캘리브레이션",
                 "all": "모든 사용자 편집"}
        if self.pano_path is None:
            QMessageBox.information(self, "초기화", "열린 파노라마가 없습니다.")
            return
        n_extra = sum(len(v) for v in self.extra_players.values())
        detail = (f"키프레임 {len(self.keyframes)} / 무시 {len(self.ignores)}"
                  f" / 승격 {len(self.promotes)} / 역할 {len(self.roles)}"
                  f" / 병합 {len(self.merges)}"
                  f" / 랜드마크 {len(self.field_points)}"
                  f" / 수동 검출 {n_extra}")
        if QMessageBox.question(
                self, "분석 원본으로 되돌리기",
                f"{names[scope]}을(를) 삭제하고 분석 결과 원본으로 "
                f"되돌립니다.\n(현재: {detail})\n계속할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        if scope in ("ball", "all"):
            self.keyframes, self.ignores, self.promotes = [], [], []
            self._box_commit = None
        if scope in ("roles", "all"):
            self.roles = {}
            self.merges = {}
            self.extra_players = {}
            self.kit_colors = {}
            self._pcache_id = None
        if scope in ("field", "all"):
            self.field_points = {}
            self.line_points = []
        if self.analysis is not None:
            self._teams = classify_teams(self.analysis, roles=self.roles)
        self._write_sidecar()
        self._refit_field()
        self._refresh_field_list()
        self._refresh_team_label()
        self._refresh_player_list()
        self._refresh_lists()
        self._recompute_tracks()
        self._plan_dirty()
        self._redraw()
        self.log(f"[ptz] {names[scope]} 초기화 — 분석 원본 상태로 되돌림")

    # ------------------------------------------------------ 내보내기 구간 마커
    def _norm_export_range(self):
        """정규화된 (f0, f1) — 마커가 하나도 없으면 None."""
        fi, fo = self.export_range
        if fi is None and fo is None:
            return None
        f0 = 0 if fi is None else int(fi)
        f1 = self.total if fo is None else int(fo)
        return (min(f0, f1), max(f0, f1))

    def _set_export_mark(self, kind, f, clear=False):
        if clear:
            self.export_range = [None, None]
            self.log("[ptz] 내보내기 구간 해제")
        else:
            i = 0 if kind == "in" else 1
            self.export_range[i] = int(f)
            self.log(f"[ptz] 내보내기 {'시작' if kind == 'in' else '끝'} "
                     f"마커 = {self._hms(f/self.fps, tenth=True)}")
        r = self._norm_export_range()
        self.trackbar.set_range(*(r if r else (None, None)))
        self._save_keyframes()

    def _timeline_menu(self, f, gpos):
        menu = QMenu(self)
        menu.addAction(f"내보내기 시작 지점 (I) — {self._hms(f/self.fps)}",
                       lambda: self._set_export_mark("in", f))
        menu.addAction(f"내보내기 끝 지점 (O) — {self._hms(f/self.fps)}",
                       lambda: self._set_export_mark("out", f))
        if self._norm_export_range():
            menu.addSeparator()
            menu.addAction("내보내기 구간 해제",
                           lambda: self._set_export_mark("", 0, clear=True))
        menu.addSeparator()
        menu.addAction("여기에 이벤트 추가...",
                       lambda: self._add_user_event(f))
        near = [i for i, (uf, _l) in enumerate(self.user_events)
                if abs(uf - f) <= 3 * self.fps]
        for i in near:
            menu.addAction(f"이벤트 '{self.user_events[i][1]}' 삭제",
                           lambda _=False, ii=i: self._del_user_event(ii))
        menu.addSeparator()
        menu.addAction("여기에 하이라이트 추가 (±8초)...",
                       lambda: self._add_manual_highlight(f))
        if self._norm_export_range():
            menu.addAction("IN/OUT 마커 구간 → 새 하이라이트...",
                           lambda: self._add_manual_highlight(None))
            menu.addAction("IN/OUT → 경기 중단 구간 (시계 정지)",
                           self._add_pause_range)
        for i, (p0, p1) in enumerate(
                (self.match_info or {}).get("pauses") or []):
            if p0 <= f <= p1:
                menu.addAction(
                    f"경기 중단 구간 삭제 — {self._hms(p0 / self.fps)}~"
                    f"{self._hms(p1 / self.fps)}",
                    lambda _=False, ii=i: self._del_pause(ii))
        hl = [i for i, h in enumerate(self.highlights)
              if h["t0"] * self.fps <= f <= h["t1"] * self.fps]
        for i in hl:
            h = self.highlights[i]
            tag = (f"'{h.get('label', '')}' "
                   f"{self._hms(h['t0'])}~{self._hms(h['t1'])}")
            menu.addSeparator()
            st = h.get("state", "cand")
            if st != "accept":
                menu.addAction(f"하이라이트 수락 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "accept"))
            if st != "reject":
                menu.addAction(f"하이라이트 제외 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "reject"))
            if st != "cand":
                menu.addAction(f"하이라이트 후보로 되돌림 — {tag}",
                               lambda _=False, ii=i:
                               self._set_hl_state(ii, "cand"))
            menu.addAction("IN/OUT 마커 ← 하이라이트 경계 (조정 시작)",
                           lambda _=False, ii=i: self._hl_to_marks(ii))
            if self._norm_export_range():
                menu.addAction("하이라이트 경계 ← 현재 IN/OUT 마커 (조정 커밋)",
                               lambda _=False, ii=i: self._marks_to_hl(ii))
            menu.addAction(f"하이라이트 삭제 — {tag}",
                           lambda _=False, ii=i: self._del_highlight(ii))
        menu.addSeparator()
        menu.addAction("여기로 이동", lambda: self.slider.setValue(int(f)))
        menu.exec(gpos)

    def _add_user_event(self, f):
        from PyQt6.QtWidgets import QInputDialog
        label, ok = QInputDialog.getText(
            self, "이벤트 추가",
            f"{self._hms(f/self.fps)} 이벤트 이름 (예: 골, 코너킥):")
        if not ok or not label.strip():
            return
        self.user_events.append([int(f), label.strip()])
        self.user_events.sort()
        self._save_keyframes()
        self._refresh_events()
        self.log(f"[ptz] 이벤트 '{label.strip()}' @ {self._hms(f/self.fps)}")

    def _del_user_event(self, i):
        if 0 <= i < len(self.user_events):
            f, label = self.user_events.pop(i)
            self._save_keyframes()
            self._refresh_events()
            self.log(f"[ptz] 이벤트 '{label}' 삭제")

    def _tl_view_changed(self, t0, vis, total):
        """타임라인 줌/팬 → 스크롤바 동기화 (전체 보기면 숨김)."""
        if vis >= total - 1 or total <= 1:
            self.tl_scroll.hide()
            return
        self.tl_scroll.blockSignals(True)
        self.tl_scroll.setRange(0, int(total - vis))
        self.tl_scroll.setPageStep(max(1, int(vis)))
        self.tl_scroll.setSingleStep(max(1, int(vis / 20)))
        self.tl_scroll.setValue(int(t0))
        self.tl_scroll.blockSignals(False)
        self.tl_scroll.show()

    def _timeline_pick(self, kind, key):
        """타임라인 바 클릭 → 해당 목록 선택(→ 프레임 이동)."""
        if kind in ("kf", "ball"):
            want = ("kf" if kind == "kf" else "track", key)
            for row, e in enumerate(getattr(self, "_top", [])):
                if e == want:
                    self.track_list.setCurrentRow(row)   # currentRowChanged=이동
                    return
            if kind == "kf" and 0 <= key < len(self.keyframes):
                self.slider.setValue(int(self.keyframes[key][0]))
        elif kind == "ignore":
            if 0 <= key < len(self.ignores):
                self.kf_list.setCurrentRow(key)
                self.slider.setValue(int(self.ignores[key][0]))
        elif kind == "player":
            rows = getattr(self, "_player_rows", [])
            if key in rows:
                self.player_list.setCurrentRow(rows.index(key))
            else:
                spans, _ = self._player_cache()
                if key in spans:
                    self.slider.setValue(int(spans[key][0]))
        elif kind == "event":
            evs = self.trackbar.events
            if 0 <= key < len(evs):
                self.slider.setValue(int(evs[key][0]))
        elif kind == "hl":
            if 0 <= key < len(self.highlights):
                self.slider.setValue(
                    int(self.highlights[key]["t0"] * self.fps))

    def _assign_selected_role(self, role):
        rows = getattr(self, "_player_rows", [])
        # 병합 멤버 선택은 대표에 지정 — 그룹(같은 사람) 전체에 적용
        sel = {self._rep(rows[i.row()])
               for i in self.player_list.selectedIndexes()
               if i.row() < len(rows)}
        for tid in sel:
            if role is None:
                self.roles.pop(tid, None)
            else:
                self.roles[tid] = role
        if sel:
            self._roles_changed()

    def _set_role(self, tid, role):
        if role is None:
            self.roles.pop(int(tid), None)
        else:
            self.roles[int(tid)] = int(role)
        self._roles_changed()

    def _roles_changed(self):
        if self.analysis is not None:
            self._teams = classify_teams(self.analysis, roles=self.roles)
        self._refresh_team_label()
        self._refresh_player_list()
        self._save_keyframes()
        self._redraw()

    # ------------------------------------------------------ 트랙릿 병합
    def suggest_tracklet_merges(self):
        """분석 메뉴: 트랙릿 병합 제안 (시공간 × 유니폼색 × 역할).

        기존 병합(수동 포함)은 유지하고 새 링크를 합친다 — 비파괴라
        목록 우클릭으로 언제든 그룹 해체/멤버 분리/추가 가능.
        """
        if self.analysis is None:
            QMessageBox.information(self, "병합", "먼저 분석이 필요합니다.")
            return
        if self._field_calib is None:
            QMessageBox.information(self, "병합",
                                    "경기장 캘리브레이션이 필요합니다 "
                                    "(시공간 근접 판단에 필드 좌표 사용).")
            return
        from ..core.tracklets import (
            merge_map, suggest_links, tracklet_summaries,
        )
        spans, _ = self._player_cache()
        n_before = len({self._rep(t) for t in spans})
        with self._busy("트랙릿 병합 제안 (시공간 × 유니폼색 × 역할)"):
            summ = tracklet_summaries(self.analysis, self._field_calib)
            roles_eff = {t: self._role_of(t) for t in summ}
            links = suggest_links(summ, roles_eff)
            all_links = ([(a, b) for a, b, _ in links]
                         + list(self.merges.items()))
            self.merges = merge_map(all_links,
                                    {t: spans[t][2] for t in spans})
        self._merges_changed()
        n_after = len({self._rep(t) for t in spans})
        self.log(f"[merge] 링크 {len(links)}개 제안 — 트랙릿 {len(spans)}개"
                 f" → 그룹 {n_before}→{n_after}개 "
                 f"(선수 목록 우클릭으로 해제/분리/추가)")

    def _merges_changed(self):
        self._save_keyframes()
        self._refresh_team_label()
        self._refresh_player_list()
        self._redraw()

    def _player_menu(self, pos):
        """선수 목록 우클릭 — 병합/분리/해체 (전부 되돌리기 가능)."""
        rows = getattr(self, "_player_rows", [])
        sel = sorted({rows[i.row()] for i in self.player_list.selectedIndexes()
                      if i.row() < len(rows)})
        menu = QMenu(self)
        if len(sel) >= 2:
            menu.addAction(
                f"선택 {len(sel)}개 트랙릿 병합 (같은 사람)",
                lambda: self._merge_tracklets(sel))
        it = self.player_list.itemAt(pos)
        tid = (rows[self.player_list.row(it)]
               if it is not None and self.player_list.row(it) < len(rows)
               else None)
        if tid is not None:
            rep = self._rep(tid)
            group = [t for t in rows if self._rep(t) == rep]
            if tid != rep:
                menu.addAction(f"#{tid} 그룹에서 분리 (원래 분류로)",
                               lambda: self._split_tracklet(tid))
            if len(group) > 1:
                menu.addAction(
                    f"그룹 해체 — #{rep} 외 {len(group) - 1}개 전부 분리",
                    lambda: self._dissolve_group(rep))
        if not menu.isEmpty():
            menu.exec(self.player_list.mapToGlobal(pos))

    def _merge_tracklets(self, tids):
        """선택 트랙릿 수동 병합 — 기존 그룹과 union (빠진 조각 추가 포함)."""
        from ..core.tracklets import merge_map
        spans, _ = self._player_cache()
        links = ([(tids[0], t) for t in tids[1:]]
                 + list(self.merges.items()))
        self.merges = merge_map(links, {t: spans[t][2] for t in spans})
        self._merges_changed()
        self.log(f"[merge] 수동 병합: {', '.join(f'#{t}' for t in tids)}"
                 f" → 대표 #{self._rep(tids[0])}")

    def _split_tracklet(self, tid):
        """멤버 하나만 그룹에서 분리 — 역할은 원래 분류로 돌아간다."""
        rep = self.merges.pop(int(tid), None)
        if rep is None:
            return
        self._merges_changed()
        self.log(f"[merge] #{tid} 를 그룹 #{rep} 에서 분리")

    def _dissolve_group(self, rep):
        n = sum(1 for r in self.merges.values() if r == rep)
        self.merges = {t: r for t, r in self.merges.items() if r != rep}
        self._merges_changed()
        self.log(f"[merge] 그룹 #{rep} 해체 ({n}개 분리)")

    # ------------------------------------------------------ 경기장 캘리브레이션
    def _review_tab_changed(self, idx):
        if idx != 2:                      # 경기장 탭을 떠나면 찍기 모드 해제
            self.btn_field_pick.setChecked(False)
        self._redraw()

    def _field_tab_active(self):
        return self.tabs_review.currentIndex() == 2

    def _cam_field_pos(self):
        """카메라의 필드 좌표 (캘리브레이션 전엔 기본 가정값)."""
        if self._field_calib is not None:
            return (self._field_calib["ex"], self._field_calib["ey"])
        return (0.0, -(self.field_size[1] / 2.0 + 5.0))

    def _click_to_field(self, x, y):
        """클릭 픽셀 → 필드 좌표. 캘리브레이션 전엔 기본 카메라 모델."""
        if self._field_calib is not None:
            fx, fy = pano_to_field(self._field_calib, [[x, y]])[0]
            return (fx, fy) if np.isfinite(fx) else None
        g = ground_positions([[x, y, 0.0, 0.0]], self.pano_w, self.pano_h)
        if not g:
            return None
        cx, cy = self._cam_field_pos()
        return (cx + g[0][0], cy + g[0][1])

    # 캘리브레이션 전 자동 매칭 후보: 간격이 넓은 최외곽 점들만.
    # 페널티박스·센터서클은 기본 카메라 모델의 거리 오차(수십 m)보다
    # 이웃 간격이 좁아 오배정되기 쉬움 — 피팅이 잡힌 뒤(픽셀 매칭)에만.
    _PRECALIB_KEYS = ("corner_far_l", "corner_far_r", "corner_near_l",
                      "corner_near_r", "half_far", "half_near",
                      "sideline_near_l", "sideline_near_r")

    def _match_positions(self):
        pos = dict(landmark_positions(*self.field_size))
        # 선 위 점(사이드라인)은 대표 위치로 매칭: 카메라 좌우 1/4 지점
        hl, hw = self.field_size[0] / 2.0, self.field_size[1] / 2.0
        pos["sideline_near_l"] = (-hl / 2.0, -hw)
        pos["sideline_near_r"] = (hl / 2.0, -hw)
        pos["center_near"] = (0.0, -hw + 5.0)   # 중앙선 가까운 끝 대표점
        return pos

    def _match_landmark(self, x, y):
        """클릭 위치를 가장 그럴듯한 랜드마크에 휴리스틱 매칭.

        피팅 후: 전체 랜드마크를 화면에 투영해 픽셀 최근접 (정확).
        피팅 전: 외곽 8종만, 카메라 기준 방향(요)을 강하게·거리는
        로그 비율로 느슨하게 비교 — 기본 모델의 높이/거리 오차가
        모든 점을 같은 배율로 밀기 때문에 방향이 훨씬 믿을 만하다.
        미지정 랜드마크 우선, 전부 지정됐으면 최근접 이동(재클릭=수정).
        """
        pos = self._match_positions()
        if self._field_calib is not None:
            keys = list(pos)
            px = field_to_pano(self._field_calib, [pos[k] for k in keys])
            order = sorted(
                range(len(keys)),
                key=lambda i: (px[i][0] - x) ** 2 + (px[i][1] - y) ** 2
                if np.isfinite(px[i][0]) else np.inf)
            for i in order:
                if keys[i] not in self.field_points:
                    return keys[i]
            return keys[order[0]]
        f = self._click_to_field(x, y)
        if f is None:
            return None
        cx, cy = self._cam_field_pos()
        a_est = np.arctan2(f[0] - cx, f[1] - cy)
        d_est = max(np.hypot(f[0] - cx, f[1] - cy), 1e-6)

        def score(k):
            lx, ly = pos[k]
            da = a_est - np.arctan2(lx - cx, ly - cy)
            dd = np.log(d_est / max(np.hypot(lx - cx, ly - cy), 1e-6))
            return (da / 0.10) ** 2 + (dd / 0.4) ** 2

        order = sorted(self._PRECALIB_KEYS, key=score)
        for k in order:
            if k not in self.field_points:
                return k
        return order[0]

    def _field_set_point(self, key, x, y):
        self.field_points[key] = [round(float(x), 1), round(float(y), 1)]
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_reassign(self, old, new):
        """찍힌 마커의 랜드마크 종류 변경 — 대상이 이미 있으면 서로 교환."""
        pt = self.field_points.pop(old, None)
        if pt is None or old == new:
            return
        if new in self.field_points:
            self.field_points[old] = self.field_points[new]
        self.field_points[new] = pt
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_remove_point(self, key):
        if self.field_points.pop(key, None) is not None:
            self._refit_field()
            self._save_keyframes()
            self._refresh_field_list()
            self._redraw()

    def _field_clear_selected(self):
        row = self.field_list.currentRow()
        if 0 <= row < len(LANDMARKS):
            self._field_remove_point(LANDMARKS[row][0])

    def _field_clear_all(self):
        if not self.field_points:
            return
        if QMessageBox.question(
                self, "랜드마크 전체 삭제",
                f"찍은 랜드마크 {len(self.field_points)}개를 모두 "
                "삭제할까요?") != QMessageBox.StandardButton.Yes:
            return
        self.field_points = {}
        self.line_points = []
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._refresh_field_list()
        self._redraw()

    def _field_pick_toggled(self, on):
        self._refit_field()          # 상태 라벨의 '다음 찍을 점' 안내 갱신
        self._refresh_field_list()   # 다음 항목 자동 선택
        self._redraw()

    def _field_next_key(self):
        """권장 순서상 다음 미지정 랜드마크 (없으면 None)."""
        for key, name, req in LANDMARKS:
            if key not in self.field_points:
                return key
        return None

    def _field_size_changed(self, _v=None):
        self.field_size = [self.spin_field_len.value(),
                           self.spin_field_w.value()]
        self._refit_field()
        self._save_keyframes()
        self._redraw()

    def _refresh_field_list(self):
        row = self.field_list.currentRow()
        self.field_list.clear()
        for i, (key, name, req) in enumerate(LANDMARKS):
            p = self.field_points.get(key)
            mark = "★" if req else "☆"
            where = f"({p[0]:.0f}, {p[1]:.0f})" if p else "— 미지정"
            self.field_list.addItem(
                f"{i+1:2d}. {mark} [{LANDMARK_TAGS[key]}] {name}  {where}")
        nk = self._field_next_key() if self.btn_field_pick.isChecked() else None
        if nk is not None:                   # 찍기 모드: 다음 권장 점 하이라이트
            self.field_list.setCurrentRow(
                [k for k, _, _ in LANDMARKS].index(nk))
        elif row >= 0:
            self.field_list.setCurrentRow(row)

    def _refine_sideline(self):
        """예측 사이드라인 주변 흰 픽셀 검출 → 라인 샘플로 재피팅."""
        if self._field_calib is None:
            self.log("[field] 캘리브레이션 먼저 (랜드마크 4점 이상)")
            return
        f = getattr(self, "_cur_frame_idx", self.slider.value())
        with self._busy("흰 선 검출 (사이드라인 정밀화)"):
            frame = self._native_frame(f)
            pts = (detect_sideline_points(self._field_calib, frame)
                   if frame is not None else [])
        if frame is None:
            return
        if len(pts) < 8:
            self.log(f"[field] 흰 선 샘플 부족 ({len(pts)}개) — "
                     "선이 프레임에 잘 보이는 프레임에서 다시 시도")
            return
        self.line_points = [[round(float(a), 1), round(float(b), 1)]
                            for a, b in pts]
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._redraw()
        self.log(f"[field] 사이드라인 흰 선 샘플 {len(pts)}개 반영")

    def _clear_line_points(self):
        """흰 선 정밀화 취소 — 샘플 제거 후 랜드마크만으로 재피팅."""
        if not self.line_points:
            self.log("[field] 지울 흰 선 샘플이 없습니다")
            return
        n = len(self.line_points)
        self.line_points = []
        self._refit_field(log_result=True)
        self._save_keyframes()
        self._redraw()
        self.log(f"[field] 흰 선 샘플 {n}개 제거 — 랜드마크만으로 재피팅")

    def _refit_field(self, log_result=False):
        self._field_calib = None
        if self.pano_w and len(self.field_points) >= 4:
            self._field_calib = fit_field_calibration(
                self.field_points, self.pano_w, self.pano_h,
                length=self.field_size[0], width=self.field_size[1],
                line_points=self.line_points)
        c = self._field_calib
        if c is not None:
            tilt = (f", 기울기 {np.degrees(c['pitch']):+.1f}°/"
                    f"{np.degrees(c['roll']):+.1f}°"
                    if c.get("pitch") or c.get("roll") else "")
            msg = (f"캘리브레이션 OK — {c['n_points']}점, 모델 잔차 "
                   f"{c['rms']:.1f}px (랜드마크는 워프로 고정), 높이 "
                   f"{c['h']:.1f}m, 터치라인 {-(c['ey'] + c['width']/2):.1f}m"
                   + tilt)
        elif len(self.field_points) >= 4:
            msg = ("캘리브레이션 실패 — 점 위치 확인 "
                   "(위치 랜드마크 3개 이상 필요, 사이드라인 점은 보조)")
        else:
            msg = (f"{len(self.field_points)}점 찍음 — 위치 랜드마크 4개"
                   "(또는 3개+사이드라인 2점)부터 풀림, ★ 먼쪽 코너부터")
        if self.btn_field_pick.isChecked():
            nk = self._field_next_key()
            if nk is not None:
                i = [k for k, _, _ in LANDMARKS].index(nk)
                msg = (f"다음 찍을 점 → {i+1}. {LANDMARKS[i][1]} "
                       f"[{LANDMARK_TAGS[nk]}]   |   {msg}")
            else:
                msg = "모든 랜드마크 지정 완료   |   " + msg
        self.lbl_field_status.setText(msg)
        if log_result:
            self.log(f"[field] {msg}")
        self.radar.set_field(
            {"length": self.field_size[0], "width": self.field_size[1],
             "cam": self._cam_field_pos()} if c is not None else None)

    def _players_row(self, si):
        """샘플 si 의 선수 행: 분석 + 사용자 수동 검출(extra) 합침."""
        if self.analysis is None or si is None:
            return []
        return list(self.analysis["players"][si]) \
            + self.extra_players.get(int(si), [])

    def _person_px_height(self, x, y):
        """(x, y) 지점에 선 사람(1.8m)의 예상 픽셀 키 — 캘리브레이션 기준.

        캘리브레이션 전이면 기본 카메라 모델로 대충 추정.
        """
        if self._field_calib is not None:
            c = self._field_calib
            g = pano_to_field(c, [[x, y]])[0]
            if not np.isfinite(g[0]):
                return None
            d = np.hypot(g[0] - c["ex"], g[1] - c["ey"])
            h, span = c["h"], c["t_top"] - c["t_bot"]
        else:
            gp = ground_positions([[x, y, 0.0, 0.0]], self.pano_w, self.pano_h)
            if not gp:
                return None
            d = np.hypot(gp[0][0], gp[0][1])
            h, span = 4.0, np.tan(np.deg2rad(10.0)) - np.tan(np.deg2rad(-38.0))
        return 1.8 / max(d, 1.0) / span * (self.pano_h - 1)

    def _native_frame(self, f):
        """프레임 f 의 원본 해상도 이미지 (프록시 표시 중이면 원본을 읽음)."""
        if self.disp_scale >= 1.0 and getattr(self, "_cur_frame_idx", -1) == f \
                and getattr(self, "_cur_frame", None) is not None:
            return self._cur_frame
        if self._native_cap is None:
            self._native_cap = cv2.VideoCapture(str(self.pano_path))
        self._native_cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self._native_cap.read()
        return frame if ok else None

    def _detect_here(self, f, x, y, gpos):
        """빈 곳 우클릭 → 주변만 네이티브 해상도로 사람 재검출.

        크롭 크기는 캘리브레이션으로 예측한 그 자리 사람 키의 ~6배
        (마진 넉넉히). 검출된 사람은 수동 검출(extra)로 추가되고 바로
        역할 지정 메뉴를 띄운다.
        """
        si = self._current_sample()
        if si is None or not ptz_available():
            return
        ph = self._person_px_height(x, y) or 120.0
        half = int(np.clip(3.0 * ph, 160, 900))
        with self._busy("주변 사람 재검출 (YOLO 타일)"):
            frame = self._native_frame(f)
            if frame is None:
                return
            x0 = int(np.clip(x - half, 0, max(self.pano_w - 2 * half, 0)))
            y0 = int(np.clip(y - half, 0, max(self.pano_h - 2 * half, 0)))
            crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if self._adhoc is None:
                from ultralytics import YOLO
                w = self._model_weights()
                self._adhoc = YOLO(str(w) if w else "yolov8n.pt")
            imgsz = int(np.clip(2 * half, 320, 1280)) // 32 * 32
            r = self._adhoc.predict(crop, imgsz=imgsz, conf=0.1,
                                    classes=[0], verbose=False)[0]
        # 주변에 여럿 잡혀도 커서를 포함하는 박스 하나만 채택
        # (포함 박스가 여럿이면 가장 작은 것 = 가장 특정한 것).
        best, best_key = None, None
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            cx, cy = x0 + (x1 + x2) / 2, y0 + (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            inside = abs(x - cx) <= bw / 2 + 8 and abs(y - cy) <= bh / 2 + 8
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if not inside and d2 > max(150.0, ph) ** 2:
                continue                 # 커서에서 먼 박스는 무시
            key = (0, bw * bh) if inside else (1, d2)
            if best_key is None or key < best_key:
                best, best_key = (cx, cy, bw, bh, float(b.conf[0])), key
        if best is None:
            self.log(f"[ptz] 주변 재검출: 커서 위치에 사람 없음 "
                     f"(크롭 {2*half}px, 검출 {len(r.boxes)}건)")
            return
        cx, cy, bw, bh, conf = best
        # 기존(분석+수동) 박스와 겹치면 중복 추가하지 않음
        if any((p[0] - cx) ** 2 + (p[1] - cy) ** 2 < (p[3] / 2) ** 2
               for p in self._players_row(si) if len(p) >= 4):
            self.log("[ptz] 주변 재검출: 이미 있는 박스와 중복 — 추가 안 함")
            return
        tid = self._next_extra_id
        self._next_extra_id += 1
        self.extra_players.setdefault(int(si), []).append(
            [round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1), tid])
        skipped = len(r.boxes) - 1
        self.log(f"[ptz] 주변 재검출: 사람 1명 추가 (conf {conf:.2f}"
                 + (f", 주변 {skipped}건은 제외" if skipped > 0 else "")
                 + f", 크롭 {2*half}px)")
        self._save_keyframes()
        self._pcache_id = None           # 선수 목록 캐시 무효화
        self._refresh_player_list()
        self._redraw()
        menu = QMenu(self)               # 바로 역할 지정
        for rr in (3, 4, 5, 0, 1):
            menu.addAction(f"{self._role_name(rr)} 지정",
                           lambda _=False, r_=rr, t=tid:
                           self._set_role(t, r_))
        menu.addAction("역할 없이 두기", lambda: None)
        menu.exec(gpos)

    def _propagate(self, f, x, y, kind, ctx=None):
        """시드 전파 시작 — 수동 인식을 앞뒤 ±4s 로 확장."""
        if self.analysis is None:
            return
        if self._seed_worker is not None and self._seed_worker.isRunning():
            self.log("[seed] 이미 실행 중")
            return
        w = SeedWorker(str(self.pano_path), self.analysis, f, x, y, kind,
                       weights=self._model_weights(), ctx=ctx)
        w.log.connect(self.log)
        w.done.connect(self._seed_done)
        w.failed.connect(lambda e: self.log(f"[seed] 실패: {e}"))
        self._seed_worker = w
        self.log(f"[seed] {'공' if kind == 'ball' else '선수'} 추적 확장 "
                 f"시작 ({f/self.fps:.1f}s ±4s)...")
        w.start()

    def _seed_done(self, kind, matches, ctx):
        if not matches:
            self.log("[seed] 연결된 샘플 없음")
            return
        if kind == "ball":
            nb = 0
            for si, x, y, w_, h_, c in matches:
                cands = self.analysis["ball_cands"][si]
                if any(np.hypot(x - p[0], y - p[1]) <= 30 for p in cands):
                    continue
                cands.append([x, y, max(0.26, c), w_, h_, c])
                if self.analysis["balls"][si] is None:
                    self.analysis["balls"][si] = [x, y, max(0.26, c), w_, h_]
                nb += 1
            self._write_analysis()
            self.log(f"[seed] 공 {nb}샘플 주입 — 트랙 재연결")
            self._start_link()
        else:
            tid = int(ctx)
            np_ = 0
            for si, x, y, w_, h_, c in matches:
                rows = self.extra_players.setdefault(int(si), [])
                if any(p[4] == tid or np.hypot(x - p[0], y - p[1])
                       < max(w_ / 2, 20) for p in rows):
                    continue
                if any(len(p) >= 5 and
                       (p[0] - x) ** 2 + (p[1] - y) ** 2 < (p[3] / 2) ** 2
                       for p in self.analysis["players"][si]):
                    continue                 # 분석 검출과 중복
                rows.append([x, y, w_, h_, tid])
                np_ += 1
            self._save_keyframes()
            self._pcache_id = None
            self._refresh_player_list()
            self._redraw()
            self.log(f"[seed] 선수 #{tid - 900000} {np_}샘플로 확장")

    def _delete_extra(self, tid):
        for si, rows in list(self.extra_players.items()):
            self.extra_players[si] = [p for p in rows if p[4] != tid]
            if not self.extra_players[si]:
                del self.extra_players[si]
        self.roles.pop(tid, None)
        self._save_keyframes()
        self._pcache_id = None
        self._refresh_player_list()
        self._redraw()

    def _player_at(self, x, y):
        """(x, y)를 포함하는 현재 샘플의 선수 박스 track id (없으면 None)."""
        si = self._current_sample()
        if self.analysis is None or si is None:
            return None
        best, bestd = None, None
        for p in self._players_row(si):
            if len(p) < 5 or p[4] < 0:
                continue
            if (abs(x - p[0]) <= p[2] / 2 + 10
                    and abs(y - p[1]) <= p[3] / 2 + 10):
                d = (x - p[0]) ** 2 + (y - p[1]) ** 2
                if best is None or d < bestd:
                    best, bestd = int(p[4]), d
        return best

    def _recompute_tracks(self):
        """수락 트랙 구간 재계산 → 트랙바 갱신 (분석/무시 구간 변경 시).

        linked 캐시가 있으면 수락 단계만 돌아 즉각 반응한다.
        """
        if self.analysis is None:
            self.track_spans = []
            self._accepted_ball = None
        else:
            _, self._accepted_ball, self.track_spans = accept_ball_tracks(
                self.analysis, ignore_ranges=[tuple(r) for r in self.ignores],
                force_ranges=[tuple(p) for p in self.promotes],
                linked=self._linked, log=self.log)
        self._recompute_airborne()
        self.trackbar.set_data(self.total, self.track_spans,
                               self.ignores, self.keyframes,
                               promotes=self.promotes)
        self._refresh_track_list()
        self._plan_dirty()

    def _goto_track(self):
        row = self.track_list.currentRow()
        if 0 <= row < len(getattr(self, "_top", [])):
            kind, i = self._top[row]
            if kind == "track":
                f0, f1 = self.track_spans[i]
                if f0 <= self.slider.value() <= f1:
                    return          # 이미 그 트랙 안 — 시작으로 되감지 않음
                self.slider.setValue(int(f0))
            else:
                self.slider.setValue(int(self.keyframes[i][0]))

    def _jump_track(self, direction: int):
        """현재 위치 기준 이전(-1)/다음(+1) 수락 트랙 시작으로 이동."""
        self._stop_play()
        f = self.slider.value()
        if direction > 0:
            nxt = [sp for sp in self.track_spans if sp[0] > f]
            if nxt:
                self.slider.setValue(int(nxt[0][0]))
        else:
            prv = [sp for sp in self.track_spans if sp[0] < f - 1]
            if prv:
                self.slider.setValue(int(prv[-1][0]))

    def _to_ignore(self):
        """≫ 버튼: 목록 선택이 있으면 그 항목, 없으면 현재 시각 트랙."""
        if 0 <= self.track_list.currentRow() < len(getattr(self, "_top", [])):
            self._ignore_selected_track()
        else:
            self._ignore_current_track()

    def _ignore_selected_track(self, advance=False):
        """위 목록에서 Del — 자동 트랙은 오인식으로, 수동 지정은 삭제."""
        row = self.track_list.currentRow()
        if not (0 <= row < len(getattr(self, "_top", []))):
            return
        kind, i = self._top[row]
        if kind == "track":
            # 재생 위치는 건드리지 않고 그 트랙만 무시 — 정적 미끼는
            # 대개 영상 시작부터 있어서 시작으로 점프하면 처음으로 튄다
            self._ignore_current_track(anchor_f=int(self.track_spans[i][0]),
                                       advance=advance)
        else:
            del self.keyframes[i]
            self._save_keyframes()
            self._refresh_lists()
            self.trackbar.set_data(self.total, self.track_spans,
                                   self.ignores, self.keyframes,
                                   promotes=self.promotes)
            self._plan_dirty()
            self._redraw()

    def _ignore_current_track(self, anchor_f=None, advance=False):
        """anchor_f(기본: 현재 시각)를 덮는 수락 트랙을 통째로 무시.

        advance=True(키보드 검수 Del/→)일 때만 다음 항목으로 이동까지 —
        미리보기/버튼 경로에서는 보던 위치를 유지한다.
        """
        f = self.slider.value() if anchor_f is None else anchor_f
        for f0, f1 in self.track_spans:
            if f0 <= f <= f1:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    spans = [(int(f0), int(f1))]
                    if self._linked is not None:
                        # 같은 자리 반복 정적 트랙 일괄 수집 (낙엽·마킹).
                        # 위치 포함 4-요소 — 같은 시간대의 진짜 공 트랙은 보호
                        spans = same_spot_spans(self._linked, f0, f1) or spans
                    added = 0
                    for sp in spans:
                        lo, hi = sp[0], sp[1]
                        if not any(a <= lo and hi <= b for a, b in
                                   ((r[0], r[1]) for r in self.ignores)):
                            self.ignores.append(list(sp))
                            added += 1
                    self.ignores.sort()
                    self._save_keyframes()
                    self._recompute_tracks()
                    self._refresh_kf_list()
                    self._redraw()
                finally:
                    QApplication.restoreOverrideCursor()
                extra = f" (같은 자리 반복 포함 {added}개 구간)" if added > 1 else ""
                self.log(f"[ptz] 트랙 무시: {f0/self.fps:.1f}s ~ "
                         f"{f1/self.fps:.1f}s{extra}")
                # 다음 항목 자동 선택 (시크 없이) — 이동은 키보드 검수만
                def _start(e):
                    return (self.track_spans[e[1]][0] if e[0] == "track"
                            else self.keyframes[e[1]][0])
                nxt = [r for r, e in enumerate(self._top) if _start(e) > f0]
                row = nxt[0] if nxt else len(self._top) - 1
                if row >= 0:
                    self.track_list.blockSignals(True)
                    self.track_list.setCurrentRow(row)
                    self.track_list.blockSignals(False)
                    if advance:
                        self._goto_track()
                return
        QMessageBox.information(self, "무시", "현재 시각을 덮는 공 트랙이 없습니다.")

    # ------------------------------------------------------------ 내보내기
    def _update_export_enabled(self):
        self.btn_export.setEnabled(self.analysis is not None)

    def _start_render(self):
        if self.analysis is None or self.pano_path is None:
            return
        self._stop_play()
        st = QSettings("PyStitch360", "PyStitch360")
        clock_avail = self._clock_config() is not None
        dlg = ExportDialog(
            self, self.total, self.fps, self._norm_export_range(),
            self.combo_mode.currentIndex(), self.encoders,
            int(st.value("ptz_export_crf", 20)),
            st.value("ptz_export_radar", "true") == "true",
            str(self.pano_path.parent), self.pano_path.stem,
            clock_on=(st.value("ptz_export_clock", "true") == "true"
                      if clock_avail else None))
        if not dlg.exec():
            return
        cfg = dlg.config()
        if not cfg["path"]:
            return
        st.setValue("ptz_export_crf", cfg["crf"])
        st.setValue("ptz_export_radar", "true" if cfg["radar"] else "false")
        if clock_avail:
            st.setValue("ptz_export_clock", "true" if cfg["clock"] else "false")
        wide = cfg["wide"]
        self.combo_mode.setCurrentIndex(1 if wide else 0)  # 미리보기 일치
        codec = self.encoders[cfg["codec_name"]]
        kfs = [tuple(k) for k in self.keyframes]
        radar = None
        if cfg["radar"]:
            spans, _ = self._player_cache()
            teams = {tid: self._role_of(tid) for tid in spans}
            radar = build_radar_data(
                self.analysis, teams, calib=self._field_calib,
                field_size=tuple(self.field_size),
                extra_players=self.extra_players,
                palette={r: self._role_color(r) for r in range(6)})
            self.log("[ptz] 미니맵 오버레이 포함 "
                     + ("(경기장 절대 좌표)" if self._field_calib is not None
                        else "(캘리브레이션 없음 — 근사 좌표)"))
        dur = (cfg["end"] - cfg["start"]) / self.fps
        self.log(f"[ptz] 내보내기 시작: {'와이드' if wide else 'PTZ'} 모드, "
                 f"구간 {self._hms(cfg['start']/self.fps)}~"
                 f"{self._hms(cfg['end']/self.fps)} ({dur/60:.1f}분), "
                 f"키프레임 {len(kfs)}개 반영")
        clock = self._clock_config() if cfg["clock"] else None
        if clock is not None:
            self.log(f"[ptz] 경기 시계 포함 ({clock['tag']}, "
                     f"중단 {len(clock['pauses'])}개"
                     + (", 스코어" if clock["score"] else "") + ")")
        w = PtzRenderWorker(str(self.pano_path), cfg["path"], self.analysis,
                            kfs, codec, cfg["crf"], wide=wide,
                            ignores=[tuple(r) for r in self.ignores],
                            far_zoom=self.spin_far_zoom.value(),
                            promotes=[tuple(p) for p in self.promotes],
                            radar=radar, start=cfg["start"], end=cfg["end"],
                            clock=clock)
        w.log.connect(self.log)
        w.progress.connect(self._render_progress)
        w.finished_ok.connect(self._render_done)
        w.failed.connect(self._render_failed)
        self._render_worker = w
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("준비 중...")
        w.start()

    def _cancel_render(self):
        if self._render_worker is not None and self._render_worker.isRunning():
            self._render_worker.cancel()

    def _render_progress(self, done, total, fps):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        remain = (total - done) / fps / 60 if fps > 0 else 0
        self.progress.setFormat(f"%p%  ({done}/{total}, {fps:.1f}fps, 남은 시간 {remain:.0f}분)")

    def _render_done(self, path):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setFormat("완료")
        self.log(f"[ptz] 저장: {path}")
        QMessageBox.information(self, "가상 PTZ", f"완료: {path}")

    def _render_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.log(f"[오류] PTZ 내보내기: {msg}")
