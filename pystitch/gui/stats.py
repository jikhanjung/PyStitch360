"""경기 지표 통계 창 (P08-3) — PtzTab(검수)과 분리된 결과 열람.

core.metrics.match_metrics 결과 dict 를 렌더만 한다 (계산은 core,
합성 테스트는 그쪽). 커버리지·미관측을 항상 함께 보여준다 (P08 원칙).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)


def _row(table, r, *cells):
    for c, v in enumerate(cells):
        table.setItem(r, c, QTableWidgetItem(str(v)))


class StatsDialog(QDialog):
    """점유율/패스 요약 — 비모달, 검수 화면과 나란히 띄운다."""

    def __init__(self, parent, metrics, team_names=("팀1", "팀2"),
                 numbers=None, passmaps=None, dist_rows=None, save_dir=None):
        super().__init__(parent)
        self.setWindowTitle("경기 지표 (점유율·패스)")
        self.setModal(False)
        self.resize(560, 520)
        v = QVBoxLayout(self)
        s = metrics["summary"] or {}
        cov = s.get("coverage", 0.0)
        v.addWidget(QLabel(
            f"<b>점유율</b> (공 관측 커버리지 {cov:.0%} — 미관측 구간은 "
            f"어느 팀에도 배정하지 않음)"))
        t1 = QTableWidget(4, 3)
        t1.setHorizontalHeaderLabels(["", team_names[0], team_names[1]])
        _row(t1, 0, "점유율", f"{s.get('share0', float('nan')):.0%}",
             f"{s.get('share1', float('nan')):.0%}")
        _row(t1, 1, "소유 시간", f"{s.get('team0_s', 0):.0f}s",
             f"{s.get('team1_s', 0):.0f}s")
        _row(t1, 2, "경합/루즈볼",
             f"{s.get('contested_s', 0):.0f}s", f"{s.get('loose_s', 0):.0f}s")
        _row(t1, 3, "미관측", f"{s.get('unobserved_s', 0):.0f}s", "")
        t1.resizeColumnsToContents()
        t1.setMaximumHeight(160)
        v.addWidget(t1)

        from ..core.metrics import pass_matrix
        mat = pass_matrix(metrics["passes"], numbers)
        v.addWidget(QLabel(
            f"<b>패스</b> {len(metrics['passes'])}회 · 턴오버 "
            f"{len(metrics['turnovers'])}회 · 미관측 전이 "
            f"{metrics['unobserved_transitions']}건 (집계 제외)"))
        top = sorted(mat.items(), key=lambda kv: -kv[1])[:15]
        t2 = QTableWidget(len(top), 2)
        t2.setHorizontalHeaderLabels(["연결 (from → to)", "횟수"])
        for r, ((a, b), n) in enumerate(top):
            _row(t2, r, f"{a} → {b}", n)
        t2.resizeColumnsToContents()
        v.addWidget(t2, 1)
        v.addWidget(QLabel(
            f"소유 구간 {len(metrics['spans'])}개 / 샘플 "
            f"{metrics['n_samples']}개. 원경 공백은 멀티캠 융합(P06-4)이 "
            f"개선 경로."))
        if dist_rows:
            v.addWidget(QLabel("<b>뛴 거리</b> (관측 비율 60% 미만은 실제보다 "
                               "적게 잡힌다 — 같은 행의 비율 참고)"))
            t3 = QTableWidget(len(dist_rows), 6)
            t3.setHorizontalHeaderLabels(
                ["팀", "번호/ID", "거리(m)", "평균(m/s)", "최고(m/s)", "관측"])
            for r, (tm, num, dm, avg, mx, obs) in enumerate(dist_rows):
                _row(t3, r, tm, num, f"{dm:.0f}", f"{avg:.1f}",
                     f"{mx:.1f}", f"{obs:.0%}")
            t3.resizeColumnsToContents()
            t3.setMinimumHeight(160)
            v.addWidget(t3, 1)
        if passmaps:
            import cv2
            from PyQt6.QtGui import QImage, QPixmap
            from PyQt6.QtWidgets import QHBoxLayout, QScrollArea, QWidget
            row = QHBoxLayout()
            for img in passmaps:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                            rgb.strides[0], QImage.Format.Format_RGB888)
                lbl = QLabel()
                lbl.setPixmap(QPixmap.fromImage(qi.copy()).scaledToWidth(430))
                row.addWidget(lbl)
            wrap = QWidget()
            wrap.setLayout(row)
            sc = QScrollArea()
            sc.setWidget(wrap)
            sc.setMinimumHeight(300)
            v.addWidget(sc, 2)
            self.resize(920, 760)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        if save_dir:
            b = bb.addButton("리포트 저장 (match.md + PNG)",
                             QDialogButtonBox.ButtonRole.ActionRole)

            def _save():
                from ..core.metrics import write_match_report
                files = write_match_report(
                    save_dir, metrics, team_names, passmaps=passmaps,
                    dist_rows=dist_rows, numbers=numbers)
                b.setText(f"저장됨: {len(files)}개 파일")
                b.setEnabled(False)

            b.clicked.connect(_save)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
