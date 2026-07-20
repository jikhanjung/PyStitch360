"""PitchStitch — 듀얼 GoPro 스티칭 앱 (P05 분리 런처).

기존 MainWindow 에서 가상 PTZ 탭과 분석 메뉴를 떼어낸 스티칭 전용
진입점 (영상·동기화 / 정합·미리보기 / 내보내기). 경기 분석은
PitchWatch (pitchwatch.py) 에서.

다른 파일 무수정 원칙: MainWindow 를 그대로 쓰고 런처에서 탭/메뉴만
제거한다 — 전용 창 구성(내보내기 완료 → PitchWatch 열기 버튼 등)은
P05 본 작업에서.
"""
import sys

from PyQt6.QtWidgets import QApplication

from pystitch.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle("PitchStitch")
    i = win.tabs.indexOf(win.ptz_tab)     # 가상 PTZ 탭 제거 (PitchWatch 로)
    if i >= 0:
        win.tabs.removeTab(i)
    for act in win.menuBar().actions():   # 분석 메뉴 제거 (PTZ 전용)
        if act.text().startswith("분석"):
            win.menuBar().removeAction(act)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
