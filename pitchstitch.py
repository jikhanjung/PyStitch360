"""PitchStitch — 듀얼 GoPro 스티칭 앱 (P05 분리 런처).

MainWindow(with_ptz=False): 가상 PTZ 탭/분석 메뉴 없이 스티칭 3탭만
(영상·동기화 / 정합·미리보기 / 내보내기). PtzTab 을 아예 생성하지
않으므로 분석 스택(cv2 검출·torch 계열)이 로드되지 않는다 — torch
미설치 환경에서도 기동. 경기 분석은 PitchWatch (pitchwatch.py) 에서.
"""
import sys

from PyQt6.QtWidgets import QApplication

from pystitch.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow(with_ptz=False)
    win._app_name = "PitchStitch"         # 프로젝트 열기/저장 제목에도 반영
    win.setWindowTitle("PitchStitch")
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
