#!/usr/bin/env python3
"""
PyStitch360 - 360도 영상 스티칭 툴
GoPro 듀얼 카메라로 촬영한 360도 영상을 전처리부터 최종 출력까지 통합 처리
"""

import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from gui.stitcher_window import StitcherWindow


def setup_logging():
    """로깅 시스템 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pystitch360.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    """메인 애플리케이션 실행"""
    logger = setup_logging()
    logger.info("PyStitch360 애플리케이션 시작")
    
    app = QApplication(sys.argv)
    app.setApplicationName("PyStitch360")
    app.setApplicationVersion("1.0.0")
    
    # 메인 윈도우 생성
    window = StitcherWindow()
    window.show()
    
    logger.info("GUI 윈도우 표시 완료")
    
    # 애플리케이션 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()