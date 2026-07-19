"""PyStitch360 실행 진입점 — 기본 GUI, --headless 로 무인 파이프라인."""
import sys


def main():
    if "--headless" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--headless"]
        from pystitch.headless import main as headless_main
        sys.exit(headless_main(argv))
    from pystitch.gui.main_window import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
