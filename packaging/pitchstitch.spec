# -*- mode: python ; coding: utf-8 -*-
# PitchStitch 배포 스펙 초안 (P05-4, 미검증 — 실제 패키징은 보류 항목).
#
# 핵심: 분석 스택(torch/ultralytics/easyocr) 완전 제외 — 스티칭 앱은
# ffmpeg/OpenCV/PyQt6 만으로 동작한다 (MainWindow(with_ptz=False) 는
# ptz_tab 모듈 자체를 import 하지 않음). ffmpeg.exe 는 동봉하지 않고
# PATH 의존 (동봉 여부는 패키징 확정 때 결정).
#
# 빌드 (Windows): pyinstaller packaging/pitchstitch.spec

a = Analysis(
    ["../pitchstitch.py"],
    pathex=[".."],
    datas=[("../presets", "presets")],
    excludes=[
        "torch", "torchvision", "ultralytics", "easyocr",
        "matplotlib", "pandas", "scipy",
    ],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts,
    exclude_binaries=True,
    name="PitchStitch",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PitchStitch")
