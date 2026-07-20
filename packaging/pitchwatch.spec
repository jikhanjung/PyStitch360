# -*- mode: python ; coding: utf-8 -*-
# PitchWatch 배포 스펙 초안 (P05-4, 미검증 — 실제 패키징은 보류 항목).
#
# 분석 앱은 torch/ultralytics/easyocr 를 지연 import 하므로
# hiddenimports 로 명시해야 onedir 에 포함된다. GB 급이라 onefile 은
# 부적합 — onedir 고정. YOLO 가중치(yolo11*.pt)는 크기 문제로 동봉하지
# 않고 첫 실행 시 다운로드/사용자 배치 (동봉 여부는 패키징 확정 때 결정).
#
# 빌드 (Windows): pyinstaller packaging/pitchwatch.spec

a = Analysis(
    ["../pitchwatch.py"],
    pathex=[".."],
    datas=[("../presets", "presets")],
    hiddenimports=["torch", "torchvision", "ultralytics", "easyocr"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts,
    exclude_binaries=True,
    name="PitchWatch",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PitchWatch")
