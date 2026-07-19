"""export_training_labels — 양성/저신뢰/네거티브/수동 분류 (P03-5)."""
from conftest import make_ball_analysis

from pystitch.core.ptz import export_training_labels


def test_label_classes():
    ana = make_ball_analysis()
    labels = export_training_labels(ana, keyframes=[(451, 500.0, 700.0)],
                                    ignore_ranges=[(600, 897)])
    by = {}
    for r in labels:
        by.setdefault(r["label"], []).append(r)
    assert len(by.get("ball", [])) >= 150              # 수락 트랙 양성
    assert len(by.get("ball_lowconf", [])) == 20       # 주입 (길이 6 행)
    assert all(abs(r["conf"] - 0.12) < 1e-6            # 원래 conf 복원
               for r in by["ball_lowconf"])
    assert len(by.get("not_ball", [])) >= 80           # 하드 네거티브
    assert len(by.get("ball_manual", [])) == 1
    assert all(r["frame"] >= 600 for r in by["not_ball"])
    assert all(labels[i]["frame"] <= labels[i + 1]["frame"]
               for i in range(len(labels) - 1))
