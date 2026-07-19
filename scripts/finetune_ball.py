"""공 검출 파인튜닝 실행 (P03-5, Windows/2080 Ti 에서 실행).

make_ball_dataset.py 가 만든 dataset.yaml 로 yolo11m 을 짧게 파인튜닝
한다 (공 1클래스). 완료 후 best.pt 경로를 출력 — presets/ 에 복사하고
A/B 로 원경 검출률 전후를 비교한다:

  python scripts/ab_source_detect.py <project.json> <pano.mp4> \
      --weights <best.pt>

주의: 결과 모델은 공 1클래스라 분석의 선수 검출을 대체하지 못한다 —
개선 확인 시 등록 방식(공 전용 2차 패스 등)은 별도 결정.

사용법: python scripts/finetune_ball.py <dataset.yaml>
        [--model yolo11m.pt] [--epochs 25] [--batch 8] [--imgsz 640]
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="make_ball_dataset.py 의 dataset.yaml")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ball_finetune")
    args = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics 미설치 — Windows 환경에서 실행하세요 "
                 "(pip install ultralytics)")
    model = YOLO(args.model)
    r = model.train(data=str(Path(args.data).resolve()),
                    epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, name=args.name, patience=10,
                    plots=True)
    best = Path(r.save_dir) / "weights" / "best.pt"
    print(f"\n완료: {best}")
    m = getattr(r, "results_dict", {}) or {}
    for k in ("metrics/precision(B)", "metrics/recall(B)",
              "metrics/mAP50(B)"):
        if k in m:
            print(f"  {k.split('/')[-1]}: {m[k]:.3f}")
    print(f"\n다음 단계:")
    print(f"  1) copy \"{best}\" presets\\ball_{args.name}.pt")
    print(f"  2) A/B: python scripts/ab_source_detect.py <project.json> "
          f"<pano.mp4> --weights \"{best}\"")


if __name__ == "__main__":
    main()
