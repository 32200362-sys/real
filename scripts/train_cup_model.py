import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Train an Ultralytics cup detector")
    p.add_argument("--data", default="data/datasets/cup/dataset.yaml")
    p.add_argument("--base-model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="runs/cup"); p.add_argument("--name", default="train")
    a = p.parse_args()
    from ultralytics import YOLO
    YOLO(a.base_model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project=a.project, name=a.name)
    return 0


if __name__ == "__main__": raise SystemExit(main())
