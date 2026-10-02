import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained cup detector")
    p.add_argument("model"); p.add_argument("--data", default="data/datasets/cup/dataset.yaml")
    a = p.parse_args()
    from ultralytics import YOLO
    print(YOLO(a.model).val(data=a.data))
    return 0


if __name__ == "__main__": raise SystemExit(main())
