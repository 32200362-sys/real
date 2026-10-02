import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Export a cup detector to ONNX")
    p.add_argument("model"); p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--simplify", action="store_true"); p.add_argument("--dynamic", action="store_true")
    a = p.parse_args()
    from ultralytics import YOLO
    print(YOLO(a.model).export(format="onnx", imgsz=a.imgsz, simplify=a.simplify, dynamic=a.dynamic))
    return 0


if __name__ == "__main__": raise SystemExit(main())
