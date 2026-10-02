from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Coaster AI + UDP controller")
    parser.add_argument("--config", default="config/settings.yaml", help="YAML config path")
    parser.add_argument("--camera", type=int, help="Camera index override")
    parser.add_argument("--video", help="Video file instead of camera")
    parser.add_argument("--udp", action="store_true", help="Enable UDP output")
    parser.add_argument("--no-udp", action="store_true", help="Force-disable UDP output")
    parser.add_argument("--host", help="ESP32 IP override")
    parser.add_argument("--port", type=int, help="ESP32 UDP port override")
    parser.add_argument("--arm", action="store_true", help="Start with motor commands armed (default: press M in the camera window)")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV window")
    parser.add_argument("--record", action="store_true", help="Save debug video")
    parser.add_argument("--profile", choices=("pc", "pi"), default="pc", help="Runtime profile")
    parser.add_argument("--model", help="YOLO weight path override")
    collection = parser.add_mutually_exclusive_group()
    collection.add_argument("--collect", action="store_true", help="Enable data collection")
    collection.add_argument("--no-collect", action="store_true", help="Disable data collection")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from smart_coaster.app import SmartCoasterApp
    from smart_coaster.config import AppConfig
    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = AppConfig.load(config_path)
    profile = config.runtime.select(args.profile)

    if args.camera is not None:
        config.camera.index = args.camera
    if args.udp:
        config.udp.enabled = True
    if args.no_udp:
        config.udp.enabled = False
    if args.host:
        config.udp.host = args.host
    if args.port:
        config.udp.port = args.port
    if args.arm:
        config.udp.start_armed = True
    if args.no_display:
        config.ui.display = False
    if args.record:
        config.logging.save_video = True
    if args.model:
        config.vision.yolo_model_path = args.model
    if args.collect:
        config.data_collection.enabled = True
    elif args.no_collect:
        config.data_collection.enabled = False
    else:
        config.data_collection.enabled = profile.collect_data

    source: int | str = args.video if args.video else config.camera.index
    app = SmartCoasterApp(config=config, source=source)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
