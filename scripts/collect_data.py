from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Smart Coaster with UDP off and data collection on")
    parser.add_argument("--profile", choices=("pc", "pi"), default="pc")
    parser.add_argument("--video")
    args = parser.parse_args()
    command = [sys.executable, "run.py", "--no-udp", "--collect", "--profile", args.profile]
    if args.video:
        command += ["--video", args.video]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
