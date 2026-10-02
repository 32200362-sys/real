from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / "models"; directory.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        destination = directory / name
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"이미 존재합니다: {destination}"); continue
        print(f"다운로드: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "smart-coaster/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        except Exception as exc:
            raise RuntimeError(f"모델 다운로드 실패: {destination}: {exc}") from exc
        print(f"저장 완료: {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
