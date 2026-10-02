from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2

from .config import LoggingConfig
from .types import MotionFeatures, RiskResult, RobotCommand, StateUpdate
from .types import FrameResult


class SessionLogger:
    FIELDNAMES = [
        "timestamp",
        "state",
        "reason",
        "risk",
        "risk_level",
        "hand_key",
        "hand_x",
        "hand_y",
        "distance_px",
        "speed_px_s",
        "distance_rate_px_s",
        "angle_deg",
        "cmd",
        "vx",
        "vy",
        "wz",
        "udp_sent",
    ]
    INTEGRATED_FIELDS = ["timestamp", "frame_index", "cup_detected", "cup_source", "cup_track_id",
        "cup_bbox", "cup_confidence", "coaster_center", "target_person_id", "hand_side",
        "wrist_x", "wrist_y", "elbow_x", "elbow_y", "shoulder_x", "shoulder_y", "hand_x", "hand_y",
        "hand_speed", "hand_cup_distance", "closing_speed", "ttc", "behavior", "risk", "robot_state",
        "command", "vx", "vy", "wz", "udp_sent"]

    def __init__(self, config: LoggingConfig, directory: Path, fps: float) -> None:
        self.config = config
        self.directory = directory
        self.fps = fps
        self.csv_file = None
        self.writer = None
        self.video_writer = None
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.coaster_marker_id = 20

        if config.enabled:
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / f"session_{self.stamp}.csv"
            self.csv_file = csv_path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.INTEGRATED_FIELDS)
            self.writer.writeheader()

    def log(
        self,
        state: StateUpdate,
        risk: RiskResult,
        feature: MotionFeatures | None,
        command: RobotCommand,
        udp_sent: bool,
    ) -> None:
        if self.writer is None:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "state": state.state.value,
            "reason": state.reason,
            "risk": f"{risk.score:.2f}",
            "risk_level": risk.level,
            "hand_key": feature.hand_key if feature else "",
            "hand_x": f"{feature.hand_position.x:.2f}" if feature else "",
            "hand_y": f"{feature.hand_position.y:.2f}" if feature else "",
            "distance_px": f"{feature.distance_to_target_px:.2f}" if feature else "",
            "speed_px_s": f"{feature.speed_px_s:.2f}" if feature else "",
            "distance_rate_px_s": f"{feature.distance_rate_px_s:.2f}" if feature else "",
            "angle_deg": f"{feature.direction_angle_deg:.2f}" if feature else "",
            "cmd": command.cmd,
            "vx": f"{command.vx:.4f}",
            "vy": f"{command.vy:.4f}",
            "wz": f"{command.wz:.4f}",
            "udp_sent": int(udp_sent),
        }
        self.writer.writerow(row)
        self.csv_file.flush()

    def log_frame(self, frame_index: int, result: FrameResult, markers, state, risk, command, udp_sent: bool) -> dict:
        track, interaction, person = result.cup_track, result.interaction, result.target_person
        coaster = next((m.center for m in markers if m.marker_id == self.coaster_marker_id), None)
        def xy(point): return ("", "") if point is None else (round(point.x, 3), round(point.y, 3))
        wrist = xy(interaction.wrist if interaction else None); elbow = xy(interaction.elbow if interaction else None)
        shoulder = xy(interaction.shoulder if interaction else None); hand = xy(interaction.hand_position if interaction else None)
        row = {
            "timestamp": result.timestamp_ms, "frame_index": frame_index, "cup_detected": int(track is not None),
            "cup_source": track.source.value if track else "", "cup_track_id": track.track_id if track else "",
            "cup_bbox": [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2] if track else "",
            "cup_confidence": track.confidence if track else "", "coaster_center": list(xy(coaster)) if coaster else "",
            "target_person_id": person.person_id if person else "", "hand_side": interaction.side if interaction else "",
            "wrist_x": wrist[0], "wrist_y": wrist[1], "elbow_x": elbow[0], "elbow_y": elbow[1],
            "shoulder_x": shoulder[0], "shoulder_y": shoulder[1], "hand_x": hand[0], "hand_y": hand[1],
            "hand_speed": ((interaction.hand_velocity.x**2 + interaction.hand_velocity.y**2)**.5) if interaction else "",
            "hand_cup_distance": interaction.distance if interaction else "",
            "closing_speed": interaction.closing_speed if interaction else "", "ttc": interaction.ttc_seconds if interaction and interaction.ttc_seconds is not None else "",
            "behavior": result.behavior.label.value, "risk": risk.score, "robot_state": state.state.value,
            "command": command.cmd, "vx": command.vx, "vy": command.vy, "wz": command.wz, "udp_sent": int(udp_sent),
        }
        if self.writer:
            self.writer.writerow(row); self.csv_file.flush()
        return row

    def write_video(self, frame) -> None:
        if not self.config.save_video:
            return
        if self.video_writer is None:
            height, width = frame.shape[:2]
            path = self.directory / f"debug_{self.stamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(str(path), fourcc, self.fps, (width, height))
        self.video_writer.write(frame)

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
        if self.csv_file is not None:
            self.csv_file.close()
