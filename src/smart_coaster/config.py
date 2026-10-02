from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = False


@dataclass
class VisionConfig:
    hand_model_path: str = "models/hand_landmarker.task"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    aruco_dictionary: str = "DICT_4X4_50"
    marker_ids: dict[str, int] = field(
        default_factory=lambda: {"cup": 10, "coaster": 20, "laptop": 30, "paper": 40}
    )
    yolo_model_path: str = "yolo11n.pt"
    yolo_confidence: float = 0.4
    yolo_max_det: int = 20
    pose_model_path: str = "models/pose_landmarker_lite.task"
    fallback_enabled: bool = True
    fallback_marker_id: int = 10


@dataclass
class ProfileConfig:
    yolo_every_n_frames: int = 1
    pose_every_n_frames: int = 1
    hand_every_n_frames: int = 1
    yolo_imgsz: int = 640
    collect_data: bool = False


@dataclass
class RuntimeConfig:
    profile: str = "pc"
    profiles: dict[str, ProfileConfig] = field(default_factory=lambda: {
        "pc": ProfileConfig(),
        "pi": ProfileConfig(3, 2, 1, 320, False),
    })

    def select(self, name: str) -> ProfileConfig:
        if name not in self.profiles:
            raise ValueError(f"unknown runtime profile: {name}")
        self.profile = name
        return self.profiles[name]


@dataclass
class TrackingConfig:
    min_confidence: float = 0.4
    min_iou: float = 0.1
    max_center_distance_px: float = 100.0
    max_missed_frames: int = 5


@dataclass
class BehaviorConfig:
    mode: str = "rule"
    model_path: str | None = None
    min_closing_speed_px_s: float = 40.0
    collision_ttc_s: float = 1.5
    reaching_ttc_s: float = 3.0
    hold_distance_px: float = 80.0
    hold_speed_px_s: float = 30.0
    safe_distance_px: float = 220.0
    confirm_frames: int = 2
    unknown_grace_frames: int = 0


@dataclass
class DataCollectionConfig:
    enabled: bool = False
    directory: str = "data/collection"
    sample_interval_s: float = 1.5
    pre_event_s: float = 2.0
    post_event_s: float = 3.0
    event_cooldown_s: float = 5.0


@dataclass
class MotionConfig:
    smoothing_alpha: float = 0.35
    velocity_smoothing_alpha: float = 0.45
    min_dt_s: float = 0.01
    max_dt_s: float = 0.25
    stale_track_s: float = 1.0


@dataclass
class RiskConfig:
    distance_far_px: float = 320.0
    distance_danger_px: float = 120.0
    speed_max_px_s: float = 700.0
    closing_rate_max_px_s: float = 500.0
    scale_rate_max_s: float = 1.5
    angle_full_deg: float = 25.0
    angle_zero_deg: float = 75.0
    warning_threshold: float = 45.0
    avoid_threshold: float = 70.0
    hold_distance_px: float = 115.0
    hold_speed_px_s: float = 80.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 0.34,
            "speed": 0.16,
            "direction": 0.22,
            "closing": 0.23,
            "scale": 0.05,
        }
    )


@dataclass
class StateConfig:
    avoid_confirm_frames: int = 3
    safe_confirm_frames: int = 10
    hold_confirm_seconds: float = 0.55
    hold_release_distance_px: float = 180.0
    return_timeout_seconds: float = 2.0


@dataclass
class PlannerConfig:
    avoid_speed_m_s: float = 0.18
    return_speed_m_s: float = 0.12
    max_speed_m_s: float = 0.25
    home_tolerance_px: float = 25.0
    command_deadband_m_s: float = 0.015


@dataclass
class UdpConfig:
    enabled: bool = False
    host: str = "192.168.0.50"
    port: int = 8888
    send_hz: float = 20.0
    # esp32_json: arduino/esp32_omni_controller typed JSON, sc1: 기존 ASCII SC1 형식
    protocol: str = "esp32_json"
    telemetry_enabled: bool = True
    telemetry_port: int = 8889
    heartbeat_interval_s: float = 1.0
    start_armed: bool = False
    speed_status: str = "RUN"
    max_linear_cm_s: float = 15.0
    max_angular_rad_s: float = 1.0
    # marker: 코스터 ArUco 방향 + offset, fixed: offset만 사용 (로봇 전방이 화면 위에서 반시계로 돈 각도)
    heading_source: str = "marker"
    heading_offset_deg: float = 0.0


@dataclass
class LoggingConfig:
    enabled: bool = True
    directory: str = "data/logs"
    save_video: bool = False


@dataclass
class UiConfig:
    display: bool = True
    window_name: str = "Smart Coaster AI"


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    state: StateConfig = field(default_factory=StateConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    udp: UdpConfig = field(default_factory=UdpConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    project_root: Path = field(default_factory=Path, repr=False)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        runtime_raw = raw.get("runtime", {})
        profiles = {name: ProfileConfig(**values) for name, values in runtime_raw.get("profiles", {}).items()}
        config = cls(
            runtime=RuntimeConfig(runtime_raw.get("profile", "pc"), profiles or RuntimeConfig().profiles),
            camera=CameraConfig(**raw.get("camera", {})),
            vision=VisionConfig(**raw.get("vision", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            behavior=BehaviorConfig(**raw.get("behavior", {})),
            data_collection=DataCollectionConfig(**raw.get("data_collection", {})),
            motion=MotionConfig(**raw.get("motion", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            state=StateConfig(**raw.get("state", {})),
            planner=PlannerConfig(**raw.get("planner", {})),
            udp=UdpConfig(**raw.get("udp", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            ui=UiConfig(**raw.get("ui", {})),
            project_root=path.parent.parent,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError("camera width/height must be positive")
        if not 0.0 < self.motion.smoothing_alpha <= 1.0:
            raise ValueError("motion.smoothing_alpha must be in (0, 1]")
        if self.risk.distance_danger_px >= self.risk.distance_far_px:
            raise ValueError("distance_danger_px must be smaller than distance_far_px")
        weight_sum = sum(self.risk.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"risk weights must sum to 1.0, got {weight_sum}")
        if not 0 < self.udp.port < 65536:
            raise ValueError("UDP port must be between 1 and 65535")
        if self.udp.protocol not in {"esp32_json", "sc1"}:
            raise ValueError("udp.protocol must be esp32_json or sc1")
        if self.udp.heading_source not in {"marker", "fixed"}:
            raise ValueError("udp.heading_source must be marker or fixed")
        if self.udp.speed_status not in {"RUN", "SLOW"}:
            raise ValueError("udp.speed_status must be RUN or SLOW")
        if not 0.0 < self.udp.max_linear_cm_s <= 30.0:
            # ESP32 HARD_LINEAR_LIMIT_CM_S(30) 이상은 latched SPEED_LIMIT fault가 된다.
            raise ValueError("udp.max_linear_cm_s must be in (0, 30]")
        if not 0.0 <= self.udp.max_angular_rad_s <= 2.0:
            raise ValueError("udp.max_angular_rad_s must be in [0, 2]")
        if not 0 <= self.vision.yolo_confidence <= 1:
            raise ValueError("vision.yolo_confidence must be in [0, 1]")
        if self.tracking.max_missed_frames < 0:
            raise ValueError("tracking.max_missed_frames must be non-negative")
        if self.behavior.mode not in {"rule", "model", "hybrid"}:
            raise ValueError("behavior.mode must be rule, model, or hybrid")
        profile = self.runtime.select(self.runtime.profile)
        if min(profile.yolo_every_n_frames, profile.pose_every_n_frames, profile.hand_every_n_frames) < 1:
            raise ValueError("detect intervals must be at least 1")
        if min(self.behavior.collision_ttc_s, self.behavior.safe_distance_px) <= 0:
            raise ValueError("behavior thresholds must be positive")
        dc = self.data_collection
        if dc.sample_interval_s <= 0 or min(dc.pre_event_s, dc.post_event_s, dc.event_cooldown_s) < 0:
            raise ValueError("invalid data collection timing")

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()
