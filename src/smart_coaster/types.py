from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


class DetectionSource(str, Enum):
    YOLO = "YOLO"
    TRACK = "TRACK"
    ARUCO_FALLBACK = "ARUCO_FALLBACK"


class BehaviorLabel(str, Enum):
    SAFE = "SAFE"
    REACHING = "REACHING"
    COLLISION_RISK = "COLLISION_RISK"
    HOLDING = "HOLDING"
    RETRACTING = "RETRACTING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        w = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        h = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = w * h
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class CupDetection:
    bbox: BoundingBox
    center: Point2D
    confidence: float
    class_id: int
    class_name: str
    source: DetectionSource = DetectionSource.YOLO


@dataclass(frozen=True)
class CupTrack:
    track_id: int
    bbox: BoundingBox
    center: Point2D
    confidence: float
    source: DetectionSource
    velocity_px_s: Point2D = Point2D(0, 0)
    age_frames: int = 1
    missed_frames: int = 0
    predicted: bool = False


@dataclass(frozen=True)
class PoseLandmarkObservation:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class ArmObservation:
    person_id: int
    side: str
    shoulder: Point2D
    elbow: Point2D
    wrist: Point2D
    confidence: float


@dataclass(frozen=True)
class PersonObservation:
    person_id: int
    arms: list[ArmObservation]
    center: Point2D
    confidence: float


@dataclass(frozen=True)
class InteractionFeatures:
    hand_key: str
    side: str
    hand_position: Point2D
    cup_position: Point2D
    hand_velocity: Point2D
    distance: float
    distance_rate: float
    closing_speed: float
    direction_angle: float
    ttc_seconds: float | None
    wrist: Point2D | None = None
    elbow: Point2D | None = None
    shoulder: Point2D | None = None
    pose_confidence: float = 0.0
    hand_confidence: float = 0.0
    data_valid: bool = False


@dataclass(frozen=True)
class BehaviorResult:
    label: BehaviorLabel
    confidence: float
    reason: str
    rule_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameResult:
    timestamp_ms: int
    cup_detections: list[CupDetection]
    cup_track: CupTrack | None
    behavior: BehaviorResult
    interaction: InteractionFeatures | None = None
    hands: list[HandObservation] = field(default_factory=list)
    people: list[PersonObservation] = field(default_factory=list)
    target_person: PersonObservation | None = None
    motion: MotionFeatures | None = None


@dataclass
class HandObservation:
    key: str
    handedness: str
    confidence: float
    center: Point2D
    palm_scale_px: float
    mean_z: float
    landmarks: list[Point3D] = field(default_factory=list)


@dataclass
class MarkerObservation:
    marker_id: int
    name: str
    center: Point2D
    corners: list[Point2D]
    area_px2: float


@dataclass
class MotionFeatures:
    hand_key: str
    hand_position: Point2D
    target_position: Point2D
    velocity_px_s: Point2D
    speed_px_s: float
    distance_to_target_px: float
    distance_rate_px_s: float
    direction_angle_deg: float
    palm_scale_rate_s: float
    approaching: bool


@dataclass
class RiskResult:
    score: float
    level: str
    components: dict[str, float] = field(default_factory=dict)


class RobotState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    WARNING = "WARNING"
    AVOIDING = "AVOIDING"
    HOLD = "HOLD"
    RETURNING = "RETURNING"


@dataclass
class StateUpdate:
    state: RobotState
    reason: str


@dataclass
class RobotCommand:
    seq: int
    timestamp_ms: int
    cmd: str
    vx: float
    vy: float
    wz: float
    risk: int

    def to_wire(self) -> bytes:
        # SC1|seq|timestamp_ms|cmd|vx|vy|wz|risk
        text = (
            f"SC1|{self.seq}|{self.timestamp_ms}|{self.cmd}|"
            f"{self.vx:.4f}|{self.vy:.4f}|{self.wz:.4f}|{self.risk}"
        )
        return text.encode("ascii")
