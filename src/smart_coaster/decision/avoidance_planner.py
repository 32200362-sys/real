from __future__ import annotations

from math import hypot

from ..config import PlannerConfig
from ..types import MotionFeatures, Point2D, RiskResult, RobotCommand, RobotState, StateUpdate


class AvoidancePlanner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.home: Point2D | None = None
        self._seq = 0

    def calibrate_home(self, robot_center: Point2D | None) -> bool:
        if robot_center is None:
            return False
        self.home = robot_center
        return True

    def plan(
        self,
        state_update: StateUpdate,
        feature: MotionFeatures | None,
        robot_center: Point2D | None,
        risk: RiskResult,
        timestamp_ms: int,
    ) -> RobotCommand:
        if self.home is None and robot_center is not None:
            self.home = robot_center

        cmd = "STOP"
        vx = vy = wz = 0.0

        if state_update.state == RobotState.AVOIDING and feature is not None and robot_center is not None:
            image_dx = robot_center.x - feature.hand_position.x
            image_dy = robot_center.y - feature.hand_position.y
            strength = 0.55 + 0.45 * min(max(risk.score / 100.0, 0.0), 1.0)
            speed = min(self.config.avoid_speed_m_s * strength, self.config.max_speed_m_s)
            vx, vy = self._image_vector_to_robot_velocity(image_dx, image_dy, speed)
            cmd = "MOVE"

        elif state_update.state == RobotState.RETURNING and robot_center is not None and self.home is not None:
            image_dx = self.home.x - robot_center.x
            image_dy = self.home.y - robot_center.y
            distance = hypot(image_dx, image_dy)
            if distance > self.config.home_tolerance_px:
                speed_scale = min(1.0, distance / 120.0)
                speed = max(self.config.command_deadband_m_s, self.config.return_speed_m_s * speed_scale)
                vx, vy = self._image_vector_to_robot_velocity(image_dx, image_dy, speed)
                cmd = "RETURN"

        elif state_update.state == RobotState.HOLD:
            cmd = "HOLD"

        if hypot(vx, vy) < self.config.command_deadband_m_s:
            vx = vy = 0.0
            if cmd in {"MOVE", "RETURN"}:
                cmd = "STOP"

        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return RobotCommand(
            seq=self._seq,
            timestamp_ms=timestamp_ms,
            cmd=cmd,
            vx=vx,
            vy=vy,
            wz=wz,
            risk=int(round(risk.score)),
        )

    def stop_command(self, timestamp_ms: int) -> RobotCommand:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return RobotCommand(self._seq, timestamp_ms, "STOP", 0.0, 0.0, 0.0, 0)

    @staticmethod
    def _image_vector_to_robot_velocity(image_dx: float, image_dy: float, speed: float) -> tuple[float, float]:
        norm = hypot(image_dx, image_dy)
        if norm < 1e-6:
            return 0.0, 0.0
        # 카메라 영상: +x 오른쪽, +y 아래. 로봇: +vx 오른쪽, +vy 앞(영상 위쪽).
        vx = image_dx / norm * speed
        vy = -image_dy / norm * speed
        return vx, vy
