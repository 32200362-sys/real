from __future__ import annotations

import cv2
import numpy as np

from ..types import (
    HandObservation,
    MarkerObservation,
    MotionFeatures,
    RiskResult,
    RobotCommand,
    StateUpdate,
)
from ..vision.hand_detector import HAND_CONNECTIONS


class DebugView:
    def render_integrated(self, frame, result, markers, risk, state, command, fps, profile, udp_enabled):
        canvas = frame.copy()
        self._draw_markers(canvas, markers)
        self._draw_hands(canvas, result.hands, result.motion)
        for detection in result.cup_detections:
            b = detection.bbox
            cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 180, 0), 2)
            cv2.putText(canvas, f"cup {detection.confidence:.2f}", (int(b.x1), int(b.y1)-5), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,180,0), 1)
        if result.cup_track:
            b = result.cup_track.bbox
            cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 3)
            cv2.putText(canvas, f"#{result.cup_track.track_id} {result.cup_track.source.value}", (int(b.x1), int(b.y2)+18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
        if result.target_person:
            for arm in result.target_person.arms:
                pts = [arm.shoulder, arm.elbow, arm.wrist]
                cv2.polylines(canvas, [np.array([[int(p.x), int(p.y)] for p in pts])], False, (0,255,0), 2)
        self._draw_panel(canvas, risk, state, command, fps, udp_enabled, result.motion)
        interaction = result.interaction
        lines = [f"PROFILE: {profile}", f"BEHAVIOR: {result.behavior.label.value}"]
        if interaction:
            lines += [f"CLOSING: {interaction.closing_speed:.1f}px/s", f"TTC: {interaction.ttc_seconds:.2f}s" if interaction.ttc_seconds is not None else "TTC: -"]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (420, 30+i*22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
        return canvas
    def draw_robot_link(self, canvas, status_lines, coaster, heading_deg, packet) -> None:
        """ESP32 연결 상태 패널과 로봇 전방/명령 속도 화살표를 그린다 (canvas를 직접 수정)."""
        if coaster is not None and heading_deg is not None:
            cx, cy = int(coaster.x), int(coaster.y)
            theta = np.radians(heading_deg)
            # 로봇 전방(+x): 빨강. heading_offset_deg 보정 시 실제 로봇 앞과 맞춘다.
            fx, fy = -np.sin(theta), -np.cos(theta)
            cv2.arrowedLine(canvas, (cx, cy), (int(cx + fx * 60), int(cy + fy * 60)), (0, 0, 255), 3, tipLength=0.25)
            cv2.putText(canvas, "FRONT", (int(cx + fx * 70) - 20, int(cy + fy * 70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            if packet is not None and packet.type == "cmd_vel":
                # 로봇 기준 (전방, 왼쪽) 속도를 화면 방향으로 되돌려 그린다: 청록.
                forward, left = packet.vx, packet.vy
                up_w = forward * np.cos(theta) - left * np.sin(theta)
                left_w = forward * np.sin(theta) + left * np.cos(theta)
                scale = 4.0  # px per cm/s
                cv2.arrowedLine(canvas, (cx, cy), (int(cx - left_w * scale), int(cy - up_w * scale)),
                                (255, 255, 0), 3, tipLength=0.2)

        if not status_lines:
            return
        height = canvas.shape[0]
        top = height - 12 - 22 * len(status_lines)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, top - 18), (470, height - 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        for i, (line, color) in enumerate(status_lines):
            cv2.putText(canvas, line, (18, top + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        hands: list[HandObservation],
        markers: list[MarkerObservation],
        selected_feature: MotionFeatures | None,
        risk: RiskResult,
        state: StateUpdate,
        command: RobotCommand,
        fps: float,
        udp_enabled: bool,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_markers(canvas, markers)
        self._draw_hands(canvas, hands, selected_feature)
        self._draw_panel(canvas, risk, state, command, fps, udp_enabled, selected_feature)
        return canvas

    def _draw_markers(self, canvas: np.ndarray, markers: list[MarkerObservation]) -> None:
        for marker in markers:
            points = np.array([[p.x, p.y] for p in marker.corners], dtype=np.int32)
            cv2.polylines(canvas, [points], True, (0, 255, 255), 2)
            center = (int(marker.center.x), int(marker.center.y))
            cv2.circle(canvas, center, 5, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"{marker.name}({marker.marker_id})",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_hands(
        self,
        canvas: np.ndarray,
        hands: list[HandObservation],
        selected_feature: MotionFeatures | None,
    ) -> None:
        selected_key = selected_feature.hand_key if selected_feature else None
        for hand in hands:
            pts = [(int(p.x), int(p.y)) for p in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(canvas, pts[a], pts[b], (255, 180, 0), 2)
            for point in pts:
                cv2.circle(canvas, point, 3, (255, 255, 255), -1)
            center = (int(hand.center.x), int(hand.center.y))
            color = (0, 0, 255) if hand.key == selected_key else (255, 0, 255)
            cv2.circle(canvas, center, 8, color, -1)
            cv2.putText(
                canvas,
                f"{hand.key} {hand.confidence:.2f}",
                (center[0] + 10, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        if selected_feature is not None:
            h = selected_feature.hand_position
            t = selected_feature.target_position
            cv2.arrowedLine(
                canvas,
                (int(h.x), int(h.y)),
                (int(t.x), int(t.y)),
                (0, 100, 255),
                2,
                tipLength=0.08,
            )
            v = selected_feature.velocity_px_s
            scale = 0.12
            cv2.arrowedLine(
                canvas,
                (int(h.x), int(h.y)),
                (int(h.x + v.x * scale), int(h.y + v.y * scale)),
                (0, 255, 0),
                2,
                tipLength=0.15,
            )

    def _draw_panel(
        self,
        canvas: np.ndarray,
        risk: RiskResult,
        state: StateUpdate,
        command: RobotCommand,
        fps: float,
        udp_enabled: bool,
        feature: MotionFeatures | None,
    ) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (410, 235), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)

        lines = [
            f"STATE: {state.state.value}  ({state.reason})",
            f"RISK: {risk.score:5.1f} / {risk.level}",
            f"CMD: {command.cmd}  vx={command.vx:+.3f} vy={command.vy:+.3f}",
            f"UDP: {'ON' if udp_enabled else 'OFF'}   FPS: {fps:4.1f}",
        ]
        if feature is not None:
            lines.extend(
                [
                    f"HAND: {feature.hand_key}",
                    f"DIST: {feature.distance_to_target_px:6.1f}px",
                    f"SPEED: {feature.speed_px_s:6.1f}px/s",
                    f"CLOSING: {feature.distance_rate_px_s:6.1f}px/s",
                    f"ANGLE: {feature.direction_angle_deg:5.1f}deg",
                ]
            )
        y = 30
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 23
