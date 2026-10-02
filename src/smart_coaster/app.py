from __future__ import annotations

import time
import traceback
from types import SimpleNamespace

from .config import AppConfig


class SmartCoasterApp:
    def __init__(self, config: AppConfig, source: int | str, components=None) -> None:
        self.config, self.source = config, source
        self.components = components
        self.frame_index = 0
        self._marker_cache: dict[int, tuple[float, object]] = {}

    def _build_components(self):
        from .behavior.rule_classifier import RuleBehaviorClassifier
        from .camera import CameraSource
        from .data_collection.collector import DataCollector
        from .decision.avoidance_planner import AvoidancePlanner
        from .decision.risk_calculator import RiskCalculator
        from .decision.state_machine import StateMachine
        from .features.motion_tracker import MotionTracker
        from .pipeline import FrameProcessor
        from .session_logger import SessionLogger
        from .tracking.cup_selector import CupSelector
        from .tracking.cup_tracker import CupTracker
        from .tracking.person_selector import PersonSelector
        from .vision.aruco_detector import ArucoDetector
        from .vision.hand_detector import HandDetector
        from .vision.pose_detector import PoseDetector
        from .vision.yolo_cup_detector import YoloCupDetector
        from .visualization.debug_view import DebugView

        profile = self.config.runtime.select(self.config.runtime.profile)
        try:
            yolo_path = self.config.resolve_path(self.config.vision.yolo_model_path)
            # 프로젝트 기준 경로에 없으면 이름 그대로 넘겨 ultralytics가 자동 다운로드하게 한다.
            cup_detector = YoloCupDetector(str(yolo_path) if yolo_path.exists() else self.config.vision.yolo_model_path,
                                           confidence=self.config.vision.yolo_confidence,
                                           imgsz=profile.yolo_imgsz)
            pose_detector = PoseDetector(self.config.resolve_path(self.config.vision.pose_model_path))
            hand_detector = HandDetector(self.config.resolve_path(self.config.vision.hand_model_path), self.config.vision)
        except Exception as exc:
            raise RuntimeError(f"AI 모델 초기화 실패: {exc}\n`python scripts/download_models.py`를 실행하고 config/settings.yaml 모델 경로를 확인하세요.") from exc
        processor = FrameProcessor(
            cup_detector, CupSelector(self.config.tracking.min_confidence), CupTracker(self.config.tracking),
            RuleBehaviorClassifier(self.config.behavior), pose_detector=pose_detector,
            hand_detector=hand_detector, person_selector=PersonSelector(), motion_tracker=MotionTracker(self.config.motion),
            profile=profile, fallback_enabled=self.config.vision.fallback_enabled,
            fallback_marker_id=self.config.vision.fallback_marker_id,
            min_closing_speed=self.config.behavior.min_closing_speed_px_s,
            coaster_marker_id=self.config.vision.marker_ids.get("coaster", 20),
            hand_only_fallback=self.config.vision.hand_only_fallback,
        )
        logger = SessionLogger(self.config.logging, self.config.resolve_path(self.config.logging.directory), float(self.config.camera.fps))
        logger.coaster_marker_id = self.config.vision.marker_ids.get("coaster", 20)
        return SimpleNamespace(
            camera=CameraSource(self.source, self.config.camera), aruco=ArucoDetector(self.config.vision),
            processor=processor, risk=RiskCalculator(self.config.risk),
            state=StateMachine(self.config.state, self.config.risk), planner=AvoidancePlanner(self.config.planner),
            udp=self._build_sender(), view=DebugView(),
            logger=logger,
            collector=DataCollector(self.config.data_collection, self.config.resolve_path(self.config.data_collection.directory), float(self.config.camera.fps)),
            hand=hand_detector, pose=pose_detector,
        )

    def _build_sender(self):
        udp = self.config.udp
        if udp.protocol == "sc1":
            from .communication.udp_sender import UdpSender
            return UdpSender(udp)
        from .communication.esp32_link import Esp32Sender, TelemetryListener
        telemetry = TelemetryListener(udp.telemetry_port) if udp.enabled and udp.telemetry_enabled else None
        sender = Esp32Sender(udp, telemetry)
        if udp.enabled:
            print(f"[ESP32] 대상 {udp.host}:{udp.port}, session={sender.session_id}, "
                  f"{'ARMED' if sender.armed else 'DISARMED (카메라 창에서 M 키로 활성화)'}")
        return sender

    def handle_key(self, key: int, timestamp_ms: int) -> bool:
        """카메라 창 키 입력 처리. 종료해야 하면 True."""
        c = self.components
        if key in (27, ord("q"), ord("Q")):
            return True
        if key in (ord("s"), ord("S")) and c.udp.enabled:
            if hasattr(c.udp, "set_armed"):
                c.udp.set_armed(False)
            c.udp.send(c.planner.stop_command(timestamp_ms), force=True)
        elif key in (ord("m"), ord("M")) and hasattr(c.udp, "set_armed") and c.udp.enabled:
            c.udp.set_armed(not c.udp.armed)
            print(f"[ESP32] {'ARMED' if c.udp.armed else 'DISARMED'}")
        elif key in (ord("r"), ord("R")) and hasattr(c.udp, "reset_fault"):
            c.udp.reset_fault()
            print("[ESP32] reset_fault 전송")
        return False

    def _hold_recent_markers(self, markers: list, now_s: float) -> list:
        """코스터/컵 마커가 1~몇 프레임 끊겨도 marker_hold_s 동안 마지막 관측을 유지한다."""
        hold_s = self.config.vision.marker_hold_s
        if hold_s <= 0:
            return markers
        wanted = {self.config.vision.marker_ids.get("coaster", 20), self.config.vision.fallback_marker_id}
        seen = {m.marker_id for m in markers}
        held = list(markers)
        for marker_id in wanted:
            if marker_id in seen:
                self._marker_cache[marker_id] = (now_s, next(m for m in markers if m.marker_id == marker_id))
            elif marker_id in self._marker_cache and now_s - self._marker_cache[marker_id][0] <= hold_s:
                held.append(self._marker_cache[marker_id][1])
        return held

    def process_frame(self, frame, timestamp_ms: int, now_s: float, fps: float = 0.0):
        c = self.components
        markers = self._hold_recent_markers(c.aruco.detect(frame), now_s)
        result = c.processor.process_observations(frame, timestamp_ms, now_s, self.frame_index, markers)
        risk = c.risk.calculate(result.motion)
        if result.behavior.label.value == "COLLISION_RISK" and risk.score < self.config.risk.avoid_threshold:
            from .types import RiskResult
            risk = RiskResult(self.config.risk.avoid_threshold, "AVOID", risk.components)
        state = c.state.update_behavior(result.behavior, risk, result.motion, now_s)
        coaster_id = self.config.vision.marker_ids.get("coaster", 20)
        coaster_marker = next((m for m in markers if m.marker_id == coaster_id), None)
        coaster = coaster_marker.center if coaster_marker else None
        if hasattr(c.udp, "update_marker_heading"):
            from .communication.esp32_link import marker_heading_deg
            c.udp.update_marker_heading(marker_heading_deg(coaster_marker.corners) if coaster_marker else None)
        command = c.planner.plan(state, result.motion, coaster, risk, timestamp_ms)
        if hasattr(c.udp, "debug_text"):
            c.udp.debug_text = f"{state.state.value} {result.behavior.label.value} r={risk.score:.0f}"
        udp_sent = c.udp.send(command) if c.udp.enabled else False
        metadata = c.logger.log_frame(self.frame_index, result, markers, state, risk, command, udp_sent)
        if hasattr(frame, "shape"):
            metadata.update(frame_height=int(frame.shape[0]), frame_width=int(frame.shape[1]))
        c.collector.collect(frame, result, metadata, now_s)
        debug_frame = c.view.render_integrated(frame, result, markers, risk, state, command, fps,
                                                self.config.runtime.profile, c.udp.enabled)
        if hasattr(c.udp, "status_lines") and hasattr(c.view, "draw_robot_link"):
            heading = c.udp.robot_heading_deg() if coaster else None
            c.view.draw_robot_link(debug_frame, c.udp.status_lines(), coaster, heading, c.udp.last_packet)
        self.frame_index += 1
        return result, state, risk, command, debug_frame

    def run(self) -> int:
        import cv2
        self.components = self.components or self._build_components()
        c = self.components
        last_timestamp_ms, last_frame_s, fps_ema = 0, time.monotonic(), 0.0
        try:
            c.camera.open()
            if self.config.ui.display:
                # 노트북 화면에서 창 크기를 자유롭게 조절할 수 있게 한다.
                cv2.namedWindow(self.config.ui.window_name, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = c.camera.read()
                if not ok:
                    break
                now_s = time.monotonic()
                timestamp_ms = max(last_timestamp_ms + 1, int(now_s * 1000)); last_timestamp_ms = timestamp_ms
                dt = max(now_s - last_frame_s, 1e-6); last_frame_s = now_s
                fps = 1 / dt; fps_ema = fps if not fps_ema else .9 * fps_ema + .1 * fps
                _, _, _, _, debug = self.process_frame(frame, timestamp_ms, now_s, fps_ema)
                if self.config.ui.display:
                    cv2.imshow(self.config.ui.window_name, debug)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 0xFF and self.handle_key(key, timestamp_ms):
                        break
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1
        finally:
            timestamp = int(time.monotonic() * 1000)
            if c.udp.enabled:
                for _ in range(3): c.udp.send(c.planner.stop_command(timestamp), force=True)
            for resource, method in ((c.collector, "close"), (c.logger, "close"), (c.udp, "close"),
                                     (getattr(c, "hand", None), "close"), (getattr(c, "pose", None), "close"),
                                     (c.camera, "release")):
                if resource and hasattr(resource, method): getattr(resource, method)()
            cv2.destroyAllWindows()
        return 0
