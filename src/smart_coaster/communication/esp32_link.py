"""ESP32 motion controller link (typed JSON UDP protocol).

AI 쪽 RobotCommand(m/s, 화면 기준 vx=오른쪽 / vy=위쪽)를 ESP32 펌웨어
(arduino/esp32_omni_controller)가 받는 typed JSON(cm/s, 로봇 기준 +x=전방 / +y=왼쪽)으로
변환해 보낸다. 소켓은 non-blocking이고 telemetry 수신은 별도 스레드라서
카메라 화면 루프를 막지 않는다.
"""

from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field

from ..config import UdpConfig
from ..types import Point2D, RobotCommand

MOTION_COMMANDS = {"MOVE", "RETURN"}
RECOVERABLE_FAULTS = {"WIFI_LOSS", "CMD_TIMEOUT", "BAD_PACKET"}


def marker_heading_deg(corners: list[Point2D]) -> float | None:
    """ArUco 마커의 '위쪽' 방향이 화면 위쪽에서 반시계로 몇 도 돌아가 있는지 계산한다.

    OpenCV 코너 순서는 마커 기준 TL, TR, BR, BL이다.
    """
    if len(corners) != 4:
        return None
    tl, tr, br, bl = corners
    top = Point2D((tl.x + tr.x) / 2.0, (tl.y + tr.y) / 2.0)
    bottom = Point2D((bl.x + br.x) / 2.0, (bl.y + br.y) / 2.0)
    dx, dy = top.x - bottom.x, top.y - bottom.y
    if math.hypot(dx, dy) < 1e-6:
        return None
    # 화면 좌표는 y가 아래로 증가한다. 화면 위쪽=0°, 왼쪽=+90°(반시계).
    return math.degrees(math.atan2(-dx, -dy))


def image_to_robot_velocity(vx_right: float, vy_up: float, robot_heading_deg: float) -> tuple[float, float]:
    """화면 기준 속도(오른쪽, 위쪽)를 로봇 기준 속도(전방, 왼쪽)로 회전한다.

    robot_heading_deg는 로봇 전방(+x)이 화면 위쪽에서 반시계로 돌아간 각도다.
    """
    forward_w, left_w = vy_up, -vx_right
    theta = math.radians(robot_heading_deg)
    c, s = math.cos(theta), math.sin(theta)
    return forward_w * c + left_w * s, -forward_w * s + left_w * c


@dataclass(frozen=True)
class Esp32Packet:
    type: str
    session_id: int
    seq: int
    vx: float | None = None
    vy: float | None = None
    w: float | None = None
    status: str | None = None

    def to_json(self) -> bytes:
        payload = {key: value for key, value in self.__dict__.items() if value is not None}
        for key in ("vx", "vy", "w"):
            if key in payload:
                payload[key] = round(float(payload[key]), 3)
        return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass
class Esp32Telemetry:
    received_at_s: float = 0.0
    state: str = "UNKNOWN"
    mode: str = "UNKNOWN"
    fault: str = "NONE"
    wheel_speed: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wheel_target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    wifi_rssi: int | None = None
    session_id: int | None = None


def parse_telemetry(data: bytes, now_s: float) -> Esp32Telemetry | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != "telemetry":
        return None

    def three_floats(key: str) -> list[float]:
        values = payload.get(key)
        if isinstance(values, list) and len(values) == 3 and all(
                type(v) in (int, float) and math.isfinite(v) for v in values):
            return [float(v) for v in values]
        return [0.0, 0.0, 0.0]

    rssi = payload.get("wifi_rssi")
    session_id = payload.get("session_id")
    return Esp32Telemetry(
        received_at_s=now_s,
        state=str(payload.get("state", "UNKNOWN")),
        mode=str(payload.get("mode", "UNKNOWN")),
        fault=str(payload.get("fault", "NONE")),
        wheel_speed=three_floats("wheel_speed"),
        wheel_target=three_floats("wheel_target"),
        wifi_rssi=rssi if type(rssi) is int else None,
        session_id=session_id if type(session_id) is int else None,
    )


class TelemetryListener:
    """ESP32가 마지막 명령 송신자에게 5Hz로 보내는 telemetry를 백그라운드에서 받는다."""

    def __init__(self, port: int, bind_host: str = "0.0.0.0") -> None:
        self._lock = threading.Lock()
        self._latest: Esp32Telemetry | None = None
        self._running = False
        self.error: str | None = None
        self._sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((bind_host, port))
            sock.settimeout(0.3)
            self._sock = sock
        except OSError as exc:
            self.error = f"telemetry port {port} bind 실패: {exc}"
            print(f"[ESP32] {self.error}")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="esp32-telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            telemetry = parse_telemetry(data, time.monotonic())
            if telemetry is not None:
                with self._lock:
                    self._latest = telemetry

    def latest(self) -> Esp32Telemetry | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            self._sock.close()


class Esp32Sender:
    """UdpSender와 같은 인터페이스(enabled/send/close)를 가진 ESP32 전용 송신기."""

    def __init__(self, config: UdpConfig, telemetry: TelemetryListener | None = None,
                 sock: socket.socket | None = None) -> None:
        self.config = config
        self.session_id = secrets.randbits(32) or 1
        self._seq = 0
        self._sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._min_interval_s = 1.0 / max(config.send_hz, 1.0)
        self._last_send_s = 0.0
        self._last_heartbeat_s = 0.0
        self.armed = bool(config.start_armed)
        self.telemetry = telemetry
        self.marker_heading_deg: float | None = None
        self.last_packet: Esp32Packet | None = None
        self.sent_count = 0
        self.error_count = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ---- 상태 조작 (카메라 화면 키 입력에서 호출) ----
    def set_armed(self, armed: bool) -> None:
        self.armed = armed
        if not armed:
            self.send_stop()

    def update_marker_heading(self, heading_deg: float | None) -> None:
        self.marker_heading_deg = heading_deg

    def robot_heading_deg(self) -> float | None:
        """로봇 전방이 화면 위쪽에서 반시계로 돌아간 각도. 알 수 없으면 None."""
        if self.config.heading_source == "marker":
            if self.marker_heading_deg is None:
                return None
            return self.marker_heading_deg + self.config.heading_offset_deg
        return self.config.heading_offset_deg

    # ---- 패킷 생성 ----
    def _next(self, packet_type: str, **values) -> Esp32Packet:
        self._seq = (self._seq + 1) & 0xFFFFFFFF or 1
        return Esp32Packet(type=packet_type, session_id=self.session_id, seq=self._seq, **values)

    def build_packet(self, command: RobotCommand) -> Esp32Packet:
        heading = self.robot_heading_deg()
        if not self.armed or command.cmd not in MOTION_COMMANDS or heading is None:
            return self._next("stop")
        forward_m_s, left_m_s = image_to_robot_velocity(command.vx, command.vy, heading)
        vx_cm_s, vy_cm_s = forward_m_s * 100.0, left_m_s * 100.0
        linear = math.hypot(vx_cm_s, vy_cm_s)
        if linear > self.config.max_linear_cm_s > 0.0:
            scale = self.config.max_linear_cm_s / linear
            vx_cm_s, vy_cm_s = vx_cm_s * scale, vy_cm_s * scale
        w = max(-self.config.max_angular_rad_s, min(self.config.max_angular_rad_s, command.wz))
        return self._next("cmd_vel", vx=vx_cm_s, vy=vy_cm_s, w=w, status=self.config.speed_status)

    # ---- 전송 ----
    def _transmit(self, packet: Esp32Packet) -> bool:
        try:
            self._sock.sendto(packet.to_json(), (self.config.host, self.config.port))
        except OSError as exc:  # BlockingIOError 포함
            self.error_count += 1
            self.last_error = str(exc)
            if self.error_count % 60 == 1:
                print(f"[ESP32] 전송 실패 ({self.error_count}회): {exc}")
            return False
        self.sent_count += 1
        self.last_packet = packet
        return True

    def send(self, command: RobotCommand, force: bool = False) -> bool:
        if not self.config.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_send_s < self._min_interval_s:
            return False
        self._last_send_s = now
        # 주기적인 heartbeat는 ESP32의 복구 가능한 fault(CMD_TIMEOUT/WIFI_LOSS/BAD_PACKET)를 해제한다.
        if now - self._last_heartbeat_s >= self.config.heartbeat_interval_s:
            self._last_heartbeat_s = now
            self._transmit(self._next("heartbeat"))
        return self._transmit(self.build_packet(command))

    def send_stop(self, repeats: int = 3) -> None:
        if not self.config.enabled:
            return
        for _ in range(repeats):
            self._transmit(self._next("stop"))

    def reset_fault(self) -> None:
        """latched fault 해제: reset_fault 후 heartbeat (펌웨어의 2단계 절차)."""
        if not self.config.enabled:
            return
        self._transmit(self._next("stop"))
        self._transmit(self._next("reset_fault"))
        self._transmit(self._next("heartbeat"))

    def status_lines(self, now_s: float | None = None) -> list[tuple[str, tuple[int, int, int]]]:
        """카메라 화면에 그릴 (문구, BGR 색) 목록. cv2.putText는 한글을 못 그리므로 영어로 쓴다."""
        if not self.config.enabled:
            return [("ESP32: UDP OFF (run with --udp)", (180, 180, 180))]
        now_s = time.monotonic() if now_s is None else now_s
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self.armed:
            lines.append(("MOTOR: ARMED  [M]=disarm [S]=stop", (0, 0, 255)))
        else:
            lines.append(("MOTOR: DISARMED  [M]=arm", (0, 200, 0)))
        lines.append((f"ESP32 {self.config.host}:{self.config.port}  sent={self.sent_count} err={self.error_count}",
                       (255, 255, 255)))
        heading = self.robot_heading_deg()
        if heading is None:
            lines.append(("HEADING: no coaster marker -> STOP", (0, 165, 255)))
        if self.last_packet is not None and self.last_packet.type == "cmd_vel":
            p = self.last_packet
            lines.append((f"TX cmd_vel x={p.vx:+.1f} y={p.vy:+.1f}cm/s w={p.w:+.2f}", (255, 255, 0)))
        elif self.last_packet is not None:
            lines.append((f"TX {self.last_packet.type}", (255, 255, 0)))

        telemetry = self.telemetry.latest() if self.telemetry else None
        if self.telemetry is None or self.telemetry.error:
            lines.append(("TELEMETRY: off", (180, 180, 180)))
        elif telemetry is None:
            lines.append(("TELEMETRY: none (check IP / Wi-Fi / firewall)", (0, 165, 255)))
        else:
            age = now_s - telemetry.received_at_s
            link_color = (0, 200, 0) if age < 1.0 else (0, 0, 255)
            rssi = "-" if telemetry.wifi_rssi is None else f"{telemetry.wifi_rssi}dBm"
            lines.append((f"LINK age={age:.1f}s rssi={rssi}  {telemetry.mode}/{telemetry.state}", link_color))
            if telemetry.fault != "NONE":
                hint = "auto-recover" if telemetry.fault in RECOVERABLE_FAULTS else "[R]=reset"
                lines.append((f"FAULT: {telemetry.fault}  ({hint})", (0, 0, 255)))
            ws = telemetry.wheel_speed
            lines.append((f"WHEEL cm/s {ws[0]:+.1f} {ws[1]:+.1f} {ws[2]:+.1f}", (255, 255, 255)))
        return lines

    def close(self) -> None:
        self.send_stop(repeats=5)
        self._sock.close()
        if self.telemetry is not None:
            self.telemetry.close()
        print(f"[ESP32] 종료. 전송 {self.sent_count}건, 실패 {self.error_count}건")
