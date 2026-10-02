import json
import math
import socket
import time

from smart_coaster.communication.esp32_link import (
    Esp32Sender,
    TelemetryListener,
    image_to_robot_velocity,
    marker_heading_deg,
    parse_telemetry,
)
from smart_coaster.config import UdpConfig
from smart_coaster.types import Point2D, RobotCommand


class FakeSocket:
    def __init__(self): self.sent = []
    def setblocking(self, flag): pass
    def sendto(self, data, addr): self.sent.append((json.loads(data), addr))
    def close(self): pass


def make_sender(**overrides):
    values = dict(enabled=True, host="10.0.0.5", start_armed=True, heading_source="fixed")
    values.update(overrides)
    sock = FakeSocket()
    return Esp32Sender(UdpConfig(**values), sock=sock), sock


def move(vx, vy, cmd="MOVE", wz=0.0):
    return RobotCommand(1, 0, cmd, vx, vy, wz, 80)


def square(cx, cy, heading_deg):
    """heading_deg만큼 반시계로 돌린 마커 코너 (TL, TR, BR, BL)."""
    t = math.radians(heading_deg)
    up = (-math.sin(t), -math.cos(t))
    right = (math.cos(t), -math.sin(t))
    pts = []
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
        pts.append(Point2D(cx + 10 * (sx * right[0] + sy * up[0]), cy + 10 * (sx * right[1] + sy * up[1])))
    return pts


def test_marker_heading_matches_rotation():
    for heading in (0.0, 30.0, 90.0, -90.0, 180.0):
        result = marker_heading_deg(square(100, 100, heading))
        assert abs(((result - heading) + 180) % 360 - 180) < 1e-6


def test_image_velocity_maps_to_robot_forward_left():
    # 로봇 전방 = 화면 위: 화면 위쪽 이동은 전방, 화면 오른쪽 이동은 -왼쪽.
    assert image_to_robot_velocity(0.0, 1.0, 0.0) == (1.0, 0.0)
    f, l = image_to_robot_velocity(1.0, 0.0, 0.0)
    assert f == 0.0 and l == -1.0
    # 로봇 전방이 화면 왼쪽(+90°)이면 화면 왼쪽 이동이 전방이다.
    f, l = image_to_robot_velocity(-1.0, 0.0, 90.0)
    assert math.isclose(f, 1.0) and math.isclose(l, 0.0, abs_tol=1e-9)


def test_move_becomes_cmd_vel_in_cm_s_with_session_and_seq():
    sender, sock = make_sender(heartbeat_interval_s=1e9)
    sender._last_heartbeat_s = time.monotonic()
    assert sender.send(move(0.0, 0.10), force=True)
    packet, addr = sock.sent[-1]
    assert addr == ("10.0.0.5", 8888)
    assert packet["type"] == "cmd_vel" and packet["status"] == "RUN"
    assert packet["vx"] == 10.0 and packet["vy"] == 0.0 and packet["w"] == 0.0
    assert packet["session_id"] == sender.session_id != 0
    sender.send(move(0.0, 0.10, "RETURN"), force=True)
    assert sock.sent[-1][0]["seq"] == packet["seq"] + 1


def test_speed_is_clamped_below_esp32_hard_limit():
    sender, sock = make_sender(heartbeat_interval_s=1e9)
    sender._last_heartbeat_s = time.monotonic()
    sender.send(move(0.30, 0.40, wz=5.0), force=True)  # 50 cm/s
    packet = sock.sent[-1][0]
    assert math.isclose(math.hypot(packet["vx"], packet["vy"]), 15.0, abs_tol=0.01)
    assert packet["w"] == 1.0


def test_hold_stop_disarmed_and_missing_heading_send_stop():
    sender, sock = make_sender(heartbeat_interval_s=1e9)
    sender._last_heartbeat_s = time.monotonic()
    for cmd in ("HOLD", "STOP"):
        sender.send(move(0.1, 0.1, cmd), force=True)
        assert sock.sent[-1][0]["type"] == "stop"
    sender.armed = False
    sender.send(move(0.1, 0.1), force=True)
    assert sock.sent[-1][0]["type"] == "stop"

    marker_sender, marker_sock = make_sender(heading_source="marker", heartbeat_interval_s=1e9)
    marker_sender._last_heartbeat_s = time.monotonic()
    marker_sender.update_marker_heading(None)
    marker_sender.send(move(0.1, 0.1), force=True)
    assert marker_sock.sent[-1][0]["type"] == "stop"


def test_disabled_sender_sends_nothing_and_heartbeat_is_periodic():
    sender, sock = make_sender(enabled=False)
    assert not sender.send(move(0.1, 0.0), force=True) and not sock.sent
    sender, sock = make_sender(heartbeat_interval_s=0.0)
    sender.send(move(0.0, 0.1), force=True)
    assert [p["type"] for p, _ in sock.sent] == ["heartbeat", "cmd_vel"]


def test_reset_fault_sequence():
    sender, sock = make_sender()
    sender.reset_fault()
    assert [p["type"] for p, _ in sock.sent] == ["stop", "reset_fault", "heartbeat"]


def test_parse_telemetry_and_loopback_listener():
    sample = {"type": "telemetry", "state": "VELOCITY", "mode": "NETWORK", "fault": "NONE",
              "wheel_speed": [1, 2.5, -3], "wheel_target": [1, 2, -3], "wifi_rssi": -55, "session_id": 7}
    parsed = parse_telemetry(json.dumps(sample).encode(), 1.0)
    assert parsed.state == "VELOCITY" and parsed.wheel_speed == [1.0, 2.5, -3.0] and parsed.wifi_rssi == -55
    assert parse_telemetry(b"not json", 1.0) is None

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]; probe.close()
    listener = TelemetryListener(port, bind_host="127.0.0.1")
    try:
        assert listener.error is None
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        out.sendto(json.dumps(sample).encode(), ("127.0.0.1", port)); out.close()
        deadline = time.monotonic() + 2.0
        while listener.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert listener.latest() is not None and listener.latest().mode == "NETWORK"
    finally:
        listener.close()
