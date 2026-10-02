from smart_coaster.types import RobotCommand


def test_wire_packet_format():
    command = RobotCommand(7, 1234, "MOVE", 0.18, -0.04, 0.0, 82)
    assert command.to_wire() == b"SC1|7|1234|MOVE|0.1800|-0.0400|0.0000|82"
