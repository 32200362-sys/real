from __future__ import annotations

import socket
import time

from ..config import UdpConfig
from ..types import RobotCommand


class UdpSender:
    def __init__(self, config: UdpConfig) -> None:
        self.config = config
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_send_s = 0.0
        self._min_interval_s = 1.0 / max(config.send_hz, 1.0)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def send(self, command: RobotCommand, force: bool = False) -> bool:
        if not self.config.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_send_s < self._min_interval_s:
            return False
        try:
            self.socket.sendto(command.to_wire(), (self.config.host, self.config.port))
            self._last_send_s = now
            return True
        except OSError as exc:
            print(f"[UDP] 전송 실패: {exc}")
            return False

    def close(self) -> None:
        self.socket.close()
