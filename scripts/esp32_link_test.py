"""노트북 <-> ESP32 Wi-Fi 연결 시험 (AI/카메라 없이).

기본(ping): 모터를 움직이지 않고 STOP/heartbeat만 보내며 telemetry 수신을 확인한다.
    python scripts/esp32_link_test.py --host 192.168.0.50

짧은 구동 시험 (바퀴를 띄운 상태에서!): 로봇 기준 cm/s (+x 전방, +y 왼쪽), 최대 3초.
    python scripts/esp32_link_test.py --host 192.168.0.50 --move 5 0 0 --duration 1

latched fault 해제:
    python scripts/esp32_link_test.py --host 192.168.0.50 --reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_coaster.communication.esp32_link import Esp32Sender, TelemetryListener  # noqa: E402
from smart_coaster.config import UdpConfig  # noqa: E402


def print_telemetry(listener: TelemetryListener, started: float) -> None:
    t = listener.latest()
    if t is None:
        print(f"  t={time.monotonic() - started:4.1f}s  telemetry 없음")
        return
    age = time.monotonic() - t.received_at_s
    ws = " ".join(f"{v:+5.1f}" for v in t.wheel_speed)
    wt = " ".join(f"{v:+5.1f}" for v in t.wheel_target)
    print(f"  t={time.monotonic() - started:4.1f}s  {t.mode}/{t.state} fault={t.fault} "
          f"rssi={t.wifi_rssi} age={age:.2f}s  target[{wt}] speed[{ws}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32 Wi-Fi link test")
    parser.add_argument("--host", required=True, help="ESP32 IP (시리얼 STATUS 참고)")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--telemetry-port", type=int, default=8889)
    parser.add_argument("--seconds", type=float, default=3.0, help="ping 시간")
    parser.add_argument("--move", nargs=3, type=float, metavar=("VX", "VY", "W"),
                        help="로봇 기준 cm/s, cm/s, rad/s")
    parser.add_argument("--duration", type=float, default=1.0, help="구동 시간(최대 3초)")
    parser.add_argument("--slow", action="store_true", help="SLOW 상태(최대 5cm/s)로 전송")
    parser.add_argument("--reset", action="store_true", help="reset_fault 전송")
    args = parser.parse_args()

    config = UdpConfig(enabled=True, host=args.host, port=args.port, start_armed=True,
                       speed_status="SLOW" if args.slow else "RUN")
    listener = TelemetryListener(args.telemetry_port)
    sender = Esp32Sender(config, listener)
    print(f"[TEST] 대상 {args.host}:{args.port}, session={sender.session_id}")
    started = time.monotonic()
    try:
        if args.reset:
            sender.reset_fault()
            print("[TEST] reset_fault 전송")

        if args.move is None:
            print("[TEST] ping: STOP/heartbeat만 전송 (모터 정지 상태 유지)")
            next_print = 0.0
            while time.monotonic() - started < args.seconds:
                sender._transmit(sender._next("heartbeat"))
                if time.monotonic() - started >= next_print:
                    print_telemetry(listener, started)
                    next_print += 0.5
                time.sleep(0.1)
        else:
            vx, vy, w = args.move
            duration = min(max(args.duration, 0.0), 3.0)
            print(f"[TEST] cmd_vel vx={vx} vy={vy} w={w} for {duration}s  (Ctrl+C = 즉시 정지)")
            sender.send_stop()
            move_started = time.monotonic()
            next_print = 0.0
            while time.monotonic() - move_started < duration:
                packet = sender._next("cmd_vel", vx=vx, vy=vy, w=w, status=config.speed_status)
                sender._transmit(packet)
                if time.monotonic() - move_started >= next_print:
                    print_telemetry(listener, started)
                    next_print += 0.25
                time.sleep(0.05)
            sender.send_stop()
            time.sleep(0.5)
            print_telemetry(listener, started)

        if listener.latest() is None:
            print("[TEST] telemetry를 받지 못했습니다. 확인할 것:\n"
                  "  - ESP32와 노트북이 같은 Wi-Fi(같은 서브넷)인지, ESP32는 2.4GHz만 지원\n"
                  "  - --host IP가 ESP32 시리얼 STATUS의 IP와 같은지\n"
                  "  - Windows 방화벽에서 Python의 UDP 수신(8889)을 허용했는지\n"
                  "  - 게스트/학교 Wi-Fi의 기기 간 통신 차단(AP isolation)")
            return 1
        print("[TEST] 연결 OK")
        return 0
    except KeyboardInterrupt:
        print("[TEST] 중단")
        return 1
    finally:
        sender.close()


if __name__ == "__main__":
    raise SystemExit(main())
