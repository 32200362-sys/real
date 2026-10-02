# UDP 명령 프로토콜 SC1 (구형)

> 현재 ESP32 펌웨어(`esp32/esp32_omni_controller`)는 이 형식을 받지 않습니다. 기본 설정(`udp.protocol: esp32_json`)은
> [typed JSON 프로토콜](esp32/ESP32_MOTION_CONTROLLER.md)을 사용합니다. 이 문서는 `udp.protocol: sc1`용 참고 자료입니다.

## 목적

Raspberry Pi/PC는 카메라와 AI 판단만 담당하고, ESP32는 3륜 역기구학·엔코더·PID·PWM을 담당합니다. AI 코드가 모터별 PWM을 직접 보내지 않습니다.

## 형식

ASCII 한 줄이며 구분자는 `|`입니다.

```text
SC1|seq|timestamp_ms|cmd|vx|vy|wz|risk
```

예시:

```text
SC1|153|12345678|MOVE|0.1800|-0.0400|0.0000|82
```

| 필드 | 형식 | 설명 |
|---|---|---|
| `SC1` | 문자열 | 프로토콜 버전 |
| `seq` | uint32 | 패킷 순서 번호 |
| `timestamp_ms` | uint64 | 송신 측 monotonic 시간 |
| `cmd` | 문자열 | `STOP`, `MOVE`, `RETURN`, `HOLD` |
| `vx` | float | 로봇 오른쪽 방향 속도, m/s |
| `vy` | float | 로봇 앞 방향 속도, m/s |
| `wz` | float | 반시계 회전 각속도, rad/s |
| `risk` | int | 0~100 위험 점수 |

## ESP32 처리

- `MOVE`, `RETURN`: `vx`, `vy`, `wz`를 역기구학으로 바퀴 목표 RPM으로 변환
- `STOP`, `HOLD`: 목표 RPM을 0으로 설정
- 마지막 정상 패킷 이후 350 ms가 지나면 강제 정지
- 잘못된 헤더 또는 필드 수의 패킷은 무시

UDP는 전달 보장을 하지 않으므로 AI 프로그램은 20 Hz로 최신 명령을 반복 전송하고, ESP32는 오래된 명령을 유지하지 않도록 타임아웃 정지를 수행합니다.
