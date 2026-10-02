# 스마트 코스터 로봇 통합 코드

카메라 영상에서 **YOLO 일반 컵 + MediaPipe Pose/Hands + ArUco 코스터**를 인식하고, 손의 closing speed와 화면 평면 TTC를 계산해 ESP32용 `vx`, `vy`, `wz` 명령을 생성하는 프로젝트입니다.

## 포함 범위

- Raspberry Pi/PC Python AI 파트
  - MediaPipe Tasks Hand Landmarker
  - OpenCV ArUco (`10=CUP`, `20=COASTER`, `30=LAPTOP`, `40=PAPER`)
  - 손 속도, 접근 방향, 컵과의 거리 변화 계산
  - 위험 점수 및 상태 머신
  - 회피/복귀 벡터 생성
  - UDP 명령 전송 및 통신 끊김 대비 STOP
  - 화면 디버그 표시와 CSV 로그
- ESP32 Arduino 파트 (`esp32/esp32_omni_controller`)
  - Wi-Fi UDP 수신 (typed JSON, 8888) / telemetry 송신 (8889)
  - 3륜 옴니휠 역기구학·오도메트리
  - 엔코더 속도 측정
  - 100 Hz PID + feed-forward + PWM 제어
  - 300 ms 명령 워치독, fault 래치/복구
- ArUco 마커 이미지와 생성 스크립트
- 단위 테스트

> 현재 코드는 **1차 통합용 기준 코드**입니다. 카메라 위치, 바퀴 장착 방향, 엔코더 분해능, 실제 하중에 따라 임계값·모터 부호·PID를 반드시 조정해야 합니다.

---

## 1. 폴더 구조

```text
smart_coaster_ai-main/
├─ config/settings.yaml          # 노트북 쪽 설정 (ESP32 IP 등)
├─ data/logs/
├─ docs/esp32/                   # ESP32 펌웨어 상세 문서
├─ docs/udp_protocol.md          # 구형 SC1 형식 (udp.protocol: sc1)
├─ esp32/esp32_omni_controller/  # ESP32에 업로드할 펌웨어
├─ markers/
├─ models/
├─ scripts/
├─ src/smart_coaster/
├─ tests/
├─ requirements.txt
└─ run.py
```

## 2. Python 설치

MediaPipe Tasks 공식 안내는 Windows, macOS, Linux, Raspberry Pi OS 64-bit와 Python 3.9 이상을 대상으로 합니다. 프로젝트에서는 Python 3.11 또는 3.12를 우선 권장합니다.

### Windows PowerShell

```powershell
cd smart_coaster_ai-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\download_models.py
```

모델 없이 로직을 검증할 때는 `python -m pytest -q`를 사용합니다. 테스트는 fake camera/detector/sender를 사용하며 모델 다운로드나 네트워크를 요구하지 않습니다.

실행 예:

```powershell
python run.py --no-udp --profile pc
python run.py --no-udp --profile pi --no-collect
```

`--collect`를 지정한 경우에만 정상 샘플과 행동 이벤트 프레임을 `data/collection/`에 저장합니다. Pi 프로필은 기본적으로 수집이 꺼져 있습니다.

### Raspberry Pi OS / Linux

```bash
cd smart_coaster_ai-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
```

Raspberry Pi에서 OpenCV 설치가 실패하면 먼저 다음 패키지가 필요할 수 있습니다.

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

## 3. 마커

`markers/`에 다음 이미지가 포함되어 있습니다.

- `marker_10_CUP.png`
- `marker_20_COASTER.png`
- `marker_30_LAPTOP.png`
- `marker_40_PAPER.png`

모든 마커는 `DICT_4X4_50` 사전을 사용합니다. 출력할 때 검은 테두리가 잘리지 않게 하고, 반사가 적은 평평한 면에 부착하세요.

다시 생성하려면:

```bash
python scripts/generate_aruco_markers.py
```

## 4. PC에서 AI만 테스트

기본 설정은 UDP가 꺼져 있어 모터가 움직이지 않습니다.

```bash
python run.py
```

키:

- `Q` 또는 `ESC`: 종료
- `C`: 현재 코스터 마커 위치를 복귀 기준점(Home)으로 재설정
- `S`: 즉시 STOP 명령 생성

동영상으로 테스트:

```bash
python run.py --video data/sample.mp4
```

카메라 번호 변경:

```bash
python run.py --camera 1
```

## 5. 노트북 + ESP32 연결 (Wi-Fi UDP)

카메라를 노트북에 USB로 연결해 AI를 돌리고, 결과를 Wi-Fi로 ESP32 모션 컨트롤러
(`smart_coaster_ai/arduino/esp32_omni_controller`)에 보냅니다. 카메라 화면은 노트북 창에 계속 표시됩니다.

```text
[USB 카메라] → [노트북: AI + 카메라 창] ──UDP 8888 (typed JSON, cm/s)──▶ [ESP32]
                                      ◀──UDP 8889 telemetry (5Hz)──────
```

1. 노트북과 ESP32를 같은 Wi-Fi에 연결합니다 (ESP32는 2.4GHz만 지원, 게스트/학교망은 기기 간 통신 차단 주의).
2. ESP32 시리얼 모니터에서 `STATUS`로 IP를 확인합니다.
3. AI 없이 통신만 확인합니다 (모터는 움직이지 않음):

   ```bash
   python scripts/esp32_link_test.py --host 192.168.0.50
   ```

   바퀴를 띄운 상태에서 짧은 구동 시험(로봇 기준 cm/s, +x 전방, +y 왼쪽):

   ```bash
   python scripts/esp32_link_test.py --host 192.168.0.50 --move 5 0 0 --duration 1
   ```

4. AI 실행:

   ```bash
   python run.py --udp --host 192.168.0.50 --profile pc
   ```

   시작 시 모터는 **DISARMED**(STOP만 전송) 상태입니다. 카메라 창을 보고 준비되면 키를 누릅니다.

| 키 | 동작 |
|---|---|
| `M` | 모터 명령 ARM / DISARM 전환 |
| `S` | 즉시 STOP + DISARM |
| `R` | latched fault 해제 (`reset_fault` → heartbeat) |
| `Q` / `ESC` | 종료 (STOP 반복 전송) |

카메라 창 아래쪽 패널에 ARM 상태, 보낸 명령(cm/s), ESP32 연결(telemetry 수신 시간, RSSI), fault,
바퀴 속도가 표시됩니다. 코스터 마커(ID 20) 위의 **빨간 FRONT 화살표**는 AI가 생각하는 로봇 전방이고,
**청록 화살표**는 보낸 이동 방향입니다.

- 좌표 변환: AI의 화면 기준 m/s(vx=오른쪽, vy=위)를 로봇 기준 cm/s(+x 전방, +y 왼쪽)로 회전합니다.
  `udp.heading_source: marker`면 코스터 마커의 방향을 따라가므로 로봇이 돌아가도 회피 방향이 유지됩니다.
- 처음 한 번 `udp.heading_offset_deg`를 맞춥니다: FRONT 화살표가 실제 로봇 앞을 가리키도록 각도(반시계 +)를 조정합니다.
  `camera.mirror`는 `false`로 둡니다 (좌우 반전 시 방향이 뒤집힘).
- 코스터 마커가 안 보이면 방향을 알 수 없으므로 STOP을 보냅니다.
- 속도는 `udp.max_linear_cm_s`(기본 15)로 제한됩니다. ESP32는 30cm/s 초과 명령을 latched fault로 처리합니다.
- 1초마다 heartbeat를 보내 CMD_TIMEOUT/WIFI_LOSS 같은 복구 가능한 fault는 자동 해제됩니다.
- telemetry를 못 받으면 Windows 방화벽에서 Python의 UDP 8889 수신을 허용하세요 (송신만은 방화벽과 무관).
- 구형 ASCII SC1 형식이 필요하면 `udp.protocol: sc1`.

처음에는 로봇 바퀴를 바닥에서 띄운 상태에서 확인하세요. 프로그램 종료 시 STOP 패킷을 여러 번 전송하며, ESP32도 주행 중 300 ms 동안 새 명령이 없으면 자동 정지합니다.

## 6. ESP32 업로드

1. Arduino IDE에서 `esp32/esp32_omni_controller/esp32_omni_controller.ino`를 엽니다 (같은 폴더 파일이 탭으로 함께 열림).
2. 같은 폴더의 `secrets.example.h`를 복사해 `secrets.h`를 만들고 Wi-Fi 이름·비밀번호(2.4GHz)를 입력합니다. `secrets.h`는 Git에 올라가지 않습니다.
3. 보드 매니저: `esp32` (Espressif) 3.x (기준 3.3.12). 라이브러리 매니저: `ArduinoJson` 7.x (기준 7.4.3).
4. 보드 `ESP32 Dev Module`과 포트를 선택해 업로드합니다.
5. 시리얼 모니터 115200 bps에서 `STATUS`를 입력해 IP를 확인하고, `config/settings.yaml`의 `udp.host`에 넣습니다.

시리얼 명령: `HELP`, `STATUS`, `PIN`, `ENC`, `ZERO`, `STOP`, `M1/M2/M3 <pwm>`, `ALL <p1> <p2> <p3>`, `VEL <vx> <vy> <w>` 등.
자세한 내용은 [docs/esp32/FIRMWARE_README.md](docs/esp32/FIRMWARE_README.md)를 참고하세요.

### 핀 배치 (`robot_config.h` 기준)

| 모터 | IN1 | IN2 | PWM | Encoder A | Encoder B |
|---|---:|---:|---:|---:|---:|
| M1 | 19 | 18 | 25 | 34 | 35 |
| M2 | 21 | 22 | 26 | 16 | 17 |
| M3 | 23 | 13 | 14 | 32 | 33 |
| STBY | 27 | | | | |

- TB6612FNG의 `STBY`는 GPIO 27로 제어합니다 (부팅 시 LOW, 초기화 후 HIGH).
- GPIO 34, 35는 입력 전용이며 내부 풀업이 없으므로 외부 풀업이 필요합니다.
- GPIO 16/17은 WROVER(PSRAM) 모듈에서는 사용할 수 없습니다. WROOM 계열 보드를 사용하세요.
- GA25-370 엔코더 전원은 ESP32 입력 보호를 위해 3.3 V 사용을 우선 검토하세요.
- 모터 전원과 ESP32 전원의 GND는 공통으로 연결해야 합니다.

## 7. 반드시 조정할 값

### Python: `config/settings.yaml`

- 카메라 번호·해상도
- 위험 거리 임계값, 손 속도 임계값, 위험 점수 기준
- 회피 속도
- `udp.host` (ESP32 IP), `udp.heading_offset_deg` (로봇 전방 보정)

### ESP32: `esp32/esp32_omni_controller/robot_config.h`

- `ENCODER_COUNTS_PER_REV` (현재 898, A상 RISING x1)
- `WHEEL_RADIUS_CM`, `ROBOT_RADIUS_CM`
- `MOTOR_REVERSED[]`, `ENCODER_REVERSED[]`
- `PID_KP`, `PID_KI`, `PID_KD`, `FF_PWM_POS/NEG` (feed-forward 표)
- 속도·가속 제한 (`BODY_LINEAR_LIMIT_CM_S` 등)

## 8. 테스트

```bash
pytest -q
```

## 9. 현재 판단 로직

```text
카메라
  → 손 랜드마크 + ArUco 마커
  → 손 속도 / 손-컵 거리 / 거리 감소율 / 접근 각도
  → 위험 점수
  → IDLE / TRACKING / WARNING / AVOIDING / HOLD / RETURNING
  → STOP / MOVE / HOLD / RETURN 명령
  → UDP
  → ESP32 역기구학 + PID + PWM
```

`HOLD`는 손이 컵 근처에서 낮은 속도로 일정 시간 머무를 때 활성화됩니다. 이는 컵을 잡으려는 행동에서 코스터가 도망가는 현상을 줄이기 위한 초기 규칙입니다. 실제 손가락-컵 접촉이나 컵 들림 판정은 후속 단계에서 센서 또는 추가 모델로 보완해야 합니다.
# Integrated cup AI pipeline

`YoloCupDetector` finds the model's `cup` class dynamically, `CupSelector` chooses the cup nearest ArUco ID 20, and `CupTracker` preserves a short-lived `track_id`. `FrameProcessor` accepts injected detector/tracker/classifier implementations so logic tests do not need a camera, MediaPipe, Ultralytics, or model weights.

Run `python scripts/download_models.py`, then place/allow loading of the configured `yolo11n.pt` and `models/pose_landmarker_lite.task`. Start with `python run.py --no-udp --profile pc`; data collection is opt-in with `--collect`. The Pi profile uses lower inference frequency and keeps collection off by default.

Limitations: COCO cup detection can fail for top-view, occluded, or unusual cups. `HOLDING` only estimates a low-speed hand near the cup; it does not prove grasp/contact. TTC is a monocular top-view image-plane approximation, not true 3D collision time. PC/Pi FPS requires measurement on the actual camera and hardware. Marker IDs 30/40 are not used for path optimization, and return positioning remains fixed-camera pixel based.
