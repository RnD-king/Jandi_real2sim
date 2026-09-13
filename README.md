# Jandi Real2Sim — MX-106 Mode 3 BAM identification

이 저장소는 MX-106R(2.0)을 **Position Control Mode(Mode 3), 내부 P Gain 850**으로
사용하는 단일 모터 식별 도구다. 연구 순서는 다음으로 고정한다.

`BAM 실험·식별 → MuJoCo DC motor 모델 → microduck_rl 방식 RL 통합 → 상태형 백래시`

구형 Mode 5 및 조립 로봇 M0/M1/equivalent-PD 파이프라인은 제거했다. 새 실험의 설정,
원시 데이터 및 결과는 각각 `configs/mode3_bam`, `data/raw/mode3_bam`,
`results/mode3_bam` 아래에만 저장한다.

## 설치와 실행

```bash
cd /home/noh/Jandi_real2sim
uv sync

# 하드웨어 없이 GUI와 작업 흐름 확인
uv run jandi-r2s-mode3-bam-gui --mock

# 실제 MX-106 실험
uv run jandi-r2s-mode3-bam-gui
```

## 하드웨어 자동 확인·부호 캘리브레이션

기본 사용법은 별도 명령을 외우지 않고 실험 GUI의 `Calibration` 탭에서 아래 네 버튼을
위에서 아래로 누르는 것이다.

1. `Auto discover / readback`: 포트·baudrate·ID 자동 탐색, Torque OFF, 레지스터와 현재 상태 확인
2. `Save upright q=0`: 사용자가 맞춘 도립 위치의 encoder tick 저장(움직임 없음)
3. `Direction sign test`: raw +32 tick(약 2.81°) 이동·복귀 후 눈으로 본 +관절 방향을 Yes/No로 판정
4. `Apply measured values to YAML`: 장치 정보·영점·세 부호와 중력 부호를 canonical YAML에 기록

여러 장치가 연결됐거나 자동 탐색 범위를 줄여야 하면 탭 상단의 Port/Baudrate/Motor ID에
필터를 입력한다. 1번 readback 보고서와 4번 최종 보고서는 `data/calibration`에 남는다.
`--mock` GUI에서는 전체 흐름을 확인할 수 있지만 YAML과 보고서를 실제로 저장하지 않는다.

아래 CLI는 GUI를 쓸 수 없을 때의 동일 기능 진단용 fallback이다.

첫 명령은 포트와 표준 baudrate를
탐색하고 연결된 Protocol 2.0 모터가 정확히 하나일 때 ID/model/firmware, Mode,
Position PID, Homing Offset, PWM·Current·Velocity·전압·온도 제한과 현재 상태를 읽는다.

```bash
cd /home/noh/Jandi_real2sim

# YAML을 바꾸지 않고 readback 보고서만 생성
uv run jandi-r2s-mode3-bam-calibrate

# 장치가 여러 개면 연결값을 명시
uv run jandi-r2s-mode3-bam-calibrate \
  --port /dev/ttyUSB0 --baudrate 3000000 --motor-id 0
```

결과는 `data/calibration/<timestamp>_mode3_calibration.json`에 저장된다. 확인한 연결값,
모델, Homing Offset, Drive Mode, D Gain, PWM/Current Limit을 YAML에 옮기려면 `--apply`를
붙인다.

영점과 부호는 눈으로 물리 방향을 확인해야 하므로 별도의 저진폭 실험을 사용한다.
추는 제거하고 혼 또는 가벼운 막대만 장착한 뒤, 추가 달렸을 때 수직 위가 될 위치를
정확한 q=0으로 잡는다. 기본 32 tick 이동은 약 2.81°다.

```bash
uv run jandi-r2s-mode3-bam-calibrate \
  --sign-test --execute \
  --confirm MOVE_MX106_MODE3_CALIBRATION \
  --apply
```

명령은 Torque Off에서 영점 tick을 여러 번 읽고, 현재 위치를 Goal로 설정한 다음
Torque On하여 raw-positive 방향으로만 저진폭 이동하고 원위치로 복귀한 뒤 Torque Off한다.
사용자가 그 움직임이 정의한 `+관절 방향`인지 `y/n`으로 답하면 `direction`과
`pwm_direction`을 계산한다. 이동 중 Present Current 부호는 중력과 역기전력의 영향을
받아 토크축 판별에 쓰지 않고, `current_direction`은 PWM과 동일한 관절축 규약을 사용한다.
q=0을 도립 위치로 정의하므로 `gravity_torque_sign=1`도 기록한다. 전류/PWM 신호가 너무
작으면 결과를 적용하지 않는다.

이 명령도 자동으로 정할 수 없는 값은 다음과 같다.

- 추·막대의 질량, COM 거리와 관성
- 실제 치구에 맞춘 software position range
- 장치 정격보다 보수적으로 정할 전류·PWM·속도·온도·전압 안전 임계값
- 100 Hz 통신 단절 시 허용할 Bus Watchdog 시간

캘리브레이션 보고서는 하드웨어 한계를 SI 단위로 함께 보여주지만, 한계까지 구동하는
위험 실험으로 안전 임계값을 자동 측정하지는 않는다.

실기체 실행 전 다음 설정의 `null` 값을 실측값으로 채운다.

- `configs/mode3_bam/hardware.yaml`: 포트, baudrate, motor ID/model, zero/sign
- `configs/mode3_bam/controller.yaml`: Mode 3 register readback 기대값
- `configs/mode3_bam/bench.yaml`: 원판·고정부·혼·막대 물성치와 거리 2개
- `configs/mode3_bam/safety.yaml`: 위치·속도·전류·PWM·전압·온도 제한
- `configs/mode3_bam/trajectories.yaml`: 각 실험 궤적의 진폭·주기·시간
- `configs/mode3_bam/fit.yaml`: 최적화 초기값·범위

GUI에서 campaign ID, 원판/고정부/혼/막대 물성치와 거리도 입력하고 검증할 수 있다.

현재 `bench.yaml`의 canonical 기본값은 실측값으로 채워져 있다.

- 무부하 probe: 0.0039 kg 혼 하나만 회전
- 부하 실험: 별도의 0.006 kg 혼과 0.0063 kg 체결볼트가 회전하며, 형상 관성은
  모든 부하 조건에 공통인 loaded 등가 armature 관성에 흡수
- 막대 0.0531 kg, COM 0.07337 m, COM 기준 관성 0.000119108 kg·m²,
  회전축 기준 관성 0.000404978 kg·m²
- 원판 1개 0.2559 kg, 지름 0.07 m
- 추 고정부 0.0524 kg: 추 개수와 관계없이 한 세트, 자체 중심관성은 무시
- 축에서 원판/고정부 공통 COM까지 거리: 0.10 m와 0.15 m

원판은 회전축이 원판 면에 수직인 이상적 원판으로 계산한다. 프로그램은 각 원판의
자체 관성 `0.5*m*(diameter/2)^2`를 자동 계산한다. 따라서 부하 조립체는 각각
0.3083/0.5642/0.8201 kg이며, 자체 중심관성은 각각
0.00015673875/0.00031347750/0.00047021625 kg·m²다. 전체 축 기준 관성에는
`막대 축 기준 관성 + 부하 조립체 질량*d² + 원판 자체 중심관성`이 들어간다.

## 실험 행렬

GUI의 한 조건 버튼은 그 조건에 필요한 궤적을 모두 순서대로 실행한다.

- `No load calibration`: `delay_probe`, `backlash_probe`
- `Mass1..3 × Distance1..2`: `sin_time_square`, `sin_sin`, `up_and_down`, `lift_and_drop`

`sin_time_square`는 현재 벤치에서 PWM 포화를 줄이기 위해 진폭 0.60 rad,
주파수 0.10→1.00 Hz로 사용한다.

현재 벤치는 `q=0`에서 추가 수직 위를 보는 도립 구조다. 따라서 원본 BAM처럼 긴 자유낙하를
사용하지 않는다. 이 프로젝트의 `lift_and_drop` profile 2는 -50°에서 Torque OFF하고,
-70°부터 측정된 위치·속도를 이어받아 감속하여 -88° 이내 정지를 목표로 한 뒤 0 rad로
복귀한다. -93°는 소프트웨어 경고이며 실측한 물리 충돌 한계는 -95°다. Drop 구간은 속도 안전한계로
중단하지 않으며, 시간은 정상
종료 조건으로 사용하지 않는다. 단, 2초 동안 상태 조건이 한 번도 발생하지 않으면 무동작
파일럿으로 판정해 0 rad로 복귀한 후 run을 invalid 처리한다. 소프트웨어만으로 정지를
보장하지 않으므로 물리적 완충장치가 필수다. 일반 구동 명령의 최대는
`up_and_down`의 +0.70 rad(+40.1°)이고, 낙하 중 실제 각도는 이보다 커질 수 있다.
과거 Mode 5의 -90/-60/-30/0/+30/+60/+90° 정적 전류 실험은 이번 Mode 3 BAM 동적 식별과
목적이 달라 canonical campaign에는 포함하지 않는다.

따라서 조건은 무부하 1개와 `질량 3 × 거리 2`의 부하 6개, 총 7개다. 각 조건을
3회 수행하며 repeat 1·2만 피팅하고 repeat 3은 검증 전용으로 사용한다. 같은 조건과
repeat를 재실행하면 기존 로그를 덮어쓰지 않고 `attempt_NNN`을 새로 만든다.
배치가 중간에 실패한 후 같은 condition/repeat 버튼을 다시 누르면, 이미 `valid: true`인
궤적은 건너뛰고 첫 미완료 궤적부터 재개한다. 또한 각 궤적은 성공 후 Torque OFF 전에
마지막 측정 위치·속도를 이어받는 제한된 등감속 구간을 거친 뒤 0 rad로 복귀한다. 이는
고속 종료 순간의 급제동과 궤적 사이 도립 벤치의 낙하를 줄이기 위한 절차다.
Drop profile이 변경되면 이전 profile의 `valid` 로그는 보존하되 진행률과 피팅에서는 제외한다.
따라서 같은 condition/repeat 버튼을 누르면 나머지 유효 궤적은 건너뛰고 Drop만 새 attempt로
자동 재측정한다.

## GUI 사용 순서

1. 모터에서 막대와 추를 제거하고 전원, USB 통신 및 비상 정지 수단을 확인한다.
2. `uv run jandi-r2s-mode3-bam-gui`로 툴을 켠다.
3. 왼쪽 `Connection`의 `Connect / Readback`을 누른다. 표시값과 실제 모터 ID,
   Mode 3, P=850 및 설정한 D gain이 일치해야 한다.
4. `Campaign ID`에 새 실험 이름을 입력한다. 공백 없이 모터와 날짜를 구분할 수 있는
   이름을 권장한다. 예: `mx106_id0_mode3_20260904`.
5. GUI에 표시된 원판, 고정부, 혼, 막대 물성치와 거리 0.10/0.15 m가 실제 치구와
   일치하는지 확인하고 `Validate & save campaign/bench values`를 누른다.
6. `Repeat 1`을 선택한다. 무부하 상태에서 `No load calibration`을 먼저 실행하고,
   같은 방식으로 무부하 repeat 2·3을 완료한다.
   각 run은 현재 위치를 Goal로 넣은 뒤 Torque ON하고, 3초 동안 첫 자세로 부드럽게
   이동한 다음 본 궤적과 telemetry 기록을 시작한다. 초기 전환 구간은 기록에서 제외된다.
7. 첫 부하 본실험 전에 `Mass1 × Distance1`을 조립하고 `Drop safety pilot`을 한 번 실행한다.
   파일럿은 본실험 데이터와 분리되어 `pilot/drop_safety` 아래 저장되며, 성공 전에는 부하
   본실험 버튼이 거부된다.
8. 해당 질량과 거리로 장치를 실제 조립한 후 일치하는 `MassN × DistanceN` 버튼을
   누른다. 확인 창의 질량·거리·궤적을 다시 보고 승인한다.
9. 여섯 부하 조건을 모두 끝낸다. 같은 순서로 repeat 2와 repeat 3을 수행한다.
   `7 × 3 progress` 탭에서 각 칸이 완성됐는지 확인한다.
10. `Ordered fitting` 탭에서 아래 순서를 위에서 아래로 한 번씩 실행한다. 앞 단계의
   결과가 없으면 다음 단계는 거부된다.
11. `M1–M5 / select`가 만든 비교 그래프와 repeat-3 검증 MAE를 확인한 뒤 마지막으로
    유효 백래시를 계산한다.
12. 실험 중 이상 진동, 충돌, 전압 이상 또는 치구 이탈이 보이면 즉시 빨간
    `TORQUE OFF` 버튼을 누른다. 작업을 마칠 때도 Torque Off 후 창을 닫는다.

`--mock`은 GUI 조작 확인용이며 canonical 실험 데이터를 만들지 않는다. 실제 실험 중
안전 한계를 피하려고 YAML 값을 임의로 키우지 말고, 원인을 확인한 뒤 측정 가능한
장치 정격과 실험 치구 범위 안에서 수정한다.

## 순차 식별

1. 시간/제어기 특성
2. M1 — Coulomb + viscous
3. M2 — M1 + Stribeck
4. M3 — M1 + 부하 의존 마찰
5. M4 — Stribeck + 부하 의존 마찰
6. M5 — 모터 구동/외력 방향을 분리한 directional load-dependent friction
7. M1~M5 repeat-3 검증 오차 비교 및 가장 단순하면서 충분한 모델 선택
8. 유효 상태형 백래시 추정

## MuJoCo 단일 진자 실행 및 검증

식별과 MuJoCo 검증은 분리한다. 기본 실행은 실측 데이터를 읽지 않고 `trajectories.yaml`의
네 궤적을 MuJoCo 단일 진자에서 모두 실행한다. 질량 3개와 거리 2개의 여섯 조건을
`--condition`으로 선택할 수 있으며, M1과 M3를 각각 또는 동시에 실행한다.

```bash
cd /home/noh/Jandi_real2sim

# 기본: 선택한 하중 조건에서 네 궤적 모두 실행, 실측 비교 없음
uv run jandi-r2s-mode3-bam-mujoco \
  --condition mass1_distance1 \
  --model both

# 특정 궤적만 실행
uv run jandi-r2s-mode3-bam-mujoco \
  --condition mass3_distance2 \
  --trajectory sin_sin \
  --model m3

# 검증 전용 repeat 3 실측값과 네 궤적 모두 비교
uv run jandi-r2s-mode3-bam-mujoco \
  --condition mass1_distance1 \
  --compare-repeat 3 \
  --model both

# repeat 3의 특정 궤적만 비교
uv run jandi-r2s-mode3-bam-mujoco \
  --condition mass3_distance2 \
  --trajectory sin_sin \
  --compare-repeat 3 \
  --model m3
```

조건 선택지는 `mass1_distance1`부터 `mass3_distance2`까지 여섯 개이고,
궤적은 `sin_time_square`, `sin_sin`, `up_and_down`, `lift_and_drop`이다.
`--trajectory`를 생략하면 네 궤적을 모두 실행한다. `--compare-repeat`을 생략하면
실측값을 전혀 불러오지 않는 순수 MuJoCo 실행이며, 이때 입력전압은 기본 12.0 V이고
`--voltage`로 변경할 수 있다. 기본 MuJoCo physics timestep은 0.001초다.

순수 MuJoCo 실행 결과는 다음 위치에 저장된다.

```text
results/mode3_bam/<campaign_id>/mujoco_simulation/
  <condition>/<trajectory>/
    summary.yaml
    simulation.csv
    simulation.png
```

`simulation.png`에는 명령 위치와 M1/M3의 위치, 속도, PWM, 전류 및 예측 토크만
표시되며 실측 곡선은 포함되지 않는다. `--compare-repeat 3`을 지정한 비교 결과는
다음 위치에 별도로 저장된다.

```text
results/mode3_bam/<campaign_id>/mujoco_validation/
  <condition>/<trajectory>/repeat_<N>/
    summary.yaml
    comparison.csv
    comparison.png
```

`comparison.png`에는 실측값과 MuJoCo의 위치, 속도, PWM, 전류 및 예측 토크가 함께 나온다.
전류 비교는 기존 `present_current_A` 열을 그대로 신뢰하지 않고 원본
`present_current_raw`를 현재 `current_direction`으로 다시 변환한다. 따라서 과거 CSV를
수정하지 않고도 현재 관절축 부호 규약으로 다시 비교할 수 있다. 새로 수집하는 run은
사용한 위치·속도·PWM·전류 방향을 `metadata.json`에도 함께 고정 기록한다.
실측 전류는 피팅에 사용되지 않았으므로 전류 오차는 독립 검증 지표다. 현재 단계는
백래시 없는 M1/M3 검증이며, 상태형 백래시는 이 기준 검증 이후 별도로 추가한다.

M5는 BAM 논문의 directional model 식을 따른다. 일반 부하 계수를 모터 토크 측과
외력 토크 측으로 나누고, Stribeck 부하 계수도 같은 방식으로 분리한다. M6의 이차
부하 항은 현재 실험 범위와 파라미터 수를 고려해 포함하지 않는다.

각 실행 폴더에는 `telemetry.csv`, `metadata.json`, `command_events.csv`,
`trajectory.png`가 함께 저장된다. 최종 모델 선택은 더 복잡한 모델의 repeat-3 위치
MAE가 `fit.yaml`의 개선 임계값 이상 좋아질 때만 허용한다.

실행 전에 전체 Goal Position 배열을 software position limit과 대조한다. 실행 중에는
모든 state sample에 안전 한계를 적용하고 Hardware Error Status를 설정 주기(기본 1 Hz)로
별도 확인한다. `command_events.csv`에는 본 궤적 구간의 각 Goal write/torque-off sample과
호스트 송신 전후 timestamp가 기록된다. Step 기반 시간 지연 계산에서는 Goal Position을
연속 신호로 선형보간하지 않고 previous-value hold(ZOH)로 재구성한다.

## 근거

- [BAM documentation](https://bam.readthedocs.io/en/latest/)
- [BAM GitHub](https://github.com/Rhoban/bam)
- [Extended Friction Models for the Physics Simulation of Servo Actuators](https://arxiv.org/abs/2410.08650)
- [ROBOTIS MX-106R(2.0) e-Manual](https://emanual.robotis.com/docs/en/dxl/mx/mx-106-2/)
