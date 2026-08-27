# Jandi Real2Sim
## MX-106R 2.0 Current-based Position Control (Mode 5) Standalone Actuator Identification

> **Status: CURRENT CANONICAL SPECIFICATION**
>
> 이 문서는 앞으로 `Jandi_real2sim` 프로젝트의 **유일한 기준 실험/식별 사양**이다.
> 과거의 조립 상태 Jandi 관절 식별, P350/P850 dual-gain 식별, PWM replay 기반 M1 식별,
> 관절군별 Kp/Kd 재식별은 **legacy 결과**로만 보존한다.
>
> Codex가 프로젝트를 수정할 때는 이 문서를 우선하며, legacy pipeline의 가정이나 수치를
> 새 Mode 5 pipeline에 자동으로 섞지 않는다.
>
> 이 문서에서 `REQUIRED` 또는 `null`로 남은 실제 하드웨어 값은 추측해서 채우지 않는다.
> 해당 값이 미입력 상태이면 실제 모터 실행을 차단해야 한다.

---

# 1. 연구 목적과 모델링 범위

## 1.1 연구 목적

본 프로젝트의 목적은 **DYNAMIXEL MX-106R 2.0 한 개를 Jandi에서 분리한 1-DOF pendulum test bench**에서 실험하여,
Jandi가 실제로 사용할 **Current-based Position Control Mode(Operating Mode 5)** 조건의
closed-loop actuator dynamics를 식별하고 이를 MuJoCo에서 재현하는 것이다.

최종적으로 필요한 것은 DYNAMIXEL 내부 firmware 전체의 복제가 아니라,
외부에서 Goal Position을 입력했을 때 실제 출력축에서 나타나는 다음 특성이다.

- Position error -> current 관계
- Current -> output torque 관계
- End-to-end command-to-current delay
- Current saturation
- Apparent/reflected inertia
- Coulomb/static friction
- Viscous friction

최종 identified actuator model은 Jandi whole-body simulation에 삽입하고,
그 이후 whole-body 실험에서는 actuator parameter를 자유롭게 다시 맞추지 않는다.

## 1.2 연구에서 주장할 수 있는 범위

최종적으로 목표하는 주장은 다음 수준이다.

> A physics-informed equivalent actuator model of the MX-106R 2.0 operated under a fixed
> Current-based Position Control configuration was identified from standalone pendulum experiments.

다음과 같이 주장하지 않는다.

> DYNAMIXEL firmware, inverter, motor electrical dynamics, gearbox contact mechanics를 완전히 복원했다.

즉 본 모델은 **embedded controller를 포함한 actuator-level effective dynamics** 또는
**closed-loop actuator dynamics** 모델이다.

## 1.3 초기 모델에서 포함할 항목

초기 canonical model은 다음 항목만 포함한다.

1. Current-domain equivalent proportional/derivative response
2. Effective current-to-output-torque coefficient
3. Effective delay
4. Current saturation
5. Apparent inertia / armature
6. M1 friction
   - Coulomb/static friction
   - Viscous friction

## 1.4 초기 모델에서 제외할 항목

다음 항목은 1차 모델에 넣지 않는다.

- Stribeck friction
- Load-dependent friction
- Directional friction
- Quadratic load-dependent friction
- Stateful backlash
- Detailed PWM/voltage/DC motor electrical model
- Temperature-dependent actuator model
- Neural-network actuator model

이 항목들은 **held-out validation residual이 반복적으로 필요성을 보여줄 때만** 후속 모델로 검토한다.

---

# 2. 실제 DYNAMIXEL 제어 구조와 MuJoCo 등가 구조

## 2.1 실제 Mode 5 제어 구조

실제 MX-106R 2.0에는 외부 PC/MCU에서 추가 joint-level PD torque controller를 사용하지 않는다.

외부 제어 측에서 보내는 최종 command는 **Goal Position**이다.

현재 실제 구조는 다음과 같이 본다.

```text
Upper-level controller / RL policy
        |
        | q_target
        v
Goal Position
        |
        v
DYNAMIXEL internal Position PID
        |
        | desired current
        v
Goal Current limit
        |
        v
DYNAMIXEL internal Current Controller
        |
        | PWM
        v
Goal PWM / PWM Limit
        |
        v
Motor + Gearbox
        |
        v
Output torque / joint motion
```

따라서 실제 로봇에서 사용자가 설정하는 `Position P Gain`, `Position D Gain`은
**DYNAMIXEL firmware 내부의 embedded controller setting**이며,
외부 joint controller에서 별도의 PD를 한 번 더 적용하지 않는다.

## 2.2 Profile

Canonical experiment에서는 DYNAMIXEL internal trajectory profile을 사용하지 않는다.

```text
Profile Velocity     = 0
Profile Acceleration = 0
```

즉 Goal Position이 갱신되면 actuator는 Profile Generator가 만든 부드러운 중간 목표가 아니라
step/ZOH 형태의 목표를 받는다.

실제 위치는 관성, current limit, friction, embedded controller 등에 의해 동적으로 추종한다.

## 2.3 Current-domain equivalent controller

Mode 5에서 embedded Position controller의 결과를 외부에서 다음과 같이 등가화한다.

```text
e = q_target - q

I_model = aP * e - aD * qdot
```

여기서:

- `aP` : equivalent current-domain proportional coefficient `[A/rad]`
- `aD` : equivalent current-domain derivative coefficient `[A*s/rad]`

중요:

- `aP`, `aD`는 DYNAMIXEL에 새로 설정하는 gain이 아니다.
- DYNAMIXEL Position P/D register는 실험 전체에서 고정한다.
- `aP`, `aD`는 고정된 내부 controller가 실제 hardware에서 만드는 current-domain 효과를 식별하기 위한 물리 단위의 등가계수다.
- `aD`는 firmware 내부 D 항의 정확한 SI 변환값이라고 단정하지 않는다.
- `aD`가 반복 실험/주파수/부하에 대해 일관되지 않으면 강제로 물리상수처럼 사용하지 않는다.

## 2.4 Current saturation

실제 Mode 5에서는 Goal Current가 Position controller가 생성한 desired current를 제한한다.

Nominal current cap은 SI 단위로 다음과 같이 구성한다.

```text
I_goal_limit = abs(Goal Current)
I_hw_limit   = Current Limit

I_cap = min(I_goal_limit, I_hw_limit)
```

모든 값은 raw register가 아니라 A 단위로 변환해서 사용한다.

하지만 높은 속도에서는 PWM Limit, supply voltage, back-EMF 등의 영향 때문에
실제로 `I_cap`까지 도달하지 못할 수 있다.

따라서 Present PWM과 Present Input Voltage는 반드시 logging하며,
PWM saturation 영역은 별도로 tag한다.

## 2.5 Current-to-torque 관계

1차 모델:

```text
tau_motor = Ktau_eff * I_model
```

`Ktau_eff` 단위:

```text
[Nm/A]
```

초기 reference 값은 MX-106R 2.0 12 V stall specification의

```text
8.4 Nm / 5.2 A ~= 1.615 Nm/A
```

를 사용할 수 있다.

그러나 `1.615 Nm/A`는 **reference/initial guess일 뿐 final truth로 고정하지 않는다.**

본 프로젝트에서 필요한 값은 motor winding 자체의 ideal torque constant가 아니라
gearbox와 transmission effect를 포함해 출력축에서 관측되는

```text
effective output torque-current coefficient
```

이다.

## 2.6 최종 torque-domain equivalent PD

식별 결과를 torque-domain으로 표현하고 싶을 경우:

```text
Kp_eq = Ktau_eff * aP
Kd_eq = Ktau_eff * aD
```

단위:

```text
Kp_eq : Nm/rad
Kd_eq : Nm*s/rad
```

이 값은 **derived result**이다.

Primary identification variable은 `aP`, `aD`, `Ktau_eff`이며,
MuJoCo 구현에서는 current-domain 구조를 유지하는 것을 기본으로 한다.

---

# 3. 실제 DYNAMIXEL 고정 설정

## 3.1 실험 전체에서 고정할 값

다음 값은 actual-use setting이며 optimizer가 변경하지 않는다.

```text
Operating Mode
Position P Gain
Position I Gain
Position D Gain
Feedforward 1st Gain
Feedforward 2nd Gain
Profile Velocity
Profile Acceleration
Goal Current
Current Limit
Goal PWM
PWM Limit
Drive Mode
Baudrate
Motor ID
```

현재 canonical requirement:

```text
Operating Mode       = 5
Position I Gain      = 0
Feedforward 1st Gain = 0
Feedforward 2nd Gain = 0
Profile Velocity     = 0
Profile Acceleration = 0
```

`Position P Gain`, `Position D Gain`, `Goal Current`, `Current Limit`,
`Goal PWM`, `PWM Limit`의 실제 값은 **실기에서 사용할 확정값을 입력해야 하며 임의값 금지**.

## 3.2 Mode 변경 후 register 재설정

Operating Mode 변경 시 일부 gain/profile/limit register가 reset될 수 있으므로
실기 초기화 순서는 반드시 아래를 따른다.

```text
Torque OFF
    |
    v
Operating Mode = 5
    |
    v
Write fixed Position P/I/D
    |
    v
Write Feedforward
    |
    v
Write Profile Velocity / Acceleration
    |
    v
Write Goal Current
    |
    v
Check Current Limit
    |
    v
Write Goal PWM
    |
    v
Check PWM Limit
    |
    v
READ BACK ALL REQUIRED REGISTERS
    |
    +--> mismatch -> ABORT
    |
    v
Torque ON
```

metadata에는 사용자가 요청한 값뿐 아니라 **실제 read-back value**를 저장한다.

---

# 4. Standalone pendulum test bench

## 4.1 기본 구성

Jandi 조립 상태에서 actuator를 식별하지 않는다.

MX-106R 2.0 한 개를 분리해 아래처럼 구성한다.

```text
Fixed frame
   |
MX-106R 2.0
   |
Output axis
   |
Rigid arm
   |
Known mass
```

## 4.2 확정된 질량 조건

추 무게는 다음 세 개로 고정한다.

```text
0.250 kg
0.500 kg
0.750 kg
```

실제 측정 질량은 nominal value와 별도로 metadata에 저장한다.

## 4.3 확정된 팔 길이 조건

팔 길이는 2개를 사용한다.

이번 canonical campaign에서 축 중심부터 추 질량중심까지의 길이는 다음으로 확정한다.

```text
L1 = 0.10 m
L2 = 0.15 m
```

GUI와 backend는 이 값을 독립적으로 복제하지 않고 `bench/geometry.yaml`을 읽는다.

## 4.4 막대 자체 물성

막대 질량은 무시하지 않는다.

최종 experiment 전 다음 값을 실제 측정/계산한다.

```text
arm_mass
arm_com_radius
arm_inertia_about_joint
```

막대 무게는 추 무게 250/500/750 g와 별개이며 모든 torque/inertia 계산에 포함한다.

## 4.5 Known external torque

좌표계 정의에 따라 gravitational torque를 계산한다.

예를 들어 `q = 0`이 중력 방향 아래쪽인 pendulum convention이면:

```text
tau_g(q)
=
-[
    m_load * r_load
    +
    m_arm * r_arm_com
 ] * g * sin(q)
```

부호는 실제 coordinate sign convention과 일치시켜야 한다.

## 4.6 Known load inertia

Point-mass approximation을 적용할 수 있는 추에 대해:

```text
J_load = m_load * r_load^2
```

전체 external inertia는 최소 다음 항을 포함한다.

```text
J_external
=
J_arm
+
m_load * r_load^2
```

이 때문에 두 개의 arm length를 사용한다.

Gravity torque는 `r`에 비례하지만 load inertia는 `r^2`에 비례하므로
길이를 바꾸면 gravity와 inertia effect를 더 잘 분리할 수 있다.

## 4.7 Bearing / counter-shaft

필요하면 output arm을 별도 bearing/counter-shaft로 지지할 수 있다.

단:

- bearing friction도 측정에 포함될 수 있음
- preload를 최소화
- 실험 전체에서 동일 fixture 유지
- fixture 변경 시 campaign을 같은 dataset으로 섞지 않음

필요할 경우 별도 free-motion check로 fixture friction 이상 여부를 기록한다.

## 4.8 좌표계 계약

실험 시작 전에 다음을 반드시 검증한다.

- q positive direction
- DYNAMIXEL raw position 증가 방향
- MuJoCo joint positive axis
- gravity torque sign
- current sign
- torque sign

한 번 확정된 sign convention은 raw data, processing, MuJoCo model에서 동일해야 한다.

---

# 5. 안전 계약

## 5.1 Hardware execution default

모든 hardware CLI는 기본적으로 dry-run이다.

실제 실행에는 최소:

```text
--execute
--confirm <EXACT_CONFIRMATION_STRING>
```

두 조건을 동시에 요구한다.

## 5.2 Required-value gate

다음 값 중 하나라도 null/invalid이면 Torque ON 및 실제 trajectory 실행을 차단한다.

- motor ID
- encoder zero / offset
- direction sign
- Position P/D
- Goal Current
- Current Limit
- Goal PWM / PWM Limit
- actual load masses
- arm mass
- arm COM
- safe angle limit
- trajectory parameters

## 5.3 Software position limit

Mode 5에서 physical fixture hard stop을 신뢰하지 않는다.

코드에서 separate software safe angle range를 강제한다.

실험 trajectory는 hard stop에서 충분한 margin을 둔다.

## 5.4 Abort 조건

다음은 즉시 abort 조건이다.

- Hardware Error Status != 0
- Communication timeout
- Position safe range 위반
- Mechanical collision
- Impossible/NaN telemetry
- Excessive current
- Excessive PWM
- Low/abnormal input voltage
- Excessive temperature
- Repeated deadline overrun
- Unexpected sustained oscillation

abort 경로는 가능한 경우 Torque OFF를 시도하고 invalid run metadata를 남긴다.

## 5.5 Physical safety

Software safety는 작업자의 실제 emergency power cutoff를 대체하지 않는다.

실험 시:

- rigid fixture
- 비상 Torque OFF 또는 전원 차단 수단
- 회전 평면 내 인체 접근 금지
- 추 탈락 방지
- fast trajectory 전 pilot 검증

이 필요하다.

## 5.6 Thermal consistency

Temperature는 actuator response에 영향을 줄 수 있으므로 모든 run에:

```text
temperature_start_C
temperature_end_C
```

를 저장한다.

가능하면 동일 warm-up procedure 후 실험하고,
run 순서는 mass/length/trajectory가 특정 thermal drift와 결합되지 않도록 섞는다.

---

# 6. Project configuration 구조

Codex는 기존 코드를 재사용할 수 있지만 canonical configuration 책임은 아래처럼 분리한다.

```text
configs/mode5/
├── hardware.yaml
├── controller.yaml
├── safety.yaml
├── bench/
│   ├── geometry.yaml
│   └── loads.yaml
├── trajectories/
│   ├── accelerated_oscillation.yaml
│   ├── slow_plus_highfreq.yaml
│   ├── slowly_raise_lower.yaml
│   ├── static_calibration.yaml
│   └── delay_probe.yaml
├── campaign.yaml
└── fit.yaml
```

## 6.1 `hardware.yaml`

저장:

- serial device
- protocol version
- baudrate
- motor ID
- encoder zero
- direction
- command rate
- telemetry target rate
- model number expectation

## 6.2 `controller.yaml`

저장:

- Operating Mode
- Position P/I/D
- Feedforward
- Profile Velocity
- Profile Acceleration
- Goal Current
- Current Limit expectation
- Goal PWM
- PWM Limit expectation
- Drive Mode

## 6.3 `bench/geometry.yaml`

저장:

```yaml
arm_lengths_m:
  L1: 0.10
  L2: 0.15

arm_mass_kg: null
arm_com_radius_m: null
arm_inertia_kg_m2: null

gravity_m_s2: 9.80665

coordinate:
  zero_definition: null
  positive_direction: null
```

## 6.4 `bench/loads.yaml`

Canonical nominal loads:

```yaml
loads:
  m250:
    nominal_mass_kg: 0.250
    measured_mass_kg: null

  m500:
    nominal_mass_kg: 0.500
    measured_mass_kg: null

  m750:
    nominal_mass_kg: 0.750
    measured_mass_kg: null
```

## 6.5 trajectory YAML

실제 amplitude/frequency/speed/angle은 아직 확정되지 않았다.

따라서 Codex는 임의 숫자를 넣지 않는다.

hardware execution은 필요한 trajectory field가 null이면 차단한다.

## 6.6 `campaign.yaml`

다음을 명시한다.

- six mechanical configurations
- trajectory list
- repetitions
- fit/validation role
- randomization/order
- output path
- safety configuration snapshot
- controller snapshot

---

# 7. Canonical CLI

아래 이름을 canonical public CLI로 사용한다.
Codex는 기존 구현을 내부적으로 재사용할 수 있으나 이 interface를 제공해야 한다.

```bash
# 전체 설정/하드웨어 준비 상태 검사
uv run jandi-r2s-mode5-check

# pilot 계획/실행
uv run jandi-r2s-mode5-pilot

# static Ktau/aP calibration
uv run jandi-r2s-mode5-static

# command-to-current delay calibration
uv run jandi-r2s-mode5-delay

# 54-run dynamic campaign
uv run jandi-r2s-mode5-collect

# preprocessing + parameter identification
uv run jandi-r2s-mode5-fit

# held-out validation
uv run jandi-r2s-mode5-validate

# plots / metrics / final report
uv run jandi-r2s-mode5-report
```

실제 실행은 예를 들어:

```bash
uv run jandi-r2s-mode5-static \
  --mechanical-configuration L1_m250 \
  --approach approach_positive \
  --repeat 1 \
  --execute \
  --confirm CALIBRATE_MX106_MODE5
```

처럼 명시적 confirmation을 요구한다.

실제 confirmation string은 한 곳에서 정의하고 README/CLI help/config가 동일해야 한다.

Canonical confirmation string은 `configs/mode5/safety.yaml`에 정의하며 현재 계약은 다음과 같다.

```text
pilot   : PILOT_MX106_MODE5
static  : CALIBRATE_MX106_MODE5
delay   : CALIBRATE_DELAY_MX106_MODE5
collect : COLLECT_MX106_MODE5
```

## 7.1 단계별 실행 분리와 one-run 원칙

Canonical 실험은 하나의 명령으로 전체 campaign을 연속 실행하지 않는다.
각 단계는 서로 다른 public CLI로 분리하며, static과 dynamic 실제 실행은 반드시
단일 mechanical configuration, trajectory/approach 및 repeat를 지정한다.

```bash
# 0. 설정 검사: 모터를 움직이지 않음
uv run jandi-r2s-mode5-check

# 1. 초기 pilot: 부호·안전·fixture·전류/PWM·통신 확인
uv run jandi-r2s-mode5-pilot \
  --execute --confirm PILOT_MX106_MODE5

# 2. Ktau_eff/aP용 static calibration: 아래 명령은 36개 중 정확히 한 sweep만 실행
uv run jandi-r2s-mode5-static \
  --mechanical-configuration L1_m250 \
  --approach approach_positive \
  --repeat 1 \
  --execute --confirm CALIBRATE_MX106_MODE5

# 3. command-to-current delay용 초기 실험
uv run jandi-r2s-mode5-delay \
  --execute --confirm CALIBRATE_DELAY_MX106_MODE5

# 4. 본실험: 아래 명령은 54개 중 정확히 한 run만 실행
uv run jandi-r2s-mode5-collect \
  --mechanical-configuration L1_m250 \
  --trajectory accelerated_oscillation \
  --repeat 1 \
  --execute --confirm COLLECT_MX106_MODE5
```

`--execute`가 없으면 dry-run이다. `static`과 `collect`에 선택 인자를 생략한 상태에서는
전체 구조만 출력할 수 있지만 실제 모터 실행은 거부한다. 완료된 raw run은 overwrite하지
않으며, `--resume`은 기존의 valid run을 그대로 두고 건너뛰는 용도다.

`jandi-r2s-mode5-check`는 완료된 valid raw run과 campaign의 고정된
`execution_order`/`randomization_seed`를 비교해 `NEXT STATIC`과 `NEXT COLLECT`를 표시한다.
실행 요청이 NEXT와 다르면 기본 차단한다. 불가피한 경우에만
`--override-order --override-reason "사유"`를 함께 쓰며 사유는 raw metadata에 남는다.

## 7.2 사용자가 실측·확정해야 하는 값

다음 값은 legacy 설정이나 추정값으로 자동 대체하지 않는다.

1. Hardware (`configs/mode5/hardware.yaml`)
   - serial device, baudrate, motor ID, model number
   - encoder zero raw
   - position/current/PWM 각각의 부호
   - telemetry와 Hardware Error polling rate
2. Mode 5 controller (`configs/mode5/controller.yaml`)
   - Drive Mode와 실제 Position P/D
   - Goal Current와 Current Limit read-back
   - Goal PWM과 PWM Limit read-back
   - Bus Watchdog raw (20 ms/count, 1..127; null이면 실행 잠금)
3. Bench geometry (`configs/mode5/bench/geometry.yaml`)
   - L1/L2는 각각 0.10/0.15 m로 고정; 실제 fixture가 일치하는지 확인
   - 추를 제외한 막대·허브·체결부 전체 질량
   - 그 assembly의 COM 반경과 회전축 기준 관성 또는 COM 기준 관성
   - 물리 fixture 축 설명, 기구 영점, 중력 토크 및 MuJoCo 양의 방향 정의
   - simulation hinge는 항상 canonical `[0, 1, 0]`; fixture 차이는 1-DOF 부호/영점 mapping으로 처리
4. Loads (`configs/mode5/bench/loads.yaml`)
   - 250/500/750 g 추 각각의 실제 측정 질량
5. Safety and pilot (`configs/mode5/safety.yaml`)
   - software 각도 범위, 온도·전압·전류·PWM·속도·오차 한계
   - transition/between-run 시간, warm-up 절차와 사람이 기록한 acknowledgment timestamp
   - pilot 시험 조건과 통과 승인
6. Trajectories (`configs/mode5/trajectories/*.yaml`)
   - static 각도는 고정; 접근 offset/settling/averaging 조건
   - delay step 진폭·반복·onset 검출 조건
   - 세 본실험 파형의 중심각·진폭·주파수·속도·시간
7. Campaign and fit
   - campaign ID, 실행 순서와 random seed
   - 데이터를 보기 전에 정하는 holdout mechanical configuration
   - MuJoCo timestep, Stage D bounds/initial/seeds/evaluation budget
   - position/velocity/current loss normalization과 weights

모든 필수값을 채운 뒤 `jandi-r2s-mode5-check`가 해당 단계를 `READY`로 표시해야만
실제 실행으로 넘어간다.

---

# 8. Logging 계약

## 8.1 기본 원칙

- raw data는 절대 수정/overwrite하지 않는다.
- command와 state는 실제 timestamp를 저장한다.
- 99 Hz state + 1 Hz exclusive Hardware Error slot 구조는 canonical Mode 5 bench에서 사용하지 않는다.
- 단일 motor bench에서는 가능한 한 연속된 state time series를 유지한다.
- Hardware Error read 때문에 state sample을 의도적으로 비우지 않는다.

## 8.2 Main bench command/state rate

Main identification campaign의 기본 interface는:

```text
Goal Position command update : 100 Hz target
State telemetry              : ~100 Hz target
```

실제 rate는 로그에서 계산하며 정확히 100 Hz라고 가정하지 않는다.

`measured_command_rate_hz`는 실제 `command_tx_after_ns` event series로,
`measured_state_rate_hz`는 state RX series로 각각 계산한다. 두 interval의 mean/std/max와
deadline overrun count/max도 metadata에 별도로 저장한다.

모든 sample에 host monotonic time을 저장한다.

## 8.3 Delay measurement rate

Delay 측정 정확도는 telemetry sampling period보다 좋아질 수 없다.

가능하면 single-motor bench에서 안정적으로 지원되는 더 높은 telemetry rate를
delay experiment에 사용할 수 있도록 구현하되, **main 100 Hz experiment와 분리한다.**

Delay loop는 Goal Position이 바뀌는 event에서만 Protocol 2.0 GroupSyncWrite의
Tx-only path를 호출하고, 사이에는 state만 polling한다. `command_tx_before_ns`와
`command_tx_after_ns`는 각각 `txPacket()` 직전/직후 host monotonic time이며 Status Packet을
기다린 시각이 아니다. 측정값은 firmware pure delay가 아니라
**host-command-to-observed-current effective delay**다. 요청 rate와 별도로 실제 state
interval에서 achieved telemetry rate와 sampling resolution을 기록한다.

고속 read가 안정적이지 않으면 100 Hz를 유지하고,
delay uncertainty를 최소 1 sample 수준으로 명시한다.

## 8.4 Runtime telemetry

가능하면 contiguous/single block read를 활용해 같은 acquisition에서 다음 값을 얻는다.

- Goal Position
- Realtime Tick
- Present PWM
- Present Current
- Present Velocity
- Present Position
- Velocity Trajectory
- Position Trajectory
- Present Input Voltage
- Present Temperature
- Moving Status

Hardware Error Status는 별도 safety polling으로 읽되 읽은 host timestamp를 저장한다.

## 8.5 Raw telemetry columns

최소 canonical columns:

```text
sample_index
host_time_ns
host_time_sec
command_seq
command_tx_before_ns
command_tx_after_ns

goal_position_raw
goal_position_rad

realtime_tick_raw
realtime_tick_unwrapped_ms

present_position_raw
present_position_rad

present_velocity_raw
present_velocity_rad_s

present_current_raw
present_current_A

present_pwm_raw
present_pwm_fraction

present_input_voltage_raw
present_input_voltage_V

present_temperature_raw
present_temperature_C

position_trajectory_raw
position_trajectory_rad

velocity_trajectory_raw
velocity_trajectory_rad_s

moving_status

valid_flag
current_saturated_flag
pwm_saturated_flag
```

Hardware Error event/read data는 별도 event log 또는 timestamped field로 저장 가능하다.

## 8.6 Raw register도 보존

SI value만 저장하고 raw를 버리지 않는다.

후속 conversion 검증을 위해 둘 다 저장한다.

## 8.7 Metadata

각 run의 `metadata.json`에 최소 저장:

- project git commit
- config file paths
- config SHA-256
- resolved configuration
- DYNAMIXEL read-back registers
- motor ID/model/firmware if available
- canonical L1/L2 selected length and physical confirmation
- load nominal/measured mass
- arm properties
- trajectory parameters
- repeat index
- split role
- start/end temperature
- average/min/max input voltage
- actual command/state sampling statistics
- software version
- valid/invalid reason

---

# 9. Unit conversion

MX-106R 2.0 register values는 processing에서 SI로 변환한다.

Canonical units:

```text
position      : rad
velocity      : rad/s
current       : A
voltage       : V
time          : s
torque        : Nm
inertia       : kg*m^2
viscous coeff : Nm*s/rad
```

Raw signed register의 two's-complement conversion을 명시적으로 테스트한다.

DYNAMIXEL e-Manual 기준 current unit 등은 implementation test로 검증한다.

---

# 10. 실험 단계 전체 개요

```text
Bench measurement / configuration
        |
        v
Pilot
        |
        v
Static calibration
  -> Ktau_eff prior
  -> aP prior
  -> static friction / load-dependence diagnostic
        |
        v
Delay calibration
  -> command-to-current delay prior
        |
        v
Main dynamic campaign
  3 masses
  x 2 arm lengths
  x 3 trajectories
  x 3 repeats
  = 54 runs
        |
        v
aD initial identification
        |
        v
M1 MuJoCo parameter optimization
        |
        v
Condition-level held-out validation
        |
        v
Residual analysis
        |
        +--> sufficient -> freeze standalone actuator model
        |
        +--> insufficient -> only then consider model extension
```

Static calibration과 delay calibration은 둘 다 main fitting 전에 끝낸다.
둘의 순서는 실무상 바뀌어도 되지만, canonical 분석에서는 static calibration을 먼저 수행한다.

---

# 11. Pilot experiment

## 11.1 목적

Pilot data는 final identification에 자동 포함하지 않는다.

목적:

- sign convention 검증
- safe range 확인
- current/PWM peak 확인
- fixture 강성 확인
- oscillation 여부 확인
- communication timing 확인
- initial thermal behavior 확인

## 11.2 방식

No-load 또는 가장 안전한 configuration에서 매우 작은 movement부터 시작한다.

실제 amplitude는 configuration에 명시하고 pilot 통과 전 임의 확대 금지.

pilot 결과가 승인되기 전에는 static/dynamic campaign actual execution을 차단한다.

---

# 12. Static calibration: Ktau_eff + aP

## 12.1 목적

Main dynamic fitting 전에 독립적인 physical prior를 확보한다.

같은 static dataset에서:

1. `Ktau_eff` prior
2. `aP` prior
3. static friction branch 정보
4. load-dependent residual 여부

를 분석한다.

## 12.2 Mechanical configurations

모든 6개 configuration을 사용한다.

```text
L1 + 250 g
L1 + 500 g
L1 + 750 g
L2 + 250 g
L2 + 500 g
L2 + 750 g
```

## 12.3 Static angle set

Static angle set은 본 Mode-5 static calibration 설계로 다음 다섯 점을 고정한다.

```yaml
static_angles_rad: [-1.0471975512, -0.5235987756, 0.0, 0.5235987756, 1.0471975512]
```

이는 각각 `-60, -30, 0, +30, +60 degree`이며 BAM/논문의 값을 복사한 것이 아니다.

Angle selection은 다음 조건을 만족해야 한다.

- gravity torque가 충분히 넓게 변화
- mechanical limit에서 충분히 떨어짐
- Current/PWM saturation이 주된 영역이 되지 않음
- positive/negative torque branch를 모두 포함 가능
- encoder/static error가 noise floor보다 충분히 큼

## 12.4 Approach direction

각 static angle은 양쪽 방향에서 접근한다.

```text
approach_positive
approach_negative
```

목적은 static friction/backlash branch effect를 관찰하고
Ktau regression에서 friction bias를 진단하기 위함이다.

## 12.5 Repetition

Canonical static sweep repeat:

```text
3 repetitions
```

한 static sweep run이 모든 configured angle을 순차 hold할 수 있다.

6 mechanical configurations x 2 approach directions x 3 repeats이면
**36 static sweep runs**가 된다.

각 sweep은 위 다섯 angle point를 모두 포함한다.

## 12.6 Settling과 averaging

각 angle에서 바로 sample 한 점을 사용하지 않는다.

다음 두 단계를 분리한다.

```text
settling interval
fixed dwell/averaging interval
```

모든 condition에서 동일한 판정 규칙을 사용한다.

가능하면:

- `|qdot|`가 threshold 아래
- current/position variation이 threshold 아래

일 때 settled로 판정한다.

그 후 고정된 averaging window를 사용한다.

Dwell time이 static friction에 영향을 줄 수 있으므로 run마다 다른 임의 dwell을 사용하지 않는다.

---

# 13. Ktau_eff static prior

## 13.1 기본 평형식

정지 상태에서:

```text
qdot ~= 0
qddot ~= 0
```

이므로:

```text
tau_motor + tau_gravity + tau_friction ~= 0
```

그리고 1차 motor model은:

```text
tau_motor = Ktau_eff * I_present
```

이다.

## 13.2 한 점의 `tau/I`를 final Ktau로 사용하지 않는다

다음 방식은 금지한다.

```text
Ktau = known_torque / current
```

을 한 조건에서 계산하고 final value로 확정.

그 이유는 static friction이 포함되어 있기 때문이다.

## 13.3 권장 추정

여러:

- masses
- arm lengths
- angles
- approach directions

의 paired data를 이용한다.

가능하면 동일 mechanical state의 opposite approach branch를 비교하고
branch midpoint/robust regression을 사용해 friction bias를 줄인 `Ktau_eff prior`를 만든다.

최소 결과:

```text
Ktau_reference_from_stall = 1.615 Nm/A
Ktau_static_prior
Ktau_static_uncertainty
Ktau_branch_difference
```

## 13.4 Load-dependent friction diagnostic

본 canonical model은 M1이지만
MX-106 계열에서 load-dependent friction이 존재할 가능성을 무시하지 않는다.

따라서 static regression residual을 최소 다음과 비교한다.

```text
residual vs |tau_gravity|
residual vs load mass
residual vs arm length
residual vs approach direction
```

residual이 load와 체계적으로 증가해도 자동으로 M3/M4를 적용하지 않는다.

단지 M1 limitation evidence로 기록하고 held-out dynamic validation 후 확장 여부를 판단한다.

---

# 14. aP static prior

## 14.1 식

Static non-saturated region에서:

```text
e = q_goal - q

I_present ~= aP * e + b
```

를 사용한다.

`b`는 diagnostic intercept이며 final controller에 자동 포함하지 않는다.

## 14.2 사용 데이터

다음 조건만 사용한다.

- settled
- `|qdot|` sufficiently small
- no current saturation
- no PWM saturation
- valid telemetry
- no thermal fault

## 14.3 Cross-condition consistency

다음 값을 따로 계산한다.

```text
aP by mass
aP by arm length
aP by approach direction
aP by angle region
```

일관된 slope가 확인될 경우 통합 `aP_prior`를 사용한다.

load/direction에 따라 slope가 크게 변하면
단순 constant-aP assumption을 자동 승인하지 않는다.

## 14.4 Position P Gain을 바꾸지 않는다

static error가 너무 작아 aP 식별이 어려워도
실제 사용할 Position P Gain을 identification 편의를 위해 변경하지 않는다.

대신:

- safe load
- arm radius
- angle range
- averaging
- repeated samples

를 이용해 excitation을 확보한다.

---

# 15. Delay calibration

## 15.1 목적

측정하려는 delay는:

```text
Host Goal Position change
        ->
Observed Present Current response
```

의 effective end-to-end delay이다.

기계적 position response onset은 사용하지 않는다.

## 15.2 Trajectory

No-load 또는 안전한 low-load에서 small positive/negative step을 반복한다.

Step amplitude 목록은 pilot 후 configuration에 확정한다.

```yaml
delay_step_amplitudes_rad: null
delay_repeats: null
```

임의값으로 채우지 않는다.

## 15.3 분석

최소 두 방법을 지원하는 것이 좋다.

1. Threshold/onset based delay
2. Cross-correlation 또는 step-response alignment based delay

결과:

```text
delay_mean_s
delay_std_s
delay_by_direction
delay_by_amplitude
sampling_resolution_s
```

Sampling period보다 작은 delay 정밀도를 주장하지 않는다.

## 15.4 Optimization에서의 역할

delay는 dynamic trajectory에서 맨땅으로 다시 찾지 않는다.

우선 static/delay calibration에서 얻은 prior를 고정하고,
필요한 경우에만 measured uncertainty 범위에서 제한적으로 refinement한다.

---

# 16. Main dynamic campaign

## 16.1 Mechanical configurations

6개:

```text
L1 + 250 g
L1 + 500 g
L1 + 750 g
L2 + 250 g
L2 + 500 g
L2 + 750 g
```

## 16.2 Trajectories

Main dynamic trajectory는 정확히 3개다.

1. Accelerated oscillations
2. Slow oscillation + small high-frequency oscillation
3. Slowly raise/lower

`lift_and_drop`은 canonical campaign에 포함하지 않는다.

기존 step/triangle/multisine은 main canonical trajectory가 아니다.

Step은 delay calibration 등 별도 diagnostic 용도로만 사용 가능하다.

## 16.3 Repetition

각 mechanical configuration x trajectory마다:

```text
3 repeats
```

## 16.4 총 main dynamic run

```text
3 masses
x 2 arm lengths
x 3 trajectories
x 3 repeats
= 54 runs
```

**54는 main dynamic dataset의 정확한 총 run 수다.**

Static calibration, pilot, delay calibration run은 이 54개에 포함하지 않는다.

---

# 17. Main trajectory 정의

## 17.1 Accelerated oscillations

목적:

- low-to-high velocity excitation
- phase lag
- current response
- aD sensitivity
- viscous friction sensitivity
- armature sensitivity

Concept:

```text
approximately constant position amplitude
+
progressively increasing oscillation frequency
```

BAM의 `sin(t^2)` 형태를 참고할 수 있으나,
실제 amplitude/frequency range는 Jandi MX-106의 safe/current/PWM range를 기준으로 확정한다.

Trajectory config는 최소:

```text
amplitude_rad
start_frequency_hz or equivalent phase law
end_frequency_hz or equivalent phase law
duration_sec
fade/ramp settings
```

를 명시한다.

현재 값은 REQUIRED/TODO이며 Codex가 임의로 결정하지 않는다.

## 17.2 Slow oscillation + small high-frequency oscillation

목적:

한 run에서 서로 다른 time-scale의 dynamics를 동시에 excite한다.

- slow component -> gravity/friction/load
- fast small component -> damping/inertia/controller dynamics

Config는 최소:

```text
slow_amplitude_rad
slow_frequency_hz
fast_amplitude_rad
fast_frequency_hz
duration_sec
```

를 명시한다.

실제 수치는 REQUIRED/TODO.

## 17.3 Slowly raise/lower

목적:

- low-speed friction
- static-to-moving transition
- drive/backdrive behavior
- load-dependence residual

Host에서 slow continuous target trajectory를 생성한다.

DYNAMIXEL Profile Generator에 이 역할을 맡기지 않는다.

Config는 최소:

```text
lower_angle_rad
upper_angle_rad
command_speed_rad_s or equivalent duration
hold/transition definition
```

를 명시한다.

실제 수치는 REQUIRED/TODO.

---

# 18. Main dynamic validation split

## 18.1 Random sample split 금지

같은 trajectory의 인접 sample을 train/validation으로 나누지 않는다.

## 18.2 Repeat 3만 validation으로 쓰지 않는다

과거 README의:

```text
repeat 1,2 = fit
repeat 3   = validation
```

방식은 canonical validation contract가 아니다.

세 repeat는 **repeatability 측정**에 사용한다.

## 18.3 Dynamic trajectory condition-level holdout

54개 dynamic run 중 **하나의 dynamic trajectory holdout configuration**을
fit 이전에 validation-only로 고정한다. Static calibration은 동일한 configuration을
포함한 6개 조건 전체를 사용하므로 이를 entirely unseen mechanical configuration이라고
부르지 않는다.

예:

```text
holdout_configuration: REQUIRED
```

선택은 데이터 결과를 본 뒤 바꾸지 않는다.

한 mechanical configuration은:

```text
3 trajectories x 3 repeats = 9 runs
```

이므로:

```text
fit dynamic runs        = 45
held-out dynamic runs   = 9
total dynamic runs      = 54
```

가 된다.

`holdout_configuration`은 campaign 시작 전에 config에 명시한다.

필요하면 이후 별도의 unseen amplitude/frequency dataset을
secondary validation으로 추가할 수 있지만 canonical 54-run count에는 포함하지 않는다.

---

# 19. aD initial identification

## 19.1 데이터

별도의 전용 sine campaign을 새로 만들 필요는 없다.

Main dynamic dataset의 **Accelerated oscillation**을 우선 사용한다.

## 19.2 Baseline current model

첫 번째 candidate는:

```text
I_model = aP * e - aD * qdot
```

이다.

aP는 static prior를 사용한다.

## 19.3 aD validity

aD를 유효한 coefficient로 채택하려면 최소:

- sign이 damping과 일관
- different mass에서 크게 무너지지 않음
- different arm length에서 크게 무너지지 않음
- accelerated-frequency 구간에 따라 극단적으로 변하지 않음
- held-out current prediction이 P-only model보다 개선

되어야 한다.

그렇지 않으면:

- aD를 weakly constrained optimization variable로 둠
- 또는 P-only current model과 비교

한다.

내부 firmware D equation을 그대로 복원했다고 주장하지 않는다.

---

# 20. MuJoCo standalone bench model

## 20.1 Bench physics

실제 bench와 동일한:

- arm mass
- arm COM
- arm inertia
- load mass
- load radius
- gravity
- joint axis

를 MuJoCo에 구성한다.

## 20.2 Actuator

MuJoCo built-in position actuator가 자동으로 별도 PD를 만들게 하지 않는다.

Torque를 직접 입력할 수 있는 motor/general actuator를 사용한다.

Canonical control law:

```python
q_ref = delayed_goal_position
e = q_ref - q

I_model = aP * e - aD * qd
I_model = clip(I_model, -I_cap, I_cap)

tau_motor = Ktau_eff * I_model
```

그 torque를 MuJoCo actuator에 입력한다.

## 20.3 Joint mechanical parameters

M1 baseline:

```text
armature     = J_arm
frictionloss = Kc
damping      = Kv
```

단위:

```text
J_arm : kg*m^2
Kc    : Nm
Kv    : Nm*s/rad
```

## 20.4 M1의 의미

Canonical M1은 Coulomb/static + viscous baseline이다.

이 모델이 MX-106의 모든 friction을 완벽히 표현한다고 가정하지 않는다.

M1을 최소 identifiable baseline으로 사용하고
held-out residual이 요구할 때만 extended friction으로 이동한다.

---

# 21. Simulation timing

## 21.1 Bench replay

실제 bench command timestamp를 가능한 그대로 재현한다.

Goal Position command는 ZOH signal이므로 resampling 시 linear interpolation하지 않는다.

```text
Goal Position -> previous-value hold
```

Position/current 등 continuous measured signal은 analysis 목적에 따라 interpolation 가능하다.

## 21.2 Physics timestep

기존 README의 `500 Hz / dt=0.002 s` 값을 자동으로 canonical truth로 사용하지 않는다.

Codex는 현재 active MuJoCo model/config의 physics timestep을 확인하고
실험 replay에 충분히 빠른지 검증한다.

변경이 필요한 경우 명시적으로 config로 관리한다.

## 21.3 Jandi deployment timing

Standalone model 확정 후 Jandi에서는:

```text
RL policy              : 50 Hz
new q_target            : every 20 ms
motor command/state I/O : 100 Hz target
MuJoCo physics          : faster internal timestep
```

를 재현한다.

Simulation에서:

```text
50 Hz policy q_target
        |
        v
20 ms ZOH
        |
        v
identified delay
        |
        v
current-domain actuator model
        |
        v
torque
        |
        v
MuJoCo rigid-body dynamics
```

Actuator torque law 자체는 physics timestep마다 평가한다.

---

# 22. Data preprocessing

## 22.1 Directory structure

권장:

```text
data/
├── raw/
│   └── mode5/
│       ├── pilot/
│       ├── static/
│       ├── delay/
│       └── dynamic/
└── processed/
    └── mode5/
```

## 22.2 Raw data immutable

Raw file은 overwrite하지 않는다.

resume 시 기존 valid run을 건드리지 않는다.

Configuration이 바뀌면 새 campaign ID를 사용한다.

## 22.3 Realtime Tick unwrap

Realtime Tick은 1 ms/count, raw 0..32767, modulus 32768 기준으로 wrap을 처리하고
host monotonic time과 함께 보존한다.

두 clock을 absolute-time으로 직접 동일시하지 않는다.

## 22.4 Saturation tag

최소:

```text
NORMAL
CURRENT_SATURATED
PWM_SATURATED
THERMAL_INVALID
COMM_INVALID
SAFETY_ABORT
```

tag를 생성한다.

## 22.5 Primary fitting region

초기 current-domain identification은 가능한:

- current non-saturated
- PWM non-saturated
- normal temperature
- valid communication

영역을 중심으로 한다.

---

# 23. Parameter identification sequence

Canonical 순서는 아래와 같다.

## 23.1 Stage A: static priors

Static calibration에서:

```text
Ktau_eff_prior
aP_prior
```

및 uncertainty/branch variation을 계산한다.

## 23.2 Stage B: delay prior

Delay calibration에서:

```text
delay_prior
delay_uncertainty
```

를 계산한다.

## 23.3 Stage C: aD initial estimate

Accelerated oscillation을 이용해:

```text
aD_initial
```

및 cross-condition consistency를 평가한다.

## 23.4 Stage D: first M1 dynamic fit

우선 고정:

```text
aP       = measured prior
Ktau_eff = measured prior
delay    = measured prior
```

Primary fit:

```text
aD
J_arm
Kc
Kv
```

## 23.5 Stage E: constrained joint refinement

Stage D가 fit/held-out data를 충분히 설명하지 못할 경우에만:

```text
aP
aD
Ktau_eff
delay
J_arm
Kc
Kv
```

를 함께 refinement할 수 있다.

단:

- `aP`는 measured uncertainty 근처
- `Ktau_eff`는 measured uncertainty 근처
- `delay`는 measured delay uncertainty 근처

에서만 움직인다.

모든 값을 무제한으로 자유롭게 두지 않는다.

---

# 24. Parameter bounds / units

## 24.1 aP

```text
unit: A/rad
constraint: aP > 0
initial: static measurement
bound: measurement uncertainty / bootstrap interval 기반
```

임의 ±20%를 기본 truth로 두지 않는다.

## 24.2 aD

```text
unit: A*s/rad
initial: accelerated-oscillation estimate
default physical constraint: aD >= 0
```

negative optimum이 반복적으로 요구되면
모델식/phase/delay/current dynamics를 먼저 점검한다.

## 24.3 Ktau_eff

```text
unit: Nm/A
reference: 1.615 Nm/A
initial: static known-load estimate
constraint: > 0
bound: static estimate uncertainty 기반
```

`1.615`를 final fixed constant로 사용하지 않는다.

## 24.4 Delay

```text
unit: s
initial: command-to-current measurement
bound: measurement resolution/uncertainty 기반
```

## 24.5 Armature

```text
unit: kg*m^2
constraint: >= 0
```

BAM MX-class actuator modeling의 참고 search scale로
대략 `0.001 ~ 0.05 kg*m^2` 수준을 initial exploratory range로 둘 수 있다.

그러나 이 범위는 우리 Mode 5 MX-106의 physical truth가 아니다.

optimum이 boundary에 붙으면:

- range 부족
- identifiability 문제
- bench inertia 오류
- model mismatch

를 검토한다.

## 24.6 Coulomb/static friction

```text
parameter: Kc
unit: Nm
constraint: >= 0
```

BAM generic M1의 reference search scale은 대략 `0 ~ 0.2 Nm` 수준을
초기 탐색 참고로 사용할 수 있다.

우리 데이터가 bound를 요구하면 범위를 재검토한다.

## 24.7 Viscous friction

```text
parameter: Kv
unit: Nm*s/rad
constraint: >= 0
```

초기 conservative reference range로 `0 ~ 0.2 Nm*s/rad`를 사용할 수 있다.

BAM generic implementation은 더 넓은 viscous bound도 허용하므로,
optimum이 상한에 붙으면 range 또는 모델구조를 재검토한다.

---

# 25. Optimization objective

## 25.1 Position만 보지 않는다

본 프로젝트는 Present Current를 직접 측정할 수 있으므로
position trajectory만 fitting하는 것보다 많은 정보를 사용한다.

가능하면:

```text
q_real     vs q_sim
qd_real    vs qd_sim
I_present  vs I_model
```

을 비교한다.

## 25.2 Normalization

단위가 서로 다르므로 raw MAE를 그대로 더하지 않는다.

예:

```text
position error / characteristic trajectory amplitude
velocity error / characteristic velocity
current error / I_cap
```

로 normalized loss를 구성한다.

## 25.3 Prior regularization

Constrained refinement에서는:

```text
(aP - aP_prior) / sigma_aP
(Ktau - Ktau_prior) / sigma_Ktau
(delay - delay_prior) / sigma_delay
```

기반 regularization을 사용할 수 있다.

## 25.4 Optimizer

CMA-ES 계열을 primary derivative-free optimizer로 사용해도 된다.

최소 여러 independent seed에서 반복해
비슷한 score와 parameter region으로 수렴하는지 확인한다.

Seed에 따라 비슷한 trajectory error인데 parameter가 크게 달라지면
parameter identifiability 문제를 의심한다.

---

# 26. Validation

## 26.1 Primary held-out validation

Campaign YAML에서 미리 지정한 1개 mechanical configuration의 9 runs를 사용한다.

Fit이 끝날 때까지 optimizer가 해당 run을 보지 않는다.

## 26.2 Metrics

최소:

```text
Position MAE
Position RMSE
Velocity MAE
Current MAE
Normalized current error
Peak/phase timing error
Steady-state error
```

를 trajectory/configuration별로 저장한다.

전체 평균 하나만 보고하지 않는다.

## 26.3 Repeatability

같은 condition의 3 repeat 간 variation도 별도로 계산한다.

Simulation error가 real-to-real repeat variation보다 작은지/큰지도 참고한다.

---

# 27. Residual analysis와 모델 확장 규칙

Validation 후 최소 다음 residual plot을 만든다.

```text
residual vs position
residual vs velocity
residual vs current
residual vs |gravity/load torque|
residual vs arm length
residual vs direction
residual vs PWM
residual vs temperature
```

해석 예:

```text
velocity와 선형적으로 증가
-> viscous model 부족 가능성

zero velocity 부근에서만 커짐
-> static/Stribeck effect 가능성

load가 커질수록 체계적으로 증가
-> load-dependent friction 가능성

방향에 따라 branch가 다름
-> directional friction/backlash 가능성

high PWM에서만 오차 증가
-> current model이 voltage/back-EMF limitation을 놓칠 가능성

temperature에 따라 drift
-> thermal dependency 가능성
```

후속 모델을 추가하려면 **held-out residual에서 반복 가능한 evidence가 있어야 한다.**

M1이 부족하다는 이유만으로 M6로 바로 가지 않는다.

---

# 28. Final standalone outputs

`results/<timestamp>_mode5_m1/`에 최소 다음을 저장한다.

```text
params_mode5_m1.yaml
metrics_fit.json
metrics_validation.json
parameter_uncertainty.json
static_calibration.json
delay_calibration.json
residual_summary.json
manifest.json
report.md
plots/
```

`params_mode5_m1.yaml` 최소 필드:

```yaml
model: mode5_current_domain_m1

aP_A_per_rad: ...
aD_A_s_per_rad: ...
Ktau_eff_Nm_per_A: ...
delay_s: ...

armature_kg_m2: ...
coulomb_friction_Nm: ...
viscous_friction_Nm_s_per_rad: ...

derived:
  Kp_eq_Nm_per_rad: ...
  Kd_eq_Nm_s_per_rad: ...

current_limit_A: ...
goal_current_limit_A: ...

controller_registers:
  position_p_gain: ...
  position_i_gain: ...
  position_d_gain: ...
  profile_velocity: 0
  profile_acceleration: 0
```

---

# 29. Whole-Jandi 적용 원칙

Standalone actuator model이 held-out validation을 통과하기 전
Jandi locomotion training model을 이 parameter로 자동 수정하지 않는다.

통과 후:

1. same Mode 5 closed-loop actuator model을 12 joints에 nominally 적용
2. 실제 motor-to-motor variation이 측정되기 전에는 무분별한 joint별 fit 금지
3. Jandi에서 차이가 남으면 먼저 다음을 점검
   - link mass/inertia
   - COM
   - joint geometry
   - foot/contact model
   - ground friction
   - sensor/interface delay
   - backlash
4. whole-body trajectory만 보고 standalone actuator Kp/Kd/friction을 자유롭게 다시 맞추지 않음

---

# 30. Legacy pipeline 처리

기존 프로젝트의 다음 데이터/코드는 삭제하지 않는다.

- Position Control legacy data
- P350/P850 campaign
- M0 equivalent Kp/Kd fits
- PWM replay M1
- joint34 PD fits
- assembled-Jandi identification
- previous figures/reports

하지만 다음 규칙을 적용한다.

## 30.1 Legacy data는 새 fit에 자동 포함 금지

Mode 5 standalone campaign과 legacy data를 같은 optimizer manifest에서 섞지 않는다.

## 30.2 Legacy command는 canonical README에서 main path로 노출하지 않음

필요하면:

```text
docs/legacy/
```

로 기존 설명을 이동한다.

## 30.3 Codex migration rule

Codex는:

- 기존 안전 코드
- raw-data immutability
- checksum/manifest
- dry-run
- resume
- config validation
- plotting/utilities

는 가능한 재사용한다.

반면 아래 실험 가정은 교체한다.

```text
assembled 12-joint identification
        -> standalone one-motor pendulum

no_load / loaded
        -> 3 masses x 2 lengths

step / triangle / sine main campaign
        -> 3 canonical trajectories

direct Kp_eff/Kd_eff fit
        -> aP/aD/Ktau current-domain model

PWM as actuator model input
        -> Present PWM as diagnostic/saturation signal

repeat-3 validation
        -> condition-level holdout

99 Hz state + 1 Hz exclusive error slot
        -> continuous state telemetry + independent safety polling
```

---

# 31. Codex implementation requirements

Codex가 이 README를 기준으로 프로젝트를 수정할 때 다음 원칙을 반드시 지킨다.

## 31.1 임의값 금지

다음 미확정값을 추측하지 않는다.

- actual arm mass/COM/inertia
- dynamic trajectory amplitudes
- dynamic trajectory frequencies
- slow raise/lower speed
- delay step amplitudes/repeats
- actual Position P/D
- Goal Current / Current Limit
- Goal PWM / PWM Limit
- safe joint range
- validation holdout configuration

필요한 값은 config에 null/REQUIRED 상태로 만들고 actual execution을 막는다.

## 31.2 Existing project 먼저 검사

코드를 수정하기 전에:

- current package entry points
- existing config loaders
- existing raw-data schema
- existing DYNAMIXEL SDK I/O
- existing safety handling
- current MuJoCo timestep
- current unit conversion
- current CLI aliases

를 확인한다.

동작 중인 안전 기능은 이유 없이 삭제하지 않는다.

## 31.3 Source of truth

Canonical source priority:

```text
1. This README
2. configs/mode5/*.yaml resolved values
3. per-run metadata/read-back
4. implementation code
5. legacy docs/results
```

README와 code가 충돌하면 code를 이 README에 맞게 수정한다.

## 31.4 No silent fallback

필수 config가 없거나 invalid이면:

```text
ERROR + explicit missing field list
```

를 내고 종료한다.

Legacy default나 임의 motor setting으로 fallback하지 않는다.

---

# 32. 최종 campaign 수량 정리

## 32.1 Main dynamic dataset

확정:

```text
Masses       = 3
Arm lengths  = 2
Trajectories = 3
Repeats      = 3

3 x 2 x 3 x 3 = 54 dynamic runs
```

## 32.2 Static dataset

Canonical design:

```text
6 mechanical configurations
x 2 approach directions
x 3 repeats
= 36 static sweep runs
```

각 sweep 안의 static angle point 수는 아직 미확정이다.

## 32.3 Delay / pilot

Pilot과 delay calibration은 별도이며
main 54와 static 36에 포함하지 않는다.

Run 수는 pilot 후 trajectory config가 확정되면 결정한다.

---

# 33. 전체 최종 workflow

```text
[1] Bench build / measurement
    - 250 / 500 / 750 g
    - L1 / L2
    - arm mass / COM / inertia
    - coordinate/sign check

                |
                v

[2] Controller setup
    - Mode 5
    - fixed real-use Position P/D
    - I = 0
    - FF = 0
    - Profile = OFF
    - fixed current/PWM limits
    - full register read-back

                |
                v

[3] Pilot
    - safety
    - sign
    - current/PWM
    - communication
    - fixture

                |
                v

[4] Static calibration
    - 6 mechanical configurations
    - several static angles
    - both approach directions
    - 3 repeats

    -> Ktau_eff prior
    -> aP prior
    -> friction/load residual diagnostic

                |
                v

[5] Delay calibration
    - small +/- Goal Position step
    - command timestamp -> Present Current onset

    -> delay prior

                |
                v

[6] Main dynamic acquisition

    3 masses
    x 2 lengths
    x 3 trajectories
    x 3 repeats
    = 54 runs

    trajectories:
      A. Accelerated oscillations
      B. Slow oscillation + small high-frequency oscillation
      C. Slowly raise/lower

                |
                v

[7] Initial dynamic identification
    -> aD candidate

                |
                v

[8] M1 first fit

    fixed priors:
      aP
      Ktau_eff
      delay

    optimize:
      aD
      armature
      Coulomb friction
      viscous friction

                |
                v

[9] Optional constrained joint refinement
    - only if required
    - priors may move only within measurement uncertainty

                |
                v

[10] Condition-level held-out validation

    45 fit runs
    9 validation runs

                |
                v

[11] Residual analysis

                |
          +-----+------+
          |            |
          v            v

      sufficient     systematic mismatch

          |            |
          |            -> consider M3/M4/backlash/electrical only with evidence
          v

[12] Freeze standalone MX-106 Mode 5 actuator model

                |
                v

[13] Jandi integration
    - 50 Hz policy
    - 20 ms target ZOH
    - 100 Hz motor interface
    - identified delay
    - current-domain actuator dynamics
    - MuJoCo rigid-body dynamics

                |
                v

[14] Whole-robot Sim2Real validation
```

---

# 34. References / methodological basis

이 canonical plan은 다음 자료의 구조를 참고하되 Jandi의 실제 Mode 5 조건에 맞게 수정한 것이다.

1. ROBOTIS, MX-106T/R(2.0) e-Manual
   - Current-based Position Control Mode
   - Position PID / Current Controller structure
   - Current/PWM limits
   - Profile behavior
   - Present Current/PWM/Position/Velocity telemetry

   https://emanual.robotis.com/docs/en/dxl/mx/mx-106-2/

2. Marc Duclusaud, Grégoire Passault, Vincent Padois, Olivier Ly,
   "Extended Friction Models for the Physics Simulation of Servo Actuators,"
   IEEE ICRA 2025.
   - Variable-load pendulum identification
   - Apparent inertia
   - M1-M6 friction models
   - Accelerated oscillation
   - Slow + high-frequency oscillation
   - Slowly raise/lower
   - CMA-ES trajectory fitting
   - MX-106 load-dependent friction evidence

3. BAM (Better Actuator Models) documentation
   - Servo actuator identification pipeline
   - Current-controlled actuator abstraction
   - Pendulum test bench
   - M1 friction
   - MuJoCo integration

   https://bam.readthedocs.io/

---

# 35. 현재 아직 확정되지 않은 값

아래 항목이 확정되기 전에는 본 hardware campaign을 실행하지 않는다.

```text
[x] L1 = 0.10 m
[x] L2 = 0.15 m
[ ] arm mass
[ ] arm COM
[ ] arm inertia
[ ] actual measured 250 g mass
[ ] actual measured 500 g mass
[ ] actual measured 750 g mass

[ ] Position P Gain
[ ] Position D Gain
[ ] Goal Current
[ ] Current Limit
[ ] Goal PWM
[ ] PWM Limit

[x] static angle set = -60/-30/0/+30/+60 deg
[ ] static settling/dwell thresholds

[ ] accelerated oscillation parameters
[ ] slow+high-frequency parameters
[ ] slowly raise/lower parameters

[ ] delay probe step amplitudes/repeats

[ ] software safe position range
[ ] dynamic trajectory holdout mechanical configuration
```

이 값들은 실험 fixture와 pilot 결과를 바탕으로 사람이 확정한다.
Codex가 자동으로 추정하거나 legacy 값으로 대체하지 않는다.

---

# 36. Canonical acquisition 세부 계약

## 36.1 Tick, watchdog, 연속 state

- MX-106R 2.0 Realtime Tick은 `1 ms/count`, `0..32767`, modulus `32768`로 unwrap한다.
- 작은 역방향 jitter는 wrap으로 승격하지 않고 비단조 표본으로 남긴다.
- Bus Watchdog는 초기화 중 0으로 clear한 뒤 Torque ON 후 설정값(1..127)을 쓰고 read-back한다.
  timeout 뒤 goal register가 read-only가 되는 공식 동작 때문에 다음 초기화는 watchdog 0 clear부터 시작한다.
- Hardware Error는 별도 cadence로 읽으며 매-cycle state block read를 대체하지 않는다.

## 36.2 Static과 trajectory 연속성

Static 각 점은 `previous target -> smooth inter-point transfer -> approach start -> controlled
approach -> fixed settling hold -> averaging` 순서다. Transfer/settling은 regression에 넣지 않고
averaging plateau만 velocity/position/current 안정성과 saturation 기준으로 사후 선별한다.
accepted/rejected plateau와 사유별 count를 `static_calibration.json`에 기록한다.

`slowly_raise_lower`는 center에서 lower까지, 마지막 endpoint에서 center까지 half-cosine으로
연속 전이한다. 세 main trajectory 모두 생성 직후 전체 합성 파형의 software position 범위,
adjacent-sample 최대 command speed, 주파수 Nyquist 조건을 검사한다.

## 36.3 Static uncertainty와 진단

기존 robust regression/Jacobian covariance와 함께 **whole static sweep run**을 단위로 bootstrap한다.
repeat count/seed/condition-number warning threshold는 `configs/mode5/fit.yaml`의 명시값이며,
Ktau/aP bootstrap standard deviation과 95% CI, regression rank/condition number를 저장한다.
정적 residual은 `|gravity torque|`, mass, arm length, approach, angle, Present Current에 대해
그린다. 추세가 보여도 M1 확장은 자동 수행하지 않으며 manual review로 남긴다.

---

# 37. Desktop GUI, immutable retries, and fit diagnostics

## 37.1 GUI 실행

GUI는 canonical YAML과 backend를 감싸는 인터페이스이며, 별도 실험 상수를 갖지 않는다.
GUI를 띄우는 것만으로 serial port를 열거나 모터를 움직이지 않는다.

```bash
# 실제 장치용. Connect 전에는 port를 열지 않는다.
uv run jandi-r2s-mode5-gui

# 하드웨어 없는 명시적 mock. 실제 backend로 자동 fallback하지 않는다.
QT_QPA_PLATFORM=xcb uv run jandi-r2s-mode5-gui --mock
```

`PREVIEW`는 port 없이 canonical trajectory를 생성해 Goal Position, duration, sample count,
범위와 최대 discrete speed를 표시한다. Static 36개와 Dynamic 54개 logical run의 상태는
`NOT_RUN`, `VALID`, `INVALID`, `MULTIPLE_VALID_ATTEMPTS`로 구분한다. Canonical RUN 전에
선택한 L1/L2, nominal/measured mass와 실제 fixture가 일치한다는
`PHYSICAL SETUP CONFIRMATION`이 필요하며 그 시각과 setup을 metadata에 기록한다.

GUI의 빨간 `TORQUE OFF`는 항상 보인다. 이는 worker에 즉시 중단 요청을 보내고 active run을
invalid/operator-abort로 남기지만, 물리 비상 전원 차단 장치를 대체하지 않는다. CLI는
low-level/debug/reproducibility 인터페이스로 계속 유지한다.

Manual Test는 `MANUAL TEST — NOT PART OF CANONICAL DATASET`으로 구분한다. Mock manual telemetry는
저장하지 않으며, 실제 manual telemetry도 canonical 경로가 아닌 `data/temp/manual/` 아래에만
immutable attempt로 저장한다. Manual의 center/amplitude/frequency/duration은 명시적 per-run
override metadata/trajectory로만 사용되며 canonical static/dynamic YAML을 바꾸지 않는다.

## 37.2 Immutable attempt 구조

Logical run은 다음과 같이 immutable attempt를 가진다.

```text
dynamic/L1_m250/accelerated_oscillation/repeat_1/
├── attempt_001/   # invalid 또는 valid; 절대 overwrite하지 않음
└── attempt_002/   # retry
```

hard crash로 metadata가 없는 incomplete attempt가 남아도 다음 번호로 retry한다. 각 metadata에는
`logical_run_id`, `attempt_index`, `retry_of`, `valid_flag`, `invalid_reason`을 저장한다. Valid가
정확히 하나면 자동 선택한다. 둘 이상이면 최신값을 몰래 고르지 않고 logical directory의
`selected_attempt.txt`에 `attempt_NNN`을 명시해야 분석한다. 기존 attempt 도입 전 raw directory는
읽기 호환한다.

```bash
uv run jandi-r2s-mode5-select-attempt \
  --logical-run dynamic/L1_m250/accelerated_oscillation/repeat_1 \
  --attempt attempt_002
```

## 37.3 Stage-D saturation 정책

Canonical M1은 current clipping은 모델링하지만 PWM/voltage/back-EMF 한계 전체는 모델링하지 않는다.
따라서 primary Stage-D loss에서는 `pwm_saturated` 표본을 제외한다. Current-saturated 표본은 current
clip model의 유효성 범위에 포함하되 flag를 보존한다. `fit_validity_region.json`에는 total, normal,
current-saturated, PWM-saturated, primary-excluded 표본 수를 기록한다. Validation/report의 full behavior
metric은 포화 표본을 버리지 않는다.

## 37.4 Repeatability 명칭 계약

- `model_error_repeat_variation`: 같은 조건의 repeat별 simulation RMSE 변동
- `real_to_real_repeatability`: 실제 repeat 1/2/3을 run-local host time으로 보간 정렬한 뒤 계산한
  r1-r2, r1-r3, r2-r3의 position/velocity/current RMSE

둘은 각각 residual summary와 `real_to_real_repeatability.json`에 분리해 저장한다.
