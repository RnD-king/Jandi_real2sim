# Jandi Real2Sim

> 새 Current-based Position Control(Mode 5) 실험은
> `docs/MODE5_EXPERIMENT_PLAN.txt`와 `configs/mode5/campaign.yaml`을 기준으로
> 한다. 아래 기존 Position Control/P350/P850 명령과 데이터는 재현용 legacy로
> 보존하며 Mode 5 결과와 섞지 않는다.

## Mode 5 최소 실험

실험은 `no_load`, `loaded` 두 조건과 각 조건의 `step`, `triangle`, `sine`
세 trajectory로만 나눈다. 각 조합은 3회 반복하여 총 18 run이다.

```bash
cd /home/noh/Jandi_real2sim
uv sync
uv run jandi-r2s-mode5-check

# 포트를 열지 않는 계획 생성
uv run jandi-r2s-mode5-collect --condition no_load
uv run jandi-r2s-mode5-collect --condition loaded

# pilot 완료 후 실측
uv run jandi-r2s-mode5-collect --condition no_load --pilot \
  --execute --confirm PILOT_MX106_MODE5
# pilot 결과 확인 뒤 YAML의 pilot_approved: true
uv run jandi-r2s-mode5-collect --condition no_load \
  --execute --confirm MOVE_MX106_MODE5
uv run jandi-r2s-mode5-collect --condition loaded \
  --execute --confirm MOVE_MX106_MODE5

# 18 run 완료 후
uv run jandi-r2s-mode5-fit
uv run jandi-r2s-mode5-compare \
  --params results/정확한_결과폴더/params_mode5.yaml
```

초기 분리형 설정에는 실제 Mode 5 register와 시험대 질량·COM이
`null`로 남아 있어 실기체 실행이 잠겨 있다. 임의값으로 해제하지 말고 실제
사용값과 실측 치수를 입력한다. 자세한 장치 구성, 안전 절차, 식별식,
논문 Methods 기록 항목은 계획서를 따른다.

설정 책임은 다음처럼 나뉜다.

- `configs/mode5/hardware.yaml`: 모터 ID·영점·방향·통신 주기
- `configs/mode5/controller.yaml`: Mode 5 고정 레지스터
- `configs/mode5/bench/no_load.yaml`: 모터+혼만인 진짜 무부하 조건
- `configs/mode5/bench/loaded.yaml`: 막대기와 무게추의 질량·COM·관성
- `configs/mode5/trajectories/*.yaml`: 파형별 진폭·주파수·주기
- `configs/mode5/campaign.yaml`: 위 파일 조합, 반복, 안전 한계, 출력 경로

두 조건은 물리 구성이 다르므로 서로 상속하지 않는다. 각 run의
`metadata.json`에는 모든 구성 파일의 절대경로,
SHA-256, 해석된 값, 계산된 등가 질량·COM·출력축 관성이 함께 저장된다.

Jandi의 MX-106 12개를 조립 상태(몸통 강체 고정, 발 무접촉)에서 100 Hz로
측정하고, 같은 `q_cmd`를 MuJoCo에 재생해 액추에이터 등가 모델을 식별하기
위한 독립 프로젝트다. 기존 `/home/noh/Jandi_mjlab`의 locomotion·vision
코드는 수정하거나 의존하지 않는다.

## 현재 제공 범위

- MuJoCo 관절 순서 `RL1..RL6, LL1..LL6` 고정
- 현재 locomotion의 보행 기본자세와 XML 관절 한계 저장
- command 100 Hz, state 99 Hz, Hardware Error 1 Hz 설정 검증
- 현재각에서 보행 기본자세로 cosine 저속 전환
- 보행 기본자세에서 단일 관절 `center,+A,center,-A,center` 시험
- 모든 하드웨어 CLI는 기본 dry-run
- ID, 영점 tick, 방향이 하나라도 비어 있으면 Torque On 차단
- 실제 실행에도 `--execute --confirm MOVE_JANDI`를 동시에 요구

Torque Off 상태에서 중력으로 관절이 XML 범위 밖까지 처진 경우, 현재 위치를
유지하는 첫 명령과 보행 자세까지의 5초 복구 전환에만 범위 밖 명령을 허용한다.
보행 자세 도달 후 모든 시험 trajectory에는 XML 관절 한계를 다시 강제한다.

- 초기자세까지의 안전 전환은 측정 CSV에 넣지 않고 실제 step 시험만 기록한다.
- 단일관절 실제 시험은 100 Hz로 명령한다. 각 1초의 100개 수신 슬롯 중
  99개는 state, 마지막 1개는 Hardware Error만 읽는다. Error 행의 state와
  state 행의 Hardware Error는 빈 칸으로 명시해 `data/raw/`에 저장한다.
- 각 실행은 `data/raw/<timestamp_joint_step>/` 아래 `telemetry.csv`와
  `metadata.json` 한 쌍으로 저장한다. dry-run도 `data/plans/<run>/` 아래
  `plan.csv`와 `metadata.json`으로 묶는다.
- RL6·LL6 compact-step 반복 1·2를 fit, 반복 3을 validation으로 고정한 M0 식별
- 원본 MJCF를 수정하지 않고 자유관절·XML actuator만 메모리에서 제거한 고정베이스 replay
- 500 Hz MuJoCo 외부 PD(토크 ±5.5 N·m)와 100 Hz 실측 명령의 시간축 재현
- 공통 지연, 공통 `Kp_eff`, `Kd_eff` 탐색과 baseline/fit/validation 그래프·지표 저장

## 환경

```bash
jandi_real2sim
jandi-r2s-check
```

alias가 아직 현재 shell에 반영되지 않았다면:

```bash
source /home/noh/alias_settings.sh
```

## Dry-run

```bash
uv run jandi-r2s-pose
uv run jandi-r2s-joint-test RL3_joint
uv run python scripts/set_walking_pose.py
uv run python scripts/test_single_joint.py RL3_joint
```

dry-run은 모터 포트에 접근하지 않는다. 생성된 command 계획은
`data/plans/`에 저장된다.

## 실기체 사용 전 필수 작업

`configs/jandi_mx106.yaml`의 각 관절에 다음 실측값을 입력한다.

- `id`: MX-106 ID
- `zero_tick`: 기구적 관절각 0 rad의 Present Position
- `direction`: DXL tick 증가가 시뮬레이터 양의 관절축이면 `+1`, 반대면 `-1`

설정 후:

```bash
uv run jandi-r2s-check
```

모든 관절이 `READY`가 되기 전에는 실제 실행이 차단된다.

실제 실행 형식은 다음과 같지만, 몸통 지그·관절 간섭·비상 Torque Off를
준비하고 설정값을 검증하기 전에는 실행하지 않는다.

```bash
uv run jandi-r2s-pose --execute --confirm MOVE_JANDI
uv run jandi-r2s-joint-test RL3_joint \
  --amplitude-rad 0.03 --execute --confirm MOVE_JANDI
```

기본 동작은 종료 시 Torque Off다. pose 명령 후에도 유지해야 하는 특별한
경우에만 `--keep-torque-on`을 사용한다.

## 클럭 계약

향후 MuJoCo 식별 환경은 다음 시간축으로 만든다.

```text
physics                 500 Hz (dt=0.002 s)
MX command              100 Hz
MX state/error        99/1 Hz (서로 배타적인 수신 슬롯)
locomotion policy        50 Hz (별도 계층)
```

## 식별용 측정 명령

모든 명령은 기본적으로 dry-run이다. 실제 모터를 움직이려면 기존과 동일하게
`--execute --confirm MOVE_JANDI`를 둘 다 붙여야 한다. 보행 자세까지의 5초
전환은 수행하지만 식별 CSV에는 기록하지 않는다.

### 정적 hold

자세별 처짐과 noise floor를 위해 각 repeat를 독립 run으로 저장한다.

```bash
jandi-r2s-hold --pose-id A --duration-sec 10 --repeat-index 1
jandi-r2s-hold --pose-id A --duration-sec 10 --repeat-index 2
jandi-r2s-hold --pose-id A --duration-sec 10 --repeat-index 3
```

### 2진폭 compact step

한 run은 `center,+small,center,-small,center,+medium,center,-medium,center`의
9단계다. repeat 1·2는 fit, repeat 3은 validation으로 metadata에 기록한다.

```bash
jandi-r2s-step-id RL6_joint --pose-id A \
  --small-amplitude-rad 0.05 --medium-amplitude-rad 0.10 \
  --hold-sec 1.5 --repeat-index 1
```

### 저속 triangle

마찰·stiction·백래시의 방향전환 잔차를 확인한다. 진폭, 주파수, 반복 주기와
정책에서 확인한 최대 명령속도를 반드시 명시한다.

```bash
jandi-r2s-triangle RL6_joint --pose-id A \
  --amplitude-rad AMPLITUDE --frequency-hz FREQUENCY --cycles CYCLES \
  --max-command-speed-rad-s POLICY_V95 --repeat-index 1
```

### multisine

주파수 4~6개는 정책 command PSD에서 고르고, 최저주파수 5주기 이상 길이를
확보한다. 같은 seed는 동일 trajectory를 재생한다.

```bash
jandi-r2s-multisine RL6_joint --pose-id A \
  --amplitude-rad AMPLITUDE \
  --frequencies-hz F1 F2 F3 F4 \
  --duration-sec DURATION --seed 1 --split-role fit \
  --max-command-speed-rad-s POLICY_V95
```

### 자세 B/C 입력

기본 자세 A는 `jandi_mx106.yaml`의 walking pose다. B/C는 임의로 만들지 않고
정책에서 뽑은 12관절 자세를 JSON으로 저장한 뒤 전달한다.

```json
{
  "RL1_joint": 0.0,
  "RL2_joint": 0.0,
  "RL3_joint": 0.98,
  "RL4_joint": -0.40,
  "RL5_joint": 0.84,
  "RL6_joint": 0.0,
  "LL1_joint": 0.0,
  "LL2_joint": 0.0,
  "LL3_joint": -0.98,
  "LL4_joint": 0.40,
  "LL5_joint": -0.84,
  "LL6_joint": 0.0
}
```

```bash
jandi-r2s-step-id RL6_joint --pose-id B --pose-json poses/pose_b.json ...
```

### 실행 직후 검증

```bash
jandi-r2s-validate data/raw/RUN_DIRECTORY --strict
```

표본 수, 99:1 수신 슬롯, cycle 연속성, 실제 Hz, deadline overrun,
Hardware Error, 전압·온도·전류·PWM 범위를 출력한다.

## M0 식별: RL6 + LL6

다음 명령은 Dynamixel 포트를 열거나 실기체를 움직이지 않는다. 기본값은
`configs/campaign_20260811_all_joints_A.yaml`에 정확한 이름으로 고정한
36개 run을 먼저 검증한 뒤, 그중 RL6·LL6의 반복 1·2만 최적화하고 반복 3은
결과가 정해진 뒤 검증에만 사용한다. timestamp나 mtime으로 최신 run을
자동 선택하지 않는다.

```bash
cd /home/noh/Jandi_real2sim
uv sync
jandi-r2s-campaign-check
jandi-r2s-fit-m0
```

결과는 `results/<timestamp>_m0_ankle_roll/`에 저장된다.

- `params_m0.yaml`: 공통 delay, `Kp_eff`, `Kd_eff`
- `metrics.json`: run별 baseline/식별 후 MAE·RMSE·NRMSE와 torque saturation
- `delay_candidates.json`: 2 ms 간격 delay 전 후보와 5% 응답 지연 진단
- `manifest.json`: 사용한 6개 로그·설정·MJCF 경로와 SHA-256
- `report.md`: fit/validation 핵심 요약과 경계값 경고
- `*.png`: 명령, 실측, baseline, 식별 모델의 관절각 비교

다른 이름 manifest를 사용하려면 `--campaign PATH`를 준다. manifest 대신
실행 폴더를 직접 고정하려면 `--run-dir PATH`를 정확히 6번 준다.
식별 설정과 탐색 범위는 `configs/m0_ankle_roll.yaml`에 있으며, 현재 M0는
마찰·백래시를 일부러 넣지 않는다. 검증 잔차의 방향 의존성과 히스테리시스를
확인한 뒤에만 M1을 추가한다.

## P350 다진폭 전 관절 자동 수집

반복해서 조건을 바꿀 때는 Python 코드를 수정하지 않고
`configs/collection_campaign.yaml`만 편집한 뒤 다음 명령을 사용한다.

```bash
jandi-r2s-collect-campaign --spec configs/collection_campaign.yaml
```

YAML에서 다음을 직접 지정할 수 있다.

- 12개 모터 공통 Position P/I/D와 관절별 override
- experiment별 활성화 여부
- compact step의 관절별 진폭·hold 시간·repeat
- triangle 진폭·주파수·cycles·명령속도 한계
- multisine 진폭·주파수·duration·fade·seed별 fit/validation 역할
- static hold 시간과 repeat
- campaign 이름과 `data/raw/` 아래 output group
- 전류·PWM·온도·입력전압·관절 추종오차의 실시간 중단 한계
- run 사이 대기와 주기적인 Torque Off 냉각 시간

실제 실행은 YAML의 모든 trajectory·관절 한계·속도·PID를 먼저 정적 검증한
다음 시작한다.

```bash
jandi-r2s-collect-campaign \
  --spec configs/collection_campaign.yaml \
  --execute \
  --confirm MOVE_JANDI_CAMPAIGN
```

중단 후에는 YAML을 바꾸지 않고 같은 명령에 `--resume`을 추가한다. 최초
실행 시 YAML snapshot, 12관절에 해석된 PID JSON, 실시간 안전 한계 JSON을
campaign 폴더에 복사한다. resume 중 원본 YAML이나 해석된 설정이 달라지면
실행을 차단한다.

MX-106 Position PID는 RAM 값이므로 각 run에서 Torque On 전에 전 모터에
`P=350, I=0, D=0`을 다시 쓰고 readback한다. 현재 compact step은 일반
관절에 `[0.05, 0.10, 0.15] rad`, RL2/LL2에 `[0.03, 0.05, 0.07] rad`,
RL4/LL4에 `[0.05, 0.10, 0.14] rad`를 사용한다. 반복 순서는 repeat 1의
12관절, repeat 2의 12관절, repeat 3의 12관절이다. triangle·multisine·
static hold 조건도 같은 YAML에 명시한다.

현재 초기 안전값은 12개 모터 각각에 대해 온도 55 °C 이상, 입력전압 9.6 V
이하, 절대전류 2.5 A 이상, 절대 PWM 85% 이상, 절대 관절 추종오차 0.25 rad
이상을 감시한다. 상태 한계는 통신 spike 하나로 중단하지 않도록 5개의 연속
state 표본(약 50 ms)에서 지속될 때 예외를 발생시키며, Hardware Error는
1 Hz 수신 즉시 중단한다. 예외가 발생하면 해당 run은 `valid_flag=false`로
남고 `finally` 경로에서 전 모터 Torque Off를 요청한다. 이 소프트웨어 감시는
기구적 비상정지나 작업자 감시를 대신하지 않는다.

각 job 종료 뒤에는 Torque Off로 3초 대기하고, 새로 실행한 12개 job마다
60초 추가 냉각한다. 값은 `configs/collection_campaign.yaml`의 `safety`에서
수정할 수 있다.

먼저 하드웨어를 열지 않는 계획 검증:

```bash
jandi-r2s-collect-p350 --campaign-id P350_multiamp_A_dryrun
```

실제 전체 수집은 몸통 지지·Wizard 연결 해제·비상 Torque Off 준비 후 실행한다.

```bash
jandi-r2s-collect-p350 \
  --campaign-id P350_multiamp_A_20260812 \
  --execute \
  --confirm MOVE_JANDI_P350_CAMPAIGN
```

실측은 `data/raw/P350/<campaign-id>/runs/`에 run별 `telemetry.csv`와
`metadata.json`을 저장한다. 중단됐다면 같은 ID로 이어간다.

```bash
jandi-r2s-collect-p350 \
  --campaign-id P350_multiamp_A_20260812 \
  --resume \
  --execute \
  --confirm MOVE_JANDI_P350_CAMPAIGN
```

각 run은 보행 자세 전환 구간을 기록하지 않고, 100 Hz 명령·99 Hz 상태·1 Hz
Hardware Error만 기존과 동일하게 기록한다. 모든 run이 끝나면 정확한 폴더
이름을 고정한 `campaign_manifest.yaml`을 함께 만든다.

## P350/P850 공통 지연·조건별 PD 식별

기존 `jandi-r2s-fit-m0`는 한 gain 조건의 RL6/LL6만 독립적으로 맞추므로
지연과 PD가 서로 보상할 수 있다. 두 조건을 함께 사용할 때는 다음 명령을 쓴다.

```bash
jandi-r2s-fit-m0-dual \
  --p350-campaign data/raw/P350/P350_multiamp_A_20260812 \
  --p850-campaign data/raw/P850/P850_multiamp_A_20260812
```

설정은 `configs/m0_dual_gain.yaml`에 있다. 공통 지연 후보 `0~10 ms`를 2 ms
간격으로 모두 정밀화하고, 각 후보에서 P350/P850의 유효 Kp/Kd를 독립적으로
최적화한다. 고정 초기 PD로 지연 shortlist를 고르지 않는다.

repeat 1·2만 fit에 사용하고 repeat 3은 validation 전용이다. 각 step 응답은
실측과 시뮬레이션 각각의 plateau를 기준으로 중심화한 뒤 전환 후 0.4초 구간의
position/velocity를 비교한다. 따라서 아직 모델에 없는 정지마찰·백래시의
정상상태 잔류오차가 command delay를 길게 만드는 현상을 줄인다. 결과는
`results/<timestamp>_m0_dual_gain/`에 저장된다.

## 실측 PWM 입력 기반 M1 식별

`Present PWM`과 `Position Trajectory` 분석으로 MX-106(2.0)의 내부
Position P→PWM 변환이 `P_register / 128`과 일치함을 확인했다. 따라서 다음
단계에서는 command에서 등가 PD와 지연을 다시 추정하지 않고, 실측 PWM을
MuJoCo의 입력으로 직접 재생한다.

```bash
jandi-r2s-fit-m1-pwm \
  --p350-campaign data/raw/P350/P350_multiamp_A_20260812 \
  --p850-campaign data/raw/P850/P850_multiamp_A_20260812
```

설정은 `configs/m1_pwm.yaml`에 있다. P350/P850, RL6/LL6에 공통인 다음 네
출력축 등가 파라미터를 여러 시작점에서 최적화한다.

- full-duty·12 V 기준 등가 drive torque gain
- reflected armature
- Coulomb friction
- viscous friction(back-EMF 등가 감쇠 포함)

12개 관절의 `Present PWM`과 입력전압을 모두 재생하므로 식별 대상 외 관절도
기존 baseline PD로 대체하지 않는다. Dynamixel raw PWM에는
`configs/jandi_mx106.yaml`의 관절 방향을 적용하고, 각 run metadata의 실제
PWM limit으로 duty를 정규화한다. repeat 1·2만 fit, repeat 3은 validation에만
사용한다.

결과는 `results/<timestamp>_m1_pwm/`에 저장된다. 이 M1의 drive gain은
모터축 `Kt`가 아니라 모터상수·기어비·전달효율을 합친 출력축 등가값이다.
backlash, Stribeck, load-dependent friction은 validation 잔차에서 필요성이
확인된 뒤 다음 단계에서 추가한다.

### 단계식 M1 재식별

step만으로 네 값을 동시에 맞춘 결과가 시작점에 따라 달라질 때는 기존 실측을
다시 수집하지 않고 다음 명령을 사용한다.

```bash
jandi-r2s-fit-m1-staged \
  --p350-campaign data/raw/P350/P350_multiamp_A_20260812 \
  --p850-campaign data/raw/P850/P850_multiamp_A_20260812
```

정확한 `campaign_status.json` 이름으로 RL6/LL6의 step·slow triangle·policy
band multisine을 읽는다. 단계는 다음과 같다.

1. 0.1 Hz triangle에서 drive gain과 Coulomb friction을 식별한다.
2. multisine에서 Coulomb을 고정하고 drive gain·armature·viscous를 식별한다.
3. step·triangle·multisine 전체에서 좁아진 bounds로 네 값을 공동 미세조정한다.

각 단계는 두 시작점을 사용하며 repeat/seed 1·2만 fit, 3은 최종 validation에
사용한다. 설정과 단계별 평가 한도는 `configs/m1_pwm_staged.yaml`에 있다.
결과는 `results/<timestamp>_m1_pwm_staged/`에 저장된다.

## 3·4번 관절군 전용 등가 PD 재식별

RL6/LL6에서 채택한 공통 nominal은 보존하고, 중력 처짐이 집중된 3번과
4번 관절군만 다시 맞출 때는 다음 명령을 사용한다.

```bash
jandi-r2s-fit-joint34-pd \
  --p350-campaign data/raw/P350/P350_multiamp_A_20260812 \
  --p850-campaign data/raw/P850/P850_multiamp_A_20260812
```

설정은 `configs/joint34_pd.yaml`에 있다. 식별 계약은 다음과 같다.

- RL3/LL3은 하나의 Kp/Kd를 공유하고 RL4/LL4도 하나를 공유한다.
- 1·2·5·6번은 P350 `Kp=6.0`, `Kd=0.6`으로 고정한다.
- delay 10 ms, backlash 0.009 rad, joint friction 0.003 Nm 및 encoder
  tick도 고정한다.
- P350 repeat/seed 1·2만 fit에 사용하고 repeat/seed 3은 validation이다.
- Kp는 slow triangle, static hold, step plateau로 먼저 맞춘다.
- Kd는 step transient와 multisine으로 먼저 맞춘다.
- 마지막에는 3번 Kp/Kd와 4번 Kp/Kd 네 값만 공동 미세조정한다.
- P850은 최적화에 넣지 않고 Kp의 register 비례 외삽을 검증한다.

replay는 학습 패키지와 같은 encoder 양자화 및 상태형 play operator를
사용한다. 결과는 `results/<timestamp>_joint34_pd/`에 저장되며
`params_joint34_pd.yaml`의 `recommended_p350`을 확인한 뒤 Jandi 학습
설정에 반영한다. 식별 명령 자체는 학습이나 GUI를 실행하지 않는다.
