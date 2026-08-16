# Jandi joint 3/4 equivalent PD identification

## Result

- final fit loss: 0.00441770
- joint3 (RL3_joint, LL3_joint): P350 Kp=13.421668, Kd=1.432032
- joint4 (RL4_joint, LL4_joint): P350 Kp=11.060912, Kd=1.241423

## Fixed parameters

- joints 1/2/5/6: Kp=6.000000, Kd=0.600000
- delay: 10.000 ms
- backlash total width: 0.009000 rad
- joint Coulomb friction: 0.003000 Nm

## Validation means

| condition | split | trajectory | q MAE deg | q RMSE deg | dq MAE rad/s | peak tau Nm | saturation |
|---|---|---|---:|---:|---:|---:|---:|
| P350 | fit | compact_step | 0.3767 | 0.4522 | 0.05781 | 1.0430 | 0.000000 |
| P350 | fit | multisine | 0.3543 | 0.4180 | 0.04784 | 0.3228 | 0.000000 |
| P350 | fit | static_hold | 0.1777 | 0.1780 | 0.00132 | 0.3008 | 0.000000 |
| P350 | fit | triangle | 0.3700 | 0.4682 | 0.01436 | 0.3456 | 0.000000 |
| P350 | validation | compact_step | 0.3809 | 0.4555 | 0.05808 | 1.0594 | 0.000000 |
| P350 | validation | multisine | 0.3503 | 0.4232 | 0.04784 | 0.3248 | 0.000000 |
| P350 | validation | static_hold | 0.1793 | 0.1808 | 0.00135 | 0.3020 | 0.000000 |
| P350 | validation | triangle | 0.3697 | 0.4685 | 0.01435 | 0.3452 | 0.000000 |
| P850 | validation | compact_step | 0.3158 | 0.4012 | 0.11025 | 2.1445 | 0.000000 |
| P850 | validation | multisine | 0.1931 | 0.2193 | 0.05056 | 0.3646 | 0.000000 |
| P850 | validation | static_hold | 0.0886 | 0.0898 | 0.00667 | 0.3321 | 0.000000 |
| P850 | validation | triangle | 0.1592 | 0.1720 | 0.01396 | 0.3923 | 0.000000 |

## Model contract

- P350 repeat/seed 1·2만 최적화에 사용했습니다.
- P350 repeat/seed 3과 P850 전체는 최적화에서 제외한 validation입니다.
- RL3/LL3은 한 PD를, RL4/LL4는 한 PD를 공유합니다.
- 1·2·5·6번 PD와 delay/backlash/friction/tick은 고정했습니다.
- Kp는 triangle·static hold·step plateau로 먼저 맞췄습니다.
- Kd는 step transient·multisine으로 먼저 맞췄습니다.
- 최종 단계에서 네 PD 값만 공동 미세조정했습니다.
- replay는 Jandi 학습 actuator와 같은 encoder tick 및 stateful play operator를 사용합니다.
