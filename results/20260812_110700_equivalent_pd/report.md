# Jandi equivalent PD actuator identification

- shared delay: 10.000 ms
- equivalent backlash total width: 0.008843 rad (0.507 deg)
- MuJoCo Coulomb frictionloss: 0.003191 Nm
- Kp_eff / Position-P register: 0.017118842
- fixed Kd_eff: 0.600000
- P350: Kp_eff=5.991595, Kd_eff=0.600000
- P850: Kp_eff=14.551015, Kd_eff=0.600000

## 12-joint held-out validation

| condition | joint | MAE deg | RMSE deg | combined loss |
|---|---:|---:|---:|---:|
| P350 | RL1_joint | 0.4677 | 0.5459 | 0.01766444 |
| P350 | RL2_joint | 0.3236 | 0.3840 | 0.02285136 |
| P350 | RL3_joint | 1.8118 | 1.8790 | 0.05513164 |
| P350 | RL4_joint | 1.7759 | 1.8390 | 0.05766446 |
| P350 | RL5_joint | 0.7411 | 0.8839 | 0.02665419 |
| P350 | RL6_joint | 0.1049 | 0.1423 | 0.00368559 |
| P350 | LL1_joint | 0.2574 | 0.2938 | 0.00869191 |
| P350 | LL2_joint | 0.4369 | 0.5174 | 0.02710151 |
| P350 | LL3_joint | 1.8803 | 1.9359 | 0.05739923 |
| P350 | LL4_joint | 1.6561 | 1.7333 | 0.05268410 |
| P350 | LL5_joint | 0.7396 | 0.8522 | 0.02627545 |
| P350 | LL6_joint | 0.1188 | 0.1502 | 0.00414658 |
| P850 | RL1_joint | 0.1048 | 0.1459 | 0.00469237 |
| P850 | RL2_joint | 0.2857 | 0.3651 | 0.02033374 |
| P850 | RL3_joint | 0.9948 | 1.0834 | 0.03108127 |
| P850 | RL4_joint | 1.0267 | 1.0528 | 0.03119390 |
| P850 | RL5_joint | 0.4372 | 0.4661 | 0.01281886 |
| P850 | RL6_joint | 0.1723 | 0.2366 | 0.00601093 |
| P850 | LL1_joint | 0.1189 | 0.2033 | 0.00656312 |
| P850 | LL2_joint | 0.3739 | 0.4282 | 0.03028350 |
| P850 | LL3_joint | 0.9921 | 1.0602 | 0.02897131 |
| P850 | LL4_joint | 0.9381 | 0.9678 | 0.02800724 |
| P850 | LL5_joint | 0.4395 | 0.4634 | 0.01279915 |
| P850 | LL6_joint | 0.1548 | 0.2046 | 0.00535636 |

## Model contract

- Kp는 Position-P register에 정비례하도록 묶었습니다.
- Kd=0.60은 고정했으며 최적화하지 않았습니다.
- RL6·LL6 repeat 1·2만 fit에 사용했습니다.
- 12개 관절 repeat 3은 재피팅 없는 held-out validation입니다.
- 각 조건의 PD는 12개 관절 전체에 적용했습니다.
- backlash 값은 상태를 가진 기어 치합 모델이 아니라 위치 오차의 등가 deadband 전체 폭입니다.
- 정상상태 plateau 절대오차를 loss에 포함했습니다.
- viscous friction은 Kd와 식별 불가능하므로 별도 피팅하지 않았습니다.
