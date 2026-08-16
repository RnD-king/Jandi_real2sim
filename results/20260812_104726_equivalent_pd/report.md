# Jandi equivalent PD actuator identification

- shared delay: 10.000 ms
- equivalent backlash total width: 0.006324 rad (0.362 deg)
- MuJoCo Coulomb frictionloss: 0.001639 Nm
- Kp_eff / Position-P register: 0.003041940
- shared Kd_eff: 0.111103
- P350: Kp_eff=1.064679, Kd_eff=0.111103
- P850: Kp_eff=2.585649, Kd_eff=0.111103

## Model contract

- Kp는 Position-P register에 정비례하도록 묶었습니다.
- Kd는 P350/P850에 공통 적용했습니다.
- 각 조건의 PD는 12개 관절 전체에 적용했습니다.
- backlash 값은 상태를 가진 기어 치합 모델이 아니라 위치 오차의 등가 deadband 전체 폭입니다.
- 정상상태 plateau 절대오차를 loss에 포함했습니다.
- viscous friction은 Kd와 식별 불가능하므로 별도 피팅하지 않았습니다.
- repeat 1·2는 fit, repeat 3은 validation 전용입니다.
