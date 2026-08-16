# Jandi equivalent PD actuator identification

- shared delay: 12.000 ms
- equivalent backlash total width: 0.007628 rad (0.437 deg)
- MuJoCo Coulomb frictionloss: 0.006056 Nm
- P350: Kp_eff=1.599344, Kd_eff=0.157442
- P850: Kp_eff=18.854038, Kd_eff=0.803098

## Model contract

- P350/P850의 PD는 각각 12개 관절 전체에 적용했습니다.
- backlash 값은 상태를 가진 기어 치합 모델이 아니라 위치 오차의 등가 deadband 전체 폭입니다.
- 정상상태 plateau 절대오차를 loss에 포함했습니다.
- viscous friction은 Kd와 식별 불가능하므로 별도 피팅하지 않았습니다.
- repeat 1·2는 fit, repeat 3은 validation 전용입니다.
