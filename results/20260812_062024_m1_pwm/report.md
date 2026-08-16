# Jandi measured-PWM M1 identification

## 선택된 출력축 등가 파라미터

- drive_gain_nm_per_duty: 4.26061542
- armature_kg_m2: 0.0227509807
- coulomb_friction_nm: 0.0708954639
- viscous_friction_nm_s_per_rad: 0.931167003
- fit loss: 0.00491575661
- validation loss mean: 0.00609967825

## 모델 계약

- Position P/D와 command delay는 피팅하지 않고 실측 Present PWM을 입력으로 사용했습니다.
- drive gain은 motor constant·기어비·효율을 합친 출력축 등가값입니다.
- Coulomb/viscous/armature는 12개 MX-106에 공통 적용했습니다.
- repeat 1·2만 fit, repeat 3은 validation 전용입니다.
- backlash, Stribeck, load-dependent friction은 아직 포함하지 않았습니다.
