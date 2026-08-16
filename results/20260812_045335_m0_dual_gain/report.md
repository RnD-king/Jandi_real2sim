# Jandi shared-delay dual-gain M0

- 공통 command delay: 10.0 ms
- P350: Kp_eff=3.584982, Kd_eff=0.328671, optimizer_success=False
- P850: Kp_eff=15.553853, Kd_eff=0.686869, optimizer_success=False

## 해석 제한

- repeat 1·2만 fit에 사용했고 repeat 3은 validation 전용입니다.
- step edge 뒤 transient는 자기 plateau 기준으로 비교해 static hysteresis가 지연을 보상하지 않게 했습니다.
- friction/backlash 자체는 아직 모델링하지 않았으므로 full-trajectory 정상상태 오차는 남을 수 있습니다.
