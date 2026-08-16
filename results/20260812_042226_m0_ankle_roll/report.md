# Jandi M0 ankle-roll identification

- 공통 지연: 4.0 ms
- 공통 Kp_eff: 13.418904
- 공통 Kd_eff: 0.672097
- fit NRMSE: 0.029128
- validation NRMSE: 0.027747
- baseline validation NRMSE: 0.053925

## 판정 주의

- 반복 1·2만 최적화에 사용했고 반복 3은 완전히 분리해 검증했습니다.
- 이 값은 MX-106 레지스터 게인이 아니라 현재 고정베이스 MuJoCo 모델의 유효 PD입니다.
- M0 잔차에 히스테리시스나 방향 의존성이 남으면 다음 M1에서 마찰·백래시를 추가합니다.
