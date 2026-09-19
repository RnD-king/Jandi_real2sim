# Mode-3 M3 follow-up validation

이 폴더의 `run_all.sh`는 새 하드웨어 실험이나 재피팅 없이 기존 repeat-3 데이터로
후속 검증 전체를 실행한다.

실행 항목:

1. M3 등가 제어기 갱신 주기 1 ms / 10 ms sensitivity
2. 실측 및 시뮬레이션 position에 동일한 5/7/9-point Savitzky-Golay derivative 적용
3. 1 ms canonical M3 baseline 전체/궤적별/하중별 집계
4. 식별된 유효 백래시 폭의 0/0.5/1.0/1.5배 sensitivity
5. actuator-side 및 output-side encoder feedback 가설 비교
6. 백래시 후보와 baseline을 동일한 0.1 ms physics timestep으로 비교
7. passive joint limit 침범률이 반폭의 5%를 넘으면 후보 자동 탈락
8. 명령 방향 반전 주변 ±100 ms local error
9. 상세 CSV/그래프와 Codex용 압축 summary 생성

```bash
cd /home/noh/Jandi_real2sim
bash scripts/mode3_bam_followup/run_all.sh
```

기본값은 일반 physics 1 ms, 백래시 검증 physics 0.1 ms, 제어기 후보
1/10 ms, canonical 제어기 1 ms, derived velocity 100 Hz 및 repeat 3이다.
0배 baseline도 백래시 후보와 동일한 미세 timestep에서 다시 계산하므로
physics 해상도 차이가 백래시 효과로 섞이지 않는다. 옵션은 다음 명령으로 확인한다.

```bash
uv run jandi-r2s-mode3-followup --help
```

결과는 다음에 새 timestamp 폴더로 저장되며 raw data와 기존 fitting 결과는 수정하지 않는다.

```text
results/mode3_bam/<campaign_id>/followup_validation/<timestamp>/
  REPORT.txt
  codex_summary.yaml
  full_summary.yaml
  all_run_metrics.csv
  reversal_metrics.csv
  cadence_overall.csv
  backlash_overall.csv
  plots/cadence/*.png
  plots/backlash/*.png
```

완료 후 Codex에는 터미널에 출력된 `codex_summary.yaml` 경로만 먼저 전달한다.
필요한 경우에만 상세 CSV나 이상 run의 PNG를 추가로 확인하면 된다.
