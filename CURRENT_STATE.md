# Jandi Real2Sim Current State

This file contains changing project state.

Unlike `AGENTS.md`, it should be updated when the selected campaign, fitted model, validation result, or next experiment changes.

Before trusting this file, compare it with the current result YAML and repository state.

## Current canonical campaign

Campaign:

`mx106_id24_p850_20260913_v2`

Target actuator:

- DYNAMIXEL MX-106
- Mode 3 Position Control
- P gain 850

## Selected model

Selected model:

`M3`

Selection rule:

Prefer the simpler model unless validation position MAE improves by at least 2%.

Current selected M3 parameters from `selected_model.yaml`:

- Kt: 2.2161716898261687 Nm/A
- resistance: 2.3243950944304173 ohm
- equivalent armature: 0.025799168257662458 kg·m²
- Coulomb friction: 0.1717060794144954 Nm
- viscous friction: approximately zero at the fitted boundary
- load friction coefficient: 0.1573951561611039

M3 repeat-3 validation:

- position MAE: 0.0015541848181378636 rad
- position RMSE: 0.0022164332815748874 rad
- validation runs: 24

Always re-read `results/mode3_bam/<campaign>/selected_model.yaml` before using these numbers for new work.

## Controller timing

Canonical equivalent controller update:

`0.001 s`

Follow-up validation compared:

- 1 ms
- 10 ms

Latest follow-up aggregate position MAE:

- 1 ms: 0.003946796996600313 rad
- 10 ms: 0.004180640832095606 rad

The current canonical choice remains 1 ms.

## MuJoCo timing

Canonical physics timestep:

`0.001 s`

Backlash sensitivity tests may use a smaller timestep to avoid mixing timestep effects with backlash effects.

## Validation split

- repeat 1: fitting
- repeat 2: fitting
- repeat 3: validation-only

Do not silently mix repeat 3 into fitting.

## Latest local follow-up validation

Local follow-up result:

`results/mode3_bam/mx106_id24_p850_20260913_v2/followup_validation/20260914_061601/`

Primary compact summary:

`codex_summary.yaml`

Full summary:

`full_summary.yaml`

Validation run count:

24

No parameter refit was performed during this follow-up validation.

## Derived velocity

The follow-up study evaluated Savitzky-Golay derivative windows:

- 5
- 7
- 9

The canonical/primary configured window in the follow-up run is 7.

Derived velocity must not be confused with DYNAMIXEL Present Velocity.

## Effective backlash

Canonical fitted effective total backlash width used by the follow-up analysis:

`0.005854386182393846 rad`

Interpret this as an effective-state/dead-zone parameter rather than confirmed physical gear backlash.

M3 nominal remains the canonical baseline.

Backlash variants are comparison models unless explicitly promoted after validation.

## Known limitations

- Predicted current does not perfectly reproduce measured DYNAMIXEL Present Current.
- Current was not independently fitted as the primary objective.
- Position reproduction currently has priority over forcing current waveform agreement.
- Effective backlash can absorb several low-speed/unmodeled effects.
- Validation is actuator/bench level and does not by itself prove full-robot sim-to-real success.

## Storage policy

Track in Git:

- source code
- tests
- configs
- scripts
- documentation
- this current-state summary

Keep locally under `results/`:

- fitted parameter YAML
- selected model YAML
- validation summaries
- simulation time series
- comparison time series
- plots
- detailed follow-up outputs

When important experimental conclusions change, reflect the adopted values
and conclusions in this file.

## Next-state update checklist

Update this file when any of the following changes:

- canonical campaign
- selected model
- fitted parameters
- equivalent controller update period
- canonical physics timestep
- backlash interpretation/model
- validation conclusion
- next planned experiment
