# Mode 5 module routing

The canonical README-v3 public path is:

- `spec.py`
- `canonical_config.py`
- `canonical_trajectories.py`
- `canonical_bus.py`
- `canonical_acquisition.py`
- `canonical_model.py`
- `canonical_analysis.py`
- `cli.py`

The older `config.py`, `trajectories.py`, `bus.py`, `collector.py`,
`mujoco_model.py`, and `analysis.py` are retained as legacy source only.  None
of the eight `jandi-r2s-mode5-*` public entry points imports them.
