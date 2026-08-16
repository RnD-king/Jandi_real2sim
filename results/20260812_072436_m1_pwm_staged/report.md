# Jandi staged measured-PWM M1 identification

## Final parameters

- drive_gain_nm_per_duty: 2.42051077
- armature_kg_m2: 0.00405198047
- coulomb_friction_nm: 0.57615522
- viscous_friction_nm_s_per_rad: 0.0499435049
- joint-refinement fit loss: 0.0276546712
- all-trajectory validation loss mean: 0.0278263923

## Stage contract

- Triangle: drive gain + Coulomb friction
- Multisine: drive gain + armature + viscous friction; Coulomb fixed
- Joint refinement: all four parameters in narrowed bounds
- repeat/seed 1·2 fit, 3 validation
- backlash/Stribeck/load-dependent friction are not included yet
