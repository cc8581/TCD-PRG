# Known limitations

- The current intermediate-state renderer is a deterministic PyBullet
  reconstruction, not a photorealistic sensor simulator. Domain adaptation and
  real-camera calibration remain experimental work.
- Exact FR5/AG certification validates IK, joint margins, collision and sampled
  approach waypoints. It is not a full time-parameterized industrial motion
  planner and should be replaced/wrapped by the deployment planner.
- PICK_REMOVE transport and placement are deliberately non-learning macro
  execution. The dataset does not supervise universal transport feasibility.
- The main PUSH primitive is fixed at 0.15 m. Variable distance is outside the
  main data semantics and may only be introduced as a separately labelled
  extension/ablation.
- Original GAPG inference requires three external checkpoints that are not
  present in or distributed by this repository.
- Offline closed-loop replay can only follow evaluated labelled transitions.
  UNKNOWN branches are reported as non-evaluable rather than failures.
- No trained TCD-PRG checkpoint is supplied. Reported performance must come
  from a completed, versioned training run rather than untrained smoke tests.
