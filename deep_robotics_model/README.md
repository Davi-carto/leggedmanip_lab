# DeepRobotics model snapshot

This directory vendors the M20 and M20_Piper model directories used by LeggedManip Lab, including USD layers, URDF, MJCF, and meshes. They are ordinary tracked files, not a Git submodule or Git LFS pointers, so cloning LeggedManip Lab also retrieves the model resources.

- Upstream: https://github.com/DeepRoboticsLab/deep_robotics_model
- Pinned commit: `549eb9ac08ab0674ad388202f0a276a1b140833b`
- Included upstream paths: `M20/`, `M20_Piper/`, `LICENSE.txt`, `Contributors.md`.
- License: upstream BSD 3-Clause, preserved in [LICENSE.txt](LICENSE.txt).
- Model files are unmodified copies of the pinned commit; unrelated robot models are omitted.

With `DEEP_ROBOTICS_MODEL_DIR` unset, the project's M20-Piper configuration uses this directory automatically. If the variable is set, it overrides this bundled snapshot.

The project loads `M20_Piper/usd/M20_Piper.usd` and copies the 16 chassis joints' axes, local rotations, and limits from `M20/usd/M20.usd` into the simulation stage before environment cloning. This preserves the previously validated M20 joint-direction correction without editing these upstream files. Opening the raw USD outside the project's loader does not apply that correction.

See [M20-Piper setup and validation](../docs/M20_PIPER_CN.md).
