# DeepRobotics model snapshot

This directory vendors the M20, M20_Piper, and X30 model directories used by LeggedManip Lab, including USD layers, URDF, MJCF, and meshes. They are ordinary files, not a Git submodule or Git LFS pointers.

- Upstream: https://github.com/DeepRoboticsLab/deep_robotics_model
- Pinned commit: `549eb9ac08ab0674ad388202f0a276a1b140833b`
- Included upstream paths: `M20/`, `M20_Piper/`, `X30/`, `LICENSE.txt`, `Contributors.md`.
- License: upstream BSD 3-Clause, preserved in [LICENSE.txt](LICENSE.txt).
- These upstream model files are unmodified copies of the pinned commit; unrelated robot models are omitted.

With `DEEP_ROBOTICS_MODEL_DIR` unset, the project's M20-Piper configuration uses this directory automatically. If the variable is set, it overrides this bundled snapshot.

The project loads `M20_Piper/usd/M20_Piper.usd` and copies the 16 chassis joints' axes, local rotations, and limits from `M20/usd/M20.usd` into the simulation stage before environment cloning. This preserves the previously validated M20 joint-direction correction without editing these upstream files. Opening the raw USD outside the project's loader does not apply that correction.

See [M20-Piper setup and validation](../docs/M20_PIPER_CN.md).

`X30_Piper/` is a derived assembly, not an upstream official robot: the original X30 chassis plus the Piper arm from M20_Piper, mounted at torso-relative `[0, 0, 0.14]` m with identity rotation. It includes a composed URDF and a standalone flattened USD with local Preview Surface materials. Source geometry, inertias, and joints remain covered by the upstream BSD 3-Clause license. Four visual foot spheres are added to match the original collision spheres. See [X30-Piper assembly and validation](../docs/X30_PIPER_CN.md) and `scripts/build_x30_piper.py` for reproduction and modeling assumptions.
