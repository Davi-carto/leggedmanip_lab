import gymnasium as gym

from . import agents


for task, config_name in (("Flat", "M20PiperFlat"), ("WBC", "M20PiperWBC")):
    for suffix in ("", "-Play"):
        gym.register(
            id=f"M20-PIPER-{task}{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.{task.lower()}_env_cfg:{config_name}EnvCfg"
                    f"{'_PLAY' if suffix else ''}"
                ),
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{config_name}PPORunnerCfg",
            },
        )
