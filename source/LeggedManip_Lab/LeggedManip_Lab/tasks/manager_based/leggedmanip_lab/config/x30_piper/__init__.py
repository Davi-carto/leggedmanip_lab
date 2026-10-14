import gymnasium as gym

from . import agents


for task, config_name in (("Flat", "X30PiperFlat"), ("WBC", "X30PiperWBC")):
    for suffix in ("", "-Play"):
        gym.register(
            id=f"X30-PIPER-{task}{suffix}",
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
