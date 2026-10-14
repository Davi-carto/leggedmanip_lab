from isaaclab.utils import configclass

from ...m20_piper.agents.rsl_rl_ppo_cfg import M20PiperFlatPPORunnerCfg


@configclass
class X30PiperFlatPPORunnerCfg(M20PiperFlatPPORunnerCfg):
    experiment_name = "x30_piper_flat"


@configclass
class X30PiperWBCPPORunnerCfg(X30PiperFlatPPORunnerCfg):
    experiment_name = "x30_piper_wbc"
