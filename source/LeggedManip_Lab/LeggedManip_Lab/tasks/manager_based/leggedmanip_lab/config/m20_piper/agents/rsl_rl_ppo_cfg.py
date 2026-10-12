from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class M20PiperFlatPPORunnerCfg(RslRlBaseRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    save_interval = 500
    experiment_name = "m20_piper_flat"
    clip_actions = 5.0
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    }
    critic = {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
    }
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms:PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class M20PiperWBCPPORunnerCfg(M20PiperFlatPPORunnerCfg):
    experiment_name = "m20_piper_wbc"
