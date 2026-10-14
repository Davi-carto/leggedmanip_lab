from isaaclab.utils import configclass

from ... import mdp
from .flat_env_cfg import X30PiperFlatEnvCfg


@configclass
class X30PiperWBCEnvCfg(X30PiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose = mdp.command_cfg.UniformPoseWBCCommandCfg(
            asset_name="robot",
            body_name="gripper_base",
            link_name="arm_base_link",
            resampling_time_range=(8.0, 10.0),
            debug_vis=False,
            curriculum_enabled=True,
            ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
                pos_x=(0.20, 0.24), pos_y=(-0.02, 0.02), pos_z=(1.10, 1.17),
                roll=(0.0, 0.0), pitch=(1.4836, 1.4836), yaw=(0.0, 0.0),
            ),
            limit_ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
                pos_x=(0.20, 0.40), pos_y=(-0.15, 0.15), pos_z=(0.82, 1.19),
                roll=(-0.25, 0.25), pitch=(1.28, 1.68), yaw=(-0.25, 0.25),
            ),
        )
        self.rewards.end_effector_position_tracking_exp.func = mdp.position_command_error_exp
        self.rewards.end_effector_position_tracking_exp.weight = 4.5
        self.rewards.end_effector_orientation_tracking.weight = -4.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_base_height_exp.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -0.5


@configclass
class X30PiperWBCEnvCfg_PLAY(X30PiperWBCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
