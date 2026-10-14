import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.x30_piper.x30_piper_articulation_cfg import (
    ARM_JOINT_NAMES,
    CONTROLLED_JOINT_NAMES,
    FOOT_NAMES,
    X30_PIPER_CFG,
)

from ... import mdp
from ...leggedmanip_lab_env_cfg import ActionsCfg, LeggedManipLabEnvCfg


@configclass
class X30PiperActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale={".*_HipX_joint": 0.125, ".*_(HipY|Knee)_joint": 0.25, "joint[1-6]": 0.25},
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class X30PiperFlatEnvCfg(LeggedManipLabEnvCfg):
    actions: X30PiperActionsCfg = X30PiperActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = X30_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 1024
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
        self.scene.terrain.physics_material.restitution = 0.0
        self.scene.sky_light.spawn.texture_file = None
        self.commands.ee_pose.body_name = "gripper_base"
        self.commands.ee_pose.root_name = "arm_base_link"
        self.commands.ee_pose.debug_vis = False
        self.commands.ee_pose.curriculum_enabled = True
        self.commands.ee_pose.ranges = mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.24), pos_y=(-0.02, 0.02), pos_z=(0.46, 0.49),
            roll=(0.0, 0.0), pitch=(1.4836, 1.4836), yaw=(0.0, 0.0),
        )
        self.commands.ee_pose.limit_ranges = mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.40), pos_y=(-0.15, 0.15), pos_z=(0.20, 0.50),
            roll=(-0.25, 0.25), pitch=(1.28, 1.68), yaw=(-0.25, 0.25),
        )
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.curriculum_enabled = True
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.8, 0.8)
        self.events.push_robot = None
        self.events.base_com.params["asset_cfg"].body_names = "TORSO"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "TORSO"
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "TORSO"

        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = isaac_mdp.joint_pos_rel
            group.joint_vel.func = isaac_mdp.joint_vel_rel
            for term_name in ("joint_pos", "joint_vel"):
                getattr(group, term_name).params = {
                    "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES, preserve_order=True)
                }
        self.observations.critic.joint_torques.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES, preserve_order=True)
        }
        self.observations.critic.feet_contact.params["sensor_cfg"].body_names = FOOT_NAMES
        self.observations.critic.ee_link0_rel_pose.params = {
            "end_effector_name": "gripper_base", "reference_body_name": "arm_base_link"
        }
        self.rewards.end_effector_position_tracking_exp.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="gripper_base"), link0_name="arm_base_link"
        )
        self.rewards.end_effector_orientation_tracking.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="gripper_base"), reference_body_name="arm_base_link"
        )
        self.rewards.track_base_height_exp.params["target_height"] = 0.50
        self.rewards.hip_torques_max.params["joint_names"] = ".*_HipX_joint"
        self.rewards.thigh_torques_max.params["joint_names"] = ".*_HipY_joint"
        self.rewards.calf_torques_max.params["joint_names"] = ".*_Knee_joint"
        self.rewards.hip_deviation.params["asset_cfg"].joint_names = [".*_HipX_joint"]
        self.rewards.joint_deviation.params["asset_cfg"].joint_names = [".*_HipY_joint", ".*_Knee_joint"]
        self.rewards.arm_deviation.params["asset_cfg"].joint_names = ARM_JOINT_NAMES
        for reward_name in ("dof_pos_limits", "dof_acc_l2", "dof_torques_l2", "joint_power"):
            getattr(self.rewards, reward_name).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=CONTROLLED_JOINT_NAMES
            )
        for reward_name in ("feet_slide", "feet_air_time", "feet_long_air", "air_time_variance"):
            getattr(self.rewards, reward_name).params["sensor_cfg"].body_names = FOOT_NAMES
        self.rewards.feet_slide.params["asset_cfg"].body_names = FOOT_NAMES
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_.*_joint", "HL_.*_joint"], ["FL_.*_joint", "HR_.*_joint"]
        ]


@configclass
class X30PiperFlatEnvCfg_PLAY(X30PiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
