import os
from pathlib import Path

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


MODEL_DIR = Path(os.environ.get(
    "DEEP_ROBOTICS_MODEL_DIR", Path(__file__).resolve().parents[5] / "deep_robotics_model"
)).expanduser()

LEG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("fl", "fr", "hl", "hr")
    for joint in ("hipx", "hipy", "knee")
]
ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
WHEEL_JOINT_NAMES = [f"{leg}_wheel_joint" for leg in ("fl", "fr", "hl", "hr")]
POSITION_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
CONTROLLED_JOINT_NAMES = POSITION_JOINT_NAMES + WHEEL_JOINT_NAMES


@sim_utils.clone
def spawn_m20_piper(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    reference_stage = Usd.Stage.Open(str(MODEL_DIR / "M20" / "usd" / "M20.usd"))
    reference_joints = {
        joint.GetName(): joint
        for joint in reference_stage.Traverse()
        if joint.GetName() in LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
    }
    for joint in Usd.PrimRange(prim):
        if joint.GetName() not in reference_joints:
            continue
        reference_joint = reference_joints[joint.GetName()]
        for attribute_name in (
            "physics:axis", "physics:localRot0", "physics:localRot1",
            "physics:lowerLimit", "physics:upperLimit",
        ):
            joint.GetAttribute(attribute_name).Set(reference_joint.GetAttribute(attribute_name).Get())
    return prim


M20_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_m20_piper,
        usd_path=str(MODEL_DIR / "M20_Piper" / "usd" / "M20_Piper.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*_hipx_joint": 0.0,
            "f[lr]_hipy_joint": -0.3,
            "h[lr]_hipy_joint": 0.3,
            "f[lr]_knee_joint": 0.6,
            "h[lr]_knee_joint": -0.6,
            ".*_wheel_joint": 0.0,
            "joint1": 0.0,
            "joint2": 1.0,
            "joint3": -1.2,
            "joint4": 0.0,
            "joint5": 0.2,
            "joint6": 0.0,
            "joint[78]": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit=76.4,
            effort_limit_sim=76.4,
            velocity_limit=22.4,
            velocity_limit_sim=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit=21.6,
            effort_limit_sim=21.6,
            velocity_limit=79.3,
            velocity_limit_sim=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit={"joint[12]": 20.0, "joint3": 15.0, "joint4": 7.0, "joint[56]": 5.0},
            effort_limit_sim={"joint[12]": 20.0, "joint3": 15.0, "joint4": 7.0, "joint[56]": 5.0},
            velocity_limit={"joint[1-5]": 5.0, "joint6": 3.0},
            velocity_limit_sim={"joint[1-5]": 5.0, "joint6": 3.0},
            stiffness={"joint[12]": 50.0, "joint3": 80.0, "joint[45]": 30.0, "joint6": 20.0},
            damping={"joint[134]": 3.0, "joint2": 2.0, "joint5": 2.5, "joint6": 1.0},
            armature=0.01,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[78]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)
