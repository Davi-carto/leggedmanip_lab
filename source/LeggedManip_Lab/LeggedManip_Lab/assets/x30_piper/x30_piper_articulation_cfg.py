from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


MODEL_DIR = Path(__file__).resolve().parents[5] / "deep_robotics_model"
LEG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "HL", "HR")
    for joint in ("HipX", "HipY", "Knee")
]
ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
CONTROLLED_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
FOOT_NAMES = [f"{leg}_FOOT" for leg in ("FL", "FR", "HL", "HR")]


X30_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(MODEL_DIR / "X30_Piper/usd/X30_Piper.usd"),
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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_HipX_joint": 0.0,
            ".*_HipY_joint": -0.65,
            ".*_Knee_joint": 1.30,
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
            effort_limit={".*_Hip[XY]_joint": 84.0, ".*_Knee_joint": 150.0},
            effort_limit_sim={".*_Hip[XY]_joint": 84.0, ".*_Knee_joint": 150.0},
            velocity_limit={".*_Hip[XY]_joint": 17.5, ".*_Knee_joint": 16.1},
            velocity_limit_sim={".*_Hip[XY]_joint": 17.5, ".*_Knee_joint": 16.1},
            stiffness={".*_Hip[XY]_joint": 200.0, ".*_Knee_joint": 240.0},
            damping={".*_Hip[XY]_joint": 5.0, ".*_Knee_joint": 6.0},
            friction=0.0,
            armature=0.0,
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
