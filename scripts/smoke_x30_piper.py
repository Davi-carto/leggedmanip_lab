"""Check X30-Piper environments and signed policy actions without training."""

import argparse
import importlib
import sys
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["X30-PIPER-Flat", "X30-PIPER-WBC"], default="X30-PIPER-WBC")
parser.add_argument("--check_actions", action="store_true")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps < 1:
    parser.error("--steps must be positive")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import omni.kit.app
import torch
from pxr import Usd, UsdPhysics, UsdUtils
from scipy.spatial.transform import Rotation
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from LeggedManip_Lab.assets.x30_piper.x30_piper_articulation_cfg import (
    CONTROLLED_JOINT_NAMES, FOOT_NAMES, MODEL_DIR,
)

importlib.import_module("LeggedManip_Lab.tasks")


def check_kinematics(robot):
    urdf = ET.parse(MODEL_DIR / "X30_Piper/urdf/X30_Piper.urdf").getroot()
    relative_pos, relative_quat = subtract_frame_transforms(
        robot.data.root_pos_w[:, None, :].expand(-1, robot.num_bodies, -1),
        robot.data.root_quat_w[:, None, :].expand(-1, robot.num_bodies, -1),
        robot.data.body_pos_w, robot.data.body_quat_w,
    )
    positions = relative_pos.cpu().numpy()
    quaternions = relative_quat.cpu().numpy()
    max_position_error = 0.0
    max_rotation_error = 0.0
    for env_index in range(robot.num_instances):
        angles = dict(zip(robot.joint_names, robot.data.joint_pos[env_index].cpu().tolist()))
        transforms = {"TORSO": np.eye(4)}
        remaining = list(urdf.findall("joint"))
        while remaining:
            available = [joint for joint in remaining if joint.find("parent").get("link") in transforms]
            assert available, "Disconnected URDF"
            for joint in available:
                origin = joint.find("origin")
                frame = np.eye(4)
                if origin is not None:
                    frame[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    frame[:3, :3] = Rotation.from_euler(
                        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    ).as_matrix()
                motion = np.eye(4)
                if joint.get("type") != "fixed":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    if joint.get("type") == "prismatic":
                        motion[:3, 3] = axis * angles[joint.get("name")]
                    else:
                        motion[:3, :3] = Rotation.from_rotvec(axis * angles[joint.get("name")]).as_matrix()
                transforms[joint.find("child").get("link")] = (
                    transforms[joint.find("parent").get("link")] @ frame @ motion
                )
                remaining.remove(joint)
        for body_index, body_name in enumerate(robot.body_names):
            expected = transforms[body_name]
            actual_rotation = Rotation.from_quat(quaternions[env_index, body_index, [1, 2, 3, 0]])
            max_position_error = max(max_position_error, np.linalg.norm(
                positions[env_index, body_index] - expected[:3, 3]
            ))
            max_rotation_error = max(max_rotation_error, (
                Rotation.from_matrix(expected[:3, :3]).inv() * actual_rotation
            ).magnitude())
    assert max_position_error < 2e-3, max_position_error
    assert max_rotation_error < 2e-3, max_rotation_error
    print(f"FK_PASS: URDF vs PhysX positions {max_position_error:.3g} m, rotations {max_rotation_error:.3g} rad", flush=True)


def main():
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(MODEL_DIR / "X30_Piper/usd/X30_Piper.usd"))
    assert len(layers) == 1 and not assets and not unresolved
    stage = Usd.Stage.Open(str(MODEL_DIR / "X30_Piper/usd/X30_Piper.usd"))
    assert sum(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse()) == 1
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=36 if args_cli.check_actions else 4)
    cfg.seed = 42
    if args_cli.check_actions:
        cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.5)
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        cfg.observations.policy.enable_corruption = False
        for event_name in (
            "physics_material", "add_mass", "base_com", "randomize_actuator_gains",
            "randomize_rigid_body_inertia", "base_external_force_torque", "push_robot",
        ):
            setattr(cfg.events, event_name, None)
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        robot = env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(CONTROLLED_JOINT_NAMES, preserve_order=True)
        assert joint_names == CONTROLLED_JOINT_NAMES
        assert robot.num_joints == 20 and env.action_manager.total_action_dim == 18
        assert env.action_manager.active_terms == ["joint_pos"]
        assert wrapper.get_observations()["policy"].shape == (env.num_envs, 210)
        assert wrapper.get_observations()["critic"].shape == (env.num_envs, 102)
        assert len(env.scene["contact_forces"].find_bodies(FOOT_NAMES)[0]) == 4
        assert not set(joint_ids).intersection(robot.find_joints("joint[78]")[0])
        assert robot.body_names[0] == "TORSO"
        check_kinematics(robot)
        arm_id = robot.find_bodies("arm_base_link")[0][0]
        flange_id = robot.find_bodies("gripper_base")[0][0]
        relative_pos, relative_quat = subtract_frame_transforms(
            robot.data.body_pos_w[:, arm_id], robot.data.body_quat_w[:, arm_id],
            robot.data.body_pos_w[:, flange_id], robot.data.body_quat_w[:, flange_id],
        )
        command = env.command_manager.get_command("ee_pose")
        saved_command = command.clone()
        command[:, :3], command[:, 3:] = relative_pos, relative_quat
        if args_cli.task.endswith("WBC"):
            command[:, 2] = robot.data.body_pos_w[:, flange_id, 2]
        reward_cfg = env.reward_manager.get_term_cfg("end_effector_position_tracking_exp")
        torch.testing.assert_close(reward_cfg.func(env, **reward_cfg.params), torch.ones(env.num_envs, device=env.device))
        reward_cfg = env.reward_manager.get_term_cfg("end_effector_orientation_tracking")
        assert torch.all(reward_cfg.func(env, **reward_cfg.params) < 1e-3)
        command.copy_(saved_command)
        scales = torch.tensor([0.125 if "HipX" in name else 0.25 for name in joint_names], device=env.device)

        def check_targets(actions):
            expected = robot.data.default_joint_pos[:, joint_ids] + actions.clamp(-5.0, 5.0) * scales
            torch.testing.assert_close(robot.data.joint_pos_target[:, joint_ids], expected)
            gripper_ids = robot.find_joints("joint[78]")[0]
            torch.testing.assert_close(robot.data.joint_pos_target[:, gripper_ids], torch.zeros_like(robot.data.joint_pos[:, gripper_ids]))

        actions = torch.zeros(env.num_envs, 18, device=env.device)
        if args_cli.check_actions:
            torch.testing.assert_close(
                robot.data.joint_pos[:, joint_ids], robot.data.default_joint_pos[:, joint_ids], atol=1e-5, rtol=0
            )
            for _ in range(10):
                wrapper.step(actions)
            initial_positions = robot.data.joint_pos.clone()
            actions = torch.cat((torch.eye(18), -torch.eye(18)), dim=0).to(env.device) * 0.2
            for _ in range(10):
                observations, rewards, dones, _ = wrapper.step(actions)
                assert not dones.any()
                assert torch.isfinite(rewards).all()
                check_targets(actions)
            for action_index, joint_id in enumerate(joint_ids):
                response = robot.data.joint_pos[:, joint_id] - initial_positions[:, joint_id]
                positive, negative = response[action_index].item(), response[action_index + 18].item()
                assert positive > 1e-5 and negative < -1e-5, (joint_names[action_index], positive, negative)
                print(f"ACTION {action_index:02d} {joint_names[action_index]}: {positive:+.6f} / {negative:+.6f} PASS", flush=True)
            check_kinematics(robot)
            for term_name, expected in (
                ("joint_pos", robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]),
                ("joint_vel", robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]),
            ):
                term = getattr(env.observation_manager.cfg.policy, term_name)
                torch.testing.assert_close(term.func(env, **term.params), expected)
            actions = actions.sign() * 8.0
            wrapper.step(actions)
            check_targets(actions)
            runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=env.device)
            policy = runner.get_inference_policy(device=env.device)
            actions = policy(wrapper.get_observations())
            assert actions.shape == (36, 18) and torch.isfinite(actions).all()
            wrapper.step(actions)
            check_targets(actions)
            print(f"ACTION_DIRECTION_CHECK_PASS {args_cli.task}: 18 joints x 2 signs; actual actor routing and clipping", flush=True)
        else:
            resets = 0
            for _ in range(args_cli.steps):
                observations, rewards, dones, _ = wrapper.step(actions)
                for values in (*observations.values(), rewards, robot.data.joint_pos, robot.data.joint_vel):
                    assert torch.isfinite(values).all()
                resets += int(dones.sum())
                check_targets(actions)
            print(f"OPEN_LOOP: resets={resets}; mean torso height={robot.data.root_pos_w[:, 2].mean().item():.3f} m", flush=True)
            env.episode_length_buf[:] = env.max_episode_length - 1
            observations, rewards, dones, extras = wrapper.step(actions)
            assert dones.all() and extras["time_outs"].all() and (env.episode_length_buf == 0).all()
            print(f"PASS {args_cli.task}: 20 DOF, 18 actions, obs 210/102, {args_cli.steps} steps and timeout reset", flush=True)
    finally:
        env.sim._disable_app_control_on_stop_handle = True
        wrapper.close()


exit_code = 0
try:
    with torch.inference_mode():
        main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    omni.kit.app.get_app().post_quit(exit_code)
    simulation_app.close()
sys.exit(exit_code)
