"""Check M20-Piper joint mapping, rolling, command frames and automatic resets."""

import argparse
import importlib
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["M20-PIPER-Flat", "M20-PIPER-WBC"], default="M20-PIPER-Flat")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps < 60:
    parser.error("--steps must be at least 60")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.kit.app
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from LeggedManip_Lab.assets.m20_piper.m20_piper_articulation_cfg import (
    CONTROLLED_JOINT_NAMES,
    LEG_JOINT_NAMES,
    MODEL_DIR,
    POSITION_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
)

importlib.import_module("LeggedManip_Lab.tasks")


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = 42
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    try:
        observations, _ = env.reset()
        robot = env.scene["robot"]
        position_ids, position_names = robot.find_joints(POSITION_JOINT_NAMES, preserve_order=True)
        wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
        assert position_names + wheel_names == CONTROLLED_JOINT_NAMES
        assert robot.num_joints == 24
        assert env.action_manager.total_action_dim == 22
        assert observations["policy"].shape == (env.num_envs, 234)
        assert observations["critic"].shape == (env.num_envs, 114)
        assert torch.all(robot.actuators["wheels"].stiffness == 0)
        assert len(env.scene["contact_forces"].find_bodies(".*_wheel")[0]) == 4
        controlled_ids = set(position_ids + wheel_ids)
        gripper_ids = set(robot.find_joints("joint[78]")[0])
        assert len(controlled_ids) == 22 and not controlled_ids.intersection(gripper_ids)
        assert sum(actuator.num_joints for actuator in robot.actuators.values()) == 24

        reference_stage = Usd.Stage.Open(str(MODEL_DIR / "M20" / "usd" / "M20.usd"))
        reference_joints = {
            joint.GetName(): joint for joint in reference_stage.Traverse()
            if joint.GetName() in LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
        }
        original_stage = Usd.Stage.Open(str(MODEL_DIR / "M20_Piper" / "usd" / "M20_Piper.usd"))
        original_arm_joints = {
            joint.GetName(): joint for joint in original_stage.Traverse()
            if joint.GetName() in [f"joint{index}" for index in range(1, 9)]
        }
        robot_prims = sim_utils.find_matching_prims(robot.cfg.prim_path)
        assert len(robot_prims) == env.num_envs
        for robot_prim in robot_prims:
            joints = {
                joint.GetName(): joint for joint in Usd.PrimRange(robot_prim)
                if joint.IsA(UsdPhysics.RevoluteJoint) and joint.GetName() in reference_joints
            }
            assert joints.keys() == reference_joints.keys()
            for name, joint in joints.items():
                for attribute_name in (
                    "physics:axis", "physics:localRot0", "physics:localRot1",
                    "physics:lowerLimit", "physics:upperLimit",
                ):
                    expected = reference_joints[name].GetAttribute(attribute_name).Get()
                    assert joint.GetAttribute(attribute_name).Get() == expected
            for joint in Usd.PrimRange(robot_prim):
                if joint.GetName() not in original_arm_joints:
                    continue
                for attribute_name in (
                    "physics:axis", "physics:localRot0", "physics:localRot1",
                    "physics:localPos0", "physics:localPos1", "physics:lowerLimit", "physics:upperLimit",
                ):
                    expected = original_arm_joints[joint.GetName()].GetAttribute(attribute_name).Get()
                    assert joint.GetAttribute(attribute_name).Get() == expected
        print(f"Official M20 joint frames and limits match in all {env.num_envs} environments", flush=True)

        reference_id = robot.find_bodies("arm_base_link")[0][0]
        end_effector_id = robot.find_bodies("gripper_base")[0][0]
        relative_pos, relative_quat = subtract_frame_transforms(
            robot.data.body_pos_w[:, reference_id], robot.data.body_quat_w[:, reference_id],
            robot.data.body_pos_w[:, end_effector_id], robot.data.body_quat_w[:, end_effector_id],
        )
        command = env.command_manager.get_command("ee_pose")
        saved_command = command.clone()
        command[:, :3] = relative_pos
        command[:, 3:7] = relative_quat
        if args_cli.task.endswith("WBC"):
            command[:, 2] = robot.data.body_pos_w[:, end_effector_id, 2]
        reward_cfg = env.reward_manager.get_term_cfg("end_effector_position_tracking_exp")
        torch.testing.assert_close(
            reward_cfg.func(env, **reward_cfg.params), torch.ones(env.num_envs, device=env.device)
        )
        orientation_cfg = env.reward_manager.get_term_cfg("end_effector_orientation_tracking")
        assert torch.all(orientation_cfg.func(env, **orientation_cfg.params) < 1e-3)
        command.copy_(saved_command)
        print(f"Initial flange relative position: {relative_pos[0].tolist()}", flush=True)

        actions = torch.zeros(env.num_envs, 22, device=env.device)
        forward_speeds = []
        termination_count = 0
        for step in range(args_cli.steps):
            actions[:, 18:] = -1.0 if 20 <= step < 60 else 0.0
            observations, rewards, terminated, truncated, _ = env.step(actions)
            for values in (*observations.values(), rewards, robot.data.joint_pos, robot.data.joint_vel):
                assert torch.isfinite(values).all()
            termination_count += int(terminated.sum())
            if 50 <= step < 60:
                forward_speeds.append(robot.data.root_lin_vel_b[:, 0].mean())
            if step == 59:
                torch.testing.assert_close(
                    robot.data.joint_pos_target[:, position_ids], robot.data.default_joint_pos[:, position_ids]
                )
                torch.testing.assert_close(
                    robot.data.joint_vel_target[:, wheel_ids], torch.full_like(actions[:, 18:], -5.0)
                )
        forward_speed = torch.stack(forward_speeds).mean().item()
        assert forward_speed > 0.05, f"Negative wheel command did not move forward: {forward_speed}"
        print(f"Forward speed at wheel target -5 rad/s: {forward_speed:.3f} m/s", flush=True)
        print(f"Open-loop falls/reset count (not a trained balance controller): {termination_count}", flush=True)

        env.episode_length_buf[:] = env.max_episode_length - 1
        observations, rewards, _, truncated, _ = env.step(torch.zeros_like(actions))
        assert truncated.all() and (env.episode_length_buf == 0).all()
        assert torch.all(robot.data.joint_pos[:, wheel_ids].abs() < 1e-5)
        assert all(torch.isfinite(values).all() for values in observations.values())
        print(
            f"PASS {args_cli.task}: 24 DOF, 22 actions, obs 234/114, {args_cli.steps} steps and timeout reset",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
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
