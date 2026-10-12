"""Check policy action routing and signed joint motion before M20-Piper training."""

import argparse
import importlib
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["M20-PIPER-Flat", "M20-PIPER-WBC"], default="M20-PIPER-WBC")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

import omni.kit.app
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from LeggedManip_Lab.assets.m20_piper.m20_piper_articulation_cfg import (
    CONTROLLED_JOINT_NAMES,
    POSITION_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
)

importlib.import_module("LeggedManip_Lab.tasks")


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=44)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    cfg.seed = 42
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
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        robot = env.scene["robot"]
        position_ids, position_names = robot.find_joints(POSITION_JOINT_NAMES, preserve_order=True)
        wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
        assert position_names + wheel_names == CONTROLLED_JOINT_NAMES
        assert env.action_manager.active_terms == ["joint_pos", "wheel_vel"]
        scales = torch.tensor(
            [0.125 if "hipx" in name else 0.25 for name in POSITION_JOINT_NAMES], device=env.device
        )

        def check_targets(raw_actions):
            clipped = raw_actions.clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions)
            torch.testing.assert_close(env.action_manager.action, clipped)
            torch.testing.assert_close(
                robot.data.joint_pos_target[:, position_ids],
                robot.data.default_joint_pos[:, position_ids] + clipped[:, :18] * scales,
            )
            torch.testing.assert_close(robot.data.joint_vel_target[:, wheel_ids], clipped[:, 18:] * 5.0)
            gripper_ids = robot.find_joints("joint[78]")[0]
            torch.testing.assert_close(
                robot.data.joint_pos_target[:, gripper_ids], robot.data.default_joint_pos[:, gripper_ids]
            )

        zero_actions = torch.zeros(44, 22, device=env.device)
        for _ in range(10):
            wrapper.step(zero_actions)
        check_targets(zero_actions)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.update(env.physics_dt)
        initial_positions = robot.data.joint_pos.clone()
        probe_actions = torch.cat((torch.eye(22), -torch.eye(22)), dim=0).to(env.device) * 0.2
        for _ in range(10):
            observations, rewards, dones, _ = wrapper.step(probe_actions)
            assert not dones.any(), "Unexpected reset during isolated joint-direction checks"
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(values).all() for values in observations.values())
            check_targets(probe_actions)
        for action_index, joint_name in enumerate(CONTROLLED_JOINT_NAMES):
            joint_index = robot.joint_names.index(joint_name)
            response = (
                robot.data.joint_pos[:, joint_index] - initial_positions[:, joint_index]
                if action_index < 18 else robot.data.joint_vel[:, joint_index]
            )
            positive = response[action_index].item()
            negative = response[action_index + 22].item()
            assert positive > 1e-5 and negative < -1e-5, (joint_name, positive, negative)
            print(
                f"ACTION {action_index:02d} {joint_name}: positive={positive:+.6f}, negative={negative:+.6f} PASS",
                flush=True,
            )

        controlled_ids = position_ids + wheel_ids
        for term_name, expected in (
            ("joint_pos", robot.data.joint_pos[:, position_ids] - robot.data.default_joint_pos[:, position_ids]),
            ("joint_vel", robot.data.joint_vel[:, controlled_ids] - robot.data.default_joint_vel[:, controlled_ids]),
        ):
            term = getattr(env.observation_manager.cfg.policy, term_name)
            torch.testing.assert_close(term.func(env, **term.params), expected)
        saturated_actions = probe_actions.sign() * 8.0
        wrapper.step(saturated_actions)
        check_targets(saturated_actions)
        print("PASS: wrapper clipping, 18 position targets, 4 wheel velocity targets and observation signs", flush=True)

        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=env.device)
        policy = runner.get_inference_policy(device=env.device)
        policy_actions = policy(wrapper.get_observations())
        assert policy_actions.shape == (44, 22) and torch.isfinite(policy_actions).all()
        wrapper.step(policy_actions)
        check_targets(policy_actions)
        print("PASS: actual untrained policy outputs pass through the training wrapper to the correct targets", flush=True)
        print(f"ACTION_DIRECTION_CHECK_PASS {args_cli.task}: 22 joints x 2 signs", flush=True)
    finally:
        env.sim._disable_app_control_on_stop_handle = True
        wrapper.close()


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
