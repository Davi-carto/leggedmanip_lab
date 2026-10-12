"""Preview the M20-aligned M20-Piper model at its default pose without training."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from LeggedManip_Lab.assets.m20_piper.m20_piper_articulation_cfg import M20_PIPER_CFG


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=(2.1, -2.1, 1.65), target=(0.0, 0.0, 0.6))
    ground = sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.27)),
    )
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    robot = Articulation(M20_PIPER_CFG.replace(prim_path="/World/M20_Piper"))
    sim.reset()
    sim._disable_app_control_on_stop_handle = True
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.set_joint_velocity_target(robot.data.default_joint_vel)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(sim.get_physics_dt())
    torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos, atol=1e-5, rtol=0)
    sim.pause()
    print("PREVIEW_READY: M20-aligned joints; front hip/knee -0.3/+0.6, rear +0.3/-0.6 rad", flush=True)
    while simulation_app.is_running():
        sim.render()
        time.sleep(1.0 / 30.0)


try:
    with torch.inference_mode():
        main()
finally:
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        sim._disable_app_control_on_stop_handle = True
    simulation_app.close()
