"""Build a portable X30-Piper URDF and flattened USD without running training."""

import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import sys
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pxr import Gf, Sdf, Usd, UsdShade, UsdUtils

import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    model_dir = Path(__file__).resolve().parents[1] / "deep_robotics_model"
    output_dir = model_dir / "X30_Piper"
    urdf_dir = output_dir / "urdf"
    usd_dir = output_dir / "usd"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    usd_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(model_dir / "X30/urdf/X30.urdf").getroot()
    robot.set("name", "X30_Piper")
    robot.remove(robot.find("mujoco"))
    for mesh in robot.iter("mesh"):
        mesh.set("filename", "../../X30/urdf/meshes/" + Path(mesh.get("filename")).name)
    for link in robot.findall("link"):
        if link.get("name").endswith("_FOOT"):
            visual = deepcopy(link.find("collision"))
            visual.tag = "visual"
            link.append(visual)
    piper = ET.parse(model_dir / "M20_Piper/urdf/M20_Piper.urdf").getroot()
    arm_links = {"arm_base_link", "gripper_base", *(f"link{index}" for index in range(1, 9))}
    for element in piper:
        if element.tag == "link" and element.get("name") in arm_links:
            copied = deepcopy(element)
            for mesh in copied.iter("mesh"):
                mesh.set("filename", "../../M20_Piper/urdf/meshes/" + Path(mesh.get("filename")).name)
            robot.append(copied)
        elif element.tag == "joint" and element.find("child").get("link") in arm_links:
            copied = deepcopy(element)
            if copied.get("name") == "base_to_arm":
                copied.find("parent").set("link", "TORSO")
                copied.find("origin").set("xyz", "0 0 0.14")
            robot.append(copied)
    ET.indent(robot, space="  ")
    urdf_path = urdf_dir / "X30_Piper.urdf"
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    with tempfile.TemporaryDirectory(prefix="x30_piper_convert_") as temporary:
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=temporary,
            usd_file_name="X30_Piper.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            root_link_name="TORSO",
            merge_fixed_joints=False,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ))
        stage = Usd.Stage.Open(converter.usd_path)
        for prim in stage.Traverse():
            if prim.IsInstance():
                prim.SetInstanceable(False)
        stage = Usd.Stage.Open(stage.Flatten())
        materials = [prim for prim in stage.Traverse() if prim.IsA(UsdShade.Material)]
        for prim in materials:
            color = Gf.Vec3f(0.65)
            for child in prim.GetChildren():
                color_attribute = child.GetAttribute("inputs:diffuse_color_constant")
                if color_attribute and color_attribute.Get() is not None:
                    color = Gf.Vec3f(color_attribute.Get())
                stage.RemovePrim(child.GetPath())
            for property_name in prim.GetPropertyNames():
                if property_name.startswith("outputs:"):
                    prim.RemoveProperty(property_name)
            shader = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("Surface"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            UsdShade.Material(prim).CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        output_path = usd_dir / "X30_Piper.usd"
        stage.Flatten().Export(str(output_path))
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(output_path))
        assert not unresolved, unresolved
        assert not assets and len(layers) == 1, "USD must be self-contained"
    print(f"BUILT {urdf_path}", flush=True)
    print(f"BUILT {output_path} (standalone USD; torso mount xyz=0,0,0.14 m)", flush=True)


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    omni.kit.app.get_app().post_quit(exit_code)
    simulation_app.close()
sys.exit(exit_code)
