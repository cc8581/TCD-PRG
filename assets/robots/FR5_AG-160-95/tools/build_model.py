"""Build the standalone FR5 + DH-Robotics AG-160-95 URDF package."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLB = ROOT / "source" / "AG95.glb"
SOURCE_FR5_URDF = ROOT / "source" / "fr5v6_original.urdf"
SOURCE_AG_URDF = ROOT / "source" / "ag95_reference.urdf"
OUTPUT_URDF = ROOT / "urdf" / "fr5_ag160_95.urdf"


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    """Backport ElementTree.indent for Python 3.8."""
    whitespace = "\n" + level * "  "
    child_whitespace = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = whitespace
        if level and (not element.tail or not element.tail.strip()):
            element.tail = whitespace
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def _validate_step_tessellation() -> dict:
    """Check that the bundled GLB still represents the complete STEP assembly."""
    scene = trimesh.load(SOURCE_GLB, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise TypeError("AG95.glb did not load as a trimesh.Scene")
    groups = set()
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        match = re.match(r"NAUO(\d+)_", node_name)
        if match is None:
            raise ValueError(f"Unexpected STEP occurrence name: {node_name}")
        groups.add(int(match.group(1)))
        transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if groups != set(range(1, 25)):
        raise ValueError(f"Expected STEP groups 1..24, got {sorted(groups)}")
    full = trimesh.util.concatenate(meshes)
    extents = np.asarray(full.extents, dtype=np.float64)
    expected = np.asarray((0.162343, 0.188141, 0.067000), dtype=np.float64)
    if not np.allclose(extents, expected, atol=0.0006):
        raise ValueError(f"Unexpected AG95 CAD extents: {extents}; expected {expected} m")
    return {
        "source_file": "source/AG95.step",
        "tessellated_source": "source/AG95.glb",
        "source_configuration": "fully_open",
        "source_component_occurrences": len(groups),
        "source_extents_m": extents.tolist(),
    }


def _add_tiny_inertial(link: ET.Element) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1e-9")
    ET.SubElement(
        inertial,
        "inertia",
        ixx="1e-18",
        ixy="0",
        ixz="0",
        iyy="1e-18",
        iyz="0",
        izz="1e-18",
    )


def _add_fixed_frame(root: ET.Element, name: str, parent: str, z: float) -> None:
    link = ET.SubElement(root, "link", name=name)
    _add_tiny_inertial(link)
    joint = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "origin", xyz=f"0 0 {z:.9g}", rpy="0 0 0")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)


def _normalize_ag_mass(ag_root: ET.Element) -> float:
    inertials = [
        link.find("inertial")
        for link in ag_root.findall("link")
        if link.find("inertial") is not None
    ]
    source_mass = sum(float(item.find("mass").get("value")) for item in inertials)
    scale = 1.0 / source_mass
    for inertial in inertials:
        mass = inertial.find("mass")
        mass.set("value", f"{float(mass.get('value')) * scale:.17g}")
        inertia = inertial.find("inertia")
        for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            inertia.set(attribute, f"{float(inertia.get(attribute)) * scale:.17g}")
    return source_mass


def _prepare_fr5_root() -> tuple[ET.ElementTree, ET.Element]:
    tree = ET.parse(SOURCE_FR5_URDF)
    root = tree.getroot()
    root.set("name", "fr5_ag160_95")
    removed_links = {"hand_base_link", "finger_link1", "finger_link2"}
    removed_joints = {"arm_hand_joint", "fj1", "fj2"}
    for child in list(root):
        if child.tag == "link" and child.get("name") in removed_links:
            root.remove(child)
        elif child.tag == "joint" and child.get("name") in removed_joints:
            root.remove(child)
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        filename = filename.replace(
            "package://fr5_description/meshes/fr5v6/visual/",
            "../meshes/fr5/visual/",
        )
        filename = filename.replace(
            "package://fr5_description/meshes/fr5v6/collision/",
            "../meshes/fr5/collision/",
        )
        mesh.set("filename", filename)
    return tree, root


def _append_ag(root: ET.Element) -> float:
    ag_root = ET.parse(SOURCE_AG_URDF).getroot()
    source_mass = _normalize_ag_mass(ag_root)
    grasp_link = ag_root.find("./link[@name='grasp_link']")
    if grasp_link is not None and grasp_link.find("inertial") is None:
        _add_tiny_inertial(grasp_link)
    for mesh in ag_root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        filename = filename.replace(
            "package://dh_ag95_description/meshes/visual/",
            "../meshes/ag16095/kinematic/visual/",
        )
        filename = filename.replace(
            "package://dh_ag95_description/meshes/collision/",
            "../meshes/ag16095/kinematic/collision/",
        )
        mesh.set("filename", filename)
    for child in list(ag_root):
        if child.tag == "link" and child.get("name") == "world":
            continue
        if child.tag == "joint" and child.get("name") == "gripper_base_joint":
            continue
        if child.tag in {"link", "joint"}:
            root.append(child)
    mount = ET.SubElement(root, "joint", name="ag_mount_joint", type="fixed")
    ET.SubElement(mount, "origin", xyz="0 0 0.12", rpy="0 0 3.141592654")
    ET.SubElement(mount, "parent", link="j6_Link")
    ET.SubElement(mount, "child", link="ag95_base_link")
    _add_fixed_frame(root, "tcp_open_front_link", "ag95_base_link", 0.188141)
    _add_fixed_frame(root, "tcp_closed_front_link", "ag95_base_link", 0.203700)
    _add_fixed_frame(root, "tcp_link", "ag95_base_link", 0.190000)
    return source_mass


def _check_names_and_meshes(root: ET.Element) -> None:
    for tag in ("link", "joint"):
        names = [item.get("name") for item in root.findall(tag)]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate {tag} names in generated URDF")
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://"):
            raise ValueError(f"Unresolved package URI: {filename}")
        path = (OUTPUT_URDF.parent / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    metadata = _validate_step_tessellation()
    tree, root = _prepare_fr5_root()
    reference_mass = _append_ag(root)
    _check_names_and_meshes(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        _indent_xml(root)
    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    metadata.update(
        {
            "urdf": "urdf/fr5_ag160_95.urdf",
            "kinematic_model": "ag95_parallel_linkage_ros_reference_v3",
            "kinematic_reference": (
                "ian-chuang/dh_ag95_gripper_ros2, adapted and mass-normalized"
            ),
            "reference_urdf_unscaled_mass_kg": reference_mass,
            "ag_mass_kg": 1.0,
            "controlled_gripper_joint": "left_outer_knuckle_joint",
            "coupled_gripper_joints": [
                "left_outer_knuckle_joint",
                "left_finger_joint",
                "left_inner_knuckle_joint",
                "right_outer_knuckle_joint",
                "right_finger_joint",
                "right_inner_knuckle_joint",
            ],
            "gripper_joint_open_rad": 0.0,
            "gripper_joint_closed_rad": 0.93,
            "fr5_flange_to_ag_mount_m": [0.0, 0.0, 0.12],
            "tcp_from_ag_mount_m": [0.0, 0.0, 0.19],
        }
    )
    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built {OUTPUT_URDF}")
    print("AG command range: 0.00 rad (open) to 0.93 rad (closed)")


if __name__ == "__main__":
    main()
