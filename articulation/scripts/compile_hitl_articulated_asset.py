#!/usr/bin/env python3
"""Compile an N-link HITL manifest into a frame-consistent URDF asset."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent


def inertia_for_mass(mesh: trimesh.Trimesh, mass: float) -> tuple[np.ndarray, np.ndarray]:
    com = np.asarray(mesh.center_mass)
    centered = mesh.copy()
    centered.apply_translation(-com)
    inertia = np.asarray(centered.moment_inertia) * mass / max(abs(float(centered.volume)), 1e-12)
    values, vectors = np.linalg.eigh(0.5 * (inertia + inertia.T))
    values = np.maximum(values, 1e-8)
    if values[0] + values[1] <= values[2]:
        values[2] = 0.999 * (values[0] + values[1])
    return com, vectors @ np.diag(values) @ vectors.T


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"not a triangle mesh: {path}")
    return mesh


def robust_pca(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(mesh.vertices, axis=0)
    _values, axes = np.linalg.eigh(np.cov(mesh.vertices - center, rowvar=False))
    local = (mesh.vertices - center) @ axes
    observed = np.quantile(local, 0.995, axis=0) - np.quantile(local, 0.005, axis=0)
    return center, axes, observed


def measured_obb(mesh: trimesh.Trimesh, dimensions_m: list[float]) -> trimesh.Trimesh:
    dimensions = np.sort(np.asarray(dimensions_m, dtype=float))
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("measured_obb_m must contain three positive dimensions")
    center, axes, _observed = robust_pca(mesh)
    box = trimesh.creation.box(dimensions)
    transform = np.eye(4)
    transform[:3, :3] = axes
    transform[:3, 3] = center
    box.apply_transform(transform)
    return box


def automatic_collision(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    values = np.linalg.eigvalsh(np.cov(centered, rowvar=False))[::-1]
    elongation = float(np.sqrt(values[0] / max(values[1], 1e-12)))
    convexity = float(abs(mesh.volume) / max(abs(mesh.convex_hull.volume), 1e-12))
    characteristic = float(mesh.extents.max())
    thickness = float(np.clip(0.015 * characteristic, 0.0012, 0.0035))
    if elongation > 2.5:
        kind, thickness = "tubular_shell", thickness * 0.8
    elif not mesh.is_watertight or convexity < 0.65:
        kind = "open_or_concave_shell"
    else:
        kind, thickness = "solid", 0.0
    if thickness == 0.0:
        collision = mesh.copy()
    else:
        outer = mesh.copy()
        surface = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(outer.vertices),
            o3d.utility.Vector3iVector(outer.faces.astype(np.int32)),
        )
        surface.compute_vertex_normals()
        inner_vertices = outer.vertices - np.asarray(surface.vertex_normals) * thickness
        collision = trimesh.Trimesh(
            vertices=np.vstack([outer.vertices, inner_vertices]),
            faces=np.vstack([outer.faces, outer.faces[:, ::-1] + len(outer.vertices)]),
            process=True,
        )
        trimesh.repair.fill_holes(collision)
        trimesh.repair.fix_winding(collision)
        trimesh.repair.fix_inversion(collision)
    return collision, {
        "backend": "automatic_morphology_shell",
        "part_type": kind,
        "elongation_ratio": round(elongation, 3),
        "convexity_ratio": round(convexity, 3),
        "derived_thickness_mm": round(thickness * 1000, 3),
    }


def contact_origin(a: trimesh.Trimesh, b: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    ap = a.sample(50_000, seed=rng)
    bp = b.sample(50_000, seed=rng)
    distance, nearest = cKDTree(ap).query(bp, workers=-1)
    take = np.argpartition(distance, min(299, len(distance) - 1))[:300]
    return np.median((bp[take] + ap[nearest[take]]) / 2, axis=0), float(np.median(distance[take]))


def surface_p95_mm(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    rng = np.random.default_rng(0)
    ap = a.sample(50_000, seed=rng)
    bp = b.sample(50_000, seed=rng)
    ab = cKDTree(bp).query(ap, workers=-1)[0]
    ba = cKDTree(ap).query(bp, workers=-1)[0]
    return float(max(np.quantile(ab, 0.95), np.quantile(ba, 0.95)) * 1000)


def vec(values: np.ndarray) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def compile_asset(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    mode = manifest.get("mode")
    if mode not in {"automatic", "hitl", "hitl_measurement_assisted"}:
        raise ValueError("mode must be automatic, hitl, or hitl_measurement_assisted")
    links = manifest["links"]
    joints = manifest["joints"]
    names = [link["name"] for link in links]
    if len(names) != len(set(names)) or not names:
        raise ValueError("link names must be non-empty and unique")
    if len(joints) != len(links) - 1:
        raise ValueError("joints must form one tree (N links require N-1 joints)")
    if any(joint.get("snap_contact", False) for joint in joints):
        raise ValueError(
            "snap_contact is forbidden: reconstruct every link in one observed q0 frame; "
            "a contact gap is a reconstruction failure, not permission to move a subtree"
        )
    children = {joint["child"] for joint in joints}
    roots = set(names) - children
    if len(roots) != 1 or any(joint[side] not in names for joint in joints for side in ("parent", "child")):
        raise ValueError("manifest must describe one rooted tree")
    root = roots.pop()

    output.mkdir(parents=True, exist_ok=True)
    visual_dir, collision_dir = output / "visual_meshes", output / "collision_meshes"
    visual_dir.mkdir(exist_ok=True)
    collision_dir.mkdir(exist_ok=True)
    visuals: dict[str, trimesh.Trimesh] = {}
    collisions: dict[str, trimesh.Trimesh] = {}
    collision_metadata: dict[str, dict] = {}
    source_vertices: dict[str, np.ndarray] = {}
    link_specs = {link["name"]: link for link in links}
    measurement_used = False
    for link in links:
        name = link["name"]
        visual_source = (ROOT / link["visual"]).resolve()
        visual = load_mesh(visual_source)
        if "measured_obb_m" in link:
            collision = measured_obb(visual, link["measured_obb_m"])
            measurement_used = True
            collision_metadata[name] = {"backend": "measured_obb", "dimensions_m": link["measured_obb_m"]}
        elif "auto_shell_source" in link:
            collision, collision_metadata[name] = automatic_collision(
                load_mesh((ROOT / link["auto_shell_source"]).resolve())
            )
        else:
            collision = load_mesh((ROOT / link["collision"]).resolve())
            collision_metadata[name] = {"backend": "provided_collision"}
        if not collision.is_volume:
            raise ValueError(f"collision mesh is not a positive watertight solid: {name}")
        visuals[name], collisions[name] = visual, collision
        source_vertices[name] = visual.vertices.copy()
    if measurement_used != (mode == "hitl_measurement_assisted"):
        raise ValueError("measured OBB use and hitl_measurement_assisted mode must agree")

    joint_data = []
    link_frame_world = {root: np.zeros(3)}
    unresolved = list(joints)
    while unresolved:
        progress = False
        for joint in unresolved[:]:
            if joint["parent"] not in link_frame_world:
                continue
            parent, child = joint["parent"], joint["child"]
            origin_mode = joint.get("origin", "contact")
            if origin_mode == "contact":
                origin_world, contact_gap = contact_origin(visuals[parent], visuals[child])
            else:
                origin_world = np.asarray(origin_mode, dtype=float)
                contact_gap = None
            axis_mode = joint.get("axis")
            if axis_mode == "child_pca_middle":
                _center, axes, _observed = robust_pca(visuals[child])
                axis = axes[:, 1]
            else:
                axis = np.asarray(axis_mode, dtype=float)
            axis /= np.linalg.norm(axis)
            joint_data.append({
                **joint, "origin_world": origin_world, "axis_vector": axis,
                "contact_gap_m": contact_gap,
            })
            link_frame_world[child] = origin_world
            unresolved.remove(joint)
            progress = True
        if not progress:
            raise ValueError("joint graph is cyclic or disconnected")

    q0_vertex_motion_mm = max(
        float(np.linalg.norm(visuals[name].vertices - source_vertices[name], axis=1).max())
        for name in names
    ) * 1000
    if q0_vertex_motion_mm > 1e-9:
        raise RuntimeError(f"compiler changed observed q0 visual vertices by {q0_vertex_motion_mm} mm")

    for name in names:
        visuals[name].export(visual_dir / f"{name}.obj")
        collisions[name].export(collision_dir / f"{name}.obj")

    volumes = {name: abs(float(mesh.volume)) * 1e6 for name, mesh in collisions.items()}
    proxy_errors = {name: surface_p95_mm(visuals[name], collisions[name]) for name in names}
    if max(proxy_errors.values()) > 15.0:
        raise ValueError(f"visual/collision surface p95 exceeds 15 mm: {proxy_errors}")
    total_mass = manifest.get("total_mass_kg")
    explicit_masses = all("mass_kg" in link for link in links)
    if explicit_masses:
        masses = {link["name"]: float(link["mass_kg"]) for link in links}
        mass_source = "measured_per_link"
    elif total_mass is not None:
        denominator = sum(volumes.values())
        masses = {name: float(total_mass) * volume / denominator for name, volume in volumes.items()}
        mass_source = "measured_total_volume_split_provisional"
    else:
        masses = {name: volume / 1e6 * 1000.0 for name, volume in volumes.items()}
        mass_source = "default_density_1000kg_m3_provisional"

    robot = ET.Element("robot", name=manifest["robot_name"])
    palette = ["0.15 0.55 0.85 1", "0.20 0.75 0.48 1", "0.95 0.55 0.12 1", "0.70 0.35 0.85 1"]
    for index, link in enumerate(links):
        name = link["name"]
        frame = link_frame_world[name]
        element = ET.SubElement(robot, "link", name=name)
        visual = ET.SubElement(element, "visual")
        ET.SubElement(visual, "origin", xyz=vec(-frame), rpy="0 0 0")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=f"visual_meshes/{name}.obj")
        material = ET.SubElement(visual, "material", name=f"{name}_color")
        ET.SubElement(material, "color", rgba=link.get("color", palette[index % len(palette)]))
        collision = ET.SubElement(element, "collision")
        ET.SubElement(collision, "origin", xyz=vec(-frame), rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=f"collision_meshes/{name}.obj")
        com, inertia = inertia_for_mass(collisions[name], masses[name])
        inertial = ET.SubElement(element, "inertial")
        ET.SubElement(inertial, "origin", xyz=vec(com - frame), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{masses[name]:.9g}")
        ET.SubElement(inertial, "inertia", ixx=f"{inertia[0,0]:.9g}", ixy=f"{inertia[0,1]:.9g}", ixz=f"{inertia[0,2]:.9g}", iyy=f"{inertia[1,1]:.9g}", iyz=f"{inertia[1,2]:.9g}", izz=f"{inertia[2,2]:.9g}")
    for joint in joint_data:
        kind = joint["type"]
        if kind not in {"revolute", "prismatic", "fixed"}:
            raise ValueError(f"unsupported joint type: {kind}")
        element = ET.SubElement(robot, "joint", name=joint["name"], type=kind)
        ET.SubElement(element, "parent", link=joint["parent"])
        ET.SubElement(element, "child", link=joint["child"])
        local_origin = joint["origin_world"] - link_frame_world[joint["parent"]]
        ET.SubElement(element, "origin", xyz=vec(local_origin), rpy="0 0 0")
        if kind != "fixed":
            ET.SubElement(element, "axis", xyz=vec(joint["axis_vector"]))
            lower, upper = joint.get("limits", [-1.570796, 1.570796])
            ET.SubElement(element, "limit", lower=str(lower), upper=str(upper), effort="10", velocity="2")
            ET.SubElement(element, "dynamics", damping="0.05")
    urdf = output / f"{manifest['robot_name']}.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, encoding="unicode", xml_declaration=True)
    ET.parse(urdf)

    gt = manifest.get("evaluation_only", {}).get("volume_gt_cm3")
    total_volume = sum(volumes.values())
    limits_measured = all(joint["type"] == "fixed" or joint.get("limits_source") == "measured" for joint in joints)
    physics = "PASS" if mass_source == "measured_per_link" and limits_measured else "PROVISIONAL"
    audit = {
        "status": "PASS" if physics == "PASS" else "GEOMETRY_PASS_PHYSICS_PROVISIONAL",
        "mode": mode,
        "robot_name": manifest["robot_name"],
        "root_link": root,
        "q0_source_vertex_motion_max_mm": q0_vertex_motion_mm,
        "links": [{
            "name": name, "visual_source": link_specs[name]["visual"],
            "collision_source": (
                "measured_obb" if "measured_obb_m" in link_specs[name]
                else link_specs[name].get("auto_shell_source", link_specs[name].get("collision"))
            ),
            "collision_backend": collision_metadata[name],
            "collision_watertight": bool(collisions[name].is_watertight),
            "visual_collision_surface_p95_mm": round(proxy_errors[name], 3),
            "volume_cm3": round(volumes[name], 3), "mass_kg": round(masses[name], 6),
        } for name in names],
        "joints": [{
            "name": joint["name"], "parent": joint["parent"], "child": joint["child"],
            "type": joint["type"], "relation": joint.get("relation", "free"),
            "origin_source": "nearest_surface_contact" if joint.get("origin", "contact") == "contact" else "human_selected",
            "axis_source": joint.get("axis_source", joint.get("axis")),
            "contact_gap_mm": None if joint["contact_gap_m"] is None else round(joint["contact_gap_m"] * 1000, 3),
            "assembly_edit": "none_common_q0_preserved",
        } for joint in joint_data],
        "total_volume_cm3": round(total_volume, 3),
        "evaluation_volume_gt_cm3": gt,
        "evaluation_volume_ape_percent": None if gt is None else round(abs(total_volume - gt) / gt * 100, 3),
        "mass_source": mass_source,
        "joint_limits_measured": limits_measured,
        "urdf": urdf.name,
        "provenance": manifest.get("provenance", {}),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    shutil.copy2(manifest_path, output / "input_manifest.json")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_asset(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
