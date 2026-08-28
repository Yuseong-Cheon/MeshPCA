#!/usr/bin/env python3
"""Disconnected watertight link PLY -> RORA joint HITL -> verified URDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from rora_prior_split_ply import compile_urdf, review_joints, save_preview
from separate_ply_links import part_stats
from static_rora_metric_parts import (
    MAX_PENETRATING_SAMPLE_FRACTION,
    _angles,
    _descendants,
    _physical_properties,
    _rotation,
    _save_joint_previews,
    _scene,
    _signed_distance,
)


def load_links(path: Path) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError("input must be one watertight multi-body triangle PLY")
    links = list(mesh.split(only_watertight=True))
    if len(links) < 2 or len(links) != mesh.body_count:
        raise ValueError("input must contain at least two disconnected watertight bodies")
    if any(not link.is_winding_consistent or link.body_count != 1 for link in links):
        raise ValueError("every connected component must be one winding-consistent body")
    return mesh, links


def infer_tree(links: list[trimesh.Trimesh]) -> tuple[list[trimesh.Trimesh], dict[str, str], np.ndarray]:
    count = len(links)
    distance = np.full((count, count), np.inf)
    for first in range(count):
        distance[first, first] = 0.0
        tree = cKDTree(links[first].vertices)
        for second in range(first + 1, count):
            value = float(tree.query(links[second].vertices, workers=-1)[0].min())
            # scipy dense MST rounds edges below 1e-8 to zero (meaning "no edge").
            distance[first, second] = distance[second, first] = max(value, 1e-7)
    adjacency = csgraph.minimum_spanning_tree(distance).toarray()
    adjacency = (adjacency + adjacency.T) > 0
    root = int(np.argmax([abs(float(link.volume)) for link in links]))
    order, parent, pending = [], {root: None}, [root]
    while pending:
        current = pending.pop(0)
        order.append(current)
        neighbors = sorted(np.flatnonzero(adjacency[current]), key=lambda item: distance[current, item])
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor not in parent:
                parent[neighbor] = current
                pending.append(neighbor)
    names = (["base", "moving_link"] if count == 2 else
             ["base", "support", "head"] if count == 3 else
             ["base", *[f"link_{number}" for number in range(2, count + 1)]])
    position = {component: number for number, component in enumerate(order)}
    parents = {
        names[position[component]]: names[position[owner]]
        for component, owner in parent.items() if owner is not None
    }
    return [links[index] for index in order], parents, distance[np.ix_(order, order)]


def sweep(parts: dict[str, trimesh.Trimesh], names: list[str], parents: dict[str, str],
          saved_joints: list[dict], samples: int, clearance_mm: float,
          ignore_radius_mm: float) -> tuple[dict, float]:
    root = next(name for name in names if name not in parents)
    descendants = _descendants(names, parents, root)
    records, maximum = {}, 0.0
    for number, saved in enumerate(saved_joints):
        kind = saved["type"].lower()
        if kind not in {"revolute", "prismatic"}:
            raise ValueError("joint must be revolute or prismatic")
        parent = names[saved["parent"] - 1]
        child = names[saved["child"] - 1]
        origin = np.asarray(saved["axis"]["origin"], dtype=float)
        axis = np.asarray(saved["axis"]["n"], dtype=float)
        axis /= np.linalg.norm(axis)
        moving = trimesh.util.concatenate([parts[name] for name in descendants[child]])
        fixed = trimesh.util.concatenate([parts[name] for name in names
                                          if name not in descendants[child]])
        points = moving.sample(samples, seed=number)
        points = points[np.linalg.norm(points - origin, axis=1) > ignore_radius_mm / 1000.0]
        scene, angles = _scene(fixed), []
        unit = "mm" if kind == "prismatic" else "deg"
        for angle in _angles(np.asarray(saved[f"limits_{unit}"], dtype=float)):
            moved = (points + axis * angle / 1000.0
                     if kind == "prismatic" else _rotation(points, origin, axis, angle))
            distance = _signed_distance(scene, moved)
            penetration = np.maximum(-distance - clearance_mm / 1000.0, 0.0)
            inside = penetration > 0
            fraction = float(np.mean(inside)) if len(inside) else 0.0
            maximum = max(maximum, fraction)
            angles.append({
                f"position_{unit}": angle,
                "penetrating_sample_fraction": fraction,
                "mean_penetration_mm": (
                    float(np.mean(penetration[inside]) * 1000.0) if inside.any() else 0.0
                ),
            })
        records[f"{parent}--{child}"] = {"angles": angles}
    return records, maximum


def _unapproved_penetration(motion: dict, joints: list[dict]) -> list[float]:
    maxima = [max(angle["penetrating_sample_fraction"] for angle in record["angles"])
              for record in motion.values()]
    return [value for value, joint in zip(maxima, joints)
            if value > MAX_PENETRATING_SAMPLE_FRACTION
            and not joint.get("static_shell_collision_override_confirmed", False)]


def run(args: argparse.Namespace) -> dict:
    assembly, components = load_links(args.input)
    components, parents, distances = infer_tree(components)
    names = list(parents.values()) + list(parents)
    names = list(dict.fromkeys(names))
    root = next(name for name in names if name not in parents)
    names = [root, *[name for name in names if name != root]]
    if len(names) != len(components):
        raise RuntimeError("automatic component tree is inconsistent")

    args.output.mkdir(parents=True, exist_ok=True)
    part_dir = args.output / "metric_parts"
    part_dir.mkdir(exist_ok=True)
    parts = {}
    for name, component in zip(names, components):
        path = part_dir / f"{name}_metric_watertight.ply"
        component.export(path)
        parts[name] = trimesh.load_mesh(path, process=False)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    tree_hash = hashlib.sha256((source_hash + json.dumps(parents, sort_keys=True)).encode()).hexdigest()
    joints = review_joints(
        parts, names, parents, args.output / "joint_selection.json", tree_hash,
        args.review_joints, assembly,
    )
    motion, maximum = sweep(
        parts, names, parents, joints, args.samples, args.clearance_mm, args.ignore_radius_mm
    )
    unapproved = _unapproved_penetration(motion, joints)
    if unapproved:
        raise RuntimeError(
            f"joint sweep rejected: {100 * max(unapproved):.3f}% > 0.5%; review axis/limits"
        )

    stats = {name: part_stats(part) for name, part in parts.items()}
    source_volume = abs(float(assembly.volume)) * 1e6
    part_sum = sum(item["volume_cm3"] for item in stats.values())
    closure_fraction = abs(part_sum - source_volume) / max(source_volume, 1e-12)
    displacement = float(cKDTree(np.vstack([part.vertices for part in parts.values()]))
                         .query(assembly.vertices, workers=-1)[0].max() * 1000.0)
    if closure_fraction >= 1e-6 or displacement >= 1e-4:
        raise RuntimeError("component extraction changed source volume or surface vertices")
    audit = {
        "status": ("PASS_WITH_HITL_CONFIRMED_STATIC_SHELL_OVERLAP"
                   if maximum > MAX_PENETRATING_SAMPLE_FRACTION else "PASS"),
        "method": "disconnected_watertight_components_plus_RORA_joint_HITL",
        "source": str(args.input.resolve()),
        "source_sha256": source_hash,
        "input_contract": "one disconnected watertight body per physical link",
        "tree": {"root": root, "parents": parents,
                 "source": "largest-volume root plus minimum-surface-distance MST",
                 "component_distance_mm": (distances * 1000.0).tolist()},
        "joint_source": "RORA_HITL_metric_interface_candidates_and_human_limits",
        "joints": joints,
        "virtual_articulation": motion,
        "maximum_penetrating_sample_fraction": maximum,
        "static_shell_collision_overrides": [
            f"{names[item['parent'] - 1]}--{names[item['child'] - 1]}"
            for item in joints if item.get("static_shell_collision_override_confirmed", False)
        ],
        "parts": stats,
        "physical_properties": {
            name: _physical_properties(part, None) for name, part in parts.items()
        },
        "source_volume_cm3": source_volume,
        "part_volume_sum_cm3": part_sum,
        "volume_closure_error_fraction": closure_fraction,
        "max_retained_source_vertex_displacement_mm": displacement,
        "limitations": (
            "component ownership is supplied by disconnected PLY topology; joint limits are HITL, "
            "because one static PLY does not uniquely determine motion range"
        ),
    }
    (args.output / "component_tree.json").write_text(
        json.dumps(audit["tree"], indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    save_preview(list(parts.values()), [[], *[[parts[name]] for name in names]], names,
                 part_dir / "final_part_preview.png")
    joint_map = {
        names[item["child"] - 1]: {
            "parent": names[item["parent"] - 1],
            "origin": np.asarray(item["axis"]["origin"]),
            "axis": np.asarray(item["axis"]["n"]),
        }
        for item in joints
    }
    _save_joint_previews(parts, names, joint_map, part_dir)
    compile_urdf(args.output, names, joints, args.robot_name, provenance={
        "part_assignment": "input_PLY_disconnected_watertight_components",
        "metric_geometry": "upstream_thickness_corrected_components_unchanged",
    })
    return audit


def _self_test() -> None:
    links = [
        trimesh.creation.box([1.0, 1.0, 0.2], transform=trimesh.transformations.translation_matrix([0, 0, 0])),
        trimesh.creation.box([0.2, 0.2, 1.0], transform=trimesh.transformations.translation_matrix([0, 0, 0.6])),
        trimesh.creation.box([0.8, 0.2, 0.2], transform=trimesh.transformations.translation_matrix([0.4, 0, 1.1])),
    ]
    ordered, parents, _distance = infer_tree(links)
    assert len(ordered) == 3 and parents == {"support": "base", "head": "support"}
    touching = [
        trimesh.creation.box([.4, .4, .2], transform=trimesh.transformations.translation_matrix([0, 0, .2 * index]))
        for index in range(3)
    ]
    ordered, parents, _distance = infer_tree(touching)
    assert len(ordered) == 3 and parents == {"support": "base", "head": "support"}
    collision = {"base--moving_link": {"angles": [
        {"penetrating_sample_fraction": MAX_PENETRATING_SAMPLE_FRACTION + 0.01}
    ]}}
    assert _unapproved_penetration(collision, [{}])
    assert not _unapproved_penetration(
        collision, [{"static_shell_collision_override_confirmed": True}]
    )
    print("articulate_disconnected_ply self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-joints", action="store_true")
    parser.add_argument("--robot-name", default="articulated_object")
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--clearance-mm", type=float, default=0.25)
    parser.add_argument("--ignore-radius-mm", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.input or not args.output:
        parser.error("input and --output are required")
    audit = run(args)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
