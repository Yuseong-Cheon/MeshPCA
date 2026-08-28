#!/usr/bin/env python3
"""Select a metric PLY link partition using static RORA priors and hinge physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree

from rora_prior_split_ply import forbidden_config_paths, save_preview
from separate_ply_links import (
    _maximum_flow_side,
    boundary_loops,
    close_labeled_parts,
    exact_seed_component_cut,
    geometry_plane_candidates,
    graph_data,
    graph_labels,
    load_mesh,
    part_stats,
    seed_faces,
    validate_config,
)

MAX_PENETRATING_SAMPLE_FRACTION = 0.005
MAX_CAP_EXPOSURE_FRACTION = 0.01
EDIT_JOINTS_EXIT_CODE = 42


class EditJointsRequested(RuntimeError):
    pass


def _joint_map(path: Path, names: list[str], parents: dict[str, str]) -> dict[str, dict]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved_names = saved.get("links", names)
    if saved_names != names:
        raise ValueError("joint link order does not match the partition config")
    result = {}
    for item in saved["joints"]:
        parent = names[item["parent"] - 1] if isinstance(item["parent"], int) else item["parent"]
        child = names[item["child"] - 1] if isinstance(item["child"], int) else item["child"]
        kind = item.get("type", "").lower()
        if parents.get(child) != parent or kind not in {"revolute", "prismatic"}:
            raise ValueError(f"joint {parent}->{child} must be revolute or prismatic")
        axis = np.asarray(item["axis"]["n"], dtype=float)
        origin = np.asarray(item["axis"]["origin"], dtype=float)
        unit = "mm" if kind == "prismatic" else "deg"
        limits = np.asarray(item[f"limits_{unit}"], dtype=float)
        if (axis.shape != (3,) or origin.shape != (3,) or limits.shape != (2,)
                or not np.isfinite(np.r_[axis, origin, limits]).all()
                or np.linalg.norm(axis) < 1e-9 or limits[0] >= limits[1]):
            raise ValueError(f"invalid axis or limits for {parent}->{child}")
        result[child] = {
            "parent": parent,
            "type": kind,
            "origin": origin,
            "axis": axis / np.linalg.norm(axis),
            "limits": limits,
            "unit": unit,
            "static_shell_collision_override_confirmed": bool(
                item.get("static_shell_collision_override_confirmed", False)
            ),
        }
    if set(result) != set(parents):
        raise ValueError("joint file must contain exactly one movable joint per tree edge")
    return result


def _descendants(names: list[str], parents: dict[str, str], root: str) -> dict[str, set[str]]:
    result = {name: {name} for name in names}
    for _ in names:
        for child, parent in parents.items():
            result[parent] |= result[child]
    if result[root] != set(names):
        raise ValueError("link tree is disconnected")
    return result


def _rotation(points: np.ndarray, origin: np.ndarray, axis: np.ndarray,
              degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cross = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    matrix = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    return (points - origin) @ matrix.T + origin


def _joint_motion(points: np.ndarray, joint: dict, value: float) -> np.ndarray:
    if joint["type"] == "prismatic":
        return points + joint["axis"] * value / 1000.0
    return _rotation(points, joint["origin"], joint["axis"], value)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.stem}.", delete=False) as stream:
        json.dump(value, stream, indent=2); stream.write("\n"); stream.flush()
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent,
                                     prefix=f".{path.stem}.", delete=False) as stream:
        stream.write(value); stream.flush(); temporary = Path(stream.name)
    temporary.replace(path)


def _joint_matrix(joint: dict, value: float) -> np.ndarray:
    matrix = np.eye(4)
    if joint["type"] == "prismatic":
        matrix[:3, 3] = joint["axis"] * value / 1000.0
        return matrix
    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(value), joint["axis"], point=joint["origin"]
    )
    return rotation


def _pose_transforms(names: list[str], parents: dict[str, str], root: str,
                     joints: dict[str, dict], values: dict[str, float]) -> dict[str, np.ndarray]:
    """Forward kinematics for joints expressed in the captured world frame."""
    transforms = {root: np.eye(4)}
    pending = set(names) - {root}
    while pending:
        ready = [child for child in pending if parents[child] in transforms]
        if not ready:
            raise ValueError("link tree is disconnected")
        for child in ready:
            transforms[child] = (transforms[parents[child]]
                                 @ _joint_matrix(joints[child], values.get(child, 0.0)))
            pending.remove(child)
    return transforms


def _posed_parts(parts: dict[str, trimesh.Trimesh], transforms: dict[str, np.ndarray]
                  ) -> dict[str, trimesh.Trimesh]:
    result = {}
    for name, part in parts.items():
        moved = part.copy(); moved.apply_transform(transforms[name]); result[name] = moved
    return result


def _automatic_contact_faces(part: trimesh.Trimesh, cap_faces: np.ndarray,
                             radius_mm: float = 20.0) -> np.ndarray:
    """Use the cap boundary as the only seed; no contact-region painting."""
    cap_faces = np.unique(np.asarray(cap_faces, dtype=np.int64))
    if not len(cap_faces):
        return cap_faces
    adjacency = part.face_adjacency
    centers = part.triangles_center
    weights = np.linalg.norm(centers[adjacency[:, 0]] - centers[adjacency[:, 1]], axis=1)
    graph = coo_matrix((np.r_[weights, weights] + 1e-12,
                        (np.r_[adjacency[:, 0], adjacency[:, 1]],
                         np.r_[adjacency[:, 1], adjacency[:, 0]])),
                       shape=(len(part.faces), len(part.faces))).tocsr()
    distance = csgraph.dijkstra(graph, directed=False, indices=cap_faces, min_only=True,
                                limit=radius_mm / 1000.0)
    return np.flatnonzero(np.isfinite(distance))


def _cap_exposure_four_views(parts: dict[str, trimesh.Trimesh], cap_face_ids: dict[str, list[int]],
                             origin: np.ndarray, axis: np.ndarray) -> tuple[float, dict[str, list[int]]]:
    """Area-weighted cap visibility from four orthogonal views around a joint."""
    import open3d as o3d

    axis = axis / np.linalg.norm(axis)
    helper = np.asarray([1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    first = np.cross(axis, helper); first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    directions = (first, -first, second, -second)
    diagonal = max(float(np.linalg.norm(np.ptp(np.vstack([p.bounds for p in parts.values()]), axis=0))), 1e-6)
    scene = o3d.t.geometry.RaycastingScene()
    geometry = {}
    for name, part in parts.items():
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(part.vertices),
            o3d.utility.Vector3iVector(part.faces.astype(np.int32)),
        )
        geometry[int(scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy)))] = name
    visible = {name: set() for name in cap_face_ids}
    for name, ids_value in cap_face_ids.items():
        ids = np.asarray(ids_value, dtype=np.int64)
        if not len(ids):
            continue
        centers = parts[name].triangles_center[ids]
        geometry_id = next(key for key, value in geometry.items() if value == name)
        for direction in directions:
            ray_origins = centers + direction * (2.0 * diagonal)
            rays = np.c_[ray_origins, np.repeat((-direction)[None, :], len(ids), axis=0)]
            hits = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
            hit_geometry = hits["geometry_ids"].numpy().astype(np.int64)
            hit_faces = hits["primitive_ids"].numpy().astype(np.int64)
            seen = (hit_geometry == geometry_id) & (hit_faces == ids)
            visible[name].update(ids[seen].tolist())
    total_area = sum(float(parts[name].area_faces[np.asarray(ids, dtype=np.int64)].sum())
                     for name, ids in cap_face_ids.items() if ids)
    exposed_area = sum(float(parts[name].area_faces[np.asarray(sorted(ids), dtype=np.int64)].sum())
                       for name, ids in visible.items() if ids)
    return exposed_area / max(total_area, 1e-12), {name: sorted(ids) for name, ids in visible.items()}


def _angles(limits: np.ndarray) -> list[float]:
    return sorted(set(
        round(float(value), 6)
        for value in np.linspace(float(limits[0]), float(limits[1]), 9)
        if abs(value) >= 1.0
    ))


def _pose_quality(parts: dict[str, trimesh.Trimesh], names: list[str],
                  parents: dict[str, str], root: str, joints: dict[str, dict],
                  values: dict[str, float], cap_faces: dict[str, dict[str, list[int]]],
                  contact_faces: dict[str, dict[str, np.ndarray]],
                  clearance_mm: float = 0.25, samples: int = 3000) -> dict:
    transforms = _pose_transforms(names, parents, root, joints, values)
    posed = _posed_parts(parts, transforms)
    pair_records, maximum_penetration, penetrating_points = {}, 0.0, []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            adjacent_child = (second if parents.get(second) == first else
                              first if parents.get(first) == second else None)
            fractions = []
            for moving, fixed in ((first, second), (second, first)):
                face_ids = np.arange(len(parts[moving].faces))
                if adjacent_child:
                    excluded = contact_faces.get(adjacent_child, {}).get(moving, np.empty(0, int))
                    face_ids = np.setdiff1d(face_ids, excluded, assume_unique=False)
                if len(face_ids) > samples:
                    face_ids = face_ids[np.linspace(0, len(face_ids) - 1, samples, dtype=int)]
                points = posed[moving].triangles_center[face_ids]
                distance = _signed_distance(_scene(posed[fixed]), points)
                penetrating_points.extend(points[distance < -clearance_mm / 1000.0][:500].tolist())
                fractions.append(float(np.mean(distance < -clearance_mm / 1000.0))
                                 if len(distance) else 0.0)
            fraction = max(fractions, default=0.0)
            maximum_penetration = max(maximum_penetration, fraction)
            pair_records[f"{first}--{second}"] = {
                "adjacent_expected_contact_excluded": adjacent_child is not None,
                "penetrating_sample_fraction": fraction,
            }

    exposures, maximum_exposure, visible_caps = {}, 0.0, {}
    for child, by_part in cap_faces.items():
        parent = parents[child]
        parent_transform = transforms[parent]
        origin = trimesh.transform_points([joints[child]["origin"]], parent_transform)[0]
        axis = parent_transform[:3, :3] @ joints[child]["axis"]
        fraction, visible = _cap_exposure_four_views(
            {parent: posed[parent], child: posed[child]}, by_part, origin, axis
        )
        area_groups = [parts[name].area_faces[np.asarray(ids, dtype=np.int64)]
                       for name, ids in by_part.items() if ids]
        areas = np.concatenate(area_groups) if area_groups else np.empty(0)
        one_face_fraction = float(np.median(areas) / max(areas.sum(), 1e-12)) if len(areas) else 0.0
        allowed = max(MAX_CAP_EXPOSURE_FRACTION, one_face_fraction)
        maximum_exposure = max(maximum_exposure, fraction)
        visible_caps[child] = visible
        exposures[child] = {"fraction": fraction, "allowed_fraction": allowed,
                            "pass": fraction <= allowed}
    passed = (maximum_penetration <= MAX_PENETRATING_SAMPLE_FRACTION
              and all(item["pass"] for item in exposures.values()))
    return {
        "pass": passed,
        "maximum_penetrating_sample_fraction": maximum_penetration,
        "pairs": pair_records,
        "cap_exposure": exposures,
        "maximum_cap_exposure_fraction": maximum_exposure,
        "visible_cap_faces": visible_caps,
        "penetrating_points": penetrating_points,
    }


def _automatic_natural_limits(parts: dict[str, trimesh.Trimesh], names: list[str],
                              parents: dict[str, str], root: str, joints: dict[str, dict],
                              cap_faces: dict[str, dict[str, list[int]]],
                              contact_faces: dict[str, dict[str, np.ndarray]],
                              clearance_mm: float = 0.25, samples: int = 3000) -> tuple[dict, dict]:
    """Clamp each joint to the contiguous passing interval containing captured q0."""
    limited = {child: {**joint, "limits": joint["limits"].copy()}
               for child, joint in joints.items()}
    audit = {}
    for child, joint in limited.items():
        low, high = map(float, joint["limits"])
        positions = sorted(set(np.linspace(low, high, 9).tolist() + [0.0]))
        cache = {}

        def quality(value: float) -> dict:
            key = round(float(value), 9)
            if key not in cache:
                cache[key] = _pose_quality(
                    parts, names, parents, root, limited, {child: value}, cap_faces,
                    contact_faces, clearance_mm, samples,
                )
            return cache[key]

        passed = [quality(value)["pass"] for value in positions]
        zero = positions.index(0.0)
        if not passed[zero]:
            q0 = {key: value for key, value in quality(0.0).items()
                  if key not in {"visible_cap_faces", "penetrating_points"}}
            audit[child] = {"status": "UNOBSERVED_JOINT_INTERIOR_AT_Q0",
                            "original_limits": [low, high], "q0": q0}
            continue
        left = right = zero
        while left and passed[left - 1]: left -= 1
        while right + 1 < len(positions) and passed[right + 1]: right += 1

        def boundary(good: float, bad: float) -> float:
            for _ in range(8):
                middle = (good + bad) / 2.0
                if quality(middle)["pass"]: good = middle
                else: bad = middle
            return good

        limited_low = positions[left]
        limited_high = positions[right]
        if left > 0: limited_low = boundary(limited_low, positions[left - 1])
        if right + 1 < len(positions): limited_high = boundary(limited_high, positions[right + 1])
        if limited_high - limited_low < max(0.1, (high - low) / 1000.0):
            audit[child] = {"status": "UNOBSERVED_JOINT_INTERIOR_NO_USABLE_RANGE",
                            "original_limits": [low, high],
                            "natural_limits": [limited_low, limited_high]}
            continue
        joint["limits"] = np.asarray([limited_low, limited_high])
        audit[child] = {
            "status": "PASS" if limited_low == low and limited_high == high else "RANGE_LIMITED",
            "original_limits": [low, high], "natural_limits": [limited_low, limited_high],
            "sample_positions": positions,
            "sample_pass": passed,
        }
    return limited, audit


def _scene(mesh: trimesh.Trimesh):
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _signed_distance(scene, points: np.ndarray) -> np.ndarray:
    import open3d as o3d

    tensor = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return scene.compute_signed_distance(tensor).numpy()


def _candidate_score(
    mesh: trimesh.Trimesh,
    labels: np.ndarray | None,
    parts: dict[str, trimesh.Trimesh],
    interfaces: dict,
    names: list[str],
    parents: dict[str, str],
    root: str,
    joints: dict[str, dict],
    samples: int,
    clearance_mm: float,
    ignore_radius_mm: float,
    seed: int,
    plane_interfaces: dict[str, dict] | None = None,
) -> tuple[float, dict]:
    descendants = _descendants(names, parents, root)
    diagonal = float(np.linalg.norm(mesh.extents))
    records = {}
    collision_total = seam_total = 0.0
    pair = labels[mesh.face_adjacency] if labels is not None else None
    for number, (child, joint) in enumerate(joints.items()):
        moving_names = descendants[child]
        moving = trimesh.util.concatenate([parts[name] for name in names if name in moving_names])
        fixed = trimesh.util.concatenate([parts[name] for name in names if name not in moving_names])
        key = f"{joint['parent']}--{child}"
        effective_ignore_radius_mm = ignore_radius_mm
        if plane_interfaces is not None:
            plane_model = plane_interfaces[key]
            plane_distance_mm = abs(float(
                joint["origin"] @ np.asarray(plane_model["normal"], dtype=float)
                - plane_model["offset_m"]
            )) * 1000.0
            cap_radius_mm = np.sqrt(float(plane_model["cap_area_mm2"]) / np.pi)
            effective_ignore_radius_mm = max(
                ignore_radius_mm, plane_distance_mm + 1.5 * cap_radius_mm
            )
        moving_points = moving.sample(samples, seed=seed + number)
        keep = (np.linalg.norm(moving_points - joint["origin"], axis=1)
                > effective_ignore_radius_mm / 1000.0)
        moving_points = moving_points[keep]
        fixed_scene = _scene(fixed)
        angle_records, collision = [], []
        for angle in _angles(joint["limits"]):
            distance = _signed_distance(
                fixed_scene,
                _joint_motion(moving_points, joint, angle),
            )
            penetration = np.maximum(-distance - clearance_mm / 1000.0, 0.0)
            inside = penetration > 0
            fraction = float(np.mean(inside)) if len(inside) else 0.0
            mean_mm = float(np.mean(penetration[inside]) * 1000) if inside.any() else 0.0
            collision.append(fraction * 100.0 + mean_mm / max(diagonal * 1000, 1e-9))
            angle_records.append({
                f"position_{joint['unit']}": angle,
                "penetrating_sample_fraction": fraction,
                "mean_penetration_mm": mean_mm,
            })

        if plane_interfaces is not None:
            interface = plane_model
            plane_normal = np.asarray(interface["normal"], dtype=float)
            origin_plane_offset = abs(float(joint["origin"] @ plane_normal - interface["offset_m"])) / diagonal
            cap_ratio = float(interface["cap_area_mm2"]) / (diagonal * 1000.0) ** 2
            residual_ratio = 0.0
        else:
            parent_index, child_index = names.index(joint["parent"]), names.index(child)
            boundary = (((pair[:, 0] == parent_index) & (pair[:, 1] == child_index))
                        | ((pair[:, 0] == child_index) & (pair[:, 1] == parent_index)))
            boundary_points = mesh.vertices[np.unique(mesh.face_adjacency_edges[boundary])]
            center = boundary_points.mean(axis=0)
            _values, axes = np.linalg.eigh(np.cov(boundary_points - center, rowvar=False))
            plane_normal = axes[:, 0]
            origin_plane_offset = abs(float((joint["origin"] - center) @ plane_normal)) / diagonal
            interface = interfaces.get(key, interfaces.get(f"{child}--{joint['parent']}"))
            if interface is None:
                raise RuntimeError(f"missing interface {key}")
            cap_ratio = float(interface["shared_cap_area_mm2"]) / (diagonal * 1000.0) ** 2
            residual_ratio = float(interface["boundary_best_fit_residual_mm"]) / (diagonal * 1000.0)
        alignment = abs(float(plane_normal @ joint["axis"]))
        alignment_penalty = alignment if joint["type"] == "revolute" else 1.0 - alignment
        seam = alignment_penalty + origin_plane_offset + 10.0 * cap_ratio + residual_ratio
        joint_collision = float(np.mean(collision)) if collision else float("inf")
        collision_total += joint_collision
        seam_total += seam
        records[f"{joint['parent']}--{child}"] = {
            "axis_normal_absolute_dot": alignment_penalty,
            "joint_type": joint["type"],
            "motion_unit": joint["unit"],
            "axis_origin_plane_offset_mm": origin_plane_offset * diagonal * 1000.0,
            "normalized_interface_score": seam,
            "collision_score": joint_collision,
            "samples_after_joint_exclusion": int(len(moving_points)),
            "effective_joint_exclusion_radius_mm": effective_ignore_radius_mm,
            "angles": angle_records,
        }
    total = collision_total + seam_total
    return total, {
        "total_score_lower_is_better": total,
        "collision_score": collision_total,
        "interface_axis_score": seam_total,
        "joints": records,
    }


def _exact_parts(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    graph_boundaries: dict,
    active_masks: dict[str, np.ndarray],
    names: list[str],
    parents: dict[str, str],
    order: list[str],
    seeds: dict[str, np.ndarray],
    joints: dict[str, dict],
    thickness_mm: dict[str, float],
    samples: int,
    clearance_mm: float,
    ignore_radius_mm: float,
    seed: int,
) -> tuple[dict[str, trimesh.Trimesh], dict, dict, np.ndarray]:
    """Make the simplest revolute-compatible planar partition for one graph candidate."""
    centers, areas = mesh.triangles_center, mesh.area_faces
    diagonal = float(np.linalg.norm(mesh.extents))
    working, outputs, selected_planes = mesh, {}, {}
    descendants = _descendants(names, parents, next(name for name in names if name not in parents))
    for joint_number, child in enumerate(order):
        parent, active = parents[child], active_masks[child]
        origin = joints[child]["origin"]
        core_radius = max(
            0.025,
            4.0 * max(float(thickness_mm.get(parent, 0.0)),
                      float(thickness_mm.get(child, 0.0))) / 1000.0,
        )
        filtered_seeds = {}
        for name, values in seeds.items():
            far = values[np.linalg.norm(centers[values] - origin, axis=1) > core_radius]
            if len(far) < 2:
                raise RuntimeError(
                    f"insufficient safe core anchors for {name} outside the joint band"
                )
            filtered_seeds[name] = far
        rest_seeds = np.unique(np.concatenate([
            values for name, values in filtered_seeds.items()
            if name != child and active[values].all()
        ]))
        child_seeds = filtered_seeds[child]
        delta = centers[child_seeds].mean(axis=0) - centers[rest_seeds].mean(axis=0)
        axis = joints[child]["axis"]
        normal = delta - axis * float(delta @ axis)
        geometric = geometry_plane_candidates(
            mesh, labels, names, child, parent, active, seeds
        )
        raw_normals = [("RORA_axis_perpendicular_seed_centroid", normal)] + [
            (proposal["selection"], np.asarray(proposal["normal"], dtype=float))
            for proposal in geometric[:4]
        ] + [("graph_linear_svm", np.asarray(graph_boundaries[child]["normal"], dtype=float))]
        proposals, kept_normals = [], []
        for source, raw_normal in raw_normals:
            candidate_normal = raw_normal - axis * float(raw_normal @ axis)
            if np.linalg.norm(candidate_normal) < 1e-9:
                continue
            candidate_normal /= np.linalg.norm(candidate_normal)
            if (centers[child_seeds].mean(axis=0) @ candidate_normal
                    < centers[rest_seeds].mean(axis=0) @ candidate_normal):
                candidate_normal *= -1
            if any(abs(float(candidate_normal @ saved)) > 0.999 for saved in kept_normals):
                continue
            kept_normals.append(candidate_normal)
            lower = float(np.max(centers[rest_seeds] @ candidate_normal)) + 1e-8
            upper = float(np.min(centers[child_seeds] @ candidate_normal)) - 1e-8
            origin_offset = float(joints[child]["origin"] @ candidate_normal)
            search = 0.030
            start, stop = max(lower, origin_offset - search), min(upper, origin_offset + search)
            if start >= stop:
                continue
            offsets = np.unique(np.r_[
                np.linspace(start, stop, 9), np.clip(origin_offset, start, stop)
            ])
            for offset in offsets:
                predicted = centers @ candidate_normal >= offset
                agreement = float(np.sum(
                    areas[active] * (predicted[active] == (labels[active] == names.index(child)))
                ) / np.sum(areas[active]))
                proposals.append({
                    "selection": f"{source}_axis_anchored_scan",
                    "normal": candidate_normal.tolist(),
                    "offset_m": float(offset),
                    "area_weighted_graph_label_agreement": agreement,
                })

        attempts = []
        for proposal in proposals:
            candidate_normal = np.asarray(proposal["normal"], dtype=float)
            candidate_normal /= np.linalg.norm(candidate_normal)
            try:
                child_part, remainder, child_meta, _rest_meta, reassigned = exact_seed_component_cut(
                    working, candidate_normal, float(proposal["offset_m"]),
                    centers[child_seeds],
                )
                if not _valid_parts({"child": child_part, "remainder": remainder}):
                    raise RuntimeError("plane did not create two watertight single bodies")
                raw_axis_dot = abs(float(candidate_normal @ axis))
                axis_dot = (raw_axis_dot if joints[child]["type"] == "revolute"
                            else 1.0 - raw_axis_dot)
                origin_offset = abs(float(
                    joints[child]["origin"] @ candidate_normal - proposal["offset_m"]
                )) / diagonal
                cap_radius = np.sqrt(float(child_meta["cap_area_mm2"]) / np.pi) / 1000.0
                if origin_offset * diagonal > cap_radius:
                    raise RuntimeError("closed plane is farther from the RORA origin than its cap radius")
                cap_ratio = float(child_meta["cap_area_mm2"]) / (diagonal * 1000.0) ** 2
                agreement = float(proposal["area_weighted_graph_label_agreement"])
                moving = trimesh.util.concatenate([
                    child_part,
                    *[outputs[name] for name in names
                      if name != child and name in descendants[child] and name in outputs],
                ])
                moving_points = moving.sample(min(samples, 4_000), seed=seed + joint_number)
                effective_ignore_radius_mm = max(
                    ignore_radius_mm,
                    origin_offset * diagonal * 1000.0
                    + 1.5 * np.sqrt(float(child_meta["cap_area_mm2"]) / np.pi),
                )
                moving_points = moving_points[
                    np.linalg.norm(moving_points - joints[child]["origin"], axis=1)
                    > effective_ignore_radius_mm / 1000.0
                ]
                fixed_scene = _scene(remainder)
                collision_values, maximum_fraction = [], 0.0
                for angle in _angles(joints[child]["limits"]):
                    distance = _signed_distance(
                        fixed_scene,
                        _joint_motion(moving_points, joints[child], angle),
                    )
                    penetration = np.maximum(-distance - clearance_mm / 1000.0, 0.0)
                    inside = penetration > 0
                    fraction = float(np.mean(inside)) if len(inside) else 0.0
                    maximum_fraction = max(maximum_fraction, fraction)
                    mean_mm = (float(np.mean(penetration[inside]) * 1000)
                               if inside.any() else 0.0)
                    collision_values.append(
                        fraction * 100.0 + mean_mm / max(diagonal * 1000, 1e-9)
                    )
                local_collision = float(np.mean(collision_values)) if collision_values else 0.0
                if maximum_fraction > MAX_PENETRATING_SAMPLE_FRACTION:
                    raise RuntimeError("virtual articulation penetration exceeds 0.5%")
                local_score = (local_collision + axis_dot + origin_offset
                               + 100.0 * cap_ratio + (1.0 - agreement))
                attempts.append((local_score, child_part, remainder, child_meta, reassigned,
                                 proposal, axis_dot, origin_offset, local_collision,
                                 maximum_fraction, effective_ignore_radius_mm))
            except Exception:
                continue
        if not attempts:
            raise RuntimeError(f"no joint-compatible closed plane for {parent}--{child}")
        chosen = min(attempts, key=lambda item: item[0])
        (_score, outputs[child], working, child_meta, reassigned, proposal, axis_dot, offset,
         local_collision, maximum_fraction, effective_ignore_radius_mm) = chosen
        selected_planes[f"{parent}--{child}"] = {
            "selection": proposal["selection"],
            "normal": child_meta["plane_normal"],
            "offset_m": child_meta["plane_offset_m"],
            "cap_area_mm2": child_meta["cap_area_mm2"],
            "cap_faces": child_meta["cap_faces"],
            "axis_normal_absolute_dot": axis_dot,
            "axis_origin_plane_offset_mm": offset * diagonal * 1000.0,
            "area_weighted_graph_label_agreement": proposal[
                "area_weighted_graph_label_agreement"
            ],
            "local_virtual_collision_score": local_collision,
            "local_maximum_penetrating_sample_fraction": maximum_fraction,
            "effective_joint_exclusion_radius_mm": effective_ignore_radius_mm,
            "child_is_positive_halfspace": bool(
                np.mean(centers[child_seeds] @ np.asarray(child_meta["plane_normal"]))
                >= float(child_meta["plane_offset_m"])
            ),
            **reassigned,
        }
    root = next(name for name in names if name not in parents)
    outputs[root] = working
    face_labels = np.full(len(mesh.faces), names.index(root), dtype=np.int64)
    remaining = np.ones(len(mesh.faces), dtype=bool)
    for child in order:
        plane = selected_planes[f"{parents[child]}--{child}"]
        positive = centers @ np.asarray(plane["normal"], dtype=float) >= plane["offset_m"]
        child_side = positive if plane["child_is_positive_halfspace"] else ~positive
        assigned = remaining & child_side
        face_labels[assigned] = names.index(child)
        remaining[assigned] = False
    return outputs, selected_planes, {"mode": "exact_joint_planes"}, face_labels


def _valid_parts(parts: dict[str, trimesh.Trimesh]) -> bool:
    return all(
        part.is_watertight and part.is_winding_consistent and part.body_count == 1
        and not np.count_nonzero(np.bincount(part.edges_unique_inverse) != 2)
        for part in parts.values()
    )


def _maximum_penetration(candidate: dict) -> float:
    return max((
        angle["penetrating_sample_fraction"]
        for record in candidate["physics"]["joints"].values()
        for angle in record["angles"]
    ), default=0.0)


def _physics_candidates(candidates: list[dict], joints: dict[str, dict] | None = None
                        ) -> tuple[list[dict], list[dict]]:
    passed, rejected = [], []
    for candidate in candidates:
        maximum = _maximum_penetration(candidate)
        candidate["maximum_penetrating_sample_fraction"] = maximum
        candidate["rank"] = (
            candidate["physics"]["total_score_lower_is_better"],
            candidate["smoothness"],
        )
        reasons = []
        if not candidate["axis_centered_interface_gate"]:
            reasons.append("axis-centered interface gate")
        for key, record in candidate["physics"]["joints"].items():
            child = key.split("--", 1)[1]
            joint_maximum = max((angle["penetrating_sample_fraction"]
                                 for angle in record["angles"]), default=0.0)
            if (joint_maximum > MAX_PENETRATING_SAMPLE_FRACTION
                    and not (joints or {}).get(child, {}).get(
                        "static_shell_collision_override_confirmed", False)):
                reasons.append(f"{key} virtual articulation penetration > 0.5%")
        (rejected if reasons else passed).append({
            **candidate,
            "physics_gate_reasons": reasons,
        })
    return sorted(passed, key=lambda item: item["rank"]), rejected


def _candidate_id(candidate: dict, names: list[str]) -> str:
    geometry = {
        name: {
            "volume": round(abs(float(candidate["parts"][name].volume)), 12),
            "centroid": np.round(candidate["parts"][name].centroid, 9).tolist(),
            "bounds": np.round(candidate["parts"][name].bounds, 9).tolist(),
            "faces": len(candidate["parts"][name].faces),
        }
        for name in names
    }
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:16]


def _cut_review_text(candidate: dict, index: int, count: int) -> str:
    lines = [
        f"FINAL CUT REVIEW  {index + 1}/{count}",
        f"{candidate['family']}  smoothness={candidate['smoothness']:g}",
        f"physics score={candidate['physics']['total_score_lower_is_better']:.5f}  "
        f"max penetration={100 * _maximum_penetration(candidate):.3f}%",
    ]
    for key, record in candidate["physics"]["joints"].items():
        lines.append(
            f"{key}: {record.get('joint_type', 'revolute')}  "
            f"axis offset={record['axis_origin_plane_offset_mm']:.2f} mm  "
            f"interface score={record['normalized_interface_score']:.4f}"
        )
    if candidate.get("physics_gate_reasons"):
        lines.append("CANNOT SAVE: " + ", ".join(candidate["physics_gate_reasons"]))
    lines.extend((
        "Colored surfaces are the closed metric links; yellow lines are hinge axes.",
        "Drag=rotate | wheel=zoom | M=motion preview | J=edit hinges | A/S/Enter=approve | N/B | Q",
    ))
    return "\n".join(lines)


def _motion_value(percent: float, limits: np.ndarray) -> float:
    return float(limits[0] + (np.clip(percent, -100.0, 100.0) + 100.0)
                 * (limits[1] - limits[0]) / 200.0)


def _motion_preview(candidate: dict, names: list[str], joints: dict[str, dict],
                    quality_context: dict | None = None) -> None:
    """Preview one real joint at a time without rebuilding the split."""
    import vedo

    children = list(joints)
    parents = {child: joint["parent"] for child, joint in joints.items()}
    root = next(name for name in names if name not in parents)
    descendants = _descendants(names, parents, root)
    colors = ("dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    diagonal = max(np.linalg.norm(np.ptp(
        np.vstack([part.bounds for part in candidate["parts"].values()]), axis=0
    )), 1e-9)
    radius = 0.14 * diagonal
    origins = np.vstack([joint["origin"] for joint in joints.values()])
    plotter = vedo.Plotter(title="RORA joint motion preview", bg="white",
                           pos=(80, 60), size=(1200, 850))
    actors, original_points = {}, {}
    for number, name in enumerate(names):
        part = candidate["parts"][name]
        near = np.min(
            np.linalg.norm(part.triangles_center[:, None, :] - origins[None, :, :], axis=2),
            axis=1,
        ) <= radius
        stride = max(1, int(np.ceil(len(part.faces) / 45_000)))
        visible = np.unique(np.r_[np.flatnonzero(near), np.arange(0, len(part.faces), stride)])
        display = part.submesh([visible], append=True, repair=False)
        actor = vedo.Mesh([display.vertices, display.faces]).c(
            colors[number % len(colors)]).alpha(0.82)
        actors[name], original_points[name] = actor, display.vertices.copy()
    state = {"index": 0}
    status = vedo.Text2D("", pos="top-left", c="black", s=0.75)
    axis_actors, diagnostic_actors = [], []

    def q0_percent(joint: dict) -> float:
        low, high = joint["limits"]
        return float(np.clip(200.0 * (0.0 - low) / (high - low) - 100.0, -100.0, 100.0))

    def refresh(_widget=None, _event=None) -> None:
        child = children[state["index"]]
        joint = joints[child]
        value = _motion_value(float(slider.value), joint["limits"])
        moving = descendants[child]
        for name, actor in actors.items():
            points = original_points[name]
            actor.points = _joint_motion(points, joint, value) if name in moving else points
        plotter.remove(*axis_actors)
        axis_actors.clear()
        origin, axis = joint["origin"], joint["axis"]
        axis_actors.extend((
            vedo.Line(origin - axis * radius, origin + axis * radius).c("yellow").lw(6),
            vedo.Sphere(origin, r=0.012 * diagonal).c("yellow"),
        ))
        plotter.add(*axis_actors)
        diagnostic = ""
        if quality_context:
            for actor in diagnostic_actors:
                plotter.remove(actor)
            diagnostic_actors.clear()
            values = {child: value}
            transforms = _pose_transforms(
                names, parents, root, joints, values
            )
            posed = _posed_parts(candidate["parts"], transforms)
            quality = _pose_quality(
                candidate["parts"], names, parents, root, joints, values,
                quality_context["cap_faces"], quality_context["contact_faces"],
                quality_context["clearance_mm"], quality_context["samples"],
            )
            points = np.asarray(quality["penetrating_points"], dtype=float)
            if len(points):
                diagnostic_actors.append(vedo.Points(points, r=8).c("red"))
            for cap_child, visible in quality["visible_cap_faces"].items():
                for part_name, face_ids in visible.items():
                    if face_ids:
                        diagnostic_actors.append(vedo.Mesh([
                            posed[part_name].vertices,
                            posed[part_name].faces[np.asarray(face_ids, dtype=np.int64)],
                        ]).c("magenta").alpha(.95))
            if diagnostic_actors:
                plotter.add(*diagnostic_actors)
            diagnostic = (
                f"\ncap exposed={100 * quality['maximum_cap_exposure_fraction']:.3f}% | "
                f"penetration={100 * quality['maximum_penetrating_sample_fraction']:.3f}% | "
                f"{'PASS' if quality['pass'] else 'FAIL'}\n"
                "magenta=exposed paired cap | red=penetration outside contact"
            )
        status.text(
            f"MOTION PREVIEW  {state['index'] + 1}/{len(children)}  "
            f"{joint['parent']} -> {child}\n"
            f"q={value:.2f} {joint['unit']}  limits=[{joint['limits'][0]:.2f}, "
            f"{joint['limits'][1]:.2f}] {joint['unit']}"
            f"{diagnostic}\n"
            "Slider=move | N=next joint | R=q0 | Q/Close=return to Final Cut"
        )
        plotter.render()

    def next_joint(_widget=None, _event=None) -> None:
        state["index"] = (state["index"] + 1) % len(children)
        slider.value = q0_percent(joints[children[state["index"]]])
        refresh()

    def reset(_widget=None, _event=None) -> None:
        slider.value = q0_percent(joints[children[state["index"]]])
        refresh()

    def close(_widget=None, _event=None) -> None:
        plotter.close()

    def keypress(event) -> None:
        key = getattr(event, "keypress", "")
        if key in ("n", "N", "Right"):
            next_joint()
        elif key in ("r", "R", "0"):
            reset()
        elif key in ("q", "Q", "Esc", "Escape", "\x1b"):
            close()

    first = joints[children[0]]
    slider = plotter.add_slider(refresh, -100.0, 100.0, value=q0_percent(first),
                                title="joint range [%]", pos=((0.15, .14), (.85, .14)),
                                delayed=bool(quality_context))
    def add_button(callback, label, position):
        try:
            return plotter.add_button(callback, states=(label,), pos=position, size=14)
        except AttributeError as error:
            if "AddActor2D" not in str(error):
                raise
            button = vedo.Button(callback, states=(label,), pos=position, size=14)
            plotter.renderer.AddViewProp(button.actor)
            button.function_id = button.actor.AddObserver("PickEvent", button.function)
            plotter.buttons.append(button)
            return button

    add_button(next_joint, "Next joint", (.25, .06))
    add_button(reset, "Reset q0", (.50, .06))
    add_button(close, "Back to Final Cut", (.78, .06))
    def preset(percent):
        def callback(_widget=None, _event=None):
            slider.value = (q0_percent(joints[children[state["index"]]])
                            if percent is None else percent)
            refresh()
        return callback
    for percent, label, x in ((-100, "lower", .14), (-50, "25%", .27),
                              (None, "q0", .40), (50, "75%", .53),
                              (100, "upper", .66)):
        add_button(preset(percent), label, (x, .22))
    plotter.add_callback("KeyPress", keypress)
    plotter.show(*actors.values(), status, axes=1, interactive=True)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent,
                                     prefix=f".{path.stem}.", delete=False) as stream:
        np.save(stream, value); stream.flush(); temporary = Path(stream.name)
    temporary.replace(path)


def _articulation_interfaces(source: trimesh.Trimesh, labels_path: Path,
                             saved_parts: dict[str, trimesh.Trimesh], names: list[str],
                             parents: dict[str, str], masks_dir: Path,
                             contact_radius_mm: float) -> tuple[dict, dict, dict]:
    labels = np.load(labels_path)
    if labels.shape != (len(source.faces),) or not np.isin(labels, range(len(names))).all():
        raise ValueError("saved face labels do not match the source PLY")
    rebuilt, interfaces = close_labeled_parts(source, labels, names)
    for name in names:
        if (rebuilt[name].vertices.shape != saved_parts[name].vertices.shape
                or rebuilt[name].faces.shape != saved_parts[name].faces.shape
                or not np.allclose(rebuilt[name].vertices, saved_parts[name].vertices,
                                   atol=5e-8, rtol=0.0)
                or not np.array_equal(rebuilt[name].faces, saved_parts[name].faces)):
            raise RuntimeError(f"saved {name} PLY is not the exact source-label partition")
    cap_faces, contact_faces, references = {}, {}, {}
    for child, parent in parents.items():
        key = f"{parent}--{child}"
        interface = interfaces.get(key, interfaces.get(f"{child}--{parent}"))
        if interface is None:
            raise RuntimeError(f"missing paired interface {key}")
        by_part = {name: list(map(int, interface["cap_face_ids"][name]))
                   for name in (parent, child)}
        cap_faces[child], contact_faces[child], references[child] = by_part, {}, {}
        for name, ids in by_part.items():
            selected = _automatic_contact_faces(saved_parts[name], np.asarray(ids), contact_radius_mm)
            path = masks_dir / f"{parent}_to_{child}_{name}_contact_faces.npy"
            _atomic_npy(path, selected)
            contact_faces[child][name] = selected
            references[child][name] = str(path.resolve())
    return cap_faces, contact_faces, references


def _save_limited_joints(path: Path, names: list[str], limited: dict[str, dict],
                         contact_references: dict, binding: dict, range_audit: dict,
                         contact_radius_mm: float) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    for item in saved["joints"]:
        child = names[item["child"] - 1] if isinstance(item["child"], int) else item["child"]
        joint = limited[child]
        unit = "mm" if joint["type"] == "prismatic" else "deg"
        item[f"limits_{unit}"] = [float(value) for value in joint["limits"]]
        item["contact_region_masks"] = contact_references[child]
        item["editing_state"] = {
            **item.get("editing_state", {}),
            "origin_step_mm": float(item.get("editing_state", {}).get("origin_step_mm", 1.0)),
            "axis_parameterization": "yaw_pitch",
            "contact_radius_mm": contact_radius_mm,
        }
        item["natural_range"] = range_audit[child]
    saved["geometry_binding"] = binding
    _atomic_json(path, saved)


def _patch_vedo_button_api(vedo) -> None:
    """Bridge Vedo releases whose renderer removed AddActor2D."""
    original = vedo.Plotter.add_button
    if getattr(original, "_rora_compatible", False):
        return

    def compatible(plotter, callback, *args, **kwargs):
        try:
            return original(plotter, callback, *args, **kwargs)
        except AttributeError as error:
            if "AddActor2D" not in str(error):
                raise
            button = vedo.Button(callback, *args, **kwargs)
            plotter.renderer.AddViewProp(button.actor)
            button.function_id = button.actor.AddObserver("PickEvent", button.function)
            plotter.buttons.append(button)
            return button
    compatible._rora_compatible = True
    vedo.Plotter.add_button = compatible


def _review_saved_parts(parts: dict[str, trimesh.Trimesh], names: list[str],
                        contact_radius_mm: float) -> float:
    import vedo
    _patch_vedo_button_api(vedo)

    colors = ("dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    actors = [vedo.Mesh([parts[name].vertices, parts[name].faces])
              .c(colors[index % len(colors)]).alpha(0.82)
              for index, name in enumerate(names)]
    lines = ["PART REVIEW — saved exact split (read-only)"]
    for name in names:
        part = parts[name]
        lines.append(f"{name}: faces={len(part.faces)} watertight={part.is_watertight} body={part.body_count}")
    lines.append("Contact radius slider uses the exact interface automatically; no painting")
    lines.append("Enter/S=continue to joint edit | Q=cancel")
    plotter = vedo.Plotter(title="RORA parts -> joints -> motion", bg="white",
                           pos=(80, 60), size=(1200, 850))
    status = vedo.Text2D("\n".join(lines), pos="top-left", c="black", s=.72)
    state = {"approved": False, "contact_radius_mm": contact_radius_mm}

    def finish(_widget=None, _event=None): state["approved"] = True; plotter.close()
    def cancel(_widget=None, _event=None): plotter.close()
    def key(event):
        pressed = getattr(event, "keypress", "")
        if pressed in ("Return", "Enter", "s", "S"): finish()
        elif pressed in ("q", "Q", "Escape", "Esc"): cancel()
    plotter.add_button(finish, states=("Continue to joints",), pos=(.65, .06), size=14)
    plotter.add_button(cancel, states=("Cancel",), pos=(.86, .06), size=14)
    def radius(widget, _event=None):
        state["contact_radius_mm"] = float(widget.value)
    plotter.add_slider(radius, 2.0, 50.0, value=contact_radius_mm,
                       title="automatic contact radius [mm]",
                       pos=((.18, .14), (.82, .14)), delayed=True)
    plotter.add_callback("KeyPress", key)
    plotter.show(*actors, status, axes=1, interactive=True)
    if not state["approved"]:
        raise RuntimeError("articulation editor cancelled at part review")
    return state["contact_radius_mm"]


def _save_motion_renders(parts: dict[str, trimesh.Trimesh], names: list[str],
                         parents: dict[str, str], root: str, joints: dict[str, dict],
                         output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output.mkdir(parents=True, exist_ok=True)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    for child, joint in joints.items():
        low, high = map(float, joint["limits"])
        for pose_name, value in (("lower", low), ("mid", (low + high) / 2.0),
                                 ("q0", 0.0), ("upper", high)):
            transforms = _pose_transforms(names, parents, root, joints, {child: value})
            posed = _posed_parts(parts, transforms)
            figure = plt.figure(figsize=(7, 7)); axis = figure.add_subplot(111, projection="3d")
            bounds = np.asarray([part.bounds for part in posed.values()])
            lower, upper = bounds[:, 0].min(0), bounds[:, 1].max(0)
            center, radius = (lower + upper) / 2, max(float((upper - lower).max()) / 2, 1e-6)
            for index, name in enumerate(names):
                triangles = posed[name].triangles
                stride = max(1, int(np.ceil(len(triangles) / 35_000)))
                actor = Poly3DCollection(triangles[::stride], alpha=.82, linewidth=0)
                actor.set_facecolor(colors[index % len(colors)]); axis.add_collection3d(actor)
            axis.set(xlim=(center[0]-radius, center[0]+radius),
                     ylim=(center[1]-radius, center[1]+radius),
                     zlim=(center[2]-radius, center[2]+radius), title=f"{child} {pose_name}: {value:.2f}")
            axis.set_box_aspect((1, 1, 1)); figure.tight_layout()
            figure.savefig(output / f"{child}_{pose_name}.png", dpi=160); plt.close(figure)


def _run_articulation_editor(args: argparse.Namespace) -> dict:
    from rora_prior_split_ply import review_joints

    config = json.loads(args.config.read_text(encoding="utf-8"))
    names, parents = list(config["seeds"]), config["parents"]
    root = config["root"]
    source = load_mesh(args.input)
    parts = {name: load_mesh(args.motion_preview_parts / f"{name}_metric_watertight.ply")
             for name in names}
    part_paths = {name: args.motion_preview_parts / f"{name}_metric_watertight.ply"
                  for name in names}
    hashes_before = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in part_paths.items()}
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    binding = {"source_sha256": source_hash, "part_sha256": hashes_before,
               "face_labels_sha256": hashlib.sha256(args.face_labels.read_bytes()).hexdigest()}
    selection_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    draft_path = args.output / "joint_selection.draft.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8")) if draft_path.is_file() else {}
    if (draft.get("geometry_binding") != binding
            and draft.get("source_sha256") != selection_hash):
        _atomic_bytes(draft_path, args.joints.read_bytes())
    contact_radius_mm = _review_saved_parts(parts, names, args.contact_radius_mm)
    review_joints(parts, names, parents, draft_path,
                  selection_hash, True, source)
    joints = _joint_map(draft_path, names, parents)
    cap_faces, contact_faces, references = _articulation_interfaces(
        source, args.face_labels, parts, names, parents,
        args.output / "contact_regions", contact_radius_mm,
    )
    limited, range_audit = _automatic_natural_limits(
        parts, names, parents, root, joints, cap_faces, contact_faces,
        args.clearance_mm, args.samples,
    )
    blocked = {child: item for child, item in range_audit.items()
               if item["status"].startswith("UNOBSERVED_JOINT_INTERIOR")}
    if blocked:
        _atomic_json(args.output / "articulation_audit.json", {
            "status": "UNOBSERVED_JOINT_INTERIOR", "geometry_binding": binding,
            "ply_unchanged": True, "range_audit": range_audit,
        })
        raise RuntimeError(f"no natural observed range for {sorted(blocked)}; joint limits not saved")
    _save_limited_joints(draft_path, names, limited, references, binding,
                         range_audit, contact_radius_mm)
    positions = {child: np.linspace(*joint["limits"], 5) for child, joint in limited.items()}
    combined = []
    import itertools
    for values in itertools.product(*positions.values()):
        pose = dict(zip(positions, map(float, values)))
        quality = _pose_quality(parts, names, parents, root, limited, pose, cap_faces,
                                contact_faces, args.clearance_mm, args.samples)
        combined.append({"positions": pose, **{key: value for key, value in quality.items()
                                                if key not in {"visible_cap_faces",
                                                               "penetrating_points"}}})
    hashes_after = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in part_paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("articulation editor modified an exact-part PLY")
    audit = {
        "status": "PASS" if all(item["pass"] for item in combined) else "COMBINED_SWEEP_FAILED",
        "geometry_binding": binding,
        "ply_unchanged": True,
        "contact_radius_mm": contact_radius_mm,
        "cap_exposure_policy": "four_views_max_1_percent_or_one_median_cap_face",
        "range_audit": range_audit,
        "combined_5x5": combined,
    }
    _atomic_json(args.output / "articulation_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("combined 5x5 articulation sweep failed; edited joints remain in draft")
    draft_path.replace(args.joints)
    _save_motion_renders(parts, names, parents, root, limited, args.output / "motion_renders")
    _motion_preview({"parts": parts}, names, limited, {
        "cap_faces": cap_faces, "contact_faces": contact_faces,
        "clearance_mm": args.clearance_mm, "samples": args.samples,
    })
    return audit


def _plane_face_labels(mesh: trimesh.Trimesh, names: list[str], parents: dict[str, str],
                       order: list[str], planes: dict[str, dict]) -> np.ndarray:
    labels = np.full(len(mesh.faces), names.index(next(name for name in names if name not in parents)))
    remaining = np.ones(len(mesh.faces), dtype=bool)
    for child in order:
        plane = planes[child]
        positive = mesh.triangles_center @ plane["normal"] >= plane["offset_m"]
        child_side = positive if plane["child_is_positive_halfspace"] else ~positive
        selected = remaining & child_side
        labels[selected] = names.index(child)
        remaining[selected] = False
    return labels


def _review_cut_candidates(options: list[dict], names: list[str], joints: dict[str, dict],
                           edit_joints=None) -> dict:
    import vedo

    colors = ("dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    index = 0
    while True:
        candidate, action = options[index], {"value": None}
        plotter = vedo.Plotter(
            title=f"RORA HITL final cut {index + 1}/{len(options)}", bg="white",
            pos=(80, 60), size=(1200, 850),
        )
        actors = []
        origins = np.vstack([joint["origin"] for joint in joints.values()])
        diagonal = max(np.linalg.norm(np.ptp(
            np.vstack([part.bounds for part in candidate["parts"].values()]), axis=0
        )), 1e-9)
        radius = 0.14 * diagonal
        for part_number, name in enumerate(names):
            part = candidate["parts"][name]
            near = np.min(
                np.linalg.norm(part.triangles_center[:, None, :] - origins[None, :, :], axis=2),
                axis=1,
            ) <= radius
            stride = max(1, int(np.ceil(len(part.faces) / 45_000)))
            visible = np.unique(np.r_[np.flatnonzero(near), np.arange(0, len(part.faces), stride)])
            display = part.submesh([visible], append=True, repair=False)
            actors.append(
                vedo.Mesh([display.vertices, display.faces])
                .c(colors[part_number % len(colors)]).alpha(0.82)
            )
        for joint in joints.values():
            origin, axis = joint["origin"], joint["axis"]
            actors.extend((
                vedo.Line(origin - axis * radius, origin + axis * radius).c("yellow").lw(6),
                vedo.Sphere(origin, r=0.012 * diagonal).c("yellow"),
            ))
        status = vedo.Text2D(_cut_review_text(candidate, index, len(options)),
                             pos="top-left", c="black", s=0.72)

        def finish(value):
            def callback(_widget=None, _event=None):
                action["value"] = value
                plotter.close()
            return callback

        def keypress(event):
            key = getattr(event, "keypress", "")
            if key in ("a", "A", "s", "S", "Return", "Enter"):
                finish("approve")()
            elif key in ("n", "N", "Right"):
                finish("next")()
            elif key in ("b", "B", "Left"):
                finish("back")()
            elif key in ("j", "J"):
                finish("edit_joints")()
            elif key in ("m", "M"):
                finish("motion_preview")()
            elif key in ("q", "Q", "Esc", "Escape", "\x1b"):
                finish("cancel")()

        def add_button(callback, label, position):
            try:
                return plotter.add_button(
                    callback, states=(label,), pos=position, size=15
                )
            except AttributeError as error:
                if "AddActor2D" not in str(error):
                    raise
                button = vedo.Button(callback, states=(label,), pos=position, size=15)
                plotter.renderer.AddViewProp(button.actor)
                button.function_id = button.actor.AddObserver("PickEvent", button.function)
                plotter.buttons.append(button)
                return button

        plotter.add_callback("KeyPress", keypress)
        add_button(finish("back"), "< Back", (0.16, 0.06))
        add_button(finish("motion_preview"), "Motion preview", (0.31, 0.06))
        add_button(finish("edit_joints"), "Edit hinges", (0.46, 0.06))
        add_button(finish("approve"), "Approve & save", (0.62, 0.06))
        add_button(finish("next"), "Next >", (0.82, 0.06))
        plotter.show(*actors, status, axes=1, interactive=True)
        if action["value"] == "approve":
            return candidate
        if action["value"] == "edit_joints":
            if edit_joints is None:
                raise EditJointsRequested()
            joints.clear()
            joints.update(edit_joints(candidate, options))
            continue
        if action["value"] == "motion_preview":
            _motion_preview(candidate, names, joints)
            continue
        if action["value"] == "next":
            index = (index + 1) % len(options)
        elif action["value"] == "back":
            index = (index - 1) % len(options)
        else:
            raise RuntimeError("final cut review cancelled; no metric parts were overwritten")


def _manual_plane_candidate(
    mesh: trimesh.Trimesh,
    names: list[str],
    parents: dict[str, str],
    order: list[str],
    seeds: dict[str, np.ndarray],
    joints: dict[str, dict],
    root: str,
    planes: dict[str, dict],
    base: dict,
    samples: int,
    clearance_mm: float,
    ignore_radius_mm: float,
    seed: int,
) -> dict:
    """Apply HITL planes to original triangles, then run the same hard physics gates."""
    working, outputs, interfaces = mesh, {}, {}
    normalized = {}
    for child in order:
        saved = planes[child]
        normal = np.asarray(saved["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        axis_dot = abs(float(normal @ joints[child]["axis"]))
        if axis_dot > 0.15:
            raise RuntimeError(
                f"selected axis is not in the STEP 1B plane at {parents[child]}--{child} "
                f"(|normal dot axis|={axis_dot:.3f} > 0.15)"
            )
        offset = float(saved["offset_m"])
        if not bool(saved["child_is_positive_halfspace"]):
            normal, offset = -normal, -offset
        child_points = mesh.triangles_center[seeds[child]]
        if np.mean(child_points @ normal) < offset:
            normal, offset = -normal, -offset
        child_part, working, child_meta, _rest_meta, reassigned = exact_seed_component_cut(
            working, normal, offset, child_points
        )
        if not _valid_parts({"child": child_part, "remainder": working}):
            raise RuntimeError(f"manual plane did not create two closed bodies for {parents[child]}--{child}")
        outputs[child] = child_part
        key = f"{parents[child]}--{child}"
        interfaces[key] = {
            "selection": "HITL_manual_axis_compatible_plane",
            "normal": child_meta["plane_normal"],
            "offset_m": child_meta["plane_offset_m"],
            "cap_area_mm2": child_meta["cap_area_mm2"],
            "cap_faces": child_meta["cap_faces"],
            "child_is_positive_halfspace": True,
            **reassigned,
        }
        normalized[child] = {
            "normal": np.asarray(child_meta["plane_normal"], dtype=float),
            "offset_m": float(child_meta["plane_offset_m"]),
            "child_is_positive_halfspace": True,
        }
    outputs[root] = working
    if not _valid_parts(outputs):
        raise RuntimeError("manual planes did not produce one watertight body per link")
    labels = _plane_face_labels(mesh, names, parents, order, normalized)
    score, physics = _candidate_score(
        mesh, None, outputs, {}, names, parents, root, joints, samples,
        clearance_mm, ignore_radius_mm, seed, plane_interfaces=interfaces,
    )
    axis_centered = all(
        abs(float(joints[child]["origin"] @ np.asarray(interfaces[
            f"{parents[child]}--{child}"
        ]["normal"]) - interfaces[f"{parents[child]}--{child}"]["offset_m"]))
        <= np.sqrt(float(interfaces[f"{parents[child]}--{child}"]["cap_area_mm2"]) / np.pi)
        / 1000.0
        for child in order
    )
    candidate = {
        "family": "HITL_manual_revolute_planes",
        "smoothness": base["smoothness"],
        "labels": labels,
        "parts": outputs,
        "interfaces": interfaces,
        "graph_boundaries": base.get("graph_boundaries", {}),
        "joint_continuation": {},
        "axis_centered_interface_gate": axis_centered,
        "physics": physics,
        "manual_planes": {
            child: {
                "normal": normalized[child]["normal"].tolist(),
                "offset_m": normalized[child]["offset_m"],
                "child_is_positive_halfspace": True,
            }
            for child in order
        },
    }
    passed, rejected = _physics_candidates([candidate], joints)
    return (passed or rejected)[0]


def _manual_face_candidate(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    names: list[str],
    parents: dict[str, str],
    joints: dict[str, dict],
    root: str,
    base: dict,
    samples: int,
    clearance_mm: float,
    ignore_radius_mm: float,
    seed: int,
) -> dict:
    if labels.shape != (len(mesh.faces),) or not np.isin(labels, range(len(names))).all():
        raise ValueError("HITL face labels do not match the original PLY")
    parts, interfaces = close_labeled_parts(mesh, labels, names)
    expected = {frozenset((parent, child)) for child, parent in parents.items()}
    actual = {frozenset(key.split("--")) for key in interfaces}
    if actual != expected or not _valid_parts(parts):
        raise RuntimeError("HITL face labels did not produce the configured closed link tree")
    score, physics = _candidate_score(
        mesh, labels, parts, interfaces, names, parents, root, joints,
        samples, clearance_mm, ignore_radius_mm, seed,
    )
    axis_centered = all(
        physics["joints"][f"{joint['parent']}--{child}"][
            "axis_origin_plane_offset_mm"
        ] <= np.sqrt(float(interfaces[f"{joint['parent']}--{child}"][
            "shared_cap_area_mm2"
        ]) / np.pi)
        for child, joint in joints.items()
    )
    return {
        "family": "HITL_corrected_original_face_ownership",
        "smoothness": base["smoothness"],
        "labels": labels,
        "parts": parts,
        "interfaces": interfaces,
        "graph_boundaries": base.get("graph_boundaries", {}),
        "joint_continuation": {},
        "axis_centered_interface_gate": axis_centered,
        "physics": physics,
        "manual_face_labels": True,
    }


def _select_candidate(passing: list[dict], names: list[str], joints: dict[str, dict],
                      args: argparse.Namespace, edit_joints=None) -> dict:
    for candidate in passing:
        candidate["candidate_id"] = _candidate_id(candidate, names)
    unique = []
    for candidate in passing:
        if candidate["candidate_id"] not in {saved["candidate_id"] for saved in unique}:
            unique.append(candidate)

    selection_path = (getattr(args, "cut_selection", None)
                      or args.output / "cut_selection.json")
    binding = hashlib.sha256("".join(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.input, args.config, args.joints)
    ).encode()).hexdigest()
    selected, mode, human_approved = None, "automatic_physics_rank", False
    if selection_path.exists() and not getattr(args, "review_cuts", False):
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        if saved.get("input_binding_sha256") == binding:
            selected = next((candidate for candidate in passing
                             if candidate["candidate_id"] == saved.get("candidate_id")), None)
            if selected is not None:
                human_approved = bool(
                    saved.get("human_approved")
                    or saved.get("selection_mode") == "HITL_whole_candidate_approval"
                )
                mode = ("reused_HITL_selection" if human_approved
                        else "reused_automatic_selection")
    if selected is None and getattr(args, "review_cuts", False):
        review_options = unique[:3]
        while selected is None:
            reviewed = _review_cut_candidates(
                review_options, names, joints, edit_joints=edit_joints
            )
            if reviewed.get("physics_gate_reasons"):
                print("[RORA HITL] manual cut rejected: "
                      + ", ".join(reviewed["physics_gate_reasons"]))
                review_options = [reviewed]
                continue
            selected = reviewed
        mode = ("HITL_corrected_face_ownership_approval"
                if selected.get("manual_face_labels")
                else "HITL_manual_plane_approval" if selected.get("manual_planes")
                else "HITL_whole_candidate_approval")
        human_approved = True
    selected = selected or passing[0]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "input_binding_sha256": binding,
        "candidate_id": selected["candidate_id"],
        "family": selected["family"],
        "smoothness": selected["smoothness"],
        "selection_mode": mode,
        "human_approved": human_approved,
        "physics_score": selected["physics"]["total_score_lower_is_better"],
        "maximum_penetrating_sample_fraction": _maximum_penetration(selected),
        "manual_planes": selected.get("manual_planes"),
        "manual_face_labels": bool(selected.get("manual_face_labels")),
    }, indent=2) + "\n", encoding="utf-8")
    selected["selection_mode"] = mode
    selected["human_approved"] = human_approved
    selected["cut_selection_path"] = str(selection_path.resolve())
    return selected


def _refine_joint_continuations(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    names: list[str],
    joints: dict[str, dict],
    thickness_mm: dict[str, float],
    adjacency: np.ndarray,
    affinity: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, dict]:
    """Move interlocking joint faces to the link whose measured slab continues through them."""
    labels = labels.copy()
    centers, normals, areas = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    records = {}
    for child, joint in joints.items():
        parent = joint["parent"]
        if parent not in thickness_mm or child not in thickness_mm:
            records[child] = {"status": "SKIPPED_MISSING_MEASURED_THICKNESS"}
            continue
        parent_index, child_index = names.index(parent), names.index(child)
        origin, axis = joint["origin"], joint["axis"]
        models = {}
        for index, name in ((parent_index, parent), (child_index, child)):
            thickness = float(thickness_mm[name]) / 1000.0
            distance = np.linalg.norm(centers - origin, axis=1)
            body = (labels == index) & (distance > max(0.025, 4.0 * thickness))
            if np.count_nonzero(body) < 100:
                break
            longitudinal = centers[body].mean(axis=0) - origin
            longitudinal -= axis * float(longitudinal @ axis)
            if np.linalg.norm(longitudinal) < 1e-9:
                break
            longitudinal /= np.linalg.norm(longitudinal)
            thickness_axis = np.cross(axis, longitudinal)
            projection = (centers - origin) @ longitudinal
            start = max(0.025, 4.0 * thickness)
            core = ((labels == index) & (projection > start)
                    & (projection < start + 10.0 * thickness))
            if np.count_nonzero(core) < 100:
                break
            width = (centers[core] - origin) @ axis
            width_center = float(np.median(width))
            half_width = float(np.percentile(np.abs(width - width_center), 95))
            if not np.isfinite(half_width) or half_width < 1e-4:
                break
            models[index] = {
                "longitudinal": longitudinal,
                "thickness_axis": thickness_axis,
                "width_center": width_center,
                "thickness_center": float(np.median(
                    (centers[core] - origin) @ thickness_axis
                )),
                "half_width": half_width,
                "half_thickness": thickness / 2.0,
            }
        if len(models) != 2:
            records[child] = {"status": "SKIPPED_INSUFFICIENT_LINK_CORE"}
            continue

        radius = float(np.clip(
            2.5 * min(model["half_width"] for model in models.values()), 0.025, 0.080
        ))
        active = ((np.linalg.norm(centers - origin, axis=1) < radius)
                  & np.isin(labels, [parent_index, child_index]))
        old = np.flatnonzero(active)
        if len(old) < 100:
            records[child] = {"status": "SKIPPED_EMPTY_JOINT_BAND"}
            continue
        inverse = np.full(len(labels), -1, dtype=np.int64)
        inverse[old] = np.arange(len(old))
        internal = active[adjacency].all(axis=1)
        edges = inverse[adjacency[internal]]
        frontier_edges = active[adjacency[:, 0]] ^ active[adjacency[:, 1]]
        frontier = np.unique(np.where(
            active[adjacency[frontier_edges, 0]],
            adjacency[frontier_edges, 0],
            adjacency[frontier_edges, 1],
        ))
        def continuation_cost(index: int) -> np.ndarray:
            model = models[index]
            offset = centers - origin
            width = offset @ axis - model["width_center"]
            thickness = offset @ model["thickness_axis"] - model["thickness_center"]
            side = (
                np.abs(np.abs(width) - model["half_width"]) / model["half_width"]
                + 1.0 - np.abs(normals @ axis)
            )
            broad = (
                np.abs(np.abs(thickness) - model["half_thickness"])
                / model["half_thickness"]
                + 1.0 - np.abs(normals @ model["thickness_axis"])
            )
            return np.minimum(side, broad)

        parent_cost, child_cost = continuation_cost(parent_index), continuation_cost(child_index)
        local_frontier = inverse[frontier]
        anchors = {}
        for index, cost in ((parent_index, parent_cost), (child_index, child_cost)):
            labeled = local_frontier[labels[frontier] == index]
            if len(labeled):
                anchors[index] = labeled
                continue
            projection = (centers - origin) @ models[index]["longitudinal"]
            candidates = np.flatnonzero(active & (projection > 0.35 * radius))
            if len(candidates) < 8:
                break
            count = min(len(candidates), max(8, int(np.sqrt(len(old)))))
            anchors[index] = inverse[candidates[np.argsort(cost[candidates])[:count]]]
        if set(anchors) != {parent_index, child_index}:
            records[child] = {"status": "SKIPPED_UNANCHORED_JOINT_BAND"}
            continue

        denominator = np.maximum(parent_cost + child_cost, 1e-9)
        area_weight = np.clip(areas[old] / np.median(areas[old]), 0.1, 10.0)
        preservation = 0.15
        choose_child = (
            weight * child_cost[old] / denominator[old]
            + preservation * (labels[old] != child_index)
        ) * area_weight
        choose_parent = (
            weight * parent_cost[old] / denominator[old]
            + preservation * (labels[old] == child_index)
        ) * area_weight
        hard = 1e6
        child_frontier, parent_frontier = anchors[child_index], anchors[parent_index]
        choose_child[child_frontier], choose_parent[child_frontier] = 0.0, hard
        choose_child[parent_frontier], choose_parent[parent_frontier] = hard, 0.0
        before = labels.copy()
        child_side = _maximum_flow_side(
            len(old), edges,
            np.sqrt(len(old)) * np.maximum(affinity[internal], 0.02),
            choose_child, choose_parent,
        )
        labels[old] = np.where(child_side, child_index, parent_index)
        removed_islands = 0
        for index, other in ((child_index, parent_index), (parent_index, child_index)):
            faces = np.flatnonzero(labels == index)
            inverse_component = np.full(len(labels), -1, dtype=np.int64)
            inverse_component[faces] = np.arange(len(faces))
            same = (labels[adjacency] == index).all(axis=1)
            if not len(faces):
                continue
            graph = coo_matrix((
                np.ones(np.count_nonzero(same) * 2),
                (np.r_[inverse_component[adjacency[same, 0]], inverse_component[adjacency[same, 1]]],
                 np.r_[inverse_component[adjacency[same, 1]], inverse_component[adjacency[same, 0]]]),
            ), shape=(len(faces), len(faces))).tocsr()
            count, component = csgraph.connected_components(graph, directed=False)
            if count <= 1:
                continue
            owner = int(np.argmax(np.bincount(component, weights=areas[faces])))
            islands = faces[component != owner]
            labels[islands] = other
            removed_islands += len(islands)
        pair = labels[mesh.face_adjacency]
        boundary = (
            ((pair[:, 0] == parent_index) & (pair[:, 1] == child_index))
            | ((pair[:, 0] == child_index) & (pair[:, 1] == parent_index))
        )
        try:
            loops = boundary_loops(mesh.face_adjacency_edges[boundary])
            if len(loops) != 1:
                raise RuntimeError(f"expected one interface loop, got {len(loops)}")
            points = mesh.vertices[loops[0]]
            center = points.mean(axis=0)
            _values, axes = np.linalg.eigh(np.cov(points - center, rowvar=False))
            origin_plane_offset = abs(float((origin - center) @ axes[:, 0]))
            if origin_plane_offset > 0.5 * radius:
                raise RuntimeError(
                    f"interface is {origin_plane_offset * 1000:.3f} mm from the RORA origin plane"
                )
        except RuntimeError as error:
            labels = before
            records[child] = {"status": "REJECTED_TOPOLOGY_GATE", "reason": str(error)}
            continue
        records[child] = {
            "status": "PASS",
            "joint_band_radius_mm": radius * 1000.0,
            "axis_origin_plane_offset_mm": origin_plane_offset * 1000.0,
            "anchor_source": {
                names[index]: ("existing_band_frontier" if index in set(labels[frontier])
                               else "axis_direction_continuation")
                for index in (parent_index, child_index)
            },
            "resolution_normalized_smoothness": float(np.sqrt(len(old))),
            "parent_to_child_faces": int(np.count_nonzero(
                (before == parent_index) & (labels == child_index)
            )),
            "child_to_parent_faces": int(np.count_nonzero(
                (before == child_index) & (labels == parent_index)
            )),
            "interface_edges": int(np.count_nonzero(boundary)),
            "removed_island_faces": int(removed_islands),
        }
    return labels, records


def _physical_properties(part: trimesh.Trimesh, target_thickness_mm: float | None) -> dict:
    return {
        "volume_cm3": abs(float(part.volume)) * 1e6,
        "center_of_mass_m": np.asarray(part.center_mass).tolist(),
        "inertia_per_density_m5": np.asarray(part.moment_inertia).tolist(),
        "measured_target_thickness_mm": target_thickness_mm,
    }


def _save_joint_previews(parts: dict[str, trimesh.Trimesh], names: list[str],
                         joints: dict[str, dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for child, joint in joints.items():
        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        radius = 0.045
        for index, name in enumerate(names):
            part = parts[name]
            near = np.linalg.norm(part.triangles_center - joint["origin"], axis=1) <= radius
            triangles = part.triangles[near]
            if len(triangles):
                stride = max(1, int(np.ceil(len(triangles) / 30_000)))
                actor = Poly3DCollection(triangles[::stride], alpha=0.78, linewidth=0)
                actor.set_facecolor(colors[index % len(colors)])
                axis.add_collection3d(actor)
            axis.scatter([], [], [], color=colors[index % len(colors)], label=name)
        center = joint["origin"]
        axis.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[1] - radius, center[1] + radius),
                 zlim=(center[2] - radius, center[2] + radius))
        axis.view_init(elev=20, azim=-55)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(output / f"joint_{joint['parent']}_to_{child}_preview.png", dpi=180)
        plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    mesh = load_mesh(args.input)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    forbidden = forbidden_config_paths(config)
    if forbidden:
        raise ValueError(f"GT/target-volume fields are forbidden in the partition config: {forbidden}")
    allowed_semantics = {
        "thickness_corrected_material_surface",
        "watertight_surface_needs_thickness_correction",
    }
    if config.get("input_geometry_semantics") not in allowed_semantics:
        raise ValueError(f"config input_geometry_semantics must be one of {sorted(allowed_semantics)}")
    names, parents, order = validate_config(config)
    seeds, seed_audit = seed_faces(mesh, config["seeds"])
    joints = _joint_map(args.joints, names, parents)
    adjacency, affinity, geodesic = graph_data(mesh, args.crease_angle)
    thickness = config.get("measured_thickness_mm", {})
    expected_interfaces = {frozenset((parent, child)) for child, parent in parents.items()}
    candidates, rejected = [], []
    for smoothness in args.smoothness:
        try:
            labels, graph_boundaries, active_masks = graph_labels(
                mesh, names, parents, order, seeds, adjacency, affinity, geodesic, smoothness
            )
            labels, continuation = _refine_joint_continuations(
                mesh, labels, names, joints, thickness, adjacency, affinity,
                args.continuation_weight,
            )
            parts, interfaces = close_labeled_parts(mesh, labels, names)
            actual_interfaces = {frozenset(key.split("--")) for key in interfaces}
            if actual_interfaces != expected_interfaces or not _valid_parts(parts):
                raise RuntimeError("closed links do not match the configured single-body tree")
            score, physics = _candidate_score(
                mesh, labels, parts, interfaces, names, parents, config["root"], joints,
                args.samples, args.clearance_mm, args.ignore_radius_mm, args.seed,
            )
            axis_centered = all(
                physics["joints"][f"{joint['parent']}--{child}"]["axis_origin_plane_offset_mm"]
                <= np.sqrt(float((interfaces.get(f"{joint['parent']}--{child}")
                                  or interfaces[f"{child}--{joint['parent']}"])
                                 ["shared_cap_area_mm2"]) / np.pi)
                for child, joint in joints.items()
            )
            candidates.append({
                "family": "original_face_graph_cut",
                "smoothness": smoothness,
                "labels": labels,
                "parts": parts,
                "interfaces": interfaces,
                "graph_boundaries": graph_boundaries,
                "joint_continuation": continuation,
                "axis_centered_interface_gate": axis_centered,
                "physics": physics,
            })
            exact_parts, exact_interfaces, exact_meta, exact_labels = _exact_parts(
                mesh, labels, graph_boundaries, active_masks, names, parents, order,
                seeds, joints, thickness, args.samples, args.clearance_mm, args.ignore_radius_mm,
                args.seed,
            )
            exact_score, exact_physics = _candidate_score(
                mesh, None, exact_parts, {}, names, parents, config["root"], joints,
                args.samples, args.clearance_mm, args.ignore_radius_mm, args.seed,
                plane_interfaces=exact_interfaces,
            )
            mixed_labels = labels.copy()
            mixed_continuation = {name: dict(record) for name, record in continuation.items()}
            descendants = _descendants(names, parents, config["root"])
            for child, record in mixed_continuation.items():
                if record["status"] == "PASS":
                    continue
                key = f"{parents[child]}--{child}"
                plane = exact_interfaces[key]
                normal = np.asarray(plane["normal"], dtype=float)
                child_side = mesh.triangles_center @ normal >= float(plane["offset_m"])
                if np.mean(mesh.triangles_center[seeds[child]] @ normal) < float(plane["offset_m"]):
                    child_side = ~child_side
                subtree = np.asarray([names.index(name) for name in descendants[child]])
                in_subtree = np.isin(mixed_labels, subtree)
                mixed_labels[child_side & ~in_subtree] = names.index(child)
                mixed_labels[~child_side & in_subtree] = names.index(parents[child])
                record.update({
                    "status": "PLANAR_FACE_FALLBACK",
                    "source": plane["selection"],
                    "axis_origin_plane_offset_mm": plane["axis_origin_plane_offset_mm"],
                    "cap_radius_mm": np.sqrt(float(plane["cap_area_mm2"]) / np.pi),
                })
            mixed_parts, mixed_interfaces = close_labeled_parts(mesh, mixed_labels, names)
            mixed_actual = {frozenset(key.split("--")) for key in mixed_interfaces}
            if mixed_actual == expected_interfaces and _valid_parts(mixed_parts):
                mixed_score, mixed_physics = _candidate_score(
                    mesh, mixed_labels, mixed_parts, mixed_interfaces, names, parents,
                    config["root"], joints, args.samples, args.clearance_mm,
                    args.ignore_radius_mm, args.seed,
                )
                mixed_axis_centered = all(
                    mixed_physics["joints"][f"{joint['parent']}--{child}"][
                        "axis_origin_plane_offset_mm"
                    ] <= np.sqrt(float((mixed_interfaces.get(
                        f"{joint['parent']}--{child}"
                    ) or mixed_interfaces[f"{child}--{joint['parent']}"])[
                        "shared_cap_area_mm2"
                    ]) / np.pi)
                    for child, joint in joints.items()
                )
                candidates.append({
                    "family": "continuation_graph_planar_fallback",
                    "smoothness": smoothness,
                    "labels": mixed_labels,
                    "parts": mixed_parts,
                    "interfaces": mixed_interfaces,
                    "graph_boundaries": graph_boundaries,
                    "joint_continuation": mixed_continuation,
                    "axis_centered_interface_gate": mixed_axis_centered,
                    "physics": mixed_physics,
                })
            candidates.append({
                "family": "exact_revolute_planes",
                "smoothness": smoothness,
                "labels": exact_labels,
                "parts": exact_parts,
                "interfaces": exact_interfaces,
                "graph_boundaries": graph_boundaries,
                "physics": exact_physics,
                "exact_meta": exact_meta,
                "axis_centered_interface_gate": True,
            })
        except Exception as error:
            rejected.append({"smoothness": smoothness, "reason": str(error)})
    face_labels_file = config.get("hitl_face_labels_file")
    if face_labels_file:
        try:
            face_labels_path = Path(face_labels_file)
            if not face_labels_path.is_absolute():
                face_labels_path = args.config.parent / face_labels_path
            base = (min(candidates, key=lambda item: (
                item["physics"]["total_score_lower_is_better"], item["smoothness"]
            )) if candidates else {"smoothness": 0.0, "graph_boundaries": {}})
            candidates.append(_manual_face_candidate(
                mesh, np.load(face_labels_path), names, parents, joints, config["root"],
                base, args.samples, args.clearance_mm, args.ignore_radius_mm, args.seed,
            ))
        except Exception as error:
            rejected.append({"family": "HITL_corrected_original_face_ownership",
                             "reason": str(error)})
    elif config.get("hitl_cut_planes"):
        try:
            base = (min(candidates, key=lambda item: (
                item["physics"]["total_score_lower_is_better"], item["smoothness"]
            )) if candidates else {"smoothness": 0.0, "graph_boundaries": {}})
            candidates.append(_manual_plane_candidate(
                mesh, names, parents, order, seeds, joints, config["root"],
                config["hitl_cut_planes"], base, args.samples, args.clearance_mm,
                args.ignore_radius_mm, args.seed,
            ))
        except Exception as error:
            rejected.append({"family": "HITL_initial_original_PLY_cut_planes",
                             "reason": str(error)})
    if not candidates:
        raise RuntimeError(f"no static topology candidate passed: {rejected}")
    passing, physics_rejected = _physics_candidates(candidates, joints)
    if face_labels_file:
        passing = [item for item in passing if item.get("manual_face_labels")]
    elif config.get("hitl_cut_planes"):
        passing = [item for item in passing if item.get("manual_planes")]
    if not passing:
        summary = [
            {
                "family": item["family"],
                "smoothness": item["smoothness"],
                "maximum_penetrating_sample_fraction": item[
                    "maximum_penetrating_sample_fraction"
                ],
                "reasons": item["physics_gate_reasons"],
            }
            for item in physics_rejected
        ]
        raise RuntimeError(
            f"no candidate passed the hinge physics gates: {summary}; "
            "rerun STEP 1B with --reselect and correct the original-PLY cut planes"
        )

    def edit_final_cut_joints(candidate, options):
        from rora_prior_split_ply import review_joints

        review_joints(
            candidate["parts"], names, parents, args.joints,
            hashlib.sha256((str(args.input.resolve()) + str(args.config.resolve())).encode()).hexdigest(),
            True, mesh,
        )
        updated = _joint_map(args.joints, names, parents)
        for option in options:
            _score, option["physics"] = _candidate_score(
                mesh, option.get("labels"), option["parts"], option.get("interfaces", {}),
                names, parents, config["root"], updated, args.samples,
                args.clearance_mm, args.ignore_radius_mm, args.seed,
            )
            option["axis_centered_interface_gate"] = all(
                option["physics"]["joints"][f"{joint['parent']}--{child}"][
                    "axis_origin_plane_offset_mm"
                ] <= np.sqrt(float(option["interfaces"][f"{joint['parent']}--{child}"][
                    "shared_cap_area_mm2"
                ]) / np.pi)
                for child, joint in updated.items()
            )
            rescored, rejected_after_edit = _physics_candidates([option], updated)
            option["physics_gate_reasons"] = (rescored or rejected_after_edit)[0][
                "physics_gate_reasons"
            ]
        return updated

    selected = _select_candidate(
        passing, names, joints, args, edit_joints=edit_final_cut_joints
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "graph_face_labels.npy", selected["labels"])
    persisted = {}
    for name in names:
        path = args.output / f"{name}_metric_watertight.ply"
        selected["parts"][name].export(path)
        persisted[name] = trimesh.load_mesh(path, process=False)
    if not _valid_parts(persisted):
        raise RuntimeError("persisted metric parts failed the topology gate")

    stats = {name: part_stats(persisted[name]) for name in names}
    source_volume = abs(float(mesh.volume)) * 1e6
    volume_sum = sum(item["volume_cm3"] for item in stats.values())
    closure = abs(volume_sum - source_volume)
    closure_fraction = closure / max(source_volume, 1e-12)
    displacement = float(cKDTree(np.vstack([p.vertices for p in persisted.values()]))
                         .query(mesh.vertices)[0].max() * 1000)
    if closure_fraction >= 1e-6 or displacement >= 1e-4:
        raise RuntimeError(
            "persisted parts changed source volume or exterior vertices: "
            f"closure={closure:.9g} cm3 ({100 * closure_fraction:.9g}%), "
            f"source_vertex_distance={displacement:.9g} mm"
        )

    physical = {
        name: _physical_properties(persisted[name], thickness.get(name)) for name in names
    }
    maximum_penetration_fraction = _maximum_penetration(selected)
    collision_accepted = all(
        max((angle["penetrating_sample_fraction"] for angle in record["angles"]), default=0.0)
        <= MAX_PENETRATING_SAMPLE_FRACTION
        or joints[key.split("--", 1)[1]]["static_shell_collision_override_confirmed"]
        for key, record in selected["physics"]["joints"].items()
    )
    collision_override_confirmed = (
        maximum_penetration_fraction > MAX_PENETRATING_SAMPLE_FRACTION
        and collision_accepted
    )
    external_sweep_status = (
        "PASS" if maximum_penetration_fraction <= MAX_PENETRATING_SAMPLE_FRACTION
        else "HITL_CONFIRMED_STATIC_SHELL_OVERLAP"
        if collision_accepted else "PROVISIONAL_VIRTUAL_ARTICULATION_COLLISION"
    )
    housing_semantics_confirmed = bool(
        (selected.get("manual_planes") or selected.get("manual_face_labels"))
        and selected["human_approved"]
    )
    audit = {
        "status": (
            "PASS" if external_sweep_status in {
                "PASS", "HITL_CONFIRMED_STATIC_SHELL_OVERLAP"
            } and housing_semantics_confirmed
            else "GEOMETRY_AND_EXTERNAL_SWEEP_PASS_SEMANTICS_PROVISIONAL"
            if external_sweep_status == "PASS"
            else "GEOMETRY_PASS_SEMANTICS_PROVISIONAL"
        ),
        "method": "static_RORA_core_tree_cut_axis_thickness_continuation_virtual_articulation",
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_geometry_semantics": config["input_geometry_semantics"],
        "tree": {"root": config["root"], "parents": parents, "leaf_to_root_cut_order": order},
        "selection": {
            "selected_family": selected["family"],
            "selected_smoothness": selected["smoothness"],
            "candidate_id": selected["candidate_id"],
            "selection_mode": selected["selection_mode"],
            "human_approved": selected["human_approved"],
            "cut_selection_file": selected["cut_selection_path"],
            "criterion": "hard axis/topology gates; sweep >0.5% requires explicit HITL static-shell override; GT excluded",
            "parameters": {
                "crease_angle_degrees": args.crease_angle,
                "surface_samples_per_joint": args.samples,
                "penetration_clearance_mm": args.clearance_mm,
                "joint_ignore_radius_mm": args.ignore_radius_mm,
                "joint_continuation_weight": args.continuation_weight,
            },
            "candidates": [
                {"family": item["family"], "smoothness": item["smoothness"],
                 "axis_centered_interface_gate": item["axis_centered_interface_gate"],
                 "maximum_penetrating_sample_fraction": _maximum_penetration(item),
                 "physics_gate": (
                     "PASS" if item["axis_centered_interface_gate"]
                     and _maximum_penetration(item) <= MAX_PENETRATING_SAMPLE_FRACTION
                     else "REJECT"
                 ),
                 "joint_continuation": item.get("joint_continuation", {}),
                 "physics": item["physics"]}
                for item in candidates
            ],
            "rejected": rejected + [
                {"family": item["family"], "smoothness": item["smoothness"],
                 "reason": ", ".join(item["physics_gate_reasons"])}
                for item in physics_rejected
            ],
            "selected_virtual_articulation_status": external_sweep_status,
            "static_shell_collision_override_confirmed": collision_override_confirmed,
            "selected_maximum_penetrating_sample_fraction": maximum_penetration_fraction,
            "joint_housing_semantics_confirmed": housing_semantics_confirmed,
            "joint_housing_semantics_source": (
                "HITL_original_PLY_face_ownership_and_final_partition_approval"
                if housing_semantics_confirmed else None
            ),
            "joint_housing_semantic_status": (
                "HITL_CONFIRMED" if housing_semantics_confirmed
                else "PROVISIONAL_STATIC_INPUT_UNDERDETERMINED"
            ),
        },
        "semantic_prior": {
            "source": config.get("seed_source", "unspecified"),
            "ambiguous_policy": config.get(
                "ambiguous_policy",
                "excluded_from_hard_seeds_then_assigned_by_original_face_graph_and_joint_gates",
            ),
            "seeds": seed_audit,
            "RORA_joint_file": str(args.joints.resolve()),
            "link_labeled_Gaussians_used": False,
            "reason": "the supplied RORA Gaussian PLY has no per-link labels",
        },
        "forbidden_observations_used": {"RGB_D": False, "camera_poses": False, "multi_state": False},
        "graph_boundaries": selected["graph_boundaries"],
        "joint_continuation": selected.get("joint_continuation", {}),
        "interfaces": selected["interfaces"],
        "parts": stats,
        "physical_properties": physical,
        "source_volume_cm3": source_volume,
        "part_volume_sum_cm3": volume_sum,
        "volume_closure_error_cm3": closure,
        "volume_closure_error_fraction": closure_fraction,
        "max_retained_source_vertex_displacement_mm": displacement,
        "physical_limit": (
            "hidden cavities are not observable from these inputs; values describe the supplied "
            + ("thickness-corrected watertight material surface"
               if config["input_geometry_semantics"] == "thickness_corrected_material_surface"
               else "watertight surface with thickness resize still pending")
        ),
    }
    (args.output / "part_separation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    save_preview(
        list(persisted.values()), [[], *[[persisted[name]] for name in names]], names,
        args.output / "final_part_preview.png",
    )
    _save_joint_previews(persisted, names, joints, args.output)
    return audit


def _self_test() -> None:
    axis = np.asarray([0.0, 0.0, 1.0])
    point = _rotation(np.asarray([[1.0, 0.0, 0.0]]), np.zeros(3), axis, 90.0)[0]
    assert np.allclose(point, [0.0, 1.0, 0.0], atol=1e-12)
    moved = _joint_motion(np.asarray([[0.0, 0.0, 0.0]]), {
        "type": "prismatic", "axis": np.asarray([1.0, 0.0, 0.0]),
        "origin": np.zeros(3),
    }, 25.0)[0]
    assert np.allclose(moved, [0.025, 0.0, 0.0])
    assert _angles(np.asarray([-10.0, 40.0])) == [
        -10.0, -3.75, 2.5, 8.75, 15.0, 21.25, 27.5, 33.75, 40.0
    ]
    assert _motion_value(-100.0, np.asarray([-30.0, 45.0])) == -30.0
    assert _motion_value(100.0, np.asarray([-30.0, 45.0])) == 45.0
    assert _motion_value(0.0, np.asarray([-30.0, 50.0])) == 10.0
    assert _descendants(
        ["base", "support", "head"], {"support": "base", "head": "support"}, "base"
    )["support"] == {"support", "head"}
    box = trimesh.creation.box()
    assert _signed_distance(_scene(box), np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))[0] < 0
    labels = _plane_face_labels(
        box, ["root", "child"], {"child": "root"}, ["child"],
        {"child": {"normal": np.asarray([1.0, 0.0, 0.0]), "offset_m": 0.0,
                   "child_is_positive_halfspace": True}},
    )
    assert set(labels.tolist()) == {0, 1}
    sphere = trimesh.creation.icosphere(subdivisions=2)
    sphere_labels = (sphere.triangles_center[:, 2] >= 0).astype(np.int16)
    closed, interfaces = close_labeled_parts(sphere, sphere_labels, ["lower", "upper"])
    cap_ids = interfaces["lower--upper"]["cap_face_ids"]
    assert all(cap_ids[name] for name in cap_ids)
    assert set(cap_ids["lower"]).issubset(
        _automatic_contact_faces(closed["lower"], np.asarray(cap_ids["lower"]), 1000).tolist()
    )
    chain_joints = {
        "support": {"type": "revolute", "origin": np.zeros(3), "axis": axis},
        "head": {"type": "revolute", "origin": np.asarray([1., 0, 0]), "axis": axis},
    }
    transforms = _pose_transforms(
        ["base", "support", "head"], {"support": "base", "head": "support"},
        "base", chain_joints, {"support": 90.0, "head": 0.0},
    )
    assert np.allclose(trimesh.transform_points([[2., 0, 0]], transforms["head"])[0],
                       [0., 2., 0.], atol=1e-12)
    def candidate(score, penetration):
        return {
            "family": "test", "smoothness": score,
            "axis_centered_interface_gate": True,
            "physics": {"total_score_lower_is_better": score, "joints": {
                "a--b": {"angles": [{"penetrating_sample_fraction": penetration}]}
            }},
        }
    passed, rejected = _physics_candidates([
        candidate(5.0, 0.0), candidate(1.0, 0.006), candidate(2.0, 0.005)
    ])
    assert [item["smoothness"] for item in passed] == [2.0, 5.0]
    assert len(rejected) == 1 and "> 0.5%" in rejected[0]["physics_gate_reasons"][0]
    passed, rejected = _physics_candidates(
        [candidate(1.0, 0.006)],
        {"b": {"static_shell_collision_override_confirmed": True}},
    )
    assert len(passed) == 1 and not rejected
    _require_pass({"status": "PASS"})
    try:
        _require_pass({"status": "PROVISIONAL"})
    except SystemExit:
        pass
    else:
        raise AssertionError("delivery gate accepted a provisional result")
    print("static_rora_metric_parts self-test: PASS")


def _require_pass(audit: dict) -> None:
    if audit["status"] != "PASS":
        raise SystemExit(f"delivery gate rejected: {audit['status']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--joints", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoothness", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--crease-angle", type=float, default=18.0)
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--clearance-mm", type=float, default=0.25)
    parser.add_argument("--ignore-radius-mm", type=float, default=20.0)
    parser.add_argument("--continuation-weight", type=float, default=1.0)
    parser.add_argument("--review-cuts", action="store_true",
                        help="compare up to three physics-valid closed partitions and approve one")
    parser.add_argument("--cut-selection", type=Path,
                        help="saved final-cut selection JSON (default: OUTPUT/cut_selection.json)")
    parser.add_argument("--motion-preview-parts", type=Path,
                        help="open saved <link>_metric_watertight.ply files without recomputing cuts")
    parser.add_argument("--edit-articulation", action="store_true",
                        help="review saved exact parts, edit joints, clamp natural ranges, and audit")
    parser.add_argument("--face-labels", type=Path,
                        help="exact source-face labels used to recover paired-cap face IDs")
    parser.add_argument("--contact-radius-mm", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-pass", action="store_true",
                        help="return nonzero unless every geometry, sweep, and semantic gate passes")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.edit_articulation:
        if not all((args.input, args.config, args.joints, args.output,
                    args.motion_preview_parts, args.face_labels)):
            parser.error("--edit-articulation requires input, --config, --joints, --output, "
                         "--motion-preview-parts and --face-labels")
        if args.contact_radius_mm <= 0 or args.samples < 100 or args.clearance_mm < 0:
            parser.error("articulation audit parameters are out of range")
        args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(_run_articulation_editor(args), indent=2))
        return
    if args.motion_preview_parts:
        if not args.config or not args.joints:
            parser.error("--motion-preview-parts requires --config and --joints")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        names, parents = list(config["seeds"]), config["parents"]
        parts = {
            name: load_mesh(args.motion_preview_parts / f"{name}_metric_watertight.ply")
            for name in names
        }
        _motion_preview({"parts": parts}, names, _joint_map(args.joints, names, parents))
        return
    if not all((args.input, args.config, args.joints, args.output)):
        parser.error("input, --config, --joints and --output are required")
    if (min(args.smoothness) <= 0 or args.crease_angle <= 0 or args.samples < 100
            or args.clearance_mm < 0 or args.ignore_radius_mm < 0
            or args.continuation_weight <= 0):
        parser.error("candidate and physics parameters are out of range")
    try:
        audit = run(args)
    except EditJointsRequested:
        raise SystemExit(EDIT_JOINTS_EXIT_CODE)
    print(json.dumps(audit, indent=2))
    if args.require_pass:
        _require_pass(audit)


if __name__ == "__main__":
    main()
