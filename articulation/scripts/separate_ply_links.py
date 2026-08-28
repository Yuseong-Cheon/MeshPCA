#!/usr/bin/env python3
"""Seed-guided, metric-preserving link separation for watertight PLY meshes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.svm import LinearSVC
from trimesh import geometry, grouping, transformations as tf
from trimesh.intersections import slice_faces_plane


ROOT = Path(__file__).resolve().parents[1]


def boundary_loops(edges: np.ndarray) -> list[np.ndarray]:
    neighbors: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    if any(len(items) != 2 for items in neighbors.values()):
        raise RuntimeError("cut boundary is not a set of closed loops")

    unseen = {tuple(sorted(map(int, edge))) for edge in edges}
    loops = []
    while unseen:
        start, current = next(iter(unseen))
        loop = [start]
        while True:
            previous = loop[-1]
            loop.append(current)
            unseen.discard(tuple(sorted((previous, current))))
            following = neighbors[current][0]
            if following == previous:
                following = neighbors[current][1]
            if following == start:
                unseen.discard(tuple(sorted((current, following))))
                break
            current = following
            if len(loop) > len(neighbors):
                raise RuntimeError("cut boundary traversal did not close")
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def spatial_face_chunks(indices: np.ndarray, centers: np.ndarray,
                        maximum_span: float = 0.012) -> list[np.ndarray]:
    """Split a long boundary band into small clickable spatial patches."""
    pending, chunks = [np.asarray(indices, dtype=np.int64)], []
    while pending:
        faces = pending.pop()
        points = centers[faces]
        if len(faces) < 80:
            chunks.append(faces)
            continue
        _values, axes = np.linalg.eigh(np.cov(points, rowvar=False))
        projection = points @ axes[:, -1]
        if np.ptp(projection) <= maximum_span:
            chunks.append(faces)
            continue
        middle = np.median(projection)
        left, right = faces[projection <= middle], faces[projection > middle]
        if not len(left) or not len(right):
            chunks.append(faces)
        else:
            pending.extend((left, right))
    return chunks


def exact_colored_cut(
    mesh: trimesh.Trimesh, normal: np.ndarray, offset: float
) -> tuple[trimesh.Trimesh, dict]:
    """Keep one half-space and close only its planar cut boundary."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    origin = normal * offset
    colors = np.asarray(mesh.visual.vertex_colors, dtype=float)
    vertices, faces, colors = slice_faces_plane(
        mesh.vertices, mesh.faces, normal, origin, uv=colors
    )
    unique, inverse = grouping.unique_rows(np.round(vertices, decimals=10))
    vertices, colors, faces = vertices[unique], colors[unique], inverse[faces]
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    ]

    to_2d = geometry.plane_transform(origin=origin, normal=-normal)
    vertices_2d = tf.transform_points(vertices, to_2d)
    edges = geometry.faces_to_edges(faces)
    edges.sort(axis=1)
    on_plane = np.abs(vertices_2d[:, 2]) < 1e-7
    edges = edges[on_plane[edges].all(axis=1)]
    edges = edges[edges[:, 0] != edges[:, 1]]
    boundary = edges[grouping.group_rows(edges, require_count=1)]
    ordered = boundary_loops(boundary)
    loop_polygons = [shapely.Polygon(vertices_2d[loop, :2]) for loop in ordered]
    if any(not polygon.is_valid for polygon in loop_polygons):
        reason = next(
            shapely.is_valid_reason(polygon)
            for polygon in loop_polygons
            if not polygon.is_valid
        )
        raise RuntimeError(f"cut boundary is not a valid planar polygon: {reason}")
    depths = [
        sum(
            other.contains(polygon.representative_point())
            for index, other in enumerate(loop_polygons)
            if index != current
        )
        for current, polygon in enumerate(loop_polygons)
    ]

    plane_indices = np.flatnonzero(on_plane)
    tree = cKDTree(vertices_2d[plane_indices, :2])
    cap_faces, cap_area = [], 0.0
    for index, shell in enumerate(loop_polygons):
        if depths[index] % 2:
            continue
        holes = [
            other.exterior.coords
            for child, other in enumerate(loop_polygons)
            if depths[child] == depths[index] + 1
            and shell.contains(other.representative_point())
        ]
        polygon = shapely.Polygon(shell.exterior.coords, holes=holes)
        cap_area += polygon.area
        for triangle in shapely.constrained_delaunay_triangles(polygon).geoms:
            if not polygon.covers(triangle.representative_point()):
                continue
            distance, face = tree.query(np.asarray(triangle.exterior.coords)[:3])
            if distance.max() > 1e-7:
                raise RuntimeError("cap triangulation inserted a new vertex")
            face = plane_indices[face].astype(np.int64)
            face_normal = np.cross(
                vertices[face[1]] - vertices[face[0]],
                vertices[face[2]] - vertices[face[0]],
            )
            if face_normal @ -normal < 0:
                face = face[[0, 2, 1]]
            cap_faces.append(face)
    if not cap_faces:
        raise RuntimeError("cut produced no cap")
    result = trimesh.Trimesh(
        vertices=vertices,
        faces=np.vstack((faces, cap_faces)),
        vertex_colors=np.clip(np.rint(colors), 0, 255).astype(np.uint8),
        process=False,
    )
    return result, {
        "plane_normal": normal.tolist(),
        "plane_offset_m": offset,
        "cap_loops": len(ordered),
        "cap_faces": len(cap_faces),
        "cap_area_mm2": cap_area * 1e6,
    }


def merge_cancel_interfaces(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Join complementary cut bodies and remove their coincident internal caps."""
    vertices, faces, colors, offset = [], [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        colors.append(mesh.visual.vertex_colors)
        offset += len(mesh.vertices)
    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    colors = np.vstack(colors)
    unique, inverse = grouping.unique_rows(np.round(vertices, decimals=10))
    vertices, colors, faces = vertices[unique], colors[unique], inverse[faces]
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    ]
    faces = faces[grouping.group_rows(np.sort(faces, axis=1), require_count=1)]
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=colors, process=False
    )


def exact_seed_component_cut(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    offset: float,
    child_seed_points: np.ndarray,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict, dict, dict]:
    """Keep only the positive component owning all child seeds; restore incidental cuts."""
    positive, positive_meta = exact_colored_cut(mesh, normal, offset)
    negative, negative_meta = exact_colored_cut(mesh, -normal, -offset)
    bodies = positive.split(only_watertight=False)
    if len(bodies) == 1:
        return positive, negative, positive_meta, negative_meta, {
            "reassigned_incidental_bodies": 0,
            "reassigned_volume_cm3": 0.0,
        }

    distances = np.asarray(
        [cKDTree(body.vertices).query(child_seed_points)[0] for body in bodies]
    )
    owners = np.argmin(distances, axis=0)
    if np.any(owners != owners[0]):
        raise RuntimeError("child seeds lie in different positive components")
    owner = int(owners[0])
    child = bodies[owner]
    incidental = [body for index, body in enumerate(bodies) if index != owner]
    remainder = merge_cancel_interfaces([negative, *incidental])
    return child, remainder, positive_meta, negative_meta, {
        "reassigned_incidental_bodies": len(incidental),
        "reassigned_volume_cm3": float(sum(abs(body.volume) for body in incidental) * 1e6),
    }


def part_stats(mesh: trimesh.Trimesh) -> dict:
    counts = np.bincount(mesh.edges_unique_inverse)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "body_count": int(mesh.body_count),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "volume_cm3": abs(float(mesh.volume)) * 1e6,
        "pure_black_vertices": int(
            np.count_nonzero(np.all(mesh.visual.vertex_colors[:, :3] == 0, axis=1))
        ),
    }


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError("input must contain one triangle mesh")
    loaded.remove_unreferenced_vertices()
    if not loaded.is_watertight or not loaded.is_winding_consistent:
        raise ValueError("input must be watertight with consistent winding")
    if loaded.body_count != 1:
        raise ValueError("input must contain one connected body")
    if loaded.visual.kind != "vertex":
        loaded.visual.vertex_colors = np.tile([200, 200, 200, 255], (len(loaded.vertices), 1))
    return loaded


def validate_config(config: dict) -> tuple[list[str], dict[str, str], list[str]]:
    root = config.get("root")
    parents = config.get("parents", {})
    seeds = config.get("seeds", {})
    names = list(seeds)
    if not names or root not in names:
        raise ValueError("config seeds must include the root link")
    if set(parents) != set(names) - {root} or any(parent not in names for parent in parents.values()):
        raise ValueError("parents must map every non-root link to another configured link")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("link names may contain only letters, digits, '_' and '-'")

    depths = {}
    for name in names:
        current, seen, depth = name, set(), 0
        while current != root:
            if current in seen or current not in parents:
                raise ValueError("parents must describe one rooted, acyclic tree")
            seen.add(current)
            current, depth = parents[current], depth + 1
        depths[name] = depth
    return names, parents, sorted((name for name in names if name != root), key=depths.get, reverse=True)


def pick_seeds(mesh: trimesh.Trimesh, config: dict, path: Path) -> None:
    import vedo

    tree = cKDTree(mesh.vertices)
    palette = ("dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    for number, name in enumerate(config["seeds"]):
        actor = vedo.Mesh([mesh.vertices, mesh.faces]).c("white").alpha(0.82)
        selected, marker = [], [None]
        plotter = vedo.Plotter(title=f"Original PLY seeds: {name}")
        status = vedo.Text2D(
            f"{name}: RIGHT click owned surface | left drag rotates | U undo | Q accept\nSelected: 0",
            pos="top-left", s=0.8, bg="white", c="black", alpha=0.9,
        )

        def redraw() -> None:
            if marker[0] is not None:
                plotter.remove(marker[0])
            marker[0] = (vedo.Points(mesh.vertices[selected], r=14, c=palette[number % len(palette)])
                         if selected else None)
            if marker[0] is not None:
                plotter.add(marker[0])
            status.text(
                f"{name}: RIGHT click owned surface | left drag rotates | U undo | Q accept\n"
                f"Selected: {len(selected)}"
            )
            plotter.render()

        def click(event) -> None:
            point = getattr(event, "picked3d", None)
            if event.actor is actor and point is not None:
                index = int(tree.query(np.asarray(point, dtype=float))[1])
                if index not in selected:
                    selected.append(index)
                    redraw()

        def key(event) -> None:
            pressed = getattr(event, "keypress", "")
            if pressed in ("u", "U") and selected:
                selected.pop(); redraw()
            elif pressed in ("q", "Q", "Return"):
                if selected:
                    plotter.close()
                else:
                    status.text(f"{name}: select at least one point before Q")
                    plotter.render()

        plotter.add_callback("RightButtonPress", click)
        plotter.add_callback("KeyPress", key)
        plotter.show(actor, status, interactive=True)
        if not selected:
            raise RuntimeError(f"no seed selected for {name}")
        config["seeds"][name] = mesh.vertices[np.asarray(selected, dtype=int)].tolist()
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def seed_faces(mesh: trimesh.Trimesh, seeds: dict[str, list]) -> tuple[dict[str, np.ndarray], dict]:
    centers = mesh.triangles_center
    tree = cKDTree(centers)
    diagonal = float(np.linalg.norm(mesh.extents))
    result, distances = {}, {}
    for name, values in seeds.items():
        points = np.asarray(values, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{name} seeds must be a non-empty list of finite xyz points")
        distance, face = tree.query(points)
        if float(distance.max()) > diagonal * 0.05:
            raise ValueError(f"{name} contains a seed farther than 5% of the mesh diagonal")
        result[name] = np.unique(face.astype(np.int64))
        distances[name] = {
            "count": int(len(result[name])),
            "maximum_face_center_distance_mm": float(distance.max() * 1000),
        }
    if len(set(np.concatenate(list(result.values())).tolist())) != sum(map(len, result.values())):
        raise ValueError("two link labels resolve to the same seed face")
    return result, distances


def graph_data(mesh: trimesh.Trimesh, angle_degrees: float) -> tuple:
    adjacency = mesh.face_adjacency
    shared = mesh.face_adjacency_edges
    lengths = np.linalg.norm(mesh.vertices[shared[:, 0]] - mesh.vertices[shared[:, 1]], axis=1)
    median_length = float(np.median(lengths))
    concave = (~mesh.face_adjacency_convex) & (mesh.face_adjacency_angles > np.deg2rad(2.0))
    affinity = (lengths / median_length) * (
        0.02 + np.exp(-((mesh.face_adjacency_angles / np.deg2rad(angle_degrees)) ** 2))
    )
    affinity *= np.where(concave, 0.15, 1.0)
    geodesic = coo_matrix(
        (
            np.r_[lengths, lengths],
            (
                np.r_[adjacency[:, 0], adjacency[:, 1]],
                np.r_[adjacency[:, 1], adjacency[:, 0]],
            ),
        ),
        shape=(len(mesh.faces), len(mesh.faces)),
    ).tocsr()
    return adjacency, affinity, geodesic


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _surface_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    return mesh.sample(min(count, max(2_000, len(mesh.faces) * 3)), seed=seed)


def _source_contact_anchor(
    mesh: trimesh.Trimesh, labels: np.ndarray, first: int, second: int
) -> np.ndarray:
    pair = labels[mesh.face_adjacency]
    boundary = ((pair[:, 0] == first) & (pair[:, 1] == second)) | (
        (pair[:, 0] == second) & (pair[:, 1] == first)
    )
    if not boundary.any():
        raise RuntimeError("initial graph labels contain no parent-child interface")
    vertices = np.unique(mesh.face_adjacency_edges[boundary])
    return np.median(mesh.vertices[vertices], axis=0)


def _pca_axes(points: np.ndarray) -> np.ndarray:
    _values, axes = np.linalg.eigh(np.cov(points, rowvar=False))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _proper_axis_permutations() -> list[np.ndarray]:
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                result.append(matrix)
    return result


def _moving_components(
    observations: dict[str, np.ndarray],
    state: str,
    comparison_states: list[str],
    static_limit: float,
    dynamic_limit: float,
    cluster_radius: float,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Split a state into repeatable background and state-changing components."""
    points = observations[state]
    distances = np.column_stack(
        [cKDTree(observations[other]).query(points, workers=-1)[0] for other in comparison_states]
    )
    nearest_other_state = distances.min(axis=1)
    static = points[nearest_other_state <= static_limit]
    moving = points[nearest_other_state >= dynamic_limit]
    if len(static) < 100 or len(moving) < 100:
        return static, [], {
            "points": int(len(points)),
            "static_points": int(len(static)),
            "moving_points": int(len(moving)),
        }
    labels = DBSCAN(eps=cluster_radius, min_samples=12, n_jobs=-1).fit_predict(moving)
    components = [
        moving[labels == label]
        for label in sorted(set(labels))
        if label >= 0 and np.count_nonzero(labels == label) >= 200
    ]
    return static, components, {
        "points": int(len(points)),
        "static_points": int(len(static)),
        "moving_points": int(len(moving)),
        "moving_components": [int(len(component)) for component in components],
    }


def _anchored_motion_candidate(
    source_points: np.ndarray,
    source_anchor: np.ndarray,
    component: np.ndarray,
    static_points: np.ndarray,
    face_centers: np.ndarray,
    initial_labels: np.ndarray,
    child_label: int,
    joint_band: np.ndarray,
    fit_limit: float,
    transfer_limit: float,
) -> tuple[dict, np.ndarray]:
    """Match a moving component without allowing PCA to swap the joint end."""
    contact_distance, nearest = cKDTree(static_points).query(component, workers=-1)
    count = min(100, len(component))
    contact = np.argpartition(contact_distance, count - 1)[:count]
    target_anchor = np.median(
        (component[contact] + static_points[nearest[contact]]) / 2, axis=0
    )
    source_axes, target_axes = _pca_axes(source_points), _pca_axes(component)
    target_tree = cKDTree(component)
    candidates = []
    for permutation in _proper_axis_permutations():
        rotation = target_axes @ permutation @ source_axes.T
        translation = target_anchor - source_anchor @ rotation.T
        distance = target_tree.query(
            source_points @ rotation.T + translation, workers=-1
        )[0]
        inlier = distance <= fit_limit
        coverage = float(np.mean(inlier))
        residual = float(np.median(distance[inlier])) if inlier.any() else float("inf")
        candidates.append((coverage, -residual, rotation, translation))
    coverage, negative_residual, rotation, translation = max(
        candidates, key=lambda item: item[:2]
    )
    face_distance = target_tree.query(
        face_centers @ rotation.T + translation, workers=-1
    )[0]
    matched = joint_band & (face_distance <= transfer_limit)
    counts = np.bincount(initial_labels[matched], minlength=child_label + 1)
    child_fraction = float(counts[child_label] / max(1, counts.sum()))
    return {
        "component_points": int(len(component)),
        "coverage": coverage,
        "inlier_median_mm": -negative_residual * 1000,
        "matched_faces": int(matched.sum()),
        "matched_child_fraction": child_fraction,
        "target_contact_xyz_m": target_anchor.tolist(),
        "matrix": np.vstack(
            [np.column_stack([rotation, translation]), [0.0, 0.0, 0.0, 1.0]]
        ).tolist(),
    }, matched


def motion_seed_faces(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    names: list[str],
    parents: dict[str, str],
    config: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Transfer observed rigid-motion ownership into the delivery PLY."""
    manifest_path = config.get("motion_manifest")
    if not manifest_path:
        return {}, {"status": "NOT_CONFIGURED"}
    manifest = json.loads(_path(manifest_path).read_text(encoding="utf-8"))
    if [item["name"] for item in manifest["links"]] != names:
        raise ValueError("motion manifest link order must match config seeds")
    output = _path(manifest["output"])
    centers = mesh.triangles_center
    observation_paths = {
        item["name"]: output / f"state_observations/{item['name']}.ply"
        for item in manifest["states"]
    }
    observations = {
        state: np.asarray(trimesh.load(path, process=False).vertices)
        for state, path in observation_paths.items()
        if path.exists()
    }
    static_limit = float(config.get("motion_static_residual_mm", 8.0)) / 1000
    dynamic_limit = float(config.get("motion_dynamic_residual_mm", 18.0)) / 1000
    cluster_radius = float(config.get("motion_cluster_radius_mm", 12.0)) / 1000
    fit_limit = float(config.get("motion_fit_residual_mm", 12.0)) / 1000
    transfer_limit = float(config.get("motion_transfer_residual_mm", 6.0)) / 1000
    minimum_coverage = float(config.get("motion_minimum_coverage", 0.25))
    minimum_child_fraction = float(config.get("motion_minimum_child_fraction", 0.55))
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    ownership_audit = {}
    manifest_joints = {(item["parent"], item["child"]): item for item in manifest["joints"]}

    descendants = {name: {name} for name in names}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            before = len(descendants[parent])
            descendants[parent] |= descendants[child]
            changed |= len(descendants[parent]) != before

    for child, parent in parents.items():
        joint = manifest_joints.get((parent, child))
        if joint is None:
            raise ValueError(f"motion manifest has no joint {parent}->{child}")
        states = [state for state in joint["states"] if state in observations]
        if len(states) < 2:
            ownership_audit[child] = {
                "parent": parent,
                "status": "SKIPPED_NEEDS_TWO_OBSERVED_STATES",
                "states": states,
            }
            continue
        parent_index, child_index = names.index(parent), names.index(child)
        anchor = _source_contact_anchor(mesh, labels, parent_index, child_index)
        contact_pair = labels[mesh.face_adjacency]
        interface = ((contact_pair[:, 0] == parent_index) & (contact_pair[:, 1] == child_index)) | (
            (contact_pair[:, 0] == child_index) & (contact_pair[:, 1] == parent_index)
        )
        interface_points = mesh.vertices[np.unique(mesh.face_adjacency_edges[interface])]
        interface_radius = float(np.linalg.norm(interface_points - anchor, axis=1).max())
        band_radius = float(config.get("motion_joint_band_m", np.clip(2.5 * interface_radius, 0.025, 0.080)))
        band = np.linalg.norm(centers - anchor, axis=1) <= band_radius
        moving_labels = [names.index(name) for name in descendants[child]]
        source = mesh.submesh(
            [np.flatnonzero(np.isin(labels, moving_labels))], append=True, repair=False
        )
        source_points = _surface_points(source, 40_000, child_index)
        state_audit, accepted = {}, []
        for state in states:
            static, components, state_record = _moving_components(
                observations,
                state,
                [other for other in states if other != state],
                static_limit,
                dynamic_limit,
                cluster_radius,
            )
            candidates = []
            for component in components:
                candidate, matched = _anchored_motion_candidate(
                    source_points,
                    anchor,
                    component,
                    static,
                    centers,
                    labels,
                    child_index,
                    band,
                    fit_limit,
                    transfer_limit,
                )
                candidate["accepted"] = bool(
                    candidate["coverage"] >= minimum_coverage
                    and candidate["matched_child_fraction"] >= minimum_child_fraction
                )
                candidates.append((candidate, matched))
            candidates.sort(
                key=lambda item: (item[0]["accepted"], item[0]["coverage"]), reverse=True
            )
            if candidates and candidates[0][0]["accepted"]:
                best, matched = candidates[0]
                correction = np.flatnonzero(matched & (labels == parent_index))
                if len(correction):
                    correction_mask = np.zeros(len(labels), dtype=bool)
                    correction_mask[correction] = True
                    touches = (
                        ((labels[mesh.face_adjacency[:, 0]] == child_index) & correction_mask[mesh.face_adjacency[:, 1]])
                        | ((labels[mesh.face_adjacency[:, 1]] == child_index) & correction_mask[mesh.face_adjacency[:, 0]])
                    )
                    best["child_boundary_touch_edges"] = int(np.count_nonzero(touches))
                    accepted.append((bool(touches.any()), best["component_points"], correction, state))
                state_record["selected_component"] = best
                state_record["matched_parent_faces"] = int(len(correction))
            else:
                state_record["selected_component"] = None
                state_record["rejection"] = "no component passed coverage and child-ownership gates"
            state_audit[state] = state_record
        selected = max(accepted, key=lambda item: item[:2]) if accepted else None
        correction = selected[2] if selected else np.empty(0, dtype=int)
        if len(correction):
            result[child].append(correction)
        ownership_audit[child] = {
            "parent": parent,
            "status": "PASS" if len(correction) else "PASS_NO_BOUNDARY_CORRECTION",
            "states": states,
            "band_radius_mm": band_radius * 1000,
            "selected_evidence_state": selected[3] if selected else None,
            "motion_parent_faces": int(len(correction)),
            "per_state": state_audit,
        }
    return {
        name: np.unique(np.concatenate(chunks)) for name, chunks in result.items() if chunks
    }, {
        "status": "PASS",
        "method": "all_state_change_detection_plus_joint_anchored_child_surface_transfer",
        "manifest": str(_path(manifest_path).resolve()),
        "thresholds_mm": {
            "static": static_limit * 1000,
            "dynamic": dynamic_limit * 1000,
            "cluster_radius": cluster_radius * 1000,
            "fit": fit_limit * 1000,
            "transfer": transfer_limit * 1000,
        },
        "minimum_coverage": minimum_coverage,
        "minimum_child_fraction": minimum_child_fraction,
        "ownership": ownership_audit,
    }


def _maximum_flow_side(
    count: int,
    edges: np.ndarray,
    pair_weights: np.ndarray,
    child_cost: np.ndarray,
    rest_cost: np.ndarray,
) -> np.ndarray:
    source, sink = count, count + 1
    rows = np.r_[edges[:, 0], edges[:, 1], np.full(count, source), np.arange(count)]
    columns = np.r_[edges[:, 1], edges[:, 0], np.arange(count), np.full(count, sink)]
    capacities = np.r_[pair_weights, pair_weights, rest_cost, child_cost]
    graph = coo_matrix(
        (
            np.maximum(1, np.rint(capacities * 1000)).astype(np.int64),
            (rows, columns),
        ),
        shape=(count + 2, count + 2),
    ).tocsr()
    flow = csgraph.maximum_flow(graph, source, sink, method="dinic")
    residual = graph - flow.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = csgraph.breadth_first_order(
        residual, source, directed=True, return_predecessors=False
    )
    source_side = np.zeros(count + 2, dtype=bool)
    source_side[reachable] = True
    return source_side[:count]


def refine_motion_labels(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    motion_faces: dict[str, np.ndarray],
    names: list[str],
    parents: dict[str, str],
    order: list[str],
    seeds: dict[str, np.ndarray],
    config: dict,
    adjacency: np.ndarray,
    affinity: np.ndarray,
    smoothness: float,
) -> tuple[np.ndarray, dict]:
    """Apply motion as a local soft prior; keep labels fixed outside each joint band."""
    labels = labels.copy()
    centers = mesh.triangles_center
    preservation = float(config.get("motion_label_preservation", 1.0))
    motion_weight = float(config.get("motion_label_weight", 0.5))
    local_smoothness = float(config.get("motion_local_smoothness", 6.0))
    affinity_floor = float(config.get("motion_affinity_floor", 0.25))
    audit = {}
    for child in order:
        evidence = motion_faces.get(child)
        if evidence is None or not len(evidence):
            continue
        parent = parents[child]
        parent_index, child_index = names.index(parent), names.index(child)
        anchor = _source_contact_anchor(mesh, labels, parent_index, child_index)
        pair = labels[mesh.face_adjacency]
        interface = ((pair[:, 0] == parent_index) & (pair[:, 1] == child_index)) | (
            (pair[:, 0] == child_index) & (pair[:, 1] == parent_index)
        )
        interface_points = mesh.vertices[np.unique(mesh.face_adjacency_edges[interface])]
        radius = float(config.get(
            "motion_joint_band_m",
            np.clip(
                2.5 * np.linalg.norm(interface_points - anchor, axis=1).max(), 0.025, 0.080
            ),
        ))
        active = (np.linalg.norm(centers - anchor, axis=1) <= radius) & np.isin(
            labels, [parent_index, child_index]
        )
        evidence = evidence[active[evidence]]
        if not len(evidence):
            audit[child] = {"status": "SKIPPED_EVIDENCE_OUTSIDE_JOINT_BAND"}
            continue
        old = np.flatnonzero(active)
        inverse = np.full(len(labels), -1, dtype=np.int64)
        inverse[old] = np.arange(len(old))
        internal = active[adjacency].all(axis=1)
        edges = inverse[adjacency[internal]]
        frontier_edges = active[adjacency[:, 0]] ^ active[adjacency[:, 1]]
        frontier = np.unique(
            np.where(
                active[adjacency[frontier_edges, 0]],
                adjacency[frontier_edges, 0],
                adjacency[frontier_edges, 1],
            )
        )
        area_weight = np.clip(
            mesh.area_faces[old] / np.median(mesh.area_faces[old]), 0.1, 10.0
        )
        child_cost = preservation * (labels[old] != child_index) * area_weight
        rest_cost = preservation * (labels[old] == child_index) * area_weight
        local_evidence = np.isin(old, evidence)
        child_cost[local_evidence] = 0.0
        rest_cost[local_evidence] = motion_weight * area_weight[local_evidence]
        hard = 1e6
        local_frontier = inverse[frontier]
        child_frontier = local_frontier[labels[frontier] == child_index]
        parent_frontier = local_frontier[labels[frontier] == parent_index]
        child_cost[child_frontier], rest_cost[child_frontier] = 0.0, hard
        child_cost[parent_frontier], rest_cost[parent_frontier] = hard, 0.0
        child_seeds = seeds[child][active[seeds[child]]]
        parent_seeds = seeds[parent][active[seeds[parent]]]
        child_cost[inverse[child_seeds]], rest_cost[inverse[child_seeds]] = 0.0, hard
        child_cost[inverse[parent_seeds]], rest_cost[inverse[parent_seeds]] = hard, 0.0
        before = labels.copy()
        child_side = _maximum_flow_side(
            len(old),
            edges,
            local_smoothness * np.maximum(affinity[internal], affinity_floor),
            child_cost,
            rest_cost,
        )
        labels[old] = np.where(child_side, child_index, parent_index)

        child_faces = np.flatnonzero(labels == child_index)
        child_inverse = np.full(len(labels), -1, dtype=np.int64)
        child_inverse[child_faces] = np.arange(len(child_faces))
        child_edges = (labels[adjacency] == child_index).all(axis=1)
        child_graph = coo_matrix(
            (
                np.ones(np.count_nonzero(child_edges) * 2),
                (
                    np.r_[child_inverse[adjacency[child_edges, 0]], child_inverse[adjacency[child_edges, 1]]],
                    np.r_[child_inverse[adjacency[child_edges, 1]], child_inverse[adjacency[child_edges, 0]]],
                ),
            ),
            shape=(len(child_faces), len(child_faces)),
        ).tocsr()
        _count, components = csgraph.connected_components(child_graph, directed=False)
        owner = components[child_inverse[seeds[child][0]]]
        islands = child_faces[components != owner]
        labels[islands] = parent_index
        updated_pair = labels[mesh.face_adjacency]
        updated_interface = (
            ((updated_pair[:, 0] == parent_index) & (updated_pair[:, 1] == child_index))
            | ((updated_pair[:, 0] == child_index) & (updated_pair[:, 1] == parent_index))
        )
        try:
            loops = boundary_loops(mesh.face_adjacency_edges[updated_interface])
        except RuntimeError as error:
            attempted_parent_to_child = int(
                np.count_nonzero((before == parent_index) & (labels == child_index))
            )
            labels = before
            audit[child] = {
                "status": "REJECTED_TOPOLOGY_GATE",
                "reason": str(error),
                "motion_evidence_faces": int(len(evidence)),
                "attempted_parent_to_child_faces": attempted_parent_to_child,
            }
            continue
        audit[child] = {
            "status": "PASS",
            "joint_band_radius_mm": radius * 1000,
            "local_smoothness": local_smoothness,
            "affinity_floor": affinity_floor,
            "motion_evidence_faces": int(len(evidence)),
            "parent_to_child_faces": int(np.count_nonzero((before == parent_index) & (labels == child_index))),
            "child_to_parent_faces": int(np.count_nonzero((before == child_index) & (labels == parent_index))),
            "removed_child_island_faces": int(len(islands)),
            "interface_loops": int(len(loops)),
        }
    return labels, audit


def _minimum_area_loop_triangulation(vertices: np.ndarray, loop: np.ndarray,
                                     source_edge_codes: np.ndarray) -> list[np.ndarray]:
    """Triangulate an ordered non-planar loop without adding interface vertices."""
    points, count = vertices[loop], len(loop)
    cost = np.full((count, count), np.inf)
    split = np.full((count, count), -1, dtype=np.int32)
    np.fill_diagonal(cost, 0.0)
    cost[np.arange(count - 1), np.arange(1, count)] = 0.0
    for gap in range(2, count):
        for first in range(count - gap):
            last = first + gap
            code = min(loop[first], loop[last]) * len(vertices) + max(loop[first], loop[last])
            position = np.searchsorted(source_edge_codes, code)
            if (not (first == 0 and last == count - 1) and position < len(source_edge_codes)
                    and source_edge_codes[position] == code):
                continue
            middle = np.arange(first + 1, last)
            area = np.linalg.norm(np.cross(
                points[middle] - points[first], points[last] - points[first]
            ), axis=1) * 0.5
            values = cost[first, middle] + cost[middle, last] + area
            best = int(np.argmin(values))
            cost[first, last], split[first, last] = values[best], middle[best]
    triangles, pending = [], [(0, count - 1)]
    while pending:
        first, last = pending.pop()
        middle = int(split[first, last])
        if middle < 0:
            continue
        triangles.append(loop[[first, middle, last]])
        pending.extend(((first, middle), (middle, last)))
    return triangles


def close_labeled_parts(
    mesh: trimesh.Trimesh, labels: np.ndarray, names: list[str]
) -> tuple[dict[str, trimesh.Trimesh], dict]:
    """Preserve every exterior triangle and add paired joint-interface caps."""
    cap_faces: list[list[np.ndarray]] = [[] for _ in names]
    exterior_counts = [int(np.count_nonzero(labels == index)) for index in range(len(names))]
    source_edge_codes = np.sort(
        mesh.edges_unique[:, 0].astype(np.int64) * len(mesh.vertices) + mesh.edges_unique[:, 1]
    )
    pair = labels[mesh.face_adjacency]
    interfaces = {}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            boundary = ((pair[:, 0] == first) & (pair[:, 1] == second)) | (
                (pair[:, 0] == second) & (pair[:, 1] == first)
            )
            edges = mesh.face_adjacency_edges[boundary]
            if not len(edges):
                continue
            loops = boundary_loops(edges)
            cap_area, maximum_residual, interface_face_count = 0.0, 0.0, 0
            triangulation_modes = set()
            local_cap_ids = {names[first]: [], names[second]: []}
            for loop in loops:
                points = mesh.vertices[loop]
                center = points.mean(axis=0)
                _values, axes = np.linalg.eigh(np.cov(points - center, rowvar=False))
                triangles = _minimum_area_loop_triangulation(
                    mesh.vertices, loop, source_edge_codes
                )
                for face in triangles:
                    cap_area += float(np.linalg.norm(np.cross(
                        mesh.vertices[face[1]] - mesh.vertices[face[0]],
                        mesh.vertices[face[2]] - mesh.vertices[face[0]],
                    )) / 2)
                triangulation_modes.add("minimum_area_ordered_3d_loop")
                if not triangles:
                    raise RuntimeError("interface triangulation produced no cap")
                interface_face_count += len(triangles)
                maximum_residual = max(
                    maximum_residual,
                    float(np.max(np.abs((points - center) @ axes[:, 0]))),
                )
                for label in (first, second):
                    start = exterior_counts[label] + len(cap_faces[label])
                    local_cap_ids[names[label]].extend(range(start, start + len(triangles)))
                    cap_faces[label].extend(triangles)
            interfaces[f"{names[first]}--{names[second]}"] = {
                "boundary_loops": len(loops),
                "boundary_edges": int(len(edges)),
                "shared_cap_faces_per_side": int(interface_face_count),
                "shared_cap_area_mm2": cap_area * 1e6,
                "boundary_best_fit_residual_mm": maximum_residual * 1000,
                "triangulation": sorted(triangulation_modes),
                "cap_face_ids": local_cap_ids,
            }

    outputs = {}
    for index, name in enumerate(names):
        faces = np.vstack([mesh.faces[labels == index], *cap_faces[index]])
        part = trimesh.Trimesh(
            vertices=mesh.vertices, faces=faces,
            vertex_colors=mesh.visual.vertex_colors, process=False
        )
        part.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(part, multibody=True)
        outputs[name] = part
    return outputs, interfaces


def repair_tree_face_labels(mesh: trimesh.Trimesh, labels: np.ndarray,
                            names: list[str], parents: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Remove non-tree junctions and disconnected paint islands before saving."""
    repaired = np.asarray(labels, dtype=np.int64).copy()
    adjacency = mesh.face_adjacency
    index = {name: value for value, name in enumerate(names)}
    tree = {value: set() for value in range(len(names))}
    for child, parent in parents.items():
        a, b = index[parent], index[child]
        tree[a].add(b); tree[b].add(a)

    def path(start: int, goal: int) -> list[int]:
        pending, previous = [start], {start: -1}
        for current in pending:
            if current == goal:
                break
            for neighbor in tree[current]:
                if neighbor not in previous:
                    previous[neighbor] = current; pending.append(neighbor)
        route = [goal]
        while route[-1] != start:
            route.append(previous[route[-1]])
        return route[::-1]

    moved_junctions = removed_islands = repaired_pinches = 0

    def remove_illegal_edges() -> int:
        moved = 0
        pair = repaired[adjacency]
        for first, second in np.unique(np.sort(pair[pair[:, 0] != pair[:, 1]], axis=1), axis=0):
            route = path(int(first), int(second))
            if len(route) <= 2:
                continue
            rows = np.all(np.sort(pair, axis=1) == (first, second), axis=1)
            faces = adjacency[rows]
            for face in np.unique(faces[repaired[faces] == first]):
                repaired[face] = route[1]; moved += 1
            for face in np.unique(faces[repaired[faces] == second]):
                repaired[face] = route[-2]; moved += 1
        return moved

    # Keep one surface component per non-root link; brush specks return to parent.
    depths = {}
    for name in names:
        depth, current = 0, name
        while current in parents:
            depth += 1; current = parents[current]
        depths[name] = depth
    def remove_islands() -> int:
        removed = 0
        for child in sorted(parents, key=depths.get, reverse=True):
            child_index, parent_index = index[child], index[parents[child]]
            faces = np.flatnonzero(repaired == child_index)
            if not len(faces):
                continue
            inverse = np.full(len(repaired), -1, dtype=np.int64); inverse[faces] = np.arange(len(faces))
            internal = (repaired[adjacency] == child_index).all(axis=1)
            edges = inverse[adjacency[internal]]
            graph = coo_matrix((np.ones(len(edges) * 2),
                                (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                               shape=(len(faces), len(faces))).tocsr()
            _count, components = csgraph.connected_components(graph, directed=False)
            owner = int(np.argmax(np.bincount(components)))
            for component in np.unique(components[components != owner]):
                island = faces[components == component]
                mask = np.zeros(len(repaired), dtype=bool); mask[island] = True
                crossing = mask[adjacency[:, 0]] ^ mask[adjacency[:, 1]]
                outside = np.where(mask[adjacency[crossing, 0]],
                                   adjacency[crossing, 1], adjacency[crossing, 0])
                neighbors = repaired[outside]
                neighbors = neighbors[neighbors != child_index]
                target = (int(np.argmax(np.bincount(neighbors, minlength=len(names))))
                          if len(neighbors) else parent_index)
                repaired[island] = target
                removed += len(island)
        return removed

    # Brush strokes can make a pair boundary visit one vertex 4+ times.  Collapse
    # only that pinched vertex fan into the parent; the surrounding ring is closed.
    for _ in range(2 * len(names) + 2):
        moved_junctions += remove_illegal_edges()
        bad_vertices = set()
        pair = repaired[adjacency]
        for child, parent in parents.items():
            child_index, parent_index = index[child], index[parent]
            boundary = (((pair[:, 0] == parent_index) & (pair[:, 1] == child_index))
                        | ((pair[:, 0] == child_index) & (pair[:, 1] == parent_index)))
            edges = mesh.face_adjacency_edges[boundary]
            if not len(edges):
                continue
            degree = np.bincount(edges.ravel(), minlength=len(mesh.vertices))
            bad_vertices.update(np.flatnonzero((degree != 0) & (degree != 2)).tolist())
        pinched = 0
        for vertex in bad_vertices:
            faces = mesh.vertex_faces[vertex]
            faces = faces[faces >= 0]
            around, counts = np.unique(repaired[faces], return_counts=True)
            owner = min(
                around,
                key=lambda value: (
                    sum(len(path(int(value), int(other))) - 1 for other in around),
                    -counts[np.flatnonzero(around == value)[0]],
                ),
            )
            pinched += np.count_nonzero(repaired[faces] != owner)
            repaired[faces] = owner
        repaired_pinches += pinched
        removed_islands += remove_islands()
        if not pinched and not remove_illegal_edges():
            break
    return repaired, {"junction_faces": moved_junctions,
                      "pinch_faces": repaired_pinches,
                      "island_faces": removed_islands}


def paint_face_labels(mesh: trimesh.Trimesh, labels: np.ndarray, names: list[str],
                      parents: dict[str, str]) -> np.ndarray:
    """Let the user correct exact source-face ownership with a geodesic brush."""
    import vedo

    palette = np.asarray([
        [40, 130, 190, 255], [245, 125, 35, 255], [45, 165, 70, 255],
        [180, 70, 190, 255], [230, 190, 40, 255],
    ], dtype=np.uint8)
    centers = mesh.triangles_center
    tree = cKDTree(centers)
    adjacency = mesh.face_adjacency
    weights = np.linalg.norm(centers[adjacency[:, 0]] - centers[adjacency[:, 1]], axis=1)
    surface_graph = coo_matrix(
        (np.r_[weights, weights] + 1e-12,
         (np.r_[adjacency[:, 0], adjacency[:, 1]],
          np.r_[adjacency[:, 1], adjacency[:, 0]])),
        shape=(len(labels), len(labels)),
    ).tocsr()
    actor = vedo.Mesh([mesh.vertices, mesh.faces])
    actor.cellcolors = palette[labels % len(palette)]
    plotter = vedo.Plotter(title="Exact original-face correction")
    state = {"mode": 0, "radius_mm": 6.0, "finished": False, "undo": []}
    status = vedo.Text2D("", pos="top-left", s=0.72, bg="white", c="black", alpha=0.9)

    def refresh(message="") -> None:
        counts = np.bincount(labels, minlength=len(names))
        status.text(
            "EXACT ORIGINAL-FACE CORRECTION\n"
            + " | ".join(f"{i + 1}:{name}={counts[i]}" for i, name in enumerate(names))
            + f"\nBrush={names[state['mode']]}  radius={state['radius_mm']:.1f} mm"
            + "\nLeft click paints | 1/2/3 label | [ ] size | U undo | Q validate/save"
            + (f"\n{message}" if message else "")
        )
        actor.cellcolors = palette[labels % len(palette)]
        plotter.render()

    def paint(event) -> None:
        point = getattr(event, "picked3d", None)
        if event.actor is not actor or point is None:
            return
        seed = int(tree.query(np.asarray(point, dtype=float))[1])
        distance = csgraph.dijkstra(
            surface_graph, directed=False, indices=seed,
            limit=state["radius_mm"] / 1000.0,
        )
        selected = np.flatnonzero(np.isfinite(distance))
        old = labels[selected].copy()
        changed = old != state["mode"]
        if changed.any():
            state["undo"].append((selected[changed], old[changed]))
            labels[selected[changed]] = state["mode"]
            refresh(f"painted {np.count_nonzero(changed)} faces")

    def finish() -> None:
        try:
            candidate, repair = repair_tree_face_labels(mesh, labels, names, parents)
            outputs, interfaces = close_labeled_parts(mesh, candidate, names)
            allowed = {frozenset((parent, child)) for child, parent in parents.items()}
            actual = {frozenset(key.split("--")) for key in interfaces}
            if actual != allowed:
                raise RuntimeError("painted labels do not match the configured link tree")
            if any(not part.is_watertight or part.body_count != 1 for part in outputs.values()):
                raise RuntimeError("painted labels do not produce single watertight links")
        except Exception as error:
            refresh(f"SAVE BLOCKED: {error}")
            return
        labels[:] = candidate
        refresh(
            f"auto-repaired {repair['junction_faces']} junction faces, "
            f"{repair['pinch_faces']} pinched faces, {repair['island_faces']} island faces"
        )
        state["finished"] = True
        plotter.close()

    def key(event) -> None:
        pressed = (getattr(event, "keypress", None)
                   or getattr(event, "keyPressed", None)
                   or getattr(event, "key", None)
                   or getattr(event, "symbol", ""))
        if pressed in tuple(str(i) for i in range(1, len(names) + 1)):
            state["mode"] = int(pressed) - 1; refresh()
        elif pressed in ("u", "U") and state["undo"]:
            indices, old = state["undo"].pop(); labels[indices] = old; refresh("undo")
        elif pressed in ("[", "braceleft"):
            state["radius_mm"] = max(1.0, state["radius_mm"] - 1.0); refresh()
        elif pressed in ("]", "braceright"):
            state["radius_mm"] = min(20.0, state["radius_mm"] + 1.0); refresh()
        elif pressed in ("q", "Q", "Return"):
            finish()

    plotter.add_callback("LeftButtonPress", paint)
    interactor = plotter.interactor
    for event in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
        interactor.RemoveObservers(event)
    plotter.add_callback("KeyPress", key)
    plotter.add_callback("CharEvent", key)
    refresh()
    plotter.show(actor, status, interactive=True)
    if not state["finished"]:
        raise RuntimeError("face correction closed without a valid save")
    return labels


def review_boundary_patches(mesh: trimesh.Trimesh, labels: np.ndarray, names: list[str],
                            parents: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Reassign a small, disjoint set of exact surface patches around each joint."""
    import vedo

    palette = ("dodgerblue", "orange", "mediumseagreen", "violet", "gold")
    adjacency = mesh.face_adjacency
    centers = mesh.triangles_center
    weights = np.linalg.norm(centers[adjacency[:, 0]] - centers[adjacency[:, 1]], axis=1)
    graph = coo_matrix(
        (np.r_[weights, weights] + 1e-12,
         (np.r_[adjacency[:, 0], adjacency[:, 1]],
          np.r_[adjacency[:, 1], adjacency[:, 0]])),
        shape=(len(labels), len(labels)),
    ).tocsr()
    joints = [(names.index(parent), names.index(child), f"{parent}--{child}")
              for child, parent in parents.items()]
    distance_columns = []
    for parent, child, _joint_name in joints:
        pair = labels[adjacency]
        boundary = (((pair[:, 0] == parent) & (pair[:, 1] == child))
                    | ((pair[:, 0] == child) & (pair[:, 1] == parent)))
        sources = np.unique(adjacency[boundary])
        if not len(sources):
            raise RuntimeError(f"no initial boundary for {names[parent]}--{names[child]}")
        distance_columns.append(csgraph.dijkstra(
            graph, directed=False, indices=sources, min_only=True,
        ))
    distances = np.column_stack(distance_columns)
    closest_joint = distances.argmin(axis=1)
    nearest = distances[np.arange(len(labels)), closest_joint]
    width = float(np.clip(np.linalg.norm(mesh.extents) * 0.04, 0.008, 0.030))
    layers = np.asarray([0.0, 0.22, 0.48, 0.72, 1.0]) * width
    patches = []
    for joint_index, (parent, child, joint_name) in enumerate(joints):
        for label in (parent, child):
            for layer, (lower, upper) in enumerate(zip(layers[:-1], layers[1:]), 1):
                faces = np.flatnonzero(
                    (closest_joint == joint_index) & (labels == label)
                    & (nearest >= lower) & (nearest < upper)
                )
                if len(faces):
                    _count, components = csgraph.connected_components(
                        graph[faces][:, faces], directed=False)
                    for component in range(_count):
                        for chunk in spatial_face_chunks(faces[components == component], centers):
                            patches.append({"faces": chunk, "label": label, "initial": label,
                                            "joint": joint_name, "layer": layer})

    base = vedo.Mesh([mesh.vertices, mesh.faces]).alpha(0.18)
    if hasattr(base, "pickable"):
        base.pickable(False)
    actors = []
    for number, patch in enumerate(patches):
        actor = vedo.Mesh([mesh.vertices, mesh.faces[patch["faces"]]])
        actor.c(palette[patch["label"]]).alpha(0.92)
        actor.patch_index = number
        actors.append(actor)
    plotter = vedo.Plotter(title="RORA joint-boundary patch assignment")
    state = {"mode": 0, "finished": False, "undo": []}
    status = vedo.Text2D("", pos="top-left", s=0.72, bg="white", c="black", alpha=0.9)

    def refresh(message="") -> None:
        counts = np.bincount([patch["label"] for patch in patches], minlength=len(names))
        status.text(
            "RORA JOINT PATCH ASSIGNMENT (exact original faces)\n"
            + " | ".join(f"{i + 1}:{name} patches={counts[i]}" for i, name in enumerate(names))
            + f"\nCurrent={names[state['mode']]} | Left click patch | 1/2/3 label | U undo | R reset | Q validate/save"
            + (f"\n{message}" if message else "")
        )
        for actor, patch in zip(actors, patches):
            actor.c(palette[patch["label"]])
        plotter.render()

    def assign(event) -> None:
        actor = event.actor
        if actor not in actors:
            return
        patch = patches[actor.patch_index]
        if patch["label"] != state["mode"]:
            state["undo"].append((actor.patch_index, patch["label"]))
            labels[patch["faces"]] = state["mode"]
            patch["label"] = state["mode"]
            refresh(f"{patch['joint']} layer {patch['layer']} -> {names[state['mode']]}")

    def finish() -> None:
        try:
            outputs, interfaces = close_labeled_parts(mesh, labels, names)
            allowed = {frozenset((parent, child)) for child, parent in parents.items()}
            actual = {frozenset(key.split("--")) for key in interfaces}
            if actual != allowed:
                raise RuntimeError("assignments do not match the configured link tree")
            if any(not part.is_watertight or part.body_count != 1 for part in outputs.values()):
                raise RuntimeError("assignments do not produce single watertight links")
        except Exception as error:
            refresh(f"SAVE BLOCKED: {error}")
            return
        state["finished"] = True
        plotter.close()

    def key(event) -> None:
        pressed = (getattr(event, "keypress", None)
                   or getattr(event, "keyPressed", None)
                   or getattr(event, "key", None)
                   or getattr(event, "symbol", ""))
        if pressed in tuple(str(i) for i in range(1, len(names) + 1)):
            state["mode"] = int(pressed) - 1; refresh()
        elif pressed in ("u", "U") and state["undo"]:
            index, old = state["undo"].pop(); patch = patches[index]
            patch["label"] = old; labels[patch["faces"]] = old; refresh("undo")
        elif pressed in ("r", "R"):
            for patch in patches:
                patch["label"] = patch["initial"]
                labels[patch["faces"]] = patch["initial"]
            state["undo"].clear(); refresh("reset")
        elif pressed in ("q", "Q", "Return"):
            finish()

    plotter.add_callback("LeftButtonPress", assign)
    plotter.add_callback("KeyPress", key)
    refresh()
    plotter.show([base, *actors, status], interactive=True)
    if not state["finished"]:
        raise RuntimeError("joint patch assignment closed without a valid save")
    changed = sum(int(np.count_nonzero(labels[patch["faces"]] != patch["initial"]))
                  for patch in patches)
    return labels, {"patch_count": len(patches), "band_width_mm": width * 1000,
                    "changed_face_count": changed, "status": "HITL_CONFIRMED"}


def expand_child_boundaries(mesh: trimesh.Trimesh, labels: np.ndarray, names: list[str],
                            parents: dict[str, str], expansions: dict) -> tuple[np.ndarray, dict]:
    """Move a complete parent-child boundary loop toward the parent by a metric distance."""
    adjacency, centers = mesh.face_adjacency, mesh.triangles_center
    weights = np.linalg.norm(centers[adjacency[:, 0]] - centers[adjacency[:, 1]], axis=1)
    graph = coo_matrix(
        (np.r_[weights, weights] + 1e-12,
         (np.r_[adjacency[:, 0], adjacency[:, 1]],
          np.r_[adjacency[:, 1], adjacency[:, 0]])),
        shape=(len(labels), len(labels)),
    ).tocsr()
    records = {}
    for child, distance_mm in expansions.items():
        if child not in parents or child not in names or float(distance_mm) <= 0:
            raise ValueError(f"invalid closed-loop child expansion: {child}={distance_mm}")
        parent = parents[child]
        parent_index, child_index = names.index(parent), names.index(child)
        pair = labels[adjacency]
        boundary = (((pair[:, 0] == parent_index) & (pair[:, 1] == child_index))
                    | ((pair[:, 0] == child_index) & (pair[:, 1] == parent_index)))
        sources = np.unique(adjacency[boundary])
        distance = csgraph.dijkstra(graph, directed=False, indices=sources, min_only=True)
        moved = (labels == parent_index) & (distance < float(distance_mm) / 1000.0)
        candidate = labels.copy()
        candidate[moved] = child_index
        outputs, interfaces = close_labeled_parts(mesh, candidate, names)
        allowed = {frozenset((parent_name, child_name))
                   for child_name, parent_name in parents.items()}
        actual = {frozenset(key.split("--")) for key in interfaces}
        if actual != allowed or any(not part.is_watertight or part.body_count != 1
                                    for part in outputs.values()):
            raise RuntimeError(f"{parent}--{child} expansion is not a valid closed-loop split")
        labels = candidate
        records[child] = {"parent": parent, "distance_mm": float(distance_mm),
                          "moved_faces": int(moved.sum())}
    return labels, {"status": "HITL_CONFIRMED_CLOSED_LOOP", "expansions": records}


def binary_cut(
    active: np.ndarray,
    child_seeds: np.ndarray,
    rest_seeds: np.ndarray,
    adjacency: np.ndarray,
    affinity: np.ndarray,
    geodesic,
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    smoothness: float,
) -> np.ndarray:
    if not active[child_seeds].all() or not active[rest_seeds].all():
        raise RuntimeError("a seed was consumed by an earlier tree cut")
    old = np.flatnonzero(active)
    inverse = np.full(len(active), -1, dtype=np.int64)
    inverse[old] = np.arange(len(old))
    internal = active[adjacency].all(axis=1)
    edges = inverse[adjacency[internal]]
    subgraph = geodesic[old][:, old]
    child_distance = csgraph.dijkstra(
        subgraph, directed=False, indices=inverse[child_seeds], min_only=True
    )
    rest_distance = csgraph.dijkstra(
        subgraph, directed=False, indices=inverse[rest_seeds], min_only=True
    )
    _count, component = csgraph.connected_components(subgraph, directed=False)
    child_components = set(component[inverse[child_seeds]])
    rest_components = set(component[inverse[rest_seeds]])
    force_child = np.zeros(len(old), dtype=bool)
    force_rest = np.zeros(len(old), dtype=bool)
    for component_id in np.unique(component):
        selected = component == component_id
        has_child = component_id in child_components
        has_rest = component_id in rest_components
        if has_child and has_rest:
            continue
        if not has_child and not has_rest:
            center = np.average(face_centers[old[selected]], axis=0,
                                weights=face_areas[old[selected]])
            has_child = cKDTree(face_centers[child_seeds]).query(center)[0] <= \
                cKDTree(face_centers[rest_seeds]).query(center)[0]
        (force_child if has_child else force_rest)[selected] = True
        child_distance[selected] = rest_distance[selected] = 1.0

    denominator = np.maximum(child_distance + rest_distance, 1e-12)
    area_weight = np.clip(face_areas[old] / np.median(face_areas[old]), 0.1, 10.0)
    child_cost = child_distance / denominator * area_weight
    rest_cost = rest_distance / denominator * area_weight
    hard = 1e6
    child_local, rest_local = inverse[child_seeds], inverse[rest_seeds]
    child_cost[child_local], rest_cost[child_local] = 0.0, hard
    child_cost[rest_local], rest_cost[rest_local] = hard, 0.0
    child_cost[force_child], rest_cost[force_child] = 0.0, hard
    child_cost[force_rest], rest_cost[force_rest] = hard, 0.0

    result = np.zeros(len(active), dtype=bool)
    result[old] = _maximum_flow_side(
        len(old),
        edges,
        smoothness * affinity[internal],
        child_cost,
        rest_cost,
    )
    return result


def graph_labels(
    mesh: trimesh.Trimesh,
    names: list[str],
    parents: dict[str, str],
    order: list[str],
    seeds: dict[str, np.ndarray],
    adjacency: np.ndarray,
    affinity: np.ndarray,
    geodesic,
    smoothness: float,
) -> tuple[np.ndarray, dict, dict]:
    areas, centers = mesh.area_faces, mesh.triangles_center
    active = np.ones(len(mesh.faces), dtype=bool)
    labels = np.full(len(mesh.faces), -1, dtype=np.int16)
    planes, active_masks = {}, {}
    for name in order:
        active_before = active.copy()
        active_masks[name] = active_before
        child_indices = [names.index(child) for child, parent in parents.items() if parent == name]
        attached = np.zeros(len(mesh.faces), dtype=bool)
        for child_index in child_indices:
            pair = labels[adjacency]
            child_boundary = (pair[:, 0] == child_index) ^ (pair[:, 1] == child_index)
            attached[adjacency[child_boundary].ravel()] = True
        attached &= active
        child_seeds = np.unique(np.r_[seeds[name], np.flatnonzero(attached)])
        rest_names = [other for other in names if other != name and active[seeds[other]].all()]
        child = binary_cut(
            active,
            child_seeds,
            np.unique(np.concatenate([seeds[other] for other in rest_names])),
            adjacency,
            affinity,
            geodesic,
            centers,
            areas,
            smoothness,
        )
        child &= active
        normal, offset, agreement = separator_plane(
            centers, areas, active, child, float(np.linalg.norm(mesh.extents))
        )
        labels[child] = names.index(name)
        active &= ~child
        boundary = active_before[adjacency].all(axis=1) & (
            child[adjacency[:, 0]] ^ child[adjacency[:, 1]]
        )
        planes[name] = {
            "parent": parents[name],
            "normal": normal.tolist(),
            "offset_m": offset,
            "area_weighted_graph_label_agreement": agreement,
            "graph_boundary_edges": int(np.count_nonzero(boundary)),
        }
    labels[active] = names.index(next(name for name in names if name not in parents))
    if np.any(labels < 0):
        raise RuntimeError("not every source face received a link label")
    return labels, planes, active_masks


def separator_plane(
    centers: np.ndarray,
    areas: np.ndarray,
    active: np.ndarray,
    child: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(0)
    positive, negative = np.flatnonzero(child), np.flatnonzero(active & ~child)
    if not len(positive) or not len(negative):
        raise RuntimeError("graph cut produced an empty side")
    positive = rng.choice(positive, min(len(positive), 100_000), replace=False)
    negative = rng.choice(negative, min(len(negative), 100_000), replace=False)
    sample = np.r_[positive, negative]
    labels = np.r_[np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8)]
    center = centers[sample].mean(axis=0)
    classifier = LinearSVC(C=100.0, dual="auto", max_iter=20_000, tol=1e-6)
    classifier.fit((centers[sample] - center) / scale, labels)
    weights = classifier.coef_[0]
    length = np.linalg.norm(weights)
    normal = weights / length
    offset = float(center @ normal - classifier.intercept_[0] * scale / length)
    predicted = centers @ normal >= offset
    relevant = active
    agreement = float(
        np.sum(areas[relevant] * (predicted[relevant] == child[relevant]))
        / np.sum(areas[relevant])
    )
    return normal, offset, agreement


def threshold_candidate(
    method: str,
    normal: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    active: np.ndarray,
    child: np.ndarray,
    child_seeds: np.ndarray,
    rest_seeds: np.ndarray,
) -> dict | None:
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    if np.mean(centers[child_seeds] @ normal) < np.mean(centers[rest_seeds] @ normal):
        normal = -normal
    lower = float(np.max(centers[rest_seeds] @ normal))
    upper = float(np.min(centers[child_seeds] @ normal))
    if lower >= upper:
        return None

    selected = np.flatnonzero(active)
    projection = centers[selected] @ normal
    order = np.argsort(projection)
    projection = projection[order]
    target = child[selected][order]
    weights = areas[selected][order]
    positive_left = np.cumsum(weights * target)
    negative_left = np.cumsum(weights * ~target)
    error = positive_left[:-1] + negative_left[-1] - negative_left[:-1]
    thresholds = (projection[:-1] + projection[1:]) / 2
    valid = (thresholds > lower) & (thresholds < upper)
    if not valid.any():
        return None
    index = int(np.flatnonzero(valid)[np.argmin(error[valid])])
    agreement = float(1.0 - error[index] / weights.sum())
    return {
        "selection": method,
        "normal": normal.tolist(),
        "offset_m": float(thresholds[index]),
        "area_weighted_graph_label_agreement": agreement,
    }


def geometry_plane_candidates(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    names: list[str],
    name: str,
    parent: str,
    active: np.ndarray,
    seeds: dict[str, np.ndarray],
) -> list[dict]:
    centers, areas = mesh.triangles_center, mesh.area_faces
    child = labels == names.index(name)
    parent_faces = labels == names.index(parent)
    rest_seeds = np.unique(
        np.concatenate(
            [values for other, values in seeds.items() if other != name and active[values].all()]
        )
    )
    normals = []
    for prefix, mask in (("child_pca", child), ("parent_pca", parent_faces)):
        values, axes = np.linalg.eigh(np.cov(centers[mask], rowvar=False))
        elongation = float(np.sqrt(values[2] / max(values[1], 1e-12)))
        normals.extend(
            (
                f"{prefix}_{axis}",
                axes[:, axis],
                bool(prefix == "parent_pca" and axis == 2 and elongation > 2.5),
            )
            for axis in range(3)
        )
    interface = child[mesh.face_adjacency[:, 0]] ^ child[mesh.face_adjacency[:, 1]]
    points = mesh.vertices[np.unique(mesh.face_adjacency_edges[interface])]
    _values, axes = np.linalg.eigh(np.cov(points, rowvar=False))
    normals.append(("graph_boundary_pca", axes[:, 0], False))

    candidates, kept_normals = [], []
    for method, normal, preferred in normals:
        if any(abs(np.dot(normal, previous)) > 0.999 for previous in kept_normals):
            continue
        kept_normals.append(normal)
        candidate = threshold_candidate(
            method, normal, centers, areas, active, child, seeds[name], rest_seeds
        )
        if candidate is not None:
            candidate["elongated_parent_axis_preferred"] = preferred
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            item["elongated_parent_axis_preferred"],
            item["area_weighted_graph_label_agreement"],
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pick", action="store_true", help="replace config seeds in the Vedo picker")
    parser.add_argument("--paint-labels", action="store_true",
                        help="correct exact original-face ownership before closing links")
    parser.add_argument("--review-boundary-patches", action="store_true",
                        help="reassign a small set of exact patches around joints")
    parser.add_argument("--smoothness", type=float, default=2.0)
    parser.add_argument("--crease-angle", type=float, default=18.0)
    parser.add_argument("--min-plane-agreement", type=float, default=0.97)
    args = parser.parse_args()
    if args.smoothness <= 0 or args.crease_angle <= 0:
        parser.error("smoothness and crease-angle must be positive")
    if not 0.5 < args.min_plane_agreement <= 1.0:
        parser.error("min-plane-agreement must be in (0.5, 1]")

    mesh = load_mesh(args.input)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    names, parents, order = validate_config(config)
    if args.pick:
        pick_seeds(mesh, config, args.config)
    seeds, seed_audit = seed_faces(mesh, config["seeds"])
    adjacency, affinity, geodesic = graph_data(mesh, args.crease_angle)
    areas = mesh.area_faces

    root = config["root"]
    labels, graph_boundaries, active_masks = graph_labels(
        mesh, names, parents, order, seeds, adjacency, affinity, geodesic, args.smoothness
    )
    motion_faces, motion_ownership = motion_seed_faces(mesh, labels, names, parents, config)
    labels, motion_refinement = refine_motion_labels(
        mesh, labels, motion_faces, names, parents, order, seeds, config,
        adjacency, affinity, args.smoothness,
    )
    if args.paint_labels:
        labels = paint_face_labels(mesh, labels, names, parents)
    boundary_review = {"status": "NOT_RUN"}
    if config.get("closed_loop_child_expansion_mm"):
        labels, boundary_review = expand_child_boundaries(
            mesh, labels, names, parents, config["closed_loop_child_expansion_mm"]
        )
    if args.review_boundary_patches:
        labels, boundary_review = review_boundary_patches(mesh, labels, names, parents)
    outputs, interfaces = close_labeled_parts(mesh, labels, names)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "graph_face_labels.npy", labels)
    persisted = {}
    for name in names:
        path = args.output / f"{name}_metric_watertight.ply"
        outputs[name].export(path)
        persisted[name] = trimesh.load_mesh(path, process=False)

    stats = {name: part_stats(persisted[name]) for name in names}
    source_volume = abs(float(mesh.volume)) * 1e6
    volume_sum = sum(item["volume_cm3"] for item in stats.values())
    closure = abs(volume_sum - source_volume)
    all_vertices = np.vstack([part.vertices for part in persisted.values()])
    displacement = float(cKDTree(all_vertices).query(mesh.vertices)[0].max() * 1000)
    passed = (
        all(
            item["watertight"]
            and item["winding_consistent"]
            and item["body_count"] == 1
            and item["boundary_edges"] == 0
            and item["nonmanifold_edges"] == 0
            for item in stats.values()
        )
        and closure < 1e-4
        and displacement < 1e-4
    )
    if not passed:
        raise RuntimeError("metric part-separation gate failed")

    audit = {
        "status": "PASS",
        "method": "RORA_core_prior_geodesic_min_cut_constrained_interface_caps",
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "tree": {"root": root, "parents": parents, "leaf_to_root_cut_order": order},
        "parameters": {
            "smoothness": args.smoothness,
            "crease_angle_degrees": args.crease_angle,
            "minimum_plane_agreement": args.min_plane_agreement,
        },
        "semantic_seed_source": config.get("seed_source", "unspecified_HITL_config"),
        "semantic_status": ("HITL_CONFIRMED" if config.get("seed_source") in {
                            "original_surface_HITL_confirmed", "RORA_HITL_dense_safe_hulls",
                            "RORA_HITL_high_margin_core_hull"}
                            else "PROVISIONAL"),
        "exact_face_ownership": ("HITL_CORRECTED" if (args.paint_labels
                                                       or args.review_boundary_patches)
                                 else "AUTOMATIC_FROM_SEEDS"),
        "joint_boundary_review": boundary_review,
        "seeds": seed_audit,
        "graph_boundaries": graph_boundaries,
        "motion_ownership": motion_ownership,
        "motion_refinement": motion_refinement,
        "interfaces": interfaces,
        "parts": stats,
        "source_volume_cm3": source_volume,
        "part_volume_sum_cm3": volume_sum,
        "volume_closure_error_cm3": closure,
        "max_retained_source_vertex_displacement_mm": displacement,
        "thickness_preservation": "exact: source exterior vertices were not moved",
    }
    (args.output / "part_separation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
