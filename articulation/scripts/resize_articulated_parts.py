#!/usr/bin/env python3
"""Resize split watertight links to measured thickness, then verify motion and build URDF."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix, csgraph, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from articulate_disconnected_ply import sweep
from rora_prior_split_ply import compile_urdf, forbidden_config_paths
from separate_ply_links import part_stats, validate_config
from static_rora_metric_parts import MAX_PENETRATING_SAMPLE_FRACTION, _physical_properties


_PYMESHLAB = None


def _atomic_json(path: Path, value: dict) -> None:
    """Replace a checkpoint without leaving a half-written resume file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _arap_vertices(mesh: trimesh.Trimesh, protected: np.ndarray,
                   body: np.ndarray, body_targets: np.ndarray) -> np.ndarray:
    """Use Open3D's installed constrained ARAP solver for the neutral band."""
    protected_mask = np.zeros(len(mesh.vertices), dtype=bool); protected_mask[protected] = True
    active_faces = mesh.faces[~protected_mask[mesh.faces].all(axis=1)]
    used = np.unique(active_faces)
    inverse = np.full(len(mesh.vertices), -1, dtype=np.int64)
    inverse[used] = np.arange(len(used))
    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices[used]),
        o3d.utility.Vector3iVector(inverse[active_faces].astype(np.int32)),
    )
    active_protected = protected[np.isin(protected, used)]
    active_body_mask = np.isin(body, used)
    active_body = body[active_body_mask]
    indices = inverse[np.r_[active_protected, active_body]].astype(np.int32)
    positions = np.vstack((mesh.vertices[active_protected], body_targets[active_body_mask]))
    deformed = source.deform_as_rigid_as_possible(
        o3d.utility.IntVector(indices.tolist()),
        o3d.utility.Vector3dVector(positions), 30,
        o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed, 0.01,
    )
    vertices = np.asarray(mesh.vertices).copy()
    vertices[used] = np.asarray(deformed.vertices)
    if not np.isfinite(vertices).all():
        raise RuntimeError("ARAP solve produced non-finite coordinates")
    return vertices


def roi_affine_resize(
    mesh: trimesh.Trimesh,
    targets_mm: dict[str, float],
    axes: dict[str, np.ndarray],
    measurement_vertices: np.ndarray,
    protected_vertices: np.ndarray,
    solver: str = "arap",
) -> tuple[trimesh.Trimesh, dict]:
    """Fit confirmed ROI dimensions with one blended affine displacement."""
    dimensions = tuple(targets_mm)
    if not dimensions or not set(dimensions) <= {"length", "width", "thickness"}:
        raise ValueError("ROI targets must use length, width, and/or thickness")
    measurement = np.unique(np.asarray(measurement_vertices, dtype=np.int64))
    protected = np.unique(np.asarray(protected_vertices, dtype=np.int64))
    # Face regions naturally share boundary vertices; protected ownership wins.
    measurement = np.setdiff1d(measurement, protected, assume_unique=True)
    protected_faces = np.flatnonzero(np.isin(mesh.faces, protected).any(axis=1))
    if (len(measurement) < 4
            or measurement.min(initial=0) < 0 or protected.min(initial=0) < 0
            or measurement.max(initial=0) >= len(mesh.vertices)
            or protected.max(initial=0) >= len(mesh.vertices)):
        raise ValueError("measurement/protected ROI vertex masks are invalid or overlap")
    basis = np.asarray([axes[name] for name in dimensions], dtype=float)
    basis /= np.linalg.norm(basis, axis=1)[:, None]
    if not np.isfinite(basis).all() or np.max(np.abs(basis @ basis.T - np.eye(len(basis)))) > 1e-3:
        raise ValueError("confirmed dimension axes must be finite and mutually orthogonal")
    points = np.asarray(mesh.vertices)
    target = np.asarray([float(targets_mm[name]) / 1000.0 for name in dimensions])
    source_intersections = _self_intersection_faces(mesh)
    source_intersection_area = float(mesh.area_faces[source_intersections].sum())
    accepted = None
    for rings in ((4, 8, 12, 16, 24, 32) if len(protected_faces) else (0,)):
        automatic_neutral = (_face_region_handles(mesh, protected_faces, rings=rings)
                             if rings else np.empty(0, dtype=np.int64))
        body = np.setdiff1d(measurement, automatic_neutral, assume_unique=False)
        if len(body) < 4:
            continue
        center = np.median(points[body], axis=0)
        before = np.ptp((points[body] - center) @ basis.T, axis=0)
        if np.any(target <= 0) or np.any(before <= 0):
            continue
        scales = target / before
        if np.any((scales < 0.25) | (scales > 4.0)):
            continue
        local = points - center
        body_displacement = sum(
            np.outer((scales[index] - 1.0) * (local[body] @ axis), axis)
            for index, axis in enumerate(basis)
        )
        resized = mesh.copy()
        if solver == "scalar_harmonic":
            weights = _harmonic_resize_weights(mesh, protected, body)
            resized.vertices = points + sum(
                np.outer(weights * (scales[index] - 1.0) * (local @ axis), axis)
                for index, axis in enumerate(basis)
            )
        elif solver == "arap":
            resized.vertices = _arap_vertices(
                mesh, protected, body, points[body] + body_displacement
            )
        else:
            raise ValueError(f"unknown ROI deformation solver: {solver}")
        displacement = resized.vertices - points
        quality = _deformation_quality(mesh, resized.vertices)
        resized_intersections = _self_intersection_faces(resized)
        resized_intersection_area = float(
            resized.area_faces[resized_intersections].sum()
        )
        intersection_gate = (
            not np.any(resized_intersections & ~source_intersections)
            and resized_intersection_area <= source_intersection_area * 1.005 + 1e-12
        )
        if (resized.is_watertight and resized.is_winding_consistent
                and resized.body_count == 1 and quality["orientation_preserving"]
                and intersection_gate):
            measurement = body
            accepted = rings
            break
    if accepted is None:
        raise RuntimeError(
            "ROI affine deformation failed the topology/orientation gate: "
            f"watertight={resized.is_watertight}, winding={resized.is_winding_consistent}, "
            f"bodies={resized.body_count}, quality={quality}"
        )
    achieved = np.ptp((resized.vertices[measurement] - center) @ basis.T, axis=0)
    tolerance = {
        name: (max(0.2, float(np.median(mesh.edges_unique_length) * 500.0))
               if name == "thickness" else 1.0)
        for name in dimensions
    }
    error = np.abs(achieved - target) * 1000.0
    if any(error[index] > tolerance[name] for index, name in enumerate(dimensions)):
        raise RuntimeError(f"ROI dimension fit missed tolerance: {dict(zip(dimensions, error))}")
    protected_motion = (np.linalg.norm(displacement[protected], axis=1)
                        if len(protected) else np.zeros(1))
    return resized, {
        "method": f"confirmed_ROI_joint_affine_{solver}_blend",
        "dimensions": {name: {
            "before_mm": float(before[index] * 1000.0),
            "target_mm": float(target[index] * 1000.0),
            "achieved_mm": float(achieved[index] * 1000.0),
            "tolerance_mm": tolerance[name],
        } for index, name in enumerate(dimensions)},
        "measurement_vertices": int(len(measurement)),
        "protected_vertices": int(len(protected)),
        "automatic_protected_neutral_buffer_vertices": int(len(automatic_neutral)),
        "automatic_protected_neutral_buffer_rings": int(accepted),
        "maximum_protected_displacement_mm": float(protected_motion.max() * 1000.0),
        "affine_scales": scales.tolist(),
        "deformation_quality": quality,
        "self_intersection_audit": {
            "source_faces": int(source_intersections.sum()),
            "resized_faces": int(resized_intersections.sum()),
            "new_face_ids": int(np.count_nonzero(
                resized_intersections & ~source_intersections
            )),
            "source_area_mm2": source_intersection_area * 1e6,
            "resized_area_mm2": resized_intersection_area * 1e6,
            "status": "PASS",
        },
    }


def automatic_dimension_roi(mesh: trimesh.Trimesh, joint_origins: list[np.ndarray],
                            interface_vertices: np.ndarray,
                            target_thickness_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derive body/protected handles and physical axes without face painting."""
    protected = np.unique(np.asarray(interface_vertices, dtype=np.int64))
    radius = max(0.020, 3.0 * float(target_thickness_mm) / 1000.0)
    if joint_origins:
        near_joint = np.min(
            np.linalg.norm(mesh.vertices[:, None, :] - np.asarray(joint_origins)[None, :, :], axis=2),
            axis=1,
        ) <= radius
        protected = np.union1d(protected, np.flatnonzero(near_joint))
    if _proxy_has_self_intersection(mesh):
        intersections = _self_intersection_faces(mesh)
        if np.any(intersections):
            protected = np.union1d(protected, _face_region_handles(mesh, intersections))
    protected_faces = np.isin(mesh.faces, protected).any(axis=1)
    neutral = (_face_region_handles(mesh, np.flatnonzero(protected_faces), rings=8)
               if np.any(protected_faces) else protected)
    measurement = np.setdiff1d(np.arange(len(mesh.vertices)), neutral)
    if len(measurement) < 100:
        raise RuntimeError("automatic joint protection leaves too little measurement body")
    thickness, _body = _thickness_diagnostic(mesh)
    thickness_axis = np.asarray(thickness["axis"], dtype=float)
    _values, pca = np.linalg.eigh(np.cov(mesh.vertices[measurement], rowvar=False))
    length_axis = pca[:, -1] - thickness_axis * float(pca[:, -1] @ thickness_axis)
    length_axis /= np.linalg.norm(length_axis)
    width_axis = np.cross(thickness_axis, length_axis)
    width_axis /= np.linalg.norm(width_axis)
    return measurement, protected, {
        "length": length_axis.tolist(), "width": width_axis.tolist(),
        "thickness": thickness_axis.tolist(), "source": "automatic_body_PCA_and_antipodal_thickness",
        "joint_protection_radius_mm": radius * 1000.0,
    }
def _paint_dimension_roi(mesh: trimesh.Trimesh, initial: np.ndarray,
                         locked_faces: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    """Three-state geodesic face brush followed by explicit PCA-axis approval."""
    import vedo

    labels = np.asarray(initial, dtype=np.int8).copy()
    centers, adjacency = mesh.triangles_center, mesh.face_adjacency
    tree = cKDTree(centers)
    lengths = np.linalg.norm(centers[adjacency[:, 0]] - centers[adjacency[:, 1]], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths] + 1e-12,
                        (np.r_[adjacency[:, 0], adjacency[:, 1]],
                         np.r_[adjacency[:, 1], adjacency[:, 0]])),
                       shape=(len(labels), len(labels))).tocsr()
    palette = np.asarray([[150, 150, 150, 255], [40, 170, 70, 255],
                          [220, 55, 45, 255]], dtype=np.uint8)
    actor = vedo.Mesh([mesh.vertices, mesh.faces]); actor.cellcolors = palette[labels]
    plotter = vedo.Plotter(title=f"Dimension ROI — {name}")
    status = vedo.Text2D("", pos="top-left", s=.72, bg="white", c="black", alpha=.9)
    state = {"mode": 1, "radius_mm": 6.0, "undo": [], "finished": False}
    names = ("Neutral transition", "Measurement body", "Protected housing/rim")

    def refresh(message=""):
        counts = np.bincount(labels, minlength=3)
        status.text(
            f"DIMENSION ROI — {name}\n" + " | ".join(
                f"{index}:{label}={counts[index]} faces" for index, label in enumerate(names)
            ) + f"\nBrush={names[state['mode']]} radius={state['radius_mm']:.1f} mm"
            "\nLeft paint | 0/1/2 state | [ ] size | U undo | Enter validate axes"
            + (f"\n{message}" if message else "")
        )
        actor.cellcolors = palette[labels]; plotter.render()

    def paint(event):
        point = getattr(event, "picked3d", None)
        if event.actor is not actor or point is None:
            return
        seed = int(tree.query(np.asarray(point))[1])
        distance = csgraph.dijkstra(graph, directed=False, indices=seed,
                                    limit=state["radius_mm"] / 1000.0)
        selected = np.flatnonzero(np.isfinite(distance) & ~locked_faces)
        changed = selected[labels[selected] != state["mode"]]
        if len(changed):
            state["undo"].append((changed, labels[changed].copy()))
            labels[changed] = state["mode"]; refresh(f"painted {len(changed)} faces")

    def set_mode(mode):
        def callback(_widget=None, _event=None):
            state["mode"] = mode; refresh()
        return callback

    def undo(_widget=None, _event=None):
        if state["undo"]:
            indices, old = state["undo"].pop(); labels[indices] = old; refresh("undo")

    def finish(_widget=None, _event=None):
        if np.count_nonzero(labels == 1) < 10:
            refresh("SAVE BLOCKED: paint at least 10 measurement faces")
        else:
            state["finished"] = True; plotter.close()

    def radius_slider(widget, _event=None):
        state["radius_mm"] = float(widget.value); refresh()

    def add_button(callback, label, position):
        try:
            return plotter.add_button(callback, states=(label,), pos=position, size=13)
        except AttributeError as error:
            if "AddActor2D" not in str(error):
                raise
            button = vedo.Button(callback, states=(label,), pos=position, size=13)
            plotter.renderer.AddViewProp(button.actor)
            button.function_id = button.actor.AddObserver("PickEvent", button.function)
            plotter.buttons.append(button)
            return button

    def key(event):
        pressed = getattr(event, "keypress", "")
        if pressed in ("0", "1", "2"):
            state["mode"] = int(pressed); refresh()
        elif pressed in ("u", "U") and state["undo"]:
            undo()
        elif pressed in ("[", "braceleft"):
            state["radius_mm"] = max(1.0, state["radius_mm"] - 1); refresh()
        elif pressed in ("]", "braceright"):
            state["radius_mm"] = min(30.0, state["radius_mm"] + 1); refresh()
        elif pressed in ("Return", "Enter"):
            finish()

    plotter.add_callback("LeftButtonPress", paint); plotter.add_callback("KeyPress", key)
    plotter.add_slider(radius_slider, 1.0, 30.0, value=state["radius_mm"],
                       title="brush radius [mm]", pos=((.12, .19), (.88, .19)))
    add_button(set_mode(0), "Neutral", (.12, .08))
    add_button(set_mode(1), "Measurement", (.30, .08))
    add_button(set_mode(2), "Protected", (.50, .08))
    add_button(undo, "Undo", (.68, .08))
    add_button(finish, "Validate axes", (.85, .08))
    refresh(); plotter.show(actor, status, axes=1, interactive=True)
    if not state["finished"]:
        raise RuntimeError(f"dimension ROI for {name} closed without validation")

    measurement_vertices = np.unique(mesh.faces[labels == 1])
    centered = mesh.vertices[measurement_vertices] - np.median(
        mesh.vertices[measurement_vertices], axis=0)
    _values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axes = [vectors[:, 2], vectors[:, 1], vectors[:, 0]]
    axis_names = ["length", "width", "thickness"]
    approved = {"value": False}
    diagonal = float(np.linalg.norm(mesh.extents))
    center = np.median(mesh.vertices[measurement_vertices], axis=0)

    def axis_review():
        viewer = vedo.Plotter(title=f"Confirm dimension axes — {name}")
        mesh_actor = vedo.Mesh([mesh.vertices, mesh.faces]).alpha(.25)
        lines, labels3d = [], []
        colors = ("red", "green", "blue")
        for axis_name, axis, color in zip(axis_names, axes, colors):
            lines.append(vedo.Line(center - axis * diagonal * .25,
                                   center + axis * diagonal * .25).c(color).lw(7))
            labels3d.append(vedo.Text3D(axis_name, center + axis * diagonal * .27,
                                       s=diagonal * .018).c(color))
        text_actor = vedo.Text2D(
            "Red=length Green=width Blue=thickness\nSwap/cycle until physical axes match, then Approve",
            pos="top-left", c="black", bg="white", alpha=.9)
        action = {"value": None}
        def choose(value):
            def callback(_widget=None, _event=None):
                action["value"] = value; viewer.close()
            return callback
        def axis_button(callback, label, position):
            try:
                return viewer.add_button(callback, states=(label,), pos=position, size=14)
            except AttributeError as error:
                if "AddActor2D" not in str(error):
                    raise
                button = vedo.Button(callback, states=(label,), pos=position, size=14)
                viewer.renderer.AddViewProp(button.actor)
                button.function_id = button.actor.AddObserver("PickEvent", button.function)
                viewer.buttons.append(button)
                return button
        axis_button(choose("swap"), "Swap length/width", (.25, .07))
        axis_button(choose("cycle"), "Cycle axes", (.52, .07))
        axis_button(choose("approve"), "Approve axes", (.78, .07))
        viewer.show(mesh_actor, *lines, *labels3d, text_actor, axes=1, interactive=True)
        return action["value"]

    while not approved["value"]:
        action = axis_review()
        if action == "swap":
            axes[0], axes[1] = axes[1], axes[0]
        elif action == "cycle":
            axes[:] = axes[1:] + axes[:1]
        elif action == "approve":
            approved["value"] = True
        else:
            raise RuntimeError(f"dimension axes for {name} closed without approval")
    return labels, {key: np.asarray(axis).tolist() for key, axis in zip(axis_names, axes)}


def edit_dimension_rois(parts_dir: Path, config_path: Path, output: Path) -> Path:
    """Create resumable ROI masks and a B-ready config from saved exact parts."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    forbidden = forbidden_config_paths(config)
    if forbidden:
        raise ValueError(f"GT/target-volume fields are forbidden in the ROI config: {forbidden}")
    names, parents, _order = validate_config(config)
    parts = {name: trimesh.load_mesh(parts_dir / f"{name}_metric_watertight.ply", process=False)
             for name in names}
    protected = {name: set() for name in names}
    for child, parent in parents.items():
        parent_distance, child_nearest = cKDTree(parts[child].vertices).query(parts[parent].vertices)
        tolerance = 1e-8 * max(np.linalg.norm(parts[parent].extents),
                               np.linalg.norm(parts[child].extents))
        shared = np.flatnonzero(parent_distance <= max(tolerance, 1e-9))
        protected[parent].update(shared.tolist())
        protected[child].update(child_nearest[shared].tolist())
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"parts_source": str(parts_dir.resolve()), "parts": {}}
    profile = config.setdefault("dimension_constraints", {}).setdefault("B", {})
    profile.setdefault("thickness_mm", config.get("measured_thickness_mm", {}))
    for name, mesh in parts.items():
        source = parts_dir / f"{name}_metric_watertight.ply"
        binding = hashlib.sha256(source.read_bytes()).hexdigest()
        labels_path = output / f"{name}_dimension_roi_faces.npy"
        initial = np.load(labels_path) if labels_path.exists() else np.zeros(len(mesh.faces), np.int8)
        locked = np.zeros(len(mesh.faces), dtype=bool)
        if protected[name]:
            locked = np.isin(mesh.faces, np.asarray(sorted(protected[name]))).any(axis=1)
            initial[locked] = 2
        labels, axes = _paint_dimension_roi(mesh, initial, locked, name)
        np.save(labels_path, labels)
        measurement_path = output / f"{name}_measurement_vertices.npy"
        protected_path = output / f"{name}_protected_vertices.npy"
        np.save(measurement_path, np.unique(mesh.faces[labels == 1]))
        np.save(protected_path, np.unique(mesh.faces[labels == 2]))
        part_profile = profile.setdefault("parts", {}).setdefault(name, {})
        part_profile.update({"measurement_vertices_file": str(measurement_path.resolve()),
                             "protected_vertices_file": str(protected_path.resolve()),
                             "axes": axes})
        manifest["parts"][name] = {"source_sha256": binding, "axes_confirmed": True,
                                    "face_labels_file": str(labels_path.resolve())}
        _atomic_json(output / "dimension_roi_progress.json", manifest)
    ready = output / "roi_config.json"
    _atomic_json(ready, config)
    return ready


def _self_intersection_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    global _PYMESHLAB
    if _PYMESHLAB is None:
        with contextlib.redirect_stdout(io.StringIO()):
            import pymeshlab
        _PYMESHLAB = pymeshlab
    mesh_set = _PYMESHLAB.MeshSet()
    mesh_set.add_mesh(_PYMESHLAB.Mesh(
        vertex_matrix=np.asarray(mesh.vertices), face_matrix=np.asarray(mesh.faces)
    ))
    mesh_set.compute_selection_by_self_intersections_per_face()
    return np.asarray(mesh_set.current_mesh().face_selection_array(), dtype=bool)


def _proxy_has_self_intersection(mesh: trimesh.Trimesh) -> bool:
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    proxy = legacy.simplify_quadric_decimation(min(20_000, len(mesh.faces)))
    return bool(proxy.is_self_intersecting())


def _raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _local_thickness_profile(mesh: trimesh.Trimesh, axis: np.ndarray,
                             maximum_samples: int = 20_000) -> tuple[dict, np.ndarray]:
    """Measure the dominant antipodal surface separation along one candidate axis."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    normals, areas = np.asarray(mesh.face_normals), np.asarray(mesh.area_faces)
    alignment = normals @ axis
    eligible = np.flatnonzero(alignment > 0.65)
    if len(eligible) < 100:
        raise RuntimeError("too little surface area faces the candidate thickness axis")
    if len(eligible) > maximum_samples:
        rng = np.random.default_rng(0)
        eligible = rng.choice(
            eligible, maximum_samples, replace=False,
            p=areas[eligible] / areas[eligible].sum(),
        )
    epsilon = max(float(np.linalg.norm(mesh.extents)) * 1e-7, 1e-8)
    origins = mesh.triangles_center[eligible] - axis * epsilon
    rays = np.c_[origins, np.repeat((-axis)[None, :], len(origins), axis=0)]
    hits = _raycasting_scene(mesh).cast_rays(
        o3d.core.Tensor(rays.astype(np.float32))
    )
    distance = hits["t_hit"].numpy().astype(float)
    target_faces = hits["primitive_ids"].numpy().astype(np.int64)
    valid = np.isfinite(distance) & (target_faces >= 0) & (target_faces < len(mesh.faces))
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= alignment[target_faces[valid_indices]] < -0.65
    valid &= distance > 2.0 * epsilon
    source_faces = eligible[valid]
    target_faces = target_faces[valid]
    distance = distance[valid]
    if len(distance) < 500:
        raise RuntimeError("fewer than 500 antipodal thickness rays were valid")

    low, high = np.percentile(distance, [1, 99])
    in_range = (distance >= low) & (distance <= high)
    values = distance[in_range]
    density_model = gaussian_kde(values)
    grid = np.linspace(low, high, 256)
    density = density_model(grid)
    peaks = find_peaks(density)[0]
    peak = int(peaks[np.argmax(density[peaks])]) if len(peaks) else int(np.argmax(density))
    half = density[peak] * 0.5
    left = peak
    while left > 0 and density[left] >= half:
        left -= 1
    right = peak
    while right + 1 < len(grid) and density[right] >= half:
        right += 1
    mode = (distance >= grid[left]) & (distance <= grid[right])
    if np.count_nonzero(mode) < 200:
        raise RuntimeError("dominant local-thickness mode has fewer than 200 ray pairs")

    pair_midpoints = (
        mesh.triangles_center[source_faces[mode]]
        - np.outer((distance[mode] + epsilon) * 0.5, axis)
    )
    body_vertices = np.unique(mesh.faces[np.r_[source_faces[mode], target_faces[mode]]])
    aligned_area_fraction = float(areas[np.abs(alignment) > 0.65].sum() / areas.sum())
    valid_fraction = float(len(distance) / len(eligible))
    mode_fraction = float(np.count_nonzero(mode) / len(distance))
    mode_peak = float(grid[peak])
    fwhm = float(grid[right] - grid[left])
    score = aligned_area_fraction * valid_fraction * mode_fraction / max(fwhm / mode_peak, 0.02)
    return {
        "axis": axis,
        "score": score,
        "dominant_thickness_m": mode_peak,
        "dominant_fwhm_m": fwhm,
        "center_plane_offset_m": float(np.median(pair_midpoints @ axis)),
        "aligned_surface_area_fraction": aligned_area_fraction,
        "valid_antipodal_ray_fraction": valid_fraction,
        "dominant_mode_ray_fraction": mode_fraction,
        "valid_ray_count": int(len(distance)),
        "dominant_mode_ray_count": int(np.count_nonzero(mode)),
        "local_thickness_percentiles_mm": {
            str(percentile): float(np.percentile(distance, percentile) * 1000.0)
            for percentile in (5, 25, 50, 75, 95)
        },
    }, body_vertices


def _thickness_diagnostic(mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
    vertices, normals, areas = np.asarray(mesh.vertices), np.asarray(mesh.face_normals), np.asarray(mesh.area_faces)
    normal_values, normal_axes = np.linalg.eigh((normals * areas[:, None]).T @ normals)
    _pca_values, pca_axes = np.linalg.eigh(np.cov(vertices, rowvar=False))
    candidates = [("area_weighted_dominant_opposing_normals", normal_axes[:, -1])]
    if abs(float(normal_axes[:, -1] @ pca_axes[:, 0])) < 0.995:
        candidates.append(("PCA_shortest_axis_fallback", pca_axes[:, 0]))
    profiles = []
    for source, axis in candidates:
        try:
            profile, body_vertices = _local_thickness_profile(mesh, axis)
            profiles.append((profile["score"], source, profile, body_vertices))
        except RuntimeError:
            continue
    if not profiles:
        raise RuntimeError("no candidate axis produced a reliable antipodal thickness profile")
    _score, source, profile, body_vertices = max(profiles, key=lambda item: item[0])
    normal_energy = normal_values / normal_values.sum()
    if normal_energy[-1] <= 0.5:
        raise RuntimeError(
            "the link is not plate-like: its dominant normal energy does not exceed the other axes combined"
        )
    profile.update({
        "axis_source": source,
        "normal_energy_fractions": normal_energy.tolist(),
        "plate_like_gate": "PASS_dominant_normal_energy_gt_0.5",
    })
    return profile, body_vertices


def _harmonic_resize_weights(mesh: trimesh.Trimesh, interface_vertices: np.ndarray,
                             body_vertices: np.ndarray) -> np.ndarray:
    """Minimum-Dirichlet-energy blend: exact interface=0, measured body=1."""
    count = len(mesh.vertices)
    interface_vertices = np.unique(np.asarray(interface_vertices, dtype=np.int64))
    body_vertices = np.setdiff1d(np.unique(body_vertices), interface_vertices, assume_unique=False)
    if not len(interface_vertices):
        return np.ones(count)
    if len(body_vertices) < 100:
        raise RuntimeError("too few dominant-thickness body vertices for constrained resizing")
    edges = np.asarray(mesh.edges_unique)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    conductance = 1.0 / np.maximum(lengths, np.median(lengths) * 0.1)
    adjacency = coo_matrix(
        (np.r_[conductance, conductance],
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(count, count),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    known = np.zeros(count, dtype=bool)
    values = np.zeros(count)
    known[interface_vertices] = True
    known[body_vertices] = True
    values[body_vertices] = 1.0
    unknown = np.flatnonzero(~known)
    if len(unknown):
        known_indices = np.flatnonzero(known)
        values[unknown] = spsolve(
            laplacian[unknown][:, unknown],
            -(laplacian[unknown][:, known_indices] @ values[known_indices]),
        )
    if not np.isfinite(values).all():
        raise RuntimeError("harmonic resize-weight solve produced a non-finite value")
    return np.clip(values, 0.0, 1.0)


def _interface_geodesic_distance(mesh: trimesh.Trimesh,
                                 interface_vertices: np.ndarray) -> np.ndarray:
    edges = np.asarray(mesh.edges_unique)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    graph = coo_matrix(
        (np.r_[lengths, lengths],
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    return csgraph.dijkstra(
        graph, directed=False, indices=np.asarray(interface_vertices), min_only=True
    )


def _deformation_quality(mesh: trimesh.Trimesh, vertices: np.ndarray) -> dict:
    candidate = mesh.copy()
    candidate.vertices = vertices
    area_ratio = candidate.area_faces / np.maximum(mesh.area_faces, 1e-20)
    normal_dot = np.einsum("ij,ij->i", mesh.face_normals, candidate.face_normals)
    edges = np.asarray(mesh.edges_unique)
    before = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    after = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    edge_ratio = after / np.maximum(before, 1e-20)
    original_face_edges = mesh.vertices[mesh.faces[:, [1, 2, 0]]] - mesh.vertices[mesh.faces]
    original_max_edge_sq = np.max(np.einsum(
        "fij,fij->fi", original_face_edges, original_face_edges
    ), axis=1)
    source_condition = 2.0 * mesh.area_faces / np.maximum(original_max_edge_sq, 1e-30)
    log_area = np.log(np.maximum(mesh.area_faces, 1e-30))
    log_median = float(np.median(log_area))
    log_mad = float(np.median(np.abs(log_area - log_median)))
    micro = log_area < log_median - 6.0 * 1.4826 * log_mad
    reliable = (source_condition > np.sqrt(np.finfo(float).eps)) & ~micro
    return {
        "orientation_preserving": bool(np.all(normal_dot[reliable] > 0.0)),
        "flipped_or_overrotated_reliable_faces": int(np.count_nonzero(normal_dot[reliable] <= 0.0)),
        "ignored_numerically_degenerate_source_faces": int(np.count_nonzero(~reliable)),
        "minimum_original_normal_dot": float(normal_dot[reliable].min()),
        "minimum_face_area_ratio": float(area_ratio.min()),
        "maximum_face_area_ratio": float(area_ratio.max()),
        "edge_stretch_percentiles": {
            str(p): float(np.percentile(edge_ratio, p)) for p in (0, 50, 95, 99, 100)
        },
    }


def _expanded_interface_handles(mesh: trimesh.Trimesh,
                                interface_vertices: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the exact cap and its connected numerical slivers rigid."""
    exact = np.unique(np.asarray(interface_vertices, dtype=np.int64))
    if not len(exact):
        return exact, 0
    log_area = np.log(np.maximum(mesh.area_faces, 1e-30))
    median = float(np.median(log_area))
    mad = float(np.median(np.abs(log_area - median)))
    micro_faces = np.flatnonzero(log_area < median - 6.0 * 1.4826 * mad)
    fixed = np.zeros(len(mesh.vertices), dtype=bool)
    fixed[exact] = True
    cap_faces = np.count_nonzero(fixed[mesh.faces], axis=1) >= 2
    fixed[np.unique(mesh.faces[cap_faces])] = True
    while len(micro_faces):
        connected = fixed[mesh.faces[micro_faces]].any(axis=1)
        if not np.any(connected):
            break
        vertices = np.unique(mesh.faces[micro_faces[connected]])
        new = vertices[~fixed[vertices]]
        if not len(new):
            break
        fixed[new] = True
    expanded = np.flatnonzero(fixed)
    return expanded, int(len(expanded) - len(exact))


def _face_region_handles(mesh: trimesh.Trimesh, faces: np.ndarray,
                         rings: int = 4) -> np.ndarray:
    touched = np.zeros(len(mesh.vertices), dtype=bool)
    touched[np.unique(mesh.faces[faces])] = True
    for _ in range(rings):
        region_faces = touched[mesh.faces].any(axis=1)
        touched[np.unique(mesh.faces[region_faces])] = True
    return np.flatnonzero(touched)


def thickness_resize(mesh: trimesh.Trimesh, target_mm: float,
                     joint_origins: list[np.ndarray],
                     interface_vertices: np.ndarray | None = None) -> tuple[trimesh.Trimesh, dict]:
    """Measurement-constrained, interface-preserving local-thickness resize."""
    vertices = np.asarray(mesh.vertices)
    exact_interface_vertices = np.asarray(
        [] if interface_vertices is None else interface_vertices, dtype=np.int64
    )
    interface_vertices, added_micro_handle_vertices = _expanded_interface_handles(
        mesh, exact_interface_vertices
    )
    source_proxy_intersects = _proxy_has_self_intersection(mesh)
    source_self_intersections = (
        _self_intersection_faces(mesh) if source_proxy_intersects else None
    )
    inherited_intersection_handle_vertices = 0
    if source_self_intersections is not None and np.any(source_self_intersections):
        inherited_handles = _face_region_handles(mesh, source_self_intersections)
        previous_count = len(interface_vertices)
        interface_vertices = np.union1d(interface_vertices, inherited_handles)
        inherited_intersection_handle_vertices = len(interface_vertices) - previous_count
    before, body_vertices = _thickness_diagnostic(mesh)
    axis = np.asarray(before["axis"])
    current = float(before["dominant_thickness_m"])
    target = float(target_mm) / 1000.0
    scale = target / current
    if not 0.25 <= scale <= 4.0:
        raise RuntimeError(
            f"target/current local-thickness scale {scale:.3f} is unsafe; check assignment or units"
        )
    center = float(before["center_plane_offset_m"])
    signed = vertices @ axis - center
    transition_m = 0.0
    maximum_transition_m = 0.0
    candidate_weights = None
    if len(interface_vertices):
        geodesic = _interface_geodesic_distance(mesh, interface_vertices)
        finite = geodesic[np.isfinite(geodesic)]
        maximum_transition_m = float(np.partition(finite, -100)[-100])
        if maximum_transition_m <= 0:
            raise RuntimeError("shared interface consumes the complete link surface")

        def candidate_weights(distance: float) -> np.ndarray:
            handles = np.flatnonzero(geodesic >= distance)
            return _harmonic_resize_weights(mesh, interface_vertices, handles)

        upper = maximum_transition_m
        upper_weights = candidate_weights(upper)
        upper_vertices = vertices + np.outer(
            upper_weights * (scale - 1.0) * signed, axis
        )
        upper_quality = _deformation_quality(mesh, upper_vertices)
        if not upper_quality["orientation_preserving"]:
            raise RuntimeError(
                "no interface-constrained deformation preserves reliable face orientation: "
                f"quality={upper_quality}"
            )
        lower = 0.0
        for _ in range(16):
            middle = (lower + upper) * 0.5
            middle_weights = candidate_weights(middle)
            middle_vertices = vertices + np.outer(
                middle_weights * (scale - 1.0) * signed, axis
            )
            middle_quality = _deformation_quality(mesh, middle_vertices)
            if middle_quality["orientation_preserving"]:
                upper = middle
            else:
                lower = middle
        transition_m = upper
        weights = candidate_weights(transition_m)
    else:
        weights = np.ones(len(vertices))

    def fit(chosen_weights: np.ndarray, initial_scale: float) -> tuple:
        fitted_scale = initial_scale
        candidate, profile = None, None
        for _iteration in range(3):
            candidate = mesh.copy()
            candidate.vertices = vertices + np.outer(
                chosen_weights * (fitted_scale - 1.0) * signed, axis
            )
            profile, _after_body = _local_thickness_profile(candidate, axis)
            achieved = float(profile["dominant_thickness_m"])
            if abs(achieved - target) <= target * 1e-4:
                break
            fitted_scale *= target / achieved
        if candidate is None or profile is None or not 0.25 <= fitted_scale <= 4.0:
            raise RuntimeError("fitted local-thickness scale left the safe range")
        return candidate, profile, fitted_scale

    def checked_candidate(chosen_weights: np.ndarray, initial_scale: float) -> tuple:
        nonlocal source_self_intersections
        candidate, profile, fitted_scale = fit(chosen_weights, initial_scale)
        quality = _deformation_quality(mesh, candidate.vertices)
        candidate_proxy_intersects = _proxy_has_self_intersection(candidate)
        if source_proxy_intersects or candidate_proxy_intersects:
            if source_self_intersections is None:
                source_self_intersections = _self_intersection_faces(mesh)
            candidate_self_intersections = _self_intersection_faces(candidate)
            new_faces = candidate_self_intersections & ~source_self_intersections
            source_region_vertices = np.zeros(len(mesh.vertices), dtype=bool)
            source_region_vertices[np.unique(mesh.faces[source_self_intersections])] = True
            source_neighborhood = source_region_vertices[mesh.faces].any(axis=1)
            new_count = int(np.count_nonzero(new_faces))
            new_outside_count = int(np.count_nonzero(new_faces & ~source_neighborhood))
            new_face_weight_max = float(
                chosen_weights[mesh.faces[new_faces]].max() if new_count else 0.0
            )
            if new_count:
                source_vertices = np.unique(mesh.faces[source_self_intersections])
                new_vertices = np.unique(mesh.faces[new_faces])
                new_distance = cKDTree(vertices[source_vertices]).query(vertices[new_vertices])[0]
                new_distance_mm = [float(new_distance.min() * 1000.0),
                                   float(new_distance.max() * 1000.0)]
            else:
                new_distance_mm = [0.0, 0.0]
            source_area = float(mesh.area_faces[source_self_intersections].sum())
            candidate_area = float(candidate.area_faces[candidate_self_intersections].sum())
            area_tolerance = max(
                float(np.median(mesh.area_faces)),
                MAX_PENETRATING_SAMPLE_FRACTION * source_area,
            )
            intersection_safe = candidate_area <= source_area + area_tolerance
            method = "20k_proxy_then_exact_pymeshlab_face_correspondence"
        else:
            candidate_self_intersections = np.zeros(len(mesh.faces), dtype=bool)
            new_count = 0
            new_outside_count = 0
            new_face_weight_max = 0.0
            new_distance_mm = [0.0, 0.0]
            source_area = candidate_area = area_tolerance = 0.0
            intersection_safe = True
            method = "20k_proxy_clear"
        return (
            quality["orientation_preserving"] and intersection_safe,
            candidate, profile, fitted_scale, quality,
            candidate_self_intersections, new_count, method,
            {
                "new_faces_outside_inherited_one_ring": new_outside_count,
                "maximum_resize_weight_on_new_face": new_face_weight_max,
                "new_face_vertex_distance_to_inherited_region_mm": new_distance_mm,
                "source_intersecting_area_mm2": source_area * 1e6,
                "resized_intersecting_area_mm2": candidate_area * 1e6,
                "mesh_face_area_tolerance_mm2": area_tolerance * 1e6,
            },
        )

    state = checked_candidate(weights, scale)
    if not state[0] and candidate_weights is not None:
        upper_state = checked_candidate(candidate_weights(maximum_transition_m), scale)
        if not upper_state[0]:
            raise RuntimeError(
                "no interface-constrained deformation preserves face orientation and avoids new "
                f"self-intersections: source_faces={np.count_nonzero(source_self_intersections)}, "
                f"resized_faces={np.count_nonzero(upper_state[5])}, new_faces={upper_state[6]}, "
                f"regression={upper_state[8]}"
            )
        lower, upper = transition_m, maximum_transition_m
        for _ in range(10):
            middle = (lower + upper) * 0.5
            middle_state = checked_candidate(candidate_weights(middle), scale)
            if middle_state[0]:
                upper, upper_state = middle, middle_state
            else:
                lower = middle
        transition_m = upper
        weights = candidate_weights(transition_m)
        state = upper_state

    (_accepted, resized, after, scale, deformation_quality,
     resized_self_intersections, new_self_intersection_count, intersection_method,
     intersection_regression) = state
    if not resized.is_watertight or not resized.is_winding_consistent or resized.body_count != 1:
        raise RuntimeError("local-thickness resize broke the closed mesh")
    if not deformation_quality["orientation_preserving"]:
        raise RuntimeError("local-thickness resize flipped or over-rotated a source triangle")
    if not _accepted:
        raise RuntimeError(
            f"local-thickness resize introduced a new self-intersection region: {intersection_regression}"
        )
    if source_self_intersections is None:
        source_self_intersections = np.zeros(len(mesh.faces), dtype=bool)
    edge_resolution_mm = float(np.median(mesh.edges_unique_length) * 1000.0)
    numerical_tolerance_mm = max(0.2, 0.5 * edge_resolution_mm)
    achieved_mm = float(after["dominant_thickness_m"] * 1000.0)
    if abs(achieved_mm - target_mm) > numerical_tolerance_mm or abs(float(resized.volume)) <= 0:
        raise RuntimeError(
            f"local thickness {achieved_mm:.4f} mm misses {target_mm:.4f} mm by more than "
            f"mesh-derived tolerance {numerical_tolerance_mm:.4f} mm"
        )
    interface_motion = (
        np.linalg.norm(
            resized.vertices[exact_interface_vertices] - vertices[exact_interface_vertices], axis=1
        ) if len(exact_interface_vertices) else np.zeros(1)
    )
    joint_projection = (
        np.asarray(joint_origins) @ axis if joint_origins else np.asarray([center])
    )
    before_record = {key: value for key, value in before.items() if key != "axis"}
    after_record = {key: value for key, value in after.items() if key not in {"axis", "score"}}
    matrix = np.eye(3) + (scale - 1.0) * np.outer(axis, axis)
    return resized, {
        "method": "antipodal_local_thickness_harmonic_interface_constrained_v2",
        "justification": (
            "dominant opposing-surface separation is fitted to measurement; exact shared interfaces "
            "are fixed; the unconstrained blend minimizes mesh-graph Dirichlet energy"
        ),
        "thickness_axis": axis.tolist(),
        "axis_source": before["axis_source"],
        "before_local_thickness": before_record,
        "target_thickness_mm": target_mm,
        "after_local_thickness": after_record,
        "after_dominant_thickness_mm": achieved_mm,
        "mesh_derived_numerical_tolerance_mm": numerical_tolerance_mm,
        "scale": scale,
        "body_affine_matrix": matrix.tolist(),
        "body_affine_translation_m": ((1.0 - scale) * center * axis).tolist(),
        "center_plane_offset_m": center,
        "center_plane_source": "median_midpoint_of_dominant_antipodal_ray_pairs",
        "dominant_body_handle_vertices": int(len(body_vertices)),
        "exact_interface_handle_vertices": int(len(exact_interface_vertices)),
        "interface_connected_micro_handle_vertices": added_micro_handle_vertices,
        "inherited_self_intersection_four_ring_handle_vertices": (
            inherited_intersection_handle_vertices
        ),
        "total_zero_displacement_handle_vertices": int(len(interface_vertices)),
        "automatically_selected_geodesic_transition_mm": transition_m * 1000.0,
        "transition_selection": (
            "minimum interface geodesic distance whose harmonic blend preserves reliable face "
            "orientation and limits inherited intersecting-area growth to 0.5 percent"
        ),
        "harmonic_weight_percentiles": np.percentile(weights, [0, 5, 50, 95, 100]).tolist(),
        "deformation_quality": deformation_quality,
        "self_intersection_audit": {
            "method": intersection_method,
            "source_intersecting_faces": int(np.count_nonzero(source_self_intersections)),
            "resized_intersecting_faces": int(np.count_nonzero(resized_self_intersections)),
            "new_face_ids_within_inherited_region": new_self_intersection_count,
            **intersection_regression,
            "inherited_intersections_status": (
                "NONE" if not np.any(source_self_intersections)
                else "PROVISIONAL_INHERITED_FROM_PRE_RESIZE_PART"
            ),
        },
        "maximum_interface_displacement_mm": float(interface_motion.max() * 1000.0),
        "joint_origin_projection_spread_mm": float(np.ptp(joint_projection) * 1000.0),
        "joint_projection_spread_over_target": float(np.ptp(joint_projection) / target),
        "affine_joint_compatibility": (
            "COMPATIBLE" if np.ptp(joint_projection) <= numerical_tolerance_mm / 1000.0
            else "INCOMPATIBLE_REQUIRES_CONSTRAINED_BLEND"
        ),
    }


def joint_anchored_affine_resize(
    mesh: trimesh.Trimesh, target_mm: float, joint_origins: list[np.ndarray]
) -> tuple[trimesh.Trimesh, dict]:
    """Scale only the measured thickness axis while keeping joint-axis points fixed."""
    before, _body_vertices = _thickness_diagnostic(mesh)
    axis = np.asarray(before["axis"])
    current_mm = float(before["dominant_thickness_m"] * 1000.0)
    scale = target_mm / current_mm
    if not 0.25 <= scale <= 4.0:
        raise RuntimeError(f"target/current local-thickness scale {scale:.3f} is unsafe")
    center = float(np.mean(np.asarray(joint_origins) @ axis))
    resized = mesh.copy()
    resized.vertices = mesh.vertices + np.outer(
        (scale - 1.0) * (mesh.vertices @ axis - center), axis
    )
    quality = _deformation_quality(mesh, resized.vertices)
    after, _ = _thickness_diagnostic(resized)
    achieved_mm = float(after["dominant_thickness_m"] * 1000.0)
    tolerance_mm = max(0.02, float(np.median(mesh.edges_unique_length) * 250.0))
    if (not resized.is_watertight or not resized.is_winding_consistent
            or resized.body_count != 1 or not quality["orientation_preserving"]
            or abs(achieved_mm - target_mm) > tolerance_mm):
        raise RuntimeError(
            f"joint-anchored affine resize failed: achieved={achieved_mm:.4f} mm, "
            f"target={target_mm:.4f} mm, quality={quality}"
        )
    return resized, {
        "method": "joint_origin_anchored_thickness_axis_affine",
        "before_dominant_thickness_mm": current_mm,
        "target_thickness_mm": target_mm,
        "after_dominant_thickness_mm": achieved_mm,
        "mesh_derived_numerical_tolerance_mm": tolerance_mm,
        "scale": scale,
        "thickness_axis": axis.tolist(),
        "joint_anchor_projection_m": center,
        "deformation_quality": quality,
    }


def measured_dimension_affine_resize(
    mesh: trimesh.Trimesh, targets_mm: dict[str, float], axes: dict[str, np.ndarray],
    anchor: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Apply one positive 3-axis affine from independent measured dimensions."""
    dimensions = tuple(targets_mm)
    basis = np.asarray([axes[name] for name in dimensions], dtype=float)
    basis /= np.linalg.norm(basis, axis=1)[:, None]
    if (set(dimensions) != {"length", "width", "thickness"}
            or np.max(np.abs(basis @ basis.T - np.eye(3))) > 1e-3):
        raise ValueError("coupled dimension resize needs three orthogonal measured axes")
    thickness, _body = _thickness_diagnostic(mesh)
    projection = np.asarray(mesh.vertices) @ basis.T
    before = np.ptp(projection, axis=0)
    thickness_index = dimensions.index("thickness")
    before[thickness_index] = float(thickness["dominant_thickness_m"])
    centers = (projection.min(axis=0) + projection.max(axis=0)) * 0.5
    centers[thickness_index] = float(thickness["center_plane_offset_m"])
    if anchor is not None:
        centers = np.asarray(anchor, dtype=float) @ basis.T
    targets = np.asarray([targets_mm[name] for name in dimensions], dtype=float) / 1000.0
    scales = targets / before
    if np.any((scales < 0.25) | (scales > 4.0)):
        raise RuntimeError(f"measured dimension scale is unsafe: {dict(zip(dimensions, scales))}")
    vertices = np.asarray(mesh.vertices).copy()
    for index, axis in enumerate(basis):
        vertices += np.outer((scales[index] - 1.0) *
                             (projection[:, index] - centers[index]), axis)
    resized = mesh.copy(); resized.vertices = vertices
    quality = _deformation_quality(mesh, vertices)
    if (not resized.is_watertight or not resized.is_winding_consistent
            or resized.body_count != 1 or not quality["orientation_preserving"]):
        raise RuntimeError(f"measured dimension affine failed: {quality}")
    return resized, {
        "method": ("joint_origin_anchored_measured_three_axis_affine"
                   if anchor is not None else "measured_three_axis_whole_part_affine"),
        "dimensions": {name: {
            "before_mm": float(before[index] * 1000.0),
            "target_mm": float(targets[index] * 1000.0),
            "affine_achieved_mm": float(np.ptp(vertices @ basis[index]) * 1000.0)
                if name != "thickness" else float(targets[index] * 1000.0),
        } for index, name in enumerate(dimensions)},
        "axes": {name: basis[index].tolist() for index, name in enumerate(dimensions)},
        "scales": scales.tolist(),
        "anchor_m": None if anchor is None else np.asarray(anchor, dtype=float).tolist(),
        "deformation_quality": quality,
    }


def coupled_joint_thickness_resize(
    meshes: dict[str, trimesh.Trimesh], targets: dict[str, float],
    origins: dict[str, list[np.ndarray]], interface_pairs: dict,
    dimensions: dict[str, dict] | None = None,
) -> tuple[dict[str, trimesh.Trimesh], dict]:
    """Resize housings with their bodies, then reconcile each paired cap exactly."""
    affine, records = {}, {}
    for name, mesh in meshes.items():
        if dimensions and name in dimensions:
            affine[name], records[name] = measured_dimension_affine_resize(
                mesh, dimensions[name]["targets_mm"], dimensions[name]["axes"]
            )
        else:
            affine[name], records[name] = joint_anchored_affine_resize(
                mesh, float(targets[name]), origins[name]
            )
    corrections = {name: [] for name in meshes}
    interface_blends = {}
    for key, (parent, child, parent_ids, child_ids) in interface_pairs.items():
        side_cache = {}
        for name, ids in ((parent, parent_ids), (child, child_ids)):
            side_cache[name] = (
                ids,
                _interface_geodesic_distance(meshes[name], ids),
                cKDTree(meshes[name].vertices[ids]).query(
                    meshes[name].vertices, workers=-1
                )[1],
            )
        parent_cap = affine[parent].vertices[parent_ids]
        child_cap = affine[child].vertices[child_ids]
        chosen = None
        # The closest safe blend to an equal parent/child compromise avoids
        # arbitrarily treating either housing as fixed.
        for multiplier in (3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            for alpha in sorted(np.linspace(0.0, 1.0, 101),
                                key=lambda value: abs(value - 0.5)):
                target = (1.0 - alpha) * parent_cap + alpha * child_cap
                safe = True
                for name, cap in ((parent, parent_cap), (child, child_cap)):
                    ids, distance, nearest = side_cache[name]
                    radius = max(0.020, multiplier * float(targets[name]) / 1000.0)
                    weight = np.clip(1.0 - distance / radius, 0.0, 1.0)
                    weight = weight * weight * (3.0 - 2.0 * weight)
                    vertices = affine[name].vertices + weight[:, None] * (target - cap)[nearest]
                    vertices[ids] = target
                    if not _deformation_quality(meshes[name], vertices)["orientation_preserving"]:
                        safe = False
                        break
                if safe:
                    chosen = float(alpha)
                    break
            if chosen is not None:
                break
        if chosen is None:
            target = meshes[parent].vertices[parent_ids]
            for multiplier in (3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
                safe = True
                for name, cap in ((parent, parent_cap), (child, child_cap)):
                    ids, distance, nearest = side_cache[name]
                    radius = max(0.020, multiplier * float(targets[name]) / 1000.0)
                    weight = np.clip(1.0 - distance / radius, 0.0, 1.0)
                    weight = weight * weight * (3.0 - 2.0 * weight)
                    vertices = affine[name].vertices + weight[:, None] * (target - cap)[nearest]
                    vertices[ids] = target
                    if not _deformation_quality(meshes[name], vertices)["orientation_preserving"]:
                        safe = False
                        break
                if safe:
                    chosen = -1.0
                    break
            if chosen is None:
                raise RuntimeError(f"no orientation-preserving shared cap blend for {key}")
        else:
            target = (1.0 - chosen) * parent_cap + chosen * child_cap
        interface_blends[key] = chosen
        corrections[parent].append((parent_ids, target - affine[parent].vertices[parent_ids]))
        corrections[child].append((child_ids, target - affine[child].vertices[child_ids]))

    result = {}
    for name, source in meshes.items():
        cached = [(ids, cap_delta, _interface_geodesic_distance(source, ids),
                   cKDTree(source.vertices[ids]).query(source.vertices, workers=-1)[1])
                  for ids, cap_delta in corrections[name]]
        resized = quality = achieved = chosen_radius = None
        for multiplier in (3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            chosen_radius = max(0.020, multiplier * float(targets[name]) / 1000.0)
            vertices = affine[name].vertices.copy()
            for ids, cap_delta, distance, nearest in cached:
                weight = np.clip(1.0 - distance / chosen_radius, 0.0, 1.0)
                weight = weight * weight * (3.0 - 2.0 * weight)
                vertices += weight[:, None] * cap_delta[nearest]
            for ids, cap_delta, _distance, _nearest in cached:
                vertices[ids] = affine[name].vertices[ids] + cap_delta
            candidate = source.copy(); candidate.vertices = vertices
            candidate_quality = _deformation_quality(source, vertices)
            if candidate_quality["orientation_preserving"]:
                resized, quality = candidate, candidate_quality
                after, _ = _thickness_diagnostic(resized)
                achieved = float(after["dominant_thickness_m"] * 1000.0)
                break
        tolerance = max(0.2, float(np.median(source.edges_unique_length) * 500.0))
        dimension_errors = {}
        if dimensions and name in dimensions:
            spec = dimensions[name]
            for dimension in ("length", "width"):
                axis = np.asarray(spec["axes"][dimension], dtype=float)
                axis /= np.linalg.norm(axis)
                value = float(np.ptp(resized.vertices @ axis) * 1000.0)
                dimension_errors[dimension] = value - float(spec["targets_mm"][dimension])
        if (resized is None or not resized.is_watertight or not resized.is_winding_consistent
                or resized.body_count != 1 or not quality["orientation_preserving"]
                or abs(achieved - float(targets[name])) > tolerance
                or any(abs(error) > 1.0 for error in dimension_errors.values())):
            raise RuntimeError(
                f"coupled joint resize failed for {name}: achieved={achieved}, "
                f"target={targets[name]}, dimension_errors_mm={dimension_errors}, quality={quality}"
            )
        source_intersections = _self_intersection_faces(source)
        resized_intersections = _self_intersection_faces(resized)
        source_area = float(source.area_faces[source_intersections].sum())
        resized_area = float(resized.area_faces[resized_intersections].sum())
        intersection_gate = (
            not np.any(resized_intersections & ~source_intersections)
            and resized_area <= source_area * 1.005 + 1e-12
        )
        result[name] = resized
        records[name].update({
            "method": "whole_part_affine_with_paired_cap_consensus_blend",
            "after_dominant_thickness_mm": achieved,
            "mesh_derived_numerical_tolerance_mm": tolerance,
            "joint_region_policy": (
                "whole housing follows body thickness scale; each paired cap uses the closest "
                "orientation-preserving parent/child affine blend to an equal compromise"
            ),
            "interface_affine_blend_child_weight": {
                key: alpha for key, alpha in interface_blends.items() if name in key.split("--")
            },
            "coupling_radius_mm": chosen_radius * 1000.0,
            "final_length_width_error_mm": dimension_errors,
            "deformation_quality": quality,
            "self_intersection_audit": {
                "source_faces": int(source_intersections.sum()),
                "resized_faces": int(resized_intersections.sum()),
                "source_area_mm2": source_area * 1e6,
                "resized_area_mm2": resized_area * 1e6,
                "status": "PASS" if intersection_gate else "REJECTED_AREA_OR_REGION_GROWTH",
            },
            "candidate_gate_status": "PASS" if intersection_gate else "REJECTED",
        })
    return result, records


def radial_joint_resize(
    parts: dict[str, trimesh.Trimesh], source_parts: dict[str, trimesh.Trimesh],
    interface_pairs: dict, joints: list[dict], names: list[str], scales: dict[str, float],
) -> tuple[dict[str, trimesh.Trimesh], dict]:
    """Scale visible joint radius about its approved axis and blend into each link."""
    vertices = {name: np.asarray(mesh.vertices).copy() for name, mesh in parts.items()}
    audit = {}
    for joint in joints:
        parent, child = names[joint["parent"] - 1], names[joint["child"] - 1]
        key = f"{parent}--{child}"
        requested_scale = float(scales[key])
        if not 0.4 <= requested_scale <= 2.5:
            raise ValueError(f"joint radial scale is unsafe for {key}: {requested_scale}")
        _parent, _child, parent_ids, child_ids = interface_pairs[key]
        origin = np.asarray(joint["axis"]["origin"], dtype=float)
        axis = np.asarray(joint["axis"]["n"], dtype=float); axis /= np.linalg.norm(axis)
        cached = {}
        for name, ids in ((parent, parent_ids), (child, child_ids)):
            distance = _interface_geodesic_distance(source_parts[name], ids)
            weight = np.clip(1.0 - distance / 0.045, 0.0, 1.0)
            weight = weight * weight * (3.0 - 2.0 * weight)
            for other_key, (other_parent, other_child, other_parent_ids,
                            other_child_ids) in interface_pairs.items():
                if other_key == key:
                    continue
                if name == other_parent:
                    weight[other_parent_ids] = 0.0
                elif name == other_child:
                    weight[other_child_ids] = 0.0
            offset = vertices[name] - origin
            cached[name] = (vertices[name].copy(), weight,
                            offset - np.outer(offset @ axis, axis))

        def safe(alpha: float) -> tuple[bool, dict[str, np.ndarray]]:
            scale = 1.0 + alpha * (requested_scale - 1.0)
            proposed = {name: base + weight[:, None] * (scale - 1.0) * radial
                        for name, (base, weight, radial) in cached.items()}
            return all(_deformation_quality(source_parts[name], value)["orientation_preserving"]
                       for name, value in proposed.items()), proposed

        accepted, proposed = safe(1.0)
        alpha = 1.0
        if not accepted:
            lower, upper = 0.0, 1.0
            for _ in range(16):
                middle = (lower + upper) * 0.5
                accepted, candidate = safe(middle)
                if accepted:
                    lower, proposed = middle, candidate
                else:
                    upper = middle
            alpha = lower
        for name, value in proposed.items():
            vertices[name] = value
        audit[key] = {
            "requested_radial_scale": requested_scale,
            "applied_radial_scale": 1.0 + alpha * (requested_scale - 1.0),
            "requested_fraction_applied": alpha,
            "geodesic_transition_mm": 45.0,
        }
    result = {}
    for name, mesh in parts.items():
        candidate = mesh.copy(); candidate.vertices = vertices[name]
        quality = _deformation_quality(source_parts[name], vertices[name])
        if (not candidate.is_watertight or not candidate.is_winding_consistent
                or candidate.body_count != 1 or not quality["orientation_preserving"]):
            raise RuntimeError(f"radial joint resize failed for {name}: {quality}")
        result[name] = candidate
        audit[name] = quality
    return result, audit


def joint_radial_scales_from_body_volume(
    joints: list[dict], names: list[str], resize_audit: dict,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Match each hinge's volume ratio to the adjacent resized link bodies."""
    scales, policy = {}, {}
    for joint in joints:
        parent, child = names[joint["parent"] - 1], names[joint["child"] - 1]
        key = f"{parent}--{child}"
        parent_ratio = (resize_audit[parent]["after_volume_cm3"]
                        / resize_audit[parent]["before_volume_cm3"])
        child_ratio = (resize_audit[child]["after_volume_cm3"]
                       / resize_audit[child]["before_volume_cm3"])
        # Shared hinge geometry needs one scale. The geometric mean treats both links
        # symmetrically; retaining axial length means V scales with radius squared.
        joint_volume_ratio = float(np.sqrt(parent_ratio * child_ratio))
        scales[key] = float(np.sqrt(joint_volume_ratio))
        policy[key] = {
            "parent_body_volume_ratio": float(parent_ratio),
            "child_body_volume_ratio": float(child_ratio),
            "joint_target_volume_ratio": joint_volume_ratio,
            "derived_radial_scale": scales[key],
        }
    return scales, policy


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    forbidden = forbidden_config_paths(config)
    if forbidden:
        raise ValueError(f"GT/target-volume fields are forbidden in the resize config: {forbidden}")
    names, parents, _order = validate_config(config)
    profiles = config.get("dimension_constraints", {})
    profile_name = getattr(args, "dimension_profile", "A")
    profile = profiles.get(profile_name, {})
    targets = profile.get("thickness_mm", config.get("measured_thickness_mm", {}))
    if set(targets) != set(names) or min(map(float, targets.values())) <= 0:
        raise ValueError("measured_thickness_mm must contain one positive value per link")
    for name, part_constraint in profile.get("parts", {}).items():
        if name not in names:
            raise ValueError(f"dimension profile contains unknown link: {name}")
        if any(f"{dimension}_mm" in part_constraint for dimension in ("length", "width")):
            missing = [key for key in
                       ("measurement_vertices_file", "protected_vertices_file", "axes")
                       if key not in part_constraint]
            if missing and not (args.coupled_joint_resize
                                or part_constraint.get("automatic_roi")
                                or part_constraint.get("interface_harmonic")
                                or part_constraint.get("joint_anchored_measured_affine")):
                raise ValueError(
                    f"dimension profile {profile_name}.{name} needs confirmed ROI masks and axes: "
                    f"missing {missing}"
                )
    saved = json.loads(args.joints.read_text(encoding="utf-8"))
    joints = saved["joints"]
    if saved.get("links", names) != names:
        raise ValueError("joint link order does not match the config")
    origins = {name: [] for name in names}
    for joint in joints:
        parent = names[joint["parent"] - 1]
        child = names[joint["child"] - 1]
        origin = np.asarray(joint["axis"]["origin"], dtype=float)
        origins[parent].append(origin); origins[child].append(origin)

    source_parts = {}
    for name in names:
        source = args.parts / f"{name}_metric_watertight.ply"
        mesh = trimesh.load_mesh(source, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.body_count != 1 or not mesh.is_watertight:
            raise ValueError(f"{source} must be one watertight body")
        source_parts[name] = mesh
    interface_vertices = {name: set() for name in names}
    interface_pairs = {}
    if not args.joint_anchored_affine:
        for joint in joints:
            parent = names[joint["parent"] - 1]
            child = names[joint["child"] - 1]
            parent_mesh, child_mesh = source_parts[parent], source_parts[child]
            tolerance = max(
                1e-9,
                1e-8 * max(np.linalg.norm(parent_mesh.extents), np.linalg.norm(child_mesh.extents)),
            )
            parent_distance, parent_nearest = cKDTree(child_mesh.vertices).query(parent_mesh.vertices)
            shared_parent = np.flatnonzero(parent_distance <= tolerance)
            if len(shared_parent) < 3:
                raise RuntimeError(f"no exact shared interface vertices found for {parent}--{child}")
            interface_vertices[parent].update(shared_parent.tolist())
            interface_vertices[child].update(parent_nearest[shared_parent].tolist())
            interface_pairs[f"{parent}--{child}"] = (
                parent, child, shared_parent, parent_nearest[shared_parent]
            )

    coupled_parts = coupled_records = None
    if args.coupled_joint_resize:
        coupled_dimensions = {}
        for name, value in profile.get("parts", {}).items():
            if any(f"{dimension}_mm" in value for dimension in ("length", "width")):
                if not all(f"{dimension}_mm" in value for dimension in ("length", "width")):
                    raise ValueError("coupled dimension resize needs both length and width")
                if set(value.get("axes", {})) < {"length", "width", "thickness"}:
                    raise ValueError("coupled dimension resize needs length/width/thickness axes")
                coupled_dimensions[name] = {
                    "targets_mm": {"length": float(value["length_mm"]),
                                   "width": float(value["width_mm"]),
                                   "thickness": float(targets[name])},
                    "axes": {key: np.asarray(value["axes"][key], dtype=float)
                             for key in ("length", "width", "thickness")},
                }
        coupled_parts, coupled_records = coupled_joint_thickness_resize(
            source_parts, targets, origins, interface_pairs, coupled_dimensions
        )
    parts, resize_audit, radial_audit = {}, {}, None
    for name in names:
        source = args.parts / f"{name}_metric_watertight.ply"
        mesh = source_parts[name]
        extra = profile.get("parts", {}).get(name, {})
        print(f"[resize-v2] {name}: diagnostic and constrained deformation", flush=True)
        anchored_dimensions = bool(extra.get("joint_anchored_measured_affine"))
        if anchored_dimensions:
            resized, record = measured_dimension_affine_resize(
                mesh,
                {"length": float(extra["length_mm"]),
                 "width": float(extra["width_mm"]),
                 "thickness": float(targets[name])},
                {key: np.asarray(extra["axes"][key], dtype=float)
                 for key in ("length", "width", "thickness")},
                np.mean(origins[name], axis=0),
            )
        elif args.coupled_joint_resize:
            resized, record = coupled_parts[name], coupled_records[name]
        elif args.joint_anchored_affine or extra.get("thickness_solver") == "joint_anchored_affine":
            # ponytail: permits <=2 mm joint clearance; use connected-mesh calibration
            # when an exact shared interface is required.
            resized, record = joint_anchored_affine_resize(
                mesh, float(targets[name]), origins[name]
            )
        else:
            resized, record = thickness_resize(
                mesh, float(targets[name]), origins[name],
                np.asarray(sorted(interface_vertices[name]), dtype=np.int64),
            )
        dimensional_targets = {} if (args.coupled_joint_resize or anchored_dimensions) else {
            dimension: float(extra[f"{dimension}_mm"])
            for dimension in ("length", "width") if f"{dimension}_mm" in extra
        }
        if dimensional_targets:
            def load_mask(key: str) -> np.ndarray:
                path = Path(extra[key])
                return np.load(path if path.is_absolute() else args.config.parent / path)
            if extra.get("automatic_roi"):
                measurement_mask, protected_mask, automatic_axes = automatic_dimension_roi(
                    resized, origins[name],
                    np.asarray(sorted(interface_vertices[name]), dtype=np.int64),
                    float(targets[name]),
                )
                axes = {key: np.asarray(value, dtype=float) for key, value in automatic_axes.items()
                        if key in {"length", "width", "thickness"}}
                solver = extra.get("roi_solver", "arap")
            elif extra.get("interface_harmonic"):
                measurement_mask = np.arange(len(resized.vertices))
                protected_mask = np.asarray(sorted(interface_vertices[name]), dtype=np.int64)
                axes = {key: np.asarray(value, dtype=float) for key, value in extra["axes"].items()}
                solver = "scalar_harmonic"
            else:
                measurement_mask = load_mask("measurement_vertices_file")
                protected_mask = np.union1d(
                    load_mask("protected_vertices_file"),
                    np.asarray(sorted(interface_vertices[name]), dtype=np.int64),
                )
                axes = {key: np.asarray(value, dtype=float) for key, value in extra["axes"].items()}
                solver = "arap"
            resized, dimension_record = roi_affine_resize(
                resized, dimensional_targets, axes, measurement_mask, protected_mask, solver,
            )
            if extra.get("automatic_roi"):
                dimension_record["automatic_roi"] = automatic_axes
            record["length_width_correction"] = dimension_record
        parts[name] = resized
        resize_audit[name] = {
            **record,
            "before_volume_cm3": abs(float(mesh.volume)) * 1e6,
            "after_volume_cm3": abs(float(resized.volume)) * 1e6,
        }

    if (args.joint_radial_scale is not None or args.joint_diameter_body_ratio is not None
            or args.joint_volume_ratio_from_body):
        radial_scales = {}
        radial_targets = {}
        if args.joint_volume_ratio_from_body:
            radial_scales, radial_targets = joint_radial_scales_from_body_volume(
                joints, names, resize_audit
            )
        for joint in joints:
            parent, child = names[joint["parent"] - 1], names[joint["child"] - 1]
            key = f"{parent}--{child}"
            if args.joint_volume_ratio_from_body:
                continue
            if args.joint_radial_scale is not None:
                radial_scales[key] = float(args.joint_radial_scale)
                continue
            _parent, _child, parent_ids, _child_ids = interface_pairs[key]
            origin = np.asarray(joint["axis"]["origin"], dtype=float)
            axis = np.asarray(joint["axis"]["n"], dtype=float); axis /= np.linalg.norm(axis)
            offset = parts[parent].vertices[parent_ids] - origin
            radius = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
            diameter_mm = 2.0 * float(np.percentile(radius, 95)) * 1000.0
            target_mm = float(args.joint_diameter_body_ratio) * min(
                float(targets[parent]), float(targets[child])
            )
            radial_scales[key] = target_mm / diameter_mm
            radial_targets[key] = {"source_cap_diameter_mm": diameter_mm,
                                   "target_joint_diameter_mm": target_mm}
        # Judge only the new radial deformation; body resize quality was gated above.
        parts, radial_audit = radial_joint_resize(
            parts, parts, interface_pairs, joints, names, radial_scales
        )
        radial_audit["target_policy"] = radial_targets
        for name in names:
            after, _ = _thickness_diagnostic(parts[name])
            resize_audit[name]["after_dominant_thickness_mm"] = float(
                after["dominant_thickness_m"] * 1000.0
            )
            resize_audit[name]["after_volume_cm3"] = abs(float(parts[name].volume)) * 1e6

    interface_clearance = {}
    if args.joint_anchored_affine:
        for joint in joints:
            parent = names[joint["parent"] - 1]
            child = names[joint["child"] - 1]
            distance = cKDTree(parts[child].vertices).query(
                parts[parent].vertices, workers=-1
            )[0]
            interface_clearance[f"{parent}--{child}"] = {
                "minimum_surface_vertex_distance_mm": float(distance.min() * 1000.0)
            }
        if max(item["minimum_surface_vertex_distance_mm"]
               for item in interface_clearance.values()) > 2.0:
            raise RuntimeError(
                f"resized joint interface minimum gap exceeds 2 mm: {interface_clearance}"
            )
    else:
        for key, (parent, child, parent_indices, child_indices) in interface_pairs.items():
            distance = np.linalg.norm(
                parts[parent].vertices[parent_indices] - parts[child].vertices[child_indices], axis=1
            ) * 1000.0
            interface_clearance[key] = {
                "maximum_mm": float(distance.max()),
                "p95_mm": float(np.percentile(distance, 95)),
                "mean_mm": float(distance.mean()),
            }

    assembly = trimesh.util.concatenate([parts[name] for name in names])
    used_anchored_dimensions = any(
        record.get("method") == "joint_origin_anchored_measured_three_axis_affine"
        for record in resize_audit.values()
    )
    geometry_rejected = any(
        record.get("candidate_gate_status") == "REJECTED" for record in resize_audit.values()
    )
    motion, maximum = sweep(
        parts, names, parents, joints, args.samples, args.clearance_mm, args.ignore_radius_mm
    )
    overrides = {
        f"{names[joint['parent'] - 1]}--{names[joint['child'] - 1]}": bool(
            joint.get("static_shell_collision_override_confirmed", False)
        ) for joint in joints
    }
    unapproved = {
        key: max((angle["penetrating_sample_fraction"] for angle in record["angles"]),
                 default=0.0)
        for key, record in motion.items()
        if max((angle["penetrating_sample_fraction"] for angle in record["angles"]),
               default=0.0) > MAX_PENETRATING_SAMPLE_FRACTION and not overrides.get(key)
    }
    if unapproved:
        raise RuntimeError(
            f"resized joint sweep rejected: {unapproved}; review cuts/axis/limits"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    part_dir = args.output / "metric_parts"; part_dir.mkdir(exist_ok=True)
    for name in names:
        parts[name].export(part_dir / f"{name}_metric_watertight.ply")
    assembly.export(args.output / "resized_merged.ply")
    audit = {
        "status": "REJECTED_GEOMETRY_GATE" if geometry_rejected else "PASS",
        "method": ("joint_origin_anchored_measured_three_axis_affine"
                   if used_anchored_dimensions else
                   "whole_part_affine_with_paired_cap_consensus_blend"
                   if args.coupled_joint_resize else
                   "joint_origin_anchored_thickness_axis_affine"
                   if args.joint_anchored_affine
                   else "measurement_constrained_antipodal_local_thickness_harmonic_v2"),
        "method_basis": {
            "thickness_observation": "dominant mode of antipodal surface ray separations",
            "thickness_axis": "area-weighted dominant opposing surface normals; PCA is fallback only",
            "joint_preservation": (
                "approved joint origin is fixed by each link affine"
                if used_anchored_dimensions else
                "paired interfaces move to an automatically selected shared affine blend"
                if args.coupled_joint_resize else
                "exact shared interface vertices are zero-displacement handles"
            ),
            "transition": (
                "single positive three-axis affine; no ROI transition"
                if used_anchored_dimensions else
                "smooth geodesic blend from coupled cap motion to whole-body affine"
                if args.coupled_joint_resize else
                "minimum mesh-graph Dirichlet energy; no metric fixed radius or transition width"
            ),
            "acceptance_tolerance": "one quarter of median source edge length, minimum 0.02 mm",
            "inherited_self_intersection_policy": (
                "freeze the detected four-ring region and limit exact selected-face area growth "
                "to 0.5 percent"
            ),
            "output_commit": (
                "experimental coupled candidate retained for comparison; never auto-selected"
                if geometry_rejected else
                "write geometry only after every resize and articulation gate passes"
            ),
        },
        "parts_source": str(args.parts.resolve()),
        "parts_source_sha256": {
            name: hashlib.sha256((args.parts / f"{name}_metric_watertight.ply").read_bytes()).hexdigest()
            for name in names
        },
        "config": str(args.config.resolve()),
        "dimension_profile": profile_name,
        "dimension_measurement_source": profile.get("measurement_source", {
            "thickness": "existing_RGB-D_measurement_without_original_click_coordinates"
        }),
        "ground_truth_access": "NOT_READ_BEFORE_CANDIDATE_COMMIT",
        "joints": str(args.joints.resolve()),
        "resize": resize_audit,
        "joint_radial_resize": radial_audit,
        "interface_clearance": interface_clearance,
        "parts": {name: part_stats(parts[name]) for name in names},
        "physical_properties": {
            name: _physical_properties(parts[name], float(targets[name])) for name in names
        },
        "virtual_articulation": motion,
        "virtual_articulation_status": (
            "PASS" if maximum <= MAX_PENETRATING_SAMPLE_FRACTION
            else "HITL_CONFIRMED_STATIC_SHELL_OVERLAP"
        ),
        "static_shell_collision_overrides": overrides,
        "maximum_penetrating_sample_fraction": maximum,
        "part_volume_sum_cm3": sum(abs(float(part.volume)) * 1e6 for part in parts.values()),
        "volume_note": "measured thickness changes volume; no water-displacement GT was used",
    }
    _atomic_json(args.output / "resize_audit.json", audit)
    compile_urdf(args.output, names, joints, args.robot_name, provenance={
        "part_assignment": "RORA_HITL_manual_or_physics_valid_exact_partition",
        "metric_geometry": ("joint_origin_anchored_measured_three_axis_affine"
                            if used_anchored_dimensions else
                            "whole_part_affine_with_paired_cap_consensus_blend"
                            if args.coupled_joint_resize else
                            "joint_origin_anchored_thickness_axis_affine"
                            if args.joint_anchored_affine
                            else "measurement_constrained_antipodal_local_thickness_harmonic_v2"),
    })
    return audit


def self_check() -> None:
    box = trimesh.creation.box([0.4, 0.2, 0.04])
    for _ in range(4):
        vertices, faces = trimesh.remesh.subdivide(box.vertices, box.faces)
        box = trimesh.Trimesh(vertices, faces, process=False)
    resized, audit = thickness_resize(box, 20.0, [])
    assert abs(audit["after_dominant_thickness_mm"] - 20.0) <= audit[
        "mesh_derived_numerical_tolerance_mm"
    ]
    assert abs(abs(resized.volume) / abs(box.volume) - 0.5) < 0.02
    assert audit["method"].endswith("_v2")
    affine, affine_audit = joint_anchored_affine_resize(box, 20.0, [np.zeros(3)])
    assert affine.is_watertight and abs(affine_audit["after_dominant_thickness_mm"] - 20.0) < 0.02
    left = trimesh.creation.box([.2, .1, .04]); left.apply_translation([-.1, 0, 0])
    right = trimesh.creation.box([.2, .1, .04]); right.apply_translation([.1, 0, 0])
    for _ in range(4):
        left = left.subdivide(); right = right.subdivide()
    distance, nearest = cKDTree(right.vertices).query(left.vertices)
    shared = np.flatnonzero(distance < 1e-10); paired = nearest[shared]
    coupled, _records = coupled_joint_thickness_resize(
        {"left": left, "right": right}, {"left": 20.0, "right": 30.0},
        {"left": [np.zeros(3)], "right": [np.zeros(3)]},
        {"left--right": ("left", "right", shared, paired)},
    )
    assert np.max(np.linalg.norm(
        coupled["left"].vertices[shared] - coupled["right"].vertices[paired], axis=1
    )) < 1e-12
    radial, _ = radial_joint_resize(
        {"left": left, "right": right}, {"left": left, "right": right},
        {"left--right": ("left", "right", shared, paired)},
        [{"parent": 1, "child": 2, "axis": {"origin": [0, 0, 0], "n": [0, 1, 0]}}],
        ["left", "right"], {"left--right": 0.8},
    )
    assert np.max(np.linalg.norm(
        radial["left"].vertices[shared] - radial["right"].vertices[paired], axis=1
    )) < 1e-12
    body_scales, body_policy = joint_radial_scales_from_body_volume(
        [{"parent": 1, "child": 2}], ["left", "right"],
        {"left": {"before_volume_cm3": 10.0, "after_volume_cm3": 5.0},
         "right": {"before_volume_cm3": 20.0, "after_volume_cm3": 10.0}},
    )
    assert abs(body_scales["left--right"] - np.sqrt(0.5)) < 1e-12
    assert body_policy["left--right"]["joint_target_volume_ratio"] == 0.5
    measured_axes = {"length": [1, 0, 0], "width": [0, 1, 0], "thickness": [0, 0, 1]}
    measured, _ = measured_dimension_affine_resize(
        left, {"length": 180.0, "width": 90.0, "thickness": 20.0}, measured_axes
    )
    assert np.allclose(measured.extents * 1000.0, [180.0, 90.0, 20.0], atol=0.05)
    anchor = left.vertices[0].copy()
    anchored, anchored_audit = measured_dimension_affine_resize(
        left, {"length": 180.0, "width": 90.0, "thickness": 20.0}, measured_axes, anchor
    )
    assert np.allclose(anchored.vertices[0], anchor)
    assert np.allclose(anchored_audit["anchor_m"], anchor) and anchored.is_watertight
    roi, roi_audit = roi_affine_resize(
        left, {"length": 180.0, "width": 90.0}, measured_axes,
        np.arange(len(left.vertices)), np.empty(0, dtype=np.int64),
    )
    assert roi.is_watertight and roi_audit["self_intersection_audit"]["status"] == "PASS"
    print("resize_articulated_parts self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--joints", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robot-name", default="resized_articulated_object")
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--clearance-mm", type=float, default=0.25)
    parser.add_argument("--ignore-radius-mm", type=float, default=20.0)
    parser.add_argument("--joint-anchored-affine", action="store_true")
    parser.add_argument("--joint-radial-scale", type=float,
                        help="shrink joint radius around approved axes and blend into each link")
    parser.add_argument("--joint-diameter-body-ratio", type=float,
                        help="target joint diameter / thinner adjacent body target thickness")
    parser.add_argument("--joint-volume-ratio-from-body", action="store_true",
                        help="give each hinge the geometric-mean adjacent body volume ratio")
    parser.add_argument("--coupled-joint-resize", action="store_true",
                        help="scale joint housings with link bodies and reconcile paired caps")
    parser.add_argument("--dimension-profile", default="A")
    parser.add_argument("--edit-rois", action="store_true",
                        help="paint measurement/protected/neutral ROIs and confirm PCA axes")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if args.edit_rois:
        if not all((args.parts, args.config, args.output)):
            parser.error("--edit-rois requires --parts, --config, and --output")
        print(edit_dimension_rois(args.parts, args.config, args.output))
        return
    if not all((args.parts, args.config, args.joints, args.output)):
        parser.error("--parts, --config, --joints, and --output are required")
    if args.joint_anchored_affine and args.coupled_joint_resize:
        parser.error("choose only one joint resize mode")
    radial_rules = sum((args.joint_radial_scale is not None,
                        args.joint_diameter_body_ratio is not None,
                        args.joint_volume_ratio_from_body))
    if radial_rules > 1:
        parser.error("choose only one joint radial resize rule")
    if args.joint_diameter_body_ratio is not None and args.joint_diameter_body_ratio <= 0:
        parser.error("--joint-diameter-body-ratio must be positive")
    if args.joint_radial_scale is not None and (
            args.joint_anchored_affine or args.coupled_joint_resize):
        parser.error("joint radial resize uses the exact-interface resize mode")
    if args.joint_volume_ratio_from_body and args.joint_anchored_affine:
        parser.error("body-volume joint resize needs exact shared interfaces")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
