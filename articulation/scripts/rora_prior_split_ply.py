#!/usr/bin/env python3
"""RORA HITL prior -> exact metric PLY links -> joint review -> URDF asset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import cKDTree, ConvexHull, HalfspaceIntersection

from separate_ply_links import close_labeled_parts, exact_seed_component_cut, paint_face_labels


ROOT = Path(__file__).resolve().parents[1]
RORA_PATCH = next(
    (parent / "rora_patch" for parent in Path(__file__).resolve().parents
     if (parent / "rora_patch").is_dir()),
    ROOT / "docs/Object_Reconstruction/part_split_delivery/rora_patch",
)


def load_rora_selector():
    """Load only RORA's part-selection GUI without its reconstruction stack."""
    functions = types.ModuleType("functions"); functions.__path__ = [str(RORA_PATCH / "functions")]
    lib = types.ModuleType("functions.lib"); lib.__path__ = [str(RORA_PATCH / "functions/lib")]
    pymeshlab = types.ModuleType("pymeshlab"); pymeshlab.MeshSet = object
    sys.modules.update({"functions": functions, "functions.lib": lib, "pymeshlab": pymeshlab})
    for name in ("geometry", "visualization"):
        packaged = RORA_PATCH / f"functions/lib/{name}.py"
        if not packaged.is_file():
            raise RuntimeError(f"missing packaged RORA GUI component: {packaged}")
        spec = importlib.util.spec_from_file_location(f"functions.lib.{name}", packaged)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["functions.lib.visualization"].select_parts_interactive


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_progress(output_dir: Path, stage: str, status: str, **details) -> None:
    path = output_dir / "hitl_progress.json"
    value = json.dumps({"stage": stage, "status": status, **details}, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir,
                                     prefix=".hitl_progress.", delete=False) as stream:
        stream.write(value); stream.flush()
        temporary = Path(stream.name)
    temporary.replace(path)
    print(f"[RORA HITL] {stage}: {status} -> {path}")


def forbidden_config_paths(value, path="") -> list[str]:
    if isinstance(value, list):
        return [found for index, child in enumerate(value)
                for found in forbidden_config_paths(child, f"{path}[{index}]")]
    if not isinstance(value, dict):
        return []
    found = []
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else key
        if "ground_truth" in key.lower() or "target_volume" in key.lower():
            found.append(child_path)
        found.extend(forbidden_config_paths(child, child_path))
    return found


def load_source(path: Path, require_watertight: bool = True) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("input must be one triangle-mesh body")
    # Exporters may leave isolated, unreferenced points in an otherwise single-body PLY.
    mesh.remove_unreferenced_vertices()
    if mesh.body_count != 1:
        raise ValueError("input must be one triangle-mesh body")
    if require_watertight and not mesh.is_watertight:
        raise ValueError("articulated output requires a watertight input; run proposal-only first")
    return mesh


def decomposition(mesh: trimesh.Trimesh, folder: Path, source_hash: str,
                  threshold: float, maximum: int, preprocess_resolution: int,
                  resolution: int) -> list[trimesh.Trimesh]:
    manifest = folder / "manifest.json"
    expected = {"source_sha256": source_hash, "threshold": threshold,
                "maximum_convex_hulls": maximum,
                "preprocess_resolution": preprocess_resolution,
                "resolution": resolution}
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        paths = sorted(folder.glob("part_*.ply"))
        if all(saved.get(key) == value for key, value in expected.items()) and paths:
            return [trimesh.load_mesh(path, process=False) for path in paths]

    folder.mkdir(parents=True, exist_ok=True)
    import open3d as o3d
    import coacd

    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    reduced = source.simplify_quadric_decimation(20_000)
    proxy = trimesh.Trimesh(np.asarray(reduced.vertices), np.asarray(reduced.triangles),
                            process=False)
    proxy.export(folder / "selection_proxy.ply")
    result = coacd.run_coacd(
        coacd.Mesh(proxy.vertices, proxy.faces), threshold=threshold,
        max_convex_hull=maximum, preprocess_mode="auto",
        preprocess_resolution=preprocess_resolution, resolution=resolution, seed=0,
    )
    parts = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in result]
    for old in folder.glob("part_*.ply"):
        old.unlink()
    for index, part in enumerate(parts):
        part.export(folder / f"part_{index:04d}.ply")
    manifest.write_text(json.dumps({**expected, "part_count": len(parts)}, indent=2) + "\n",
                        encoding="utf-8")
    return parts


def unsafe_hull_indices(parts: list[trimesh.Trimesh], labels: list[int],
                        mesh: trimesh.Trimesh) -> list[int]:
    """Find cross-link hulls that claim the same original surface region."""
    hulls = [ConvexHull(part.vertices) for part in parts]

    def overlap_volume(first: int, second: int) -> float:
        halfspaces = np.vstack([hulls[first].equations, hulls[second].equations])
        normals, offsets = halfspaces[:, :3], halfspaces[:, 3]
        constraints = np.column_stack([normals, np.linalg.norm(normals, axis=1)])
        solved = linprog(
            [0.0, 0.0, 0.0, -1.0], A_ub=constraints, b_ub=-offsets,
            bounds=[(None, None)] * 3 + [(0.0, None)], method="highs",
        )
        if not solved.success or solved.x[3] <= 1e-7:
            return 0.0
        vertices = HalfspaceIntersection(halfspaces, solved.x[:3]).intersections
        return float(ConvexHull(vertices).volume) if len(vertices) >= 4 else 0.0

    unsafe = set()
    for first in range(len(parts)):
        if labels[first] < 0:
            continue
        for second in range(first + 1, len(parts)):
            if labels[second] < 0 or labels[first] == labels[second]:
                continue
            overlap = overlap_volume(first, second)
            fractions = (
                overlap / max(float(hulls[first].volume), 1e-12),
                overlap / max(float(hulls[second].volume), 1e-12),
            )
            if max(fractions) < 0.01:
                continue
            if min(fractions) >= 0.15:
                unsafe.update((first, second))
            else:
                unsafe.add(first if fractions[0] > fractions[1] else second)
    return sorted(unsafe)


def plane_face_labels(mesh: trimesh.Trimesh, names: list[str], parents: dict[str, str],
                      planes: dict[str, dict], centers: np.ndarray | None = None) -> np.ndarray:
    """Use STEP 1B planes only as the initial ownership proposal for face painting."""
    def depth(name: str) -> int:
        value = 0
        while name in parents:
            name, value = parents[name], value + 1
        return value

    centers = mesh.triangles_center if centers is None else np.asarray(centers, dtype=float)
    root = next(name for name in names if name not in parents)
    result = np.full(len(centers), names.index(root), dtype=np.int16)
    remaining = np.ones(len(centers), dtype=bool)
    for child in sorted(parents, key=depth, reverse=True):
        saved = planes[child]
        normal = np.asarray(saved["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        offset = float(saved["offset_m"])
        if not saved.get("child_is_positive_halfspace", True):
            normal, offset = -normal, -offset
        selected = remaining & (centers @ normal >= offset)
        result[selected] = names.index(child)
        remaining[selected] = False
    return result


def initial_cut_planes(parts: list[trimesh.Trimesh], labels: list[int],
                       mesh: trimesh.Trimesh, names: list[str], parents: dict[str, str],
                       initial: dict | None = None) -> dict[str, dict]:
    """STEP 1B: place semantic cut planes on the original PLY, never on CoACD volume."""
    import vedo

    def descendants(name: str) -> set[str]:
        result = {name}
        for _ in names:
            result |= {child for child, parent in parents.items() if parent in result}
        return result

    def depth(name: str) -> int:
        value = 0
        while name in parents:
            name, value = parents[name], value + 1
        return value

    order = sorted(parents, key=depth, reverse=True)
    diagonal = float(np.linalg.norm(mesh.extents))
    saved = {}
    for child in order:
        subtree = descendants(child)
        moving = np.vstack([
            part.vertices for part, label in zip(parts, labels)
            if label >= 0 and names[label] in subtree
        ])
        fixed = np.vstack([
            part.vertices for part, label in zip(parts, labels)
            if label >= 0 and names[label] not in subtree
        ])
        distance, nearest = cKDTree(fixed).query(moving, workers=-1)
        index = int(np.argmin(distance))
        anchor = (moving[index] + fixed[int(nearest[index])]) / 2
        normal = moving.mean(axis=0) - fixed.mean(axis=0)
        normal /= np.linalg.norm(normal)
        previous = (initial or {}).get(child)
        if previous:
            normal = np.asarray(previous["normal"], dtype=float)
            normal /= np.linalg.norm(normal)
            anchor = np.asarray(previous.get("anchor", normal * previous["offset_m"]), dtype=float)
        tangent = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(tangent) < 0.2:
            tangent = np.cross(normal, [0.0, 1.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        saved[child] = {
            "base_normal": normal, "tangent_a": tangent,
            "tangent_b": np.cross(normal, tangent), "anchor": anchor,
            "tilt_a_deg": 0.0, "tilt_b_deg": 0.0,
            "offset_mm": (float(previous["offset_m"] - normal @ anchor) * 1000.0
                          if previous else 0.0),
        }

    def rotate(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
        matrix = trimesh.transformations.rotation_matrix(np.deg2rad(degrees), axis)[:3, :3]
        return matrix @ vector

    def plane(child: str) -> tuple[np.ndarray, float]:
        item = saved[child]
        normal = rotate(item["base_normal"], item["tangent_a"], item["tilt_a_deg"])
        normal = rotate(normal, item["tangent_b"], item["tilt_b_deg"])
        normal /= np.linalg.norm(normal)
        if np.mean(np.vstack([
            part.vertices for part, label in zip(parts, labels)
            if label >= 0 and names[label] in descendants(child)
        ]) @ normal) < normal @ item["anchor"]:
            normal *= -1
        return normal, float(normal @ item["anchor"] + item["offset_mm"] / 1000.0)

    def resolved() -> dict[str, dict]:
        return {
            child: {
                "normal": plane(child)[0].tolist(), "offset_m": plane(child)[1],
                "anchor": saved[child]["anchor"].tolist(),
                "child_is_positive_halfspace": True,
                "source": "RORA_HITL_initial_original_PLY_cut_plane",
            }
            for child in order
        }

    palette = np.asarray([
        [40, 130, 190, 225], [245, 125, 35, 225], [45, 165, 70, 225],
        [180, 70, 190, 225], [230, 190, 40, 225],
    ], dtype=np.uint8)
    stride = max(1, int(np.ceil(len(mesh.faces) / 120_000)))
    visible = np.arange(0, len(mesh.faces), stride)
    visible_centers = mesh.triangles_center[visible]
    preview_mesh = mesh.submesh([visible], append=True, repair=False)
    actor = vedo.Mesh([mesh.vertices, mesh.faces[visible]])
    ambiguous = [
        vedo.Mesh([part.vertices, part.faces]).c("gray").alpha(0.18)
        for part, label in zip(parts, labels) if label < 0
    ]
    plotter = vedo.Plotter(
        title="RORA HITL 1B/3 — Original PLY cut planes", bg="white",
        pos=(80, 60), size=(1200, 850),
    )
    state = {"index": 0, "plane": None, "section": [],
             "saved": False, "cancelled": False}
    status = vedo.Text2D("", pos="top-left", c="black", bg="white", alpha=0.9, s=0.68)
    radius = 0.08 * diagonal

    def refresh() -> None:
        child = order[state["index"]]
        normal, plane_offset = plane(child)
        actor.cellcolors = palette[plane_face_labels(
            mesh, names, parents, resolved(), visible_centers
        ) % len(palette)]
        if state["plane"] is not None:
            plotter.remove(state["plane"])
        for line in state["section"]:
            plotter.remove(line)
        position = saved[child]["anchor"] + normal * saved[child]["offset_mm"] / 1000.0
        state["plane"] = vedo.Plane(
            pos=position, normal=normal, s=(2 * radius, 2 * radius)
        ).c("cyan").alpha(0.12)
        section = preview_mesh.section(plane_origin=normal * plane_offset, plane_normal=normal)
        state["section"] = ([vedo.Line(points).c("magenta").lw(5)
                             for points in section.discrete if len(points) > 1]
                            if section is not None else [])
        plotter.add(state["plane"])
        if state["section"]:
            plotter.add(*state["section"])
        status.text(
            f"STEP 1B/3 — ORIGINAL PLY CUT  {state['index'] + 1}/{len(order)}\n"
            f"{parents[child]} -> {child} | tilt=({saved[child]['tilt_a_deg']:.1f}, "
            f"{saved[child]['tilt_b_deg']:.1f}) deg | offset={saved[child]['offset_mm']:.2f} mm\n"
            "Exact part volume/closure is computed during final face-paint validation.\n"
            "Magenta=fast intersection preview; small cyan patch is display-only.\n"
            "CoACD gray hulls are display-only. Colors come from original PLY face sides.\n"
            "Adjust proposal | N=next boundary | Enter/S=continue to face paint | Q=cancel"
        )
        plotter.render()

    def sliders(_widget=None, _event=None) -> None:
        child = order[state["index"]]
        saved[child]["tilt_a_deg"] = float(tilt_a.value)
        saved[child]["tilt_b_deg"] = float(tilt_b.value)
        saved[child]["offset_mm"] = float(offset.value)
        refresh()

    def load_sliders() -> None:
        child = order[state["index"]]
        tilt_a.value = saved[child]["tilt_a_deg"]
        tilt_b.value = saved[child]["tilt_b_deg"]
        offset.value = saved[child]["offset_mm"]
        refresh()

    def next_boundary(_widget=None, _event=None) -> None:
        state["index"] = (state["index"] + 1) % len(order)
        load_sliders()

    def finish(_widget=None, _event=None) -> None:
        state["saved"] = True; plotter.close()

    def cancel(_widget=None, _event=None) -> None:
        state["cancelled"] = True; plotter.close()

    def keypress(event) -> None:
        key = getattr(event, "keypress", "")
        if key in ("n", "N", "Right"):
            next_boundary()
        elif key in ("s", "S", "Return", "Enter"):
            finish()
        elif key in ("q", "Q", "Esc", "Escape", "\x1b"):
            cancel()

    limit = max(30.0, diagonal * 100.0)
    tilt_a = plotter.add_slider(sliders, -85, 85, value=0, title="tilt A [deg]",
                                pos=((0.05, .17), (.30, .17)), delayed=True)
    tilt_b = plotter.add_slider(sliders, -85, 85, value=0, title="tilt B [deg]",
                                pos=((0.37, .17), (.62, .17)), delayed=True)
    offset = plotter.add_slider(sliders, -limit, limit, value=0, title="offset [mm]",
                               pos=((0.69, .17), (.94, .17)), delayed=True)
    plotter.add_button(next_boundary, states=("Next boundary",), pos=(.25, .07), size=14)
    plotter.add_button(finish, states=("Continue to face paint",), pos=(.57, .07), size=14)
    plotter.add_button(cancel, states=("Cancel",), pos=(.84, .07), size=14)
    plotter.add_callback("KeyPress", keypress)
    refresh()
    plotter.show(actor, *ambiguous, status, axes=1, interactive=True)
    if state["cancelled"] or not state["saved"]:
        raise RuntimeError("initial original-PLY cut-plane selection was cancelled")
    return resolved()


def native_rora_selection(parts: list[trimesh.Trimesh], mesh: trimesh.Trimesh,
                          names: list[str], parents: dict[str, str], path: Path,
                          source_hash: str,
                          initial_labels: list[int] | None = None,
                          initial_planes: dict | None = None
                          ) -> tuple[list[int], dict, np.ndarray]:
    selector = load_rora_selector()
    initial = (None if initial_labels is None else
               [0, *[label + 1 if label >= 0 else 0 for label in initial_labels]])
    while True:
        _categories, one_based, _colors = selector(
            parts, mesh.vertices, mesh.faces, len(names), True, allow_unassigned=True,
            category_names=names, initial_belongings=initial,
        )
        labels = [int(value) - 1 if value else -1 for value in one_based[1:]]
        unsafe = unsafe_hull_indices(parts, labels, mesh)
        if not unsafe:
            break
        for index in unsafe:
            labels[index] = -1
        print(f"[RORA HITL] reset crossing hulls to Ambiguous: {unsafe}")
        initial = [0, *[label + 1 if label >= 0 else 0 for label in labels]]
    if not set(range(len(names))).issubset(labels):
        raise RuntimeError("every link must own at least one unambiguous RORA fragment")
    planes = initial_cut_planes(parts, labels, mesh, names, parents, initial_planes)
    face_labels_path = path.parent / "hitl_face_labels.npy"
    record = {
        "source_sha256": source_hash,
        "links": names,
        "part_labels": labels,
        "selection": "RORA human-in-the-loop convex-part assignment with ambiguous proposals",
        "ambiguous_policy": "excluded_from_hard_seeds_then_assigned_by_original_face_graph_and_joint_gates",
        "hitl_cut_planes": planes,
        "hitl_face_labels_file": face_labels_path.name,
        "face_ownership": "PENDING_HITL_geodesic_brush_correction",
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    face_labels = paint_face_labels(
        mesh, plane_face_labels(mesh, names, parents, planes), names, parents
    )
    np.save(face_labels_path, face_labels)
    record["face_ownership"] = "HITL_geodesic_brush_corrected_original_faces"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return labels, planes, face_labels


def selected_categories(parts: list[trimesh.Trimesh], mesh: trimesh.Trimesh,
                        names: list[str], parents: dict[str, str], path: Path,
                        source_hash: str, reselect: bool):
    if path.exists() and not reselect:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved.get("source_sha256") != source_hash or saved.get("links") != names
                or not saved.get("selection", "").startswith("RORA human-in-the-loop")):
            raise RuntimeError("saved RORA selection belongs to a different input or link tree")
        belongings = saved["part_labels"]
        if (len(belongings) != len(parts)
                or not set(range(len(names))).issubset(belongings)
                or any(value not in {-1, *range(len(names))} for value in belongings)):
            raise RuntimeError("saved RORA selection is incomplete")
        planes = saved.get("hitl_cut_planes")
        if not planes:
            planes = initial_cut_planes(parts, belongings, mesh, names, parents)
            saved["hitl_cut_planes"] = planes
            path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        face_labels_path = path.parent / saved.get("hitl_face_labels_file", "")
        if face_labels_path.is_file():
            face_labels = np.load(face_labels_path)
        else:
            face_labels_path = path.parent / "hitl_face_labels.npy"
            face_labels = paint_face_labels(
                mesh, plane_face_labels(mesh, names, parents, planes), names, parents
            )
            np.save(face_labels_path, face_labels)
            saved["hitl_face_labels_file"] = face_labels_path.name
            saved["face_ownership"] = "HITL_geodesic_brush_corrected_original_faces"
            path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    else:
        initial = None
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("source_sha256") == source_hash and saved.get("links") == names
                    and len(saved.get("part_labels", [])) == len(parts)):
                initial = saved["part_labels"]
        initial_planes = saved.get("hitl_cut_planes") if path.exists() else None
        belongings, planes, face_labels = native_rora_selection(
            parts, mesh, names, parents, path, source_hash,
            initial_labels=initial, initial_planes=initial_planes)

    categories = [[] for _ in range(len(names) + 1)]
    for part, label in zip(parts, belongings):
        categories[label + 1].append(part)
    return categories, planes, face_labels


def save_preview(parts: list[trimesh.Trimesh], categories, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for part in categories[0]:
        stride = max(1, int(np.ceil(len(part.faces) / 8_000)))
        actor = Poly3DCollection(part.triangles[::stride], alpha=0.28, linewidth=0)
        actor.set_facecolor("gray")
        axis.add_collection3d(actor)
    if categories[0]:
        axis.scatter([], [], [], color="gray", alpha=0.5, label="Ambiguous (auto later)")
    for label, name in enumerate(names):
        for part in categories[label + 1]:
            stride = max(1, int(np.ceil(len(part.faces) / 8_000)))
            actor = Poly3DCollection(part.triangles[::stride], alpha=0.72, linewidth=0)
            actor.set_facecolor(colors[label % len(colors)])
            axis.add_collection3d(actor)
        axis.scatter([], [], [], color=colors[label % len(colors)], label=name)
    bounds = np.asarray([part.bounds for part in parts])
    lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    center, radius = (lower + upper) / 2, (upper - lower).max() / 2
    axis.set(xlim=(center[0] - radius, center[0] + radius),
             ylim=(center[1] - radius, center[1] + radius),
             zlim=(center[2] - radius, center[2] + radius))
    axis.view_init(elev=20, azim=-55)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rora_seeds(mesh: trimesh.Trimesh, categories, names: list[str],
               core_hulls_per_link: int = 1) -> tuple[dict, np.ndarray]:
    """Turn overlapping convex assignments into sparse, high-confidence body seeds."""
    core_parts = []
    for label in range(len(names)):
        parts = categories[label + 1]
        other_centers = np.vstack([
            part.centroid for other in range(len(names)) if other != label
            for part in categories[other + 1]
        ])
        scores = [cKDTree(other_centers).query(part.centroid)[0]
                  / max(np.linalg.norm(part.extents), 1e-9) for part in parts]
        take = np.argsort(scores)[-min(core_hulls_per_link, len(parts)):]
        core_parts.append([parts[int(index)] for index in take])
    trees = [cKDTree(np.vstack([part.vertices for part in selected]))
             for selected in core_parts]
    distances = np.column_stack([
        tree.query(mesh.vertices, workers=-1)[0] for tree in trees
    ])
    labels = distances.argmin(axis=1).astype(np.int16)
    own = distances[np.arange(len(mesh.vertices)), labels]
    runner_up = np.partition(distances, 1, axis=1)[:, 1]
    margin = runner_up - own
    if categories[0]:
        ambiguous = cKDTree(np.vstack([part.vertices for part in categories[0]])).query(
            mesh.vertices, workers=-1)[0]
        labels[ambiguous + np.finfo(float).eps < own] = -1

    seeds = {}
    for label, name in enumerate(names):
        candidates = np.flatnonzero(labels == label)
        if not len(candidates):
            raise RuntimeError(f"RORA prior produced no source vertices for {name}")
        points = mesh.vertices[candidates]
        axis = np.linalg.eigh(np.cov(points, rowvar=False))[1][:, -1]
        projection = points @ axis
        edges = np.quantile(projection, np.linspace(0.0, 1.0, 6))
        anchors = []
        for start, stop in zip(edges[:-1], edges[1:]):
            band = candidates[(projection >= start) & (projection <= stop)]
            if len(band):
                anchors.append(int(band[np.argmax(margin[band])]))
        seeds[name] = mesh.vertices[np.unique(anchors)].tolist()
    return seeds, labels


def exact_parts_from_planes(mesh: trimesh.Trimesh, names: list[str],
                            parents: dict[str, str], planes: dict[str, dict],
                            seeds: dict[str, list]) -> dict[str, trimesh.Trimesh]:
    """Apply STEP 1B planes to original triangles before joint selection."""
    def depth(name: str) -> int:
        value = 0
        while name in parents:
            name, value = parents[name], value + 1
        return value

    working, outputs = mesh, {}
    for child in sorted(parents, key=depth, reverse=True):
        saved = planes[child]
        normal = np.asarray(saved["normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        offset = float(saved["offset_m"])
        if not saved.get("child_is_positive_halfspace", True):
            normal, offset = -normal, -offset
        child_points = np.asarray(seeds[child], dtype=float)
        if np.mean(child_points @ normal) < offset:
            normal, offset = -normal, -offset
        try:
            part, working, _child_meta, _rest_meta, _reassigned = exact_seed_component_cut(
                working, normal, offset, child_points
            )
        except Exception as error:
            raise RuntimeError(f"{parents[child]}--{child}: {error}") from error
        if (not part.is_watertight or not part.is_winding_consistent
                or part.body_count != 1 or not working.is_watertight
                or not working.is_winding_consistent or working.body_count != 1):
            raise RuntimeError(f"STEP 1B plane did not make closed links at {parents[child]}--{child}")
        outputs[child] = part
    outputs[next(name for name in names if name not in parents)] = working
    return outputs


def exact_parts_from_face_labels(mesh: trimesh.Trimesh, names: list[str],
                                 parents: dict[str, str], labels: np.ndarray
                                 ) -> dict[str, trimesh.Trimesh]:
    if labels.shape != (len(mesh.faces),) or not np.isin(labels, range(len(names))).all():
        raise ValueError("HITL face labels do not match the original PLY")
    outputs, interfaces = close_labeled_parts(mesh, labels, names)
    expected = {frozenset((parent, child)) for child, parent in parents.items()}
    actual = {frozenset(key.split("--")) for key in interfaces}
    if actual != expected or any(
            not part.is_watertight or not part.is_winding_consistent or part.body_count != 1
            for part in outputs.values()):
        raise RuntimeError("HITL face correction did not produce the configured closed link tree")
    return outputs


def rora_dense_seeds(mesh: trimesh.Trimesh, categories, names: list[str],
                     maximum_per_link: int = 2000) -> tuple[dict, np.ndarray]:
    """Use every HITL-approved safe hull as a dense original-surface prior."""
    inside = np.zeros((len(mesh.vertices), len(names)), dtype=bool)
    for label in range(len(names)):
        for part in categories[label + 1]:
            equation = ConvexHull(part.vertices).equations
            lower, upper = part.bounds
            candidates = np.flatnonzero(np.all(
                (mesh.vertices >= lower - 1e-7) & (mesh.vertices <= upper + 1e-7), axis=1
            ))
            for start in range(0, len(candidates), 20_000):
                selected = candidates[start:start + 20_000]
                inside[selected, label] |= np.max(
                    mesh.vertices[selected] @ equation[:, :3].T + equation[:, 3], axis=1
                ) <= 1e-6

    confident = inside.sum(axis=1) == 1
    vertex_labels = np.full(len(mesh.vertices), -1, dtype=np.int16)
    vertex_labels[confident] = inside[confident].argmax(axis=1)
    centers = mesh.triangles_center
    rng = np.random.default_rng(0)
    seeds = {}
    for label, name in enumerate(names):
        faces = np.flatnonzero(
            np.all(confident[mesh.faces], axis=1)
            & np.all(vertex_labels[mesh.faces] == label, axis=1)
        )
        if not len(faces):
            raise RuntimeError(f"safe RORA hulls produced no original faces for {name}")
        selected = rng.choice(faces, min(maximum_per_link, len(faces)), replace=False)
        seeds[name] = centers[selected].tolist()
    return seeds, vertex_labels


def joint_candidates(parent: trimesh.Trimesh, child: trimesh.Trimesh,
                     seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Propose interface and motion-plane axes; the human still chooses the hinge."""
    parent_points = parent.sample(30_000, seed=seed)
    child_points = child.sample(30_000, seed=seed + 100)
    distance, nearest = cKDTree(parent_points).query(child_points, workers=-1)
    take = np.argpartition(distance, min(599, len(distance) - 1))[:600]
    interface = (child_points[take] + parent_points[nearest[take]]) / 2
    origin = np.median(interface, axis=0)
    _values, axes = np.linalg.eigh(np.cov(interface - origin, rowvar=False))
    proposals = list(axes.T[::-1])
    parent_long = np.linalg.eigh(np.cov(parent.vertices, rowvar=False))[1][:, -1]
    child_long = np.linalg.eigh(np.cov(child.vertices, rowvar=False))[1][:, -1]
    motion_normal = np.cross(parent_long, child_long)
    if np.linalg.norm(motion_normal) > 0.15:
        proposals.insert(0, motion_normal / np.linalg.norm(motion_normal))
    unique = []
    for axis in proposals:
        axis = axis / np.linalg.norm(axis)
        if not any(abs(axis @ saved) > 0.995 for saved in unique):
            unique.append(axis)
    return origin, unique


def selected_joints(rlps: list[dict]) -> list[dict]:
    joints = []
    for rlp in rlps:
        picked = [vector for vector in rlp["vectors"]
                  if vector.get("state") in {"Revolute", "Prismatic"}]
        if len(picked) != 1:
            raise RuntimeError("select exactly one axis per joint")
        vector = picked[0]
        joints.append({"parent": rlp["a"]["parent"], "child": rlp["a"]["child"],
                       "axis": {"origin": vector["center"], "n": vector["n"]},
                       "type": vector["state"]})
    return joints


def review_joints(meshes: dict[str, trimesh.Trimesh], names: list[str], parents: dict,
                  path: Path, source_hash: str, reselect: bool,
                  assembly: trimesh.Trimesh) -> list[dict]:
    saved = path.exists() and json.loads(path.read_text(encoding="utf-8"))
    if saved and not reselect and saved.get("source_sha256") == source_hash:
        return saved["joints"]
    saved_by_child = {}
    if saved:
        for joint in saved.get("joints", []):
            child = (names[joint["child"] - 1]
                     if isinstance(joint.get("child"), int) else joint.get("child"))
            saved_by_child[child] = joint

    load_rora_selector()
    from functions.lib.visualization import visualize_and_select_vectors_for_rlps

    index = {name: position + 1 for position, name in enumerate(names)}
    children = {name: [] for name in names}
    for child, parent in parents.items():
        children[parent].append(child)

    def descendants(name: str) -> list[int]:
        result, pending = [], [name]
        while pending:
            current = pending.pop()
            result.append(index[current]); pending.extend(children[current])
        return result

    rlps = []
    for number, child_name in enumerate((name for name in names if name in parents), 1):
        parent_name = parents[child_name]
        origin, axes = joint_candidates(meshes[parent_name], meshes[child_name], number)
        previous = saved_by_child.get(child_name)
        if previous:
            origin = np.asarray(previous["axis"]["origin"], dtype=float)
            old_axis = np.asarray(previous["axis"]["n"], dtype=float)
            axes = [old_axis, *[axis for axis in axes if abs(float(axis @ old_axis)) < 0.995]]
        rlps.append({
            "a": {"parent": index[parent_name], "child": index[child_name]},
            "parent_name": parent_name, "child_name": child_name,
            "joint_index": number, "joint_count": len(parents),
            "center": origin.tolist(), "moving_part_ids": descendants(child_name),
            "vectors": [{"center": origin.tolist(), "n": axis.tolist(), "d": [0, 0, 0],
                         "state": (previous.get("type", "Revolute") if previous and i == 0 else "None"),
                         "source": ("saved_final_cut_axis" if previous and i == 0
                                    else "RORA_metric_interface_candidate")}
                        for i, axis in enumerate(axes)],
            "selected_candidate_index": 0 if previous else None,
            "candidate_joint_type": previous.get("type", "Revolute") if previous else "Revolute",
            "ui_stage": "candidate_selection",
        })
    categories = [[], *[[meshes[name]] for name in names]]
    visualize_and_select_vectors_for_rlps(
        rlps, list(meshes.values()), categories, assembly.vertices, assembly.faces, display=True)
    if not all(rlp.get("hitl_finished") for rlp in rlps):
        raise RuntimeError("Use selected axis in every RORA joint window")
    joints = selected_joints(rlps)
    if len(joints) != len(parents):
        raise RuntimeError(f"select exactly one axis per joint; got {len(joints)}")

    from static_rora_metric_parts import _angles, _rotation, _scene, _signed_distance

    final_rlps = []
    for rlp in rlps:
        vector = next(v for v in rlp["vectors"] if v.get("state") in {"Revolute", "Prismatic"})
        previous = saved_by_child.get(rlp["child_name"], {})
        vector.update({"limits_deg": previous.get("limits_deg", [-120.0, 120.0]),
                       "limits_mm": previous.get("limits_mm", [-50.0, 50.0])})
        moving_ids = set(rlp["moving_part_ids"])
        moving = trimesh.util.concatenate([
            meshes[name] for name in names if index[name] in moving_ids
        ])
        fixed = trimesh.util.concatenate([
            meshes[name] for name in names if index[name] not in moving_ids
        ])
        moving_points = moving.sample(8_000, seed=rlp["joint_index"])
        fixed_scene = _scene(fixed)

        def validate_limits(candidate, points=moving_points, scene=fixed_scene):
            origin = np.asarray(candidate["center"], dtype=float)
            axis = np.asarray(candidate["n"], dtype=float)
            axis /= np.linalg.norm(axis)
            samples = points[np.linalg.norm(points - origin, axis=1) > 0.020]
            unit = "mm" if candidate.get("state") == "Prismatic" else "deg"
            worst = 0.0
            worst_angle = 0.0
            for angle in _angles(np.asarray(candidate[f"limits_{unit}"], dtype=float)):
                moved = (samples + axis * angle / 1000.0
                         if unit == "mm" else _rotation(samples, origin, axis, angle))
                distance = _signed_distance(scene, moved)
                fraction = float(np.mean(distance < -0.00025)) if len(distance) else 0.0
                if fraction > worst:
                    worst, worst_angle = fraction, angle
            return (worst <= 0.005,
                    f"Sweep collision: {100 * worst:.3f}% at {worst_angle:.1f} {unit} "
                    f"({'PASS' if worst <= 0.005 else 'FAIL > 0.5%'})")

        final_rlps.append({
            **rlp, "vectors": [vector], "motion_limits": [-180.0, 180.0],
            "ui_stage": "final_motion_confirmation", "limit_validator": validate_limits,
        })
    visualize_and_select_vectors_for_rlps(
        final_rlps, list(meshes.values()), categories,
        assembly.vertices, assembly.faces, display=True)
    if not all(rlp.get("hitl_finished") for rlp in final_rlps):
        raise RuntimeError("Confirm direction and limits, then Finish & save every joint")

    metadata = {frozenset((rlp["a"]["parent"], rlp["a"]["child"])):
                (rlp["a"], rlp["vectors"][0]) for rlp in final_rlps}
    for joint in joints:
        relation, vector = metadata[frozenset((joint["parent"], joint["child"]))]
        joint["parent"], joint["child"] = relation["parent"], relation["child"]
        previous = saved_by_child.get(names[joint["child"] - 1], {})
        joint["type"] = vector["state"]
        unit = "mm" if joint["type"] == "Prismatic" else "deg"
        joint.update({
            "axis": {"origin": vector["center"], "n": vector["n"]},
            f"limits_{unit}": [float(value) for value in vector[f"limits_{unit}"]],
            "positive_direction_confirmed": True, "limits_confirmed": True,
            "limits_source": "RORA_HITL_observed",
            "static_shell_collision_override_confirmed": bool(
                vector.get("static_shell_collision_override_confirmed", False)
            ),
            "contact_region_masks": previous.get("contact_region_masks", {}),
            "editing_state": previous.get("editing_state", {
                "origin_step_mm": 1.0, "axis_parameterization": "yaw_pitch"
            }),
        })
    value = {"source_sha256": source_hash, "links": names, "joints": joints}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".joint_selection.", delete=False) as stream:
        json.dump(value, stream, indent=2); stream.write("\n"); stream.flush()
        temporary = Path(stream.name)
    temporary.replace(path)
    return joints


def compile_urdf(output: Path, names: list[str], joints: list[dict], robot_name: str,
                 provenance: dict | None = None, parts_dir: Path | None = None) -> None:
    parts_dir = parts_dir or output / "metric_parts"
    manifest = {
        "mode": "hitl", "robot_name": robot_name,
        "links": [{"name": name, "visual": str((parts_dir /
                    f"{name}_metric_watertight.ply").resolve()),
                   "collision": str((parts_dir /
                    f"{name}_metric_watertight.ply").resolve())} for name in names],
        "joints": [],
        "provenance": provenance or {
            "part_assignment": "RORA_HITL_soft_prior",
            "metric_geometry": "original_PLY_exact_partition",
        },
    }
    for number, joint in enumerate(joints, 1):
        kind = "prismatic" if joint["type"] == "Prismatic" else "revolute"
        unit = "mm" if kind == "prismatic" else "deg"
        limits = np.asarray(joint[f"limits_{unit}"], dtype=float)
        limits = limits / 1000.0 if unit == "mm" else np.radians(limits)
        manifest["joints"].append({
            "name": f"joint_{number}_{names[joint['parent'] - 1]}_to_{names[joint['child'] - 1]}",
            "parent": names[joint["parent"] - 1], "child": names[joint["child"] - 1],
            "type": kind, "origin": joint["axis"]["origin"], "axis": joint["axis"]["n"],
            "limits": limits.tolist(), "limits_source": "measured",
            "axis_source": "RORA_HITL_metric_interface",
        })
    manifest_path = output / "articulated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/compile_hitl_articulated_asset.py"),
                    "--manifest", str(manifest_path), "--output", str(output / "urdf_asset")],
                   check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON containing root, parents, ordered seed keys, and measured thickness")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coacd-threshold", type=float, default=0.02)
    parser.add_argument("--max-convex-hulls", type=int, default=32)
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=75)
    parser.add_argument("--coacd-resolution", type=int, default=3000)
    parser.add_argument("--core-hulls-per-link", type=int, default=1)
    parser.add_argument("--reselect", action="store_true", help="repeat the RORA HITL assignment")
    parser.add_argument("--resume-confirmed-seeds", action="store_true",
                        help="reuse the confirmed original-surface seeds in this output")
    parser.add_argument("--articulated", action="store_true",
                        help="review joints on the preliminary split and save joint_selection.json")
    parser.add_argument("--review-joints", action="store_true")
    parser.add_argument("--review-cuts", action="store_true",
                        help="reopen final physics-valid whole-partition approval")
    parser.add_argument("--edit-articulation", action="store_true",
                        help="reuse saved exact parts and edit/audit joints without rewriting PLY")
    parser.add_argument("--contact-radius-mm", type=float, default=20.0)
    parser.add_argument("--articulation-samples", type=int, default=3000)
    parser.add_argument("--resize-to-measured-thickness", action="store_true",
                        help="deprecated: connected-mesh thickness calibration must precede split")
    parser.add_argument("--allow-thickness-pending", action="store_true",
                        help="split now while preserving needs-thickness-correction provenance")
    parser.add_argument("--robot-name", default="rora_metric_object")
    args = parser.parse_args()
    if (args.coacd_threshold <= 0 or args.max_convex_hulls < 1
            or args.coacd_preprocess_resolution < 1 or args.coacd_resolution < 1
            or args.core_hulls_per_link < 1 or args.contact_radius_mm <= 0
            or args.articulation_samples < 100):
        parser.error("CoACD threshold, hull count, and resolutions must be positive")
    if args.resize_to_measured_thickness:
        parser.error(
            "post-split resize is disabled because interface caps bias volume; calibrate the "
            "connected mesh first and set input_geometry_semantics=thickness_corrected_material_surface"
        )

    mesh, source_hash = load_source(args.input, require_watertight=args.articulated), sha256(args.input)
    if not mesh.is_watertight:
        print("[RORA] open input accepted for proposal/HITL labels only; no articulated geometry will be written")
    template = json.loads(args.config.read_text(encoding="utf-8"))
    forbidden = forbidden_config_paths(template)
    if forbidden:
        raise ValueError(f"GT/target-volume fields are forbidden in the RORA config: {forbidden}")
    names = list(template.get("seeds", {}))
    if not names or template.get("root") not in names:
        raise ValueError("config must contain an ordered seeds map including root")
    needs_resize = (template.get("input_geometry_semantics")
                    == "watertight_surface_needs_thickness_correction")
    if needs_resize and not args.allow_thickness_pending and not args.edit_articulation:
        raise ValueError(
            "connected-mesh thickness calibration is required before part split; this wrapper "
            "accepts thickness_corrected_material_surface"
        )
    if needs_resize and not args.edit_articulation:
        print("[RORA] thickness resize remains pending; producing split/joints only")

    if args.edit_articulation:
        generated_config = args.output / "rora_generated_seeds.json"
        config_path = generated_config if generated_config.is_file() else args.config
        required = [args.output / "joint_selection.json", args.output / "hitl_face_labels.npy"]
        parts_dir = next((folder for folder in (
            args.output / "metric_parts", args.output / "initial_metric_parts"
        ) if all((folder / f"{name}_metric_watertight.ply").is_file() for name in names)), None)
        if parts_dir is None or not all(path.is_file() for path in required):
            raise RuntimeError("--edit-articulation requires saved exact parts, face labels, and joints")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/static_rora_metric_parts.py"), str(args.input),
            "--config", str(config_path), "--joints", str(required[0]),
            "--output", str(args.output), "--motion-preview-parts", str(parts_dir),
            "--face-labels", str(required[1]), "--edit-articulation",
            "--contact-radius-mm", str(args.contact_radius_mm),
            "--samples", str(args.articulation_samples),
        ], check=True)
        joint_payload = json.loads(required[0].read_text(encoding="utf-8"))
        compile_urdf(args.output, names, joint_payload["joints"], args.robot_name,
                     provenance={"metric_geometry": "saved_exact_PLY_unchanged",
                                 "joint_limits": "cap_exposure_and_collision_limited"},
                     parts_dir=parts_dir)
        save_progress(args.output, "ARTICULATION_EDIT", "COMPLETE",
                      audit=str((args.output / "articulation_audit.json").resolve()))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    parts = decomposition(mesh, args.output / "rora_selection_proxies_coarse", source_hash,
                          args.coacd_threshold, args.max_convex_hulls,
                          args.coacd_preprocess_resolution, args.coacd_resolution)
    categories, hitl_cut_planes, hitl_face_labels = selected_categories(
        parts, mesh, names, template["parents"],
        args.output / "rora_part_selection.json", source_hash, args.reselect,
    )
    save_progress(
        args.output, "1/3_PART_ASSIGNMENT", "SAVED",
        hull_counts={"ambiguous": len(categories[0]), **{
            name: len(categories[index + 1]) for index, name in enumerate(names)
        }},
        output=str((args.output / "rora_part_selection.json").resolve()),
    )
    save_preview(parts, categories, names, args.output / "rora_selection_preview.png")
    config_path = args.output / "rora_generated_seeds.json"
    if args.resume_confirmed_seeds:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        if (generated.get("seed_source") not in {
                    "original_surface_HITL_confirmed", "RORA_HITL_dense_safe_hulls",
                    "RORA_HITL_high_margin_core_hull"}
                or list(generated.get("seeds", {})) != names):
            raise RuntimeError("no compatible confirmed seed selection to resume")
        generated["hitl_cut_planes"] = hitl_cut_planes
        generated["hitl_face_labels_file"] = str(
            (args.output / "hitl_face_labels.npy").resolve()
        )
        config_path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    else:
        seeds, vertex_labels = rora_seeds(
            mesh, categories, names, core_hulls_per_link=args.core_hulls_per_link
        )
        np.save(args.output / "rora_vertex_labels.npy", vertex_labels)
        generated = {
            "root": template["root"],
            "parents": template["parents"],
            "seeds": seeds,
            "seed_source": "RORA_HITL_high_margin_core_hull",
            "ambiguous_policy": "excluded_from_hard_seeds_then_assigned_by_original_face_graph_and_joint_gates",
            "hitl_cut_planes": hitl_cut_planes,
            "hitl_face_labels_file": str((args.output / "hitl_face_labels.npy").resolve()),
        }
        for key in ("input_geometry_semantics", "measured_thickness_mm",
                    "dimension_constraints", "joint_housing_semantics_confirmed"):
            if key in template:
                generated[key] = template[key]
        config_path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")

    if not args.articulated:
        save_progress(args.output, "1/3_PART_ASSIGNMENT", "COMPLETE",
                      output=str(config_path.resolve()))
        print(f"RORA core config saved: {config_path}")
        return

    coarse_meshes = exact_parts_from_face_labels(
        mesh, names, template["parents"], hitl_face_labels
    )
    initial_parts_dir = args.output / "initial_exact_parts"
    initial_parts_dir.mkdir(exist_ok=True)
    for name, part in coarse_meshes.items():
        part.export(initial_parts_dir / f"{name}_initial_exact.ply")
    selection_hash = hashlib.sha256((
        source_hash + sha256(args.output / "rora_part_selection.json")
        + hashlib.sha256(hitl_face_labels.tobytes()).hexdigest()
    ).encode()).hexdigest()
    save_progress(args.output, "2-3/3_JOINT_REVIEW", "WAITING_OR_REUSING_SAVED_GUI")
    joints = review_joints(
        coarse_meshes, names, template["parents"],
        args.output / "joint_selection.json", selection_hash,
        args.review_joints, mesh,
    )
    save_progress(args.output, "2-3/3_JOINT_REVIEW", "SAVED", joint_count=len(joints),
                  output=str((args.output / "joint_selection.json").resolve()))
    save_progress(args.output, "FINAL_SPLIT", "RUNNING")
    split_command = [
        sys.executable, str(ROOT / "scripts/static_rora_metric_parts.py"), str(args.input),
        "--config", str(config_path), "--joints", str(args.output / "joint_selection.json"),
        "--output", str(args.output / "metric_parts"),
        "--smoothness", "2", "4", "8", "16", "--samples", "8000",
    ]
    if args.review_cuts or args.review_joints:
        split_command.append("--review-cuts")
    while True:
        split = subprocess.run(split_command, check=False)
        if split.returncode != 42:
            split.check_returncode()
            break
        save_progress(args.output, "2-3/3_JOINT_REVIEW", "EDIT_REQUESTED_FROM_FINAL_CUT")
        joints = review_joints(
            coarse_meshes, names, template["parents"],
            args.output / "joint_selection.json", selection_hash,
            True, mesh,
        )
        save_progress(args.output, "FINAL_SPLIT", "RECOMPUTING_AFTER_JOINT_EDIT")
    geometry_audit = json.loads((args.output / "metric_parts" /
                                 "part_separation_audit.json").read_text(encoding="utf-8"))
    save_progress(args.output, "FINAL_SPLIT", geometry_audit["status"],
                  output=str((args.output / "metric_parts").resolve()))
    if geometry_audit["status"] != "PASS":
        raise RuntimeError(
            f"delivery gate rejected {geometry_audit['status']}; inspect joint previews/audit, "
            "correct the input or HITL approval, and rerun"
        )
    final_output = args.output
    compile_urdf(args.output, names, joints, args.robot_name)
    urdf_audit = json.loads((final_output / "urdf_asset" / "audit.json").read_text(encoding="utf-8"))
    save_progress(args.output, "COMPLETE", "PASS_GEOMETRY_DENSITY_PENDING",
                  geometry_status=geometry_audit["status"], urdf_status=urdf_audit["status"],
                  output=str((final_output / "urdf_asset").resolve()))


if __name__ == "__main__":
    main()
