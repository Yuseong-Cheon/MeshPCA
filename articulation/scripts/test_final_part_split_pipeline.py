#!/usr/bin/env python3
"""Small category-independent regression checks for the delivered splitter."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rora_prior_split_ply import (
    exact_parts_from_face_labels,
    exact_parts_from_planes,
    plane_face_labels,
    rora_seeds,
    selected_joints,
)
from resize_articulated_parts import roi_affine_resize
from separate_ply_links import close_labeled_parts, repair_tree_face_labels


def tapered_bar(sections):
    vertices = []
    for x, y, z in sections:
        vertices.extend([[x, -y, -z], [x, y, -z], [x, y, z], [x, -y, z]])
    faces = [[0, 2, 1], [0, 3, 2]]
    for index in range(len(sections) - 1):
        a, b = 4 * index, 4 * (index + 1)
        for side in range(4):
            u, v = a + side, a + (side + 1) % 4
            w, q = b + side, b + (side + 1) % 4
            faces.extend([[u, v, q], [u, q, w]])
    end = 4 * (len(sections) - 1)
    faces.extend([[end, end + 1, end + 2], [end, end + 2, end + 3]])
    return trimesh.Trimesh(vertices, faces, vertex_colors=[210, 215, 220, 255], process=False)


def transformed(mesh, points):
    transform = trimesh.transformations.euler_matrix(0.31, -0.44, 0.27)
    transform[:3, 3] = [0.4, -0.2, 0.7]
    mesh.apply_transform(transform)
    return trimesh.transform_points(np.asarray(points, float), transform).tolist()


def laptop_shape():
    profile = np.asarray([
        [-1.0, -.01], [.01, -.01], [.01, 1.0],
        [-.01, 1.0], [-.01, .01], [-1.0, .01],
    ])
    triangles = np.asarray([[0, 1, 4], [0, 4, 5], [1, 2, 3], [1, 3, 4]])
    mesh = trimesh.creation.extrude_triangulation(profile, triangles, 1.0)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.vertices[:, 1] -= .5
    mesh.faces = mesh.faces[:, ::-1]
    mesh.fix_normals()
    return mesh


def run_case(directory, name, sections, links, parents, seeds):
    mesh = tapered_bar(sections)
    for _ in range(3):
        mesh = mesh.subdivide()
    mesh.vertices += np.random.default_rng(7).normal(0.0, 1e-5, mesh.vertices.shape)
    flat = [point for values in seeds.values() for point in values]
    moved = transformed(mesh, flat)
    counts = [len(values) for values in seeds.values()]
    cursor, moved_seeds = 0, {}
    for link, count in zip(links, counts):
        moved_seeds[link] = moved[cursor:cursor + count]; cursor += count
    source = directory / f"{name}.ply"; mesh.export(source)
    config = directory / f"{name}.json"
    config.write_text(json.dumps({"root": links[0], "parents": parents,
                                  "seeds": moved_seeds,
                                  "seed_source": "original_surface_HITL_confirmed"}))
    output = directory / name
    subprocess.run([sys.executable, str(ROOT / "scripts/separate_ply_links.py"),
                    str(source), "--config", str(config), "--output", str(output)],
                   check=True, stdout=subprocess.DEVNULL)
    audit = json.loads((output / "part_separation_audit.json").read_text())
    assert audit["status"] == "PASS" and audit["semantic_status"] == "HITL_CONFIRMED"
    assert audit["volume_closure_error_cm3"] < 1e-4
    assert all(part["body_count"] == 1 and part["watertight"]
               for part in audit["parts"].values())


def run_static_laptop(directory):
    mesh = laptop_shape()
    for _ in range(3):
        mesh = mesh.subdivide()
    seed_values = {
        "base": [[-.7, -.4, -.01], [-.7, .4, -.01], [-.7, 0, .01]],
        "lid": [[-.01, -.4, .7], [.01, .4, .7], [.01, 0, .7]],
    }
    flat = [point for values in seed_values.values() for point in values]
    moved = transformed(mesh, flat)
    seed_values = {"base": moved[:3], "lid": moved[3:]}
    transform = trimesh.transformations.euler_matrix(0.31, -0.44, 0.27)
    transform[:3, 3] = [0.4, -0.2, 0.7]
    origin = trimesh.transform_points([[0, 0, 0]], transform)[0]
    axis = transform[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    source = directory / "static_laptop.ply"; mesh.export(source)
    config = directory / "static_laptop.json"
    config.write_text(json.dumps({
        "root": "base", "parents": {"lid": "base"}, "seeds": seed_values,
        "seed_source": "RORA_HITL_high_margin_core_hull",
        "input_geometry_semantics": "thickness_corrected_material_surface",
        "measured_thickness_mm": {"base": 20.0, "lid": 20.0},
        "joint_housing_semantics_confirmed": False,
    }))
    joints = directory / "static_laptop_joints.json"
    joints.write_text(json.dumps({"links": ["base", "lid"], "joints": [{
        "parent": "base", "child": "lid", "type": "revolute",
        "axis": {"origin": origin.tolist(), "n": axis.tolist()},
        "limits_deg": [-60.0, 15.0],
    }]}))
    output = directory / "static_laptop"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/static_rora_metric_parts.py"), str(source),
        "--config", str(config), "--joints", str(joints), "--output", str(output),
        "--smoothness", "2", "--samples", "100",
    ], check=True, stdout=subprocess.DEVNULL,
       env={**os.environ, "MPLCONFIGDIR": str(directory / "mpl")})
    audit = json.loads((output / "part_separation_audit.json").read_text())
    selected = next(candidate for candidate in audit["selection"]["candidates"]
                    if candidate["family"] == audit["selection"]["selected_family"])
    assert selected["axis_centered_interface_gate"]
    assert audit["volume_closure_error_fraction"] < 1e-6
    assert "ground_truth_evaluation_only" not in audit
    poisoned = json.loads(config.read_text())
    poisoned["metadata"] = {"ground_truth_volume_cm3": {"base": 1.0, "lid": 1.0}}
    poison_path = directory / "static_laptop_with_forbidden_gt.json"
    poison_path.write_text(json.dumps(poisoned))
    rejected = subprocess.run([
        sys.executable, str(ROOT / "scripts/static_rora_metric_parts.py"), str(source),
        "--config", str(poison_path), "--joints", str(joints),
        "--output", str(directory / "must_not_exist"), "--samples", "100",
    ], capture_output=True, text=True)
    assert rejected.returncode != 0 and "GT/target-volume fields are forbidden" in rejected.stderr


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        rejected_resize = subprocess.run([
            sys.executable, str(ROOT / "scripts/rora_prior_split_ply.py"), "unused.ply",
            "--config", "unused.json", "--output", str(directory / "must_not_resize"),
            "--articulated", "--resize-to-measured-thickness",
        ], capture_output=True, text=True)
        assert (rejected_resize.returncode != 0
                and "post-split resize is disabled" in rejected_resize.stderr)
        rejected_resolution = subprocess.run([
            sys.executable, str(ROOT / "scripts/rora_prior_split_ply.py"), "unused.ply",
            "--config", "unused.json", "--output", str(directory / "must_not_run"),
            "--coacd-resolution", "0",
        ], capture_output=True, text=True)
        assert (rejected_resolution.returncode != 0
                and "resolutions must be positive" in rejected_resolution.stderr)
        run_case(directory, "laptop_like", [(-1, .5, .08), (-.03, .5, .08),
                 (0, .12, .08), (.03, .5, .08), (1, .5, .08)],
                 ["base", "lid"], {"lid": "base"},
                 {"base": [[-.7, -.5, -.08], [-.7, .5, -.08],
                            [-.7, .5, .08], [-.7, -.5, .08]],
                  "lid": [[.7, -.5, -.08], [.7, .5, -.08],
                           [.7, .5, .08], [.7, -.5, .08]]})
        run_case(directory, "three_link", [(-1.5, .45, .2), (-.52, .45, .2),
                 (-.48, .12, .12), (.48, .12, .12), (.52, .4, .16), (1.5, .4, .16)],
                 ["root", "arm", "tool"], {"arm": "root", "tool": "arm"},
                 {"root": [[-.9, -.45, -.2], [-.9, .45, -.2],
                            [-.9, .45, .2], [-.9, -.45, .2]],
                  "arm": [[0, -.12, -.12], [0, .12, -.12],
                           [0, .12, .12], [0, -.12, .12]],
                  "tool": [[.9, -.4, -.16], [.9, .4, -.16],
                            [.9, .4, .16], [.9, -.4, .16]]})

        source = tapered_bar([(-1, .3, .2), (1, .3, .2)])
        for _ in range(3):
            source = source.subdivide()
        planes = {"right": {"normal": [1, 0, 0], "offset_m": 0,
                            "child_is_positive_halfspace": True}}
        full_labels = plane_face_labels(source, ["left", "right"], {"right": "left"}, planes)
        visible = np.arange(0, len(source.faces), 3)
        assert np.array_equal(
            full_labels[visible],
            plane_face_labels(source, ["left", "right"], {"right": "left"},
                              planes, source.triangles_center[visible]),
        )
        left = trimesh.creation.box([1.2, .7, .5],
            transform=trimesh.transformations.translation_matrix([-.55, 0, 0]))
        right = trimesh.creation.box([1.2, .7, .5],
            transform=trimesh.transformations.translation_matrix([.55, 0, 0]))
        crossing = trimesh.creation.box([.5, .8, .6])
        seeds, _labels = rora_seeds(source, [[crossing], [left], [right]], ["a", "b"])
        assert all(len(values) >= 3 for values in seeds.values())
        exact = exact_parts_from_planes(
            source, ["left", "right"], {"right": "left"},
            {"right": {"normal": [1, 0, 0], "offset_m": 0,
                       "child_is_positive_halfspace": True}},
            {"left": [[-.8, 0, 0]], "right": [[.8, 0, 0]]},
        )
        assert all(part.is_watertight and part.body_count == 1 for part in exact.values())
        assert abs(sum(abs(part.volume) for part in exact.values()) - abs(source.volume)) < 1e-9
        painted = exact_parts_from_face_labels(
            source, ["left", "right"], {"right": "left"},
            (source.triangles_center[:, 0] >= 0).astype(np.int16),
        )
        assert all(part.is_watertight and part.body_count == 1 for part in painted.values())
        assert abs(sum(abs(part.volume) for part in painted.values()) - abs(source.volume)) < 1e-9
        sphere = trimesh.creation.icosphere(subdivisions=3)
        labels = np.ones(len(sphere.faces), dtype=np.int16)
        labels[sphere.triangles_center[:, 2] < -.25] = 0
        labels[sphere.triangles_center[:, 2] > .25] = 2
        labels[np.argmin(np.linalg.norm(sphere.triangles_center - [1, 0, 0], axis=1))] = 2
        labels, _repair = repair_tree_face_labels(
            sphere, labels, ["base", "support", "head"],
            {"support": "base", "head": "support"},
        )
        repaired, repaired_interfaces = close_labeled_parts(
            sphere, labels, ["base", "support", "head"]
        )
        assert set(repaired_interfaces) == {"base--support", "support--head"}
        assert all(part.is_watertight and part.body_count == 1 for part in repaired.values())
        joints = selected_joints([{
            "a": {"parent": 1, "child": 2},
            "vectors": [{"center": [0, 0, 0], "n": [0, 1, 0], "state": "Revolute"}],
        }])
        assert joints == [{"parent": 1, "child": 2, "axis": {
            "origin": [0, 0, 0], "n": [0, 1, 0]}, "type": "Revolute"}]
        roi_box = trimesh.creation.box([.4, .2, .04])
        for _ in range(3):
            roi_box = roi_box.subdivide()
        protected = np.flatnonzero(roi_box.vertices[:, 0] < -.19)
        measurement = np.flatnonzero(roi_box.vertices[:, 0] > -.15)
        corrected, dimension_audit = roi_affine_resize(
            roi_box, {"width": 100.0, "thickness": 20.0},
            {"width": [0, 1, 0], "thickness": [0, 0, 1]},
            measurement, protected,
        )
        assert corrected.is_watertight
        assert dimension_audit["maximum_protected_displacement_mm"] == 0.0
        assert all(abs(item["achieved_mm"] - item["target_mm"]) <= item["tolerance_mm"]
                   for item in dimension_audit["dimensions"].values())
        run_static_laptop(directory)
    print('{"status":"PASS","cases":["laptop_like","three_link","overlapping_RORA_prior","static_RORA_laptop"]}')


if __name__ == "__main__":
    main()
