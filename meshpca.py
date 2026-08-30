#!/usr/bin/env python3
"""Measure N separated parts from robust original RGB-D medians per PCA axis."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import rgbd


SUFFIX = "_metric_watertight.ply"


class ColmapImage:
    def __init__(self, image):
        pose = image.cam_from_world()
        self.name = image.name
        self.camera_id = image.camera_id
        self.tvec = np.asarray(pose.translation)
        self._rotation = np.asarray(pose.rotation.matrix())

    def qvec2rotmat(self):
        return self._rotation


def load_colmap(path):
    reconstruction = pycolmap.Reconstruction(str(path))
    images = {key: ColmapImage(value) for key, value in reconstruction.images.items()}
    return reconstruction.cameras, images


def source_id(name):
    return Path(name).stem.rsplit("_", 1)[-1]


def camera_center(image):
    rotation = image.qvec2rotmat()
    return -rotation.T @ image.tvec


def fit_similarity(source, target, keep=None):
    if keep is None:
        keep = np.ones(len(source), bool)
    x, y = source[keep], target[keep]
    x_mean, y_mean = x.mean(0), y.mean(0)
    covariance = (y - y_mean).T @ (x - x_mean) / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    variance = np.mean(np.sum((x - x_mean) ** 2, axis=1))
    scale = np.trace(np.diag(singular) @ sign) / variance
    translation = y_mean - scale * rotation @ x_mean
    return scale, rotation, translation


def robust_similarity(mesh_images, rgb_images):
    mesh_by_source = {source_id(image.name): image for image in mesh_images.values()}
    rgb_by_source = {source_id(image.name): image for image in rgb_images.values()}
    common = sorted(set(mesh_by_source) & set(rgb_by_source))
    if len(common) < 8:
        raise RuntimeError(f"only {len(common)} common COLMAP cameras")
    source = np.asarray([camera_center(mesh_by_source[key]) for key in common])
    target = np.asarray([camera_center(rgb_by_source[key]) for key in common])
    keep = np.ones(len(common), bool)
    for _ in range(4):
        scale, rotation, translation = fit_similarity(source, target, keep)
        predicted = (scale * (rotation @ source.T)).T + translation
        error = np.linalg.norm(predicted - target, axis=1)
        median = np.median(error[keep])
        mad = np.median(np.abs(error[keep] - median))
        next_keep = error <= median + max(3 * 1.4826 * mad, 1e-4)
        if next_keep.sum() < 8 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    scale, rotation, translation = fit_similarity(source, target, keep)
    predicted = (scale * (rotation @ source.T)).T + translation
    error = np.linalg.norm(predicted - target, axis=1)
    return scale, rotation, translation, {
        "common_cameras": len(common),
        "inlier_cameras": int(keep.sum()),
        "center_error_median_colmap_units": float(np.median(error[keep])),
        "center_error_max_colmap_units": float(np.max(error[keep])),
    }


def camera_parameters(camera):
    model = getattr(camera.model, "name", camera.model)
    if model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        fy = fx
    else:
        raise ValueError(f"unsupported camera model: {camera.model}")
    return float(fx), float(fy), float(cx), float(cy)


def mesh_axes(vertices):
    centered = vertices - vertices.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    return axes


def build_scene(paths, scale, rotation, translation, mesh_scale_m):
    scene = o3d.t.geometry.RaycastingScene()
    labels, axes, geometry_to_label = [], {}, {}
    for path in paths:
        label = path.name.removesuffix(SUFFIX)
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if not mesh.has_triangles():
            raise RuntimeError(f"empty mesh: {path}")
        metric_vertices = np.asarray(mesh.vertices)
        benchmark_vertices = metric_vertices / mesh_scale_m
        vertices = (scale * (rotation @ benchmark_vertices.T)).T + translation
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        geometry_id = int(scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
        labels.append(label)
        axes[label] = mesh_axes(vertices)
        geometry_to_label[geometry_id] = label
    return scene, labels, axes, geometry_to_label


def cast_frame(scene, image, camera):
    fx, fy, cx, cy = camera_parameters(camera)
    yy, xx = np.indices((camera.height, camera.width), dtype=np.float32)
    camera_directions = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1)
    rotation = image.qvec2rotmat()
    world_directions = camera_directions @ rotation
    origin = camera_center(image).astype(np.float32)
    origins = np.broadcast_to(origin, world_directions.shape)
    rays = np.concatenate((origins, world_directions), axis=-1).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))
    return hits["geometry_ids"].numpy(), hits["t_hit"].numpy()


def depth_points(depth, mask, fx, fy, cx, cy):
    y, x = np.where(mask)
    z = depth[y, x].astype(np.float64) / 1000.0
    points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    return points, x, y


def projected_border(mask):
    y, x = np.where(mask)
    if not len(x):
        return -1
    height, width = mask.shape
    return int(min(x.min(), y.min(), width - 1 - x.max(), height - 1 - y.max()))


def depth_agreement(depth, valid, t_hit, metric_per_original_unit):
    residual = depth[valid].astype(np.float64) / 1000 - t_hit[valid] * metric_per_original_unit
    plausible = residual[np.abs(residual) < 0.12]
    if len(plausible) < 80:
        return np.zeros_like(valid), 0.0, 12.0
    edges = np.arange(-0.122, 0.126, 0.004)
    counts, _ = np.histogram(plausible, edges)
    peak = int(np.argmax(counts))
    rough = (edges[peak] + edges[peak + 1]) / 2
    cluster = plausible[np.abs(plausible - rough) <= 0.008]
    median = float(np.median(cluster))
    tolerance = 0.012
    agreement = valid.copy()
    agreement[valid] = np.abs(residual - median) <= tolerance
    return agreement, median * 1000, tolerance * 1000


def frame_measurement(depth, sam_mask, geometry_ids, t_hit, geometry_id, image, camera,
                      axis_world, metric_per_original_unit, axis_index):
    projected = geometry_ids == geometry_id
    border_px = projected_border(projected)
    projected = cv2.erode(projected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = projected & sam_mask
    valid = mask & (depth > 0)
    mask_pixels = int(mask.sum())
    if mask_pixels < 200 or valid.sum() < 120:
        return None
    agreement, depth_offset, depth_tolerance = depth_agreement(
        depth, valid, t_hit, metric_per_original_unit)
    if mask_pixels < 200 or agreement.sum() < 120:
        return None
    fx, fy, cx, cy = camera_parameters(camera)
    points, x, y = depth_points(depth, agreement, fx, fy, cx, cy)
    rotation = image.qvec2rotmat()
    axis_camera = rotation @ axis_world
    axis_camera /= np.linalg.norm(axis_camera)
    center = np.median(points, axis=0)
    coordinates = (points - center) @ axis_camera
    percentiles = ((0.1, 99.9), (0.25, 99.75), (0.5, 99.5))[axis_index]
    low, high = np.percentile(coordinates, percentiles)
    value_mm = float((high - low) * 1000)
    viewability = float(np.linalg.norm(axis_camera[:2]))
    coverage = float(valid.sum() / mask_pixels)
    agreement_fraction = float(agreement.sum() / valid.sum())
    if value_mm <= 1 or viewability < 0.35 or coverage < 0.01 or agreement_fraction < 0.25:
        return None
    score = float(np.log1p(agreement.sum()) * np.sqrt(coverage) * agreement_fraction * viewability ** 2)
    return {
        "value_mm": value_mm,
        "score": score,
        "viewability": viewability,
        "depth_coverage": coverage,
        "mesh_depth_agreement": agreement_fraction,
        "mesh_depth_offset_mm": depth_offset,
        "mesh_depth_tolerance_mm": depth_tolerance,
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": int(agreement.sum()),
        "border_px": border_px,
        "axis_camera": axis_camera,
        "center_camera": center,
        "low_m": float(low),
        "high_m": float(high),
    }


def section_measurement(depth, sam_mask, geometry_ids, t_hit, geometry_id, image, camera,
                        axes_world, axis_index, metric_per_original_unit, central_range):
    projected = geometry_ids == geometry_id
    border_px = projected_border(projected)
    projected = cv2.erode(projected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = projected & sam_mask
    valid = mask & (depth > 0)
    mask_pixels = int(mask.sum())
    if mask_pixels < 200 or valid.sum() < 100:
        return None
    agreement, depth_offset, depth_tolerance = depth_agreement(
        depth, valid, t_hit, metric_per_original_unit)
    if mask_pixels < 200 or agreement.sum() < 100:
        return None
    fx, fy, cx, cy = camera_parameters(camera)
    points, x, y = depth_points(depth, agreement, fx, fy, cx, cy)
    rotation = image.qvec2rotmat()
    axes_camera = (rotation @ axes_world.T).T
    center = np.median(points, axis=0)
    all_coordinates = (points - center) @ axes_camera.T
    long_low, long_high = np.percentile(all_coordinates[:, 0], (0.5, 99.5))
    selected = np.ones(len(points), bool)
    if axis_index:
        start, stop = central_range
        selected = ((all_coordinates[:, 0] >= long_low + start * (long_high - long_low)) &
                    (all_coordinates[:, 0] <= long_low + stop * (long_high - long_low)))
        if selected.sum() < 80:
            return None
    coordinates = all_coordinates[selected, axis_index]
    percentiles = ((0.1, 99.9), (0.25, 99.75), (0.5, 99.5))[axis_index]
    low, high = np.percentile(coordinates, percentiles)
    value_mm = float((high - low) * 1000)
    axis_camera = axes_camera[axis_index]
    viewability = float(np.linalg.norm(axis_camera[:2]))
    coverage = float(valid.sum() / mask_pixels)
    agreement_fraction = float(agreement.sum() / valid.sum())
    if value_mm <= 1 or viewability < 0.35 or coverage < 0.01 or agreement_fraction < 0.25:
        return None
    return {
        "value_mm": value_mm,
        "score": float(np.log1p(selected.sum()) * np.sqrt(coverage) * agreement_fraction * viewability ** 2),
        "viewability": viewability,
        "depth_coverage": coverage,
        "mesh_depth_agreement": agreement_fraction,
        "mesh_depth_offset_mm": depth_offset,
        "mesh_depth_tolerance_mm": depth_tolerance,
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": int(selected.sum()),
        "border_px": border_px,
        "axis_camera": axis_camera,
        "center_camera": np.median(points[selected], axis=0),
        "low_m": float(low),
        "high_m": float(high),
    }


def choose_center(rows, axis_index):
    if not rows:
        return None
    complete = [row for row in rows if row["border_px"] >= 4] or rows
    if axis_index == 2:
        best_view = max(row["viewability"] for row in complete)
        complete = [row for row in complete if row["viewability"] >= max(0.70, 0.93 * best_view)]
        return max(complete, key=lambda row: row["score"] * np.sqrt(row["mask_pixels"]))
    best_score = max(row["score"] for row in complete)
    quality = [row for row in complete if row["score"] >= 0.65 * best_score]
    values = np.asarray([row["value_mm"] for row in quality])
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sane = [row for row in quality if row["value_mm"] <= median + max(3 * 1.4826 * mad, 2.0)]
    return max(sane or quality, key=lambda row: row["value_mm"])


def open_raw_rgbd(db3, camera):
    helper = rgbd
    database = sqlite3.connect(f"file:{db3}?mode=ro&immutable=1", uri=True)
    topics = dict(database.execute("SELECT name,id FROM topics"))
    color_topic = topics["/device_0/sensor_1/Color_0/image/data"]
    depth_topic = topics["/device_0/sensor_0/Depth_0/image/data"]
    color_rows = list(database.execute(
        "SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (color_topic,)))
    depth_rows = list(database.execute(
        "SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (depth_topic,)))
    depth_times = np.asarray([row[1] for row in depth_rows], np.int64)

    def topic_string(name):
        blob = database.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (topics[name],)).fetchone()[0]
        return helper.decode_string(blob)

    depth_cal = helper.parse_camera_info(topic_string("/device_0/sensor_0/Depth_0/camera_info"))
    color_cal = helper.parse_camera_info(topic_string("/device_0/sensor_1/Color_0/camera_info"))
    color_from_depth = helper.parse_transform(topic_string("/device_0/sensor_1/Color_0/tf/ref_0"))
    fx, fy, cx, cy = camera_parameters(camera)
    original_k = np.array(((color_cal["fx"], 0, color_cal["ppx"]),
                           (0, color_cal["fy"], color_cal["ppy"]), (0, 0, 1)), float)
    distortion = np.asarray(color_cal["coeffs"], float)
    output_k = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), float)
    map_x, map_y = cv2.initUndistortRectifyMap(
        original_k, distortion, None, output_k, (camera.width, camera.height), cv2.CV_32FC1)
    return {
        "helper": helper, "database": database, "color_rows": color_rows,
        "depth_rows": depth_rows, "depth_times": depth_times,
        "depth_cal": depth_cal, "color_cal": color_cal, "color_from_depth": color_from_depth,
        "map_x": map_x, "map_y": map_y,
    }


def load_raw_frame(raw, index):
    index = int(np.clip(index, 0, len(raw["color_rows"]) - 1))
    color_id, timestamp = raw["color_rows"][index]
    blob = raw["database"].execute("SELECT data FROM messages WHERE id=?", (color_id,)).fetchone()[0]
    rgb = raw["helper"].decode_image(blob)
    at = int(np.searchsorted(raw["depth_times"], timestamp))
    choices = [max(0, at - 1), min(len(raw["depth_rows"]) - 1, at)]
    nearest = min(choices, key=lambda i: abs(int(raw["depth_times"][i]) - timestamp))
    depth_id, depth_time = raw["depth_rows"][nearest]
    blob = raw["database"].execute("SELECT data FROM messages WHERE id=?", (depth_id,)).fetchone()[0]
    aligned_m = raw["helper"].align_depth(
        raw["helper"].decode_image(blob), raw["depth_cal"], raw["color_cal"], raw["color_from_depth"])
    rgb = cv2.remap(rgb, raw["map_x"], raw["map_y"], cv2.INTER_LINEAR, borderValue=0)
    depth_m = cv2.remap(aligned_m, raw["map_x"], raw["map_y"], cv2.INTER_NEAREST, borderValue=0)
    depth = np.clip(np.rint(depth_m * 1000), 0, 65535).astype(np.uint16)
    return rgb, depth, {"source_index": index, "color_time_ns": timestamp,
                        "depth_time_ns": depth_time, "sync_dt_ms": (depth_time - timestamp) / 1e6}


def estimate_raw_warp(center_rgb, raw_rgb):
    template = cv2.cvtColor(center_rgb, cv2.COLOR_BGR2GRAY)
    moving = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
    except cv2.error:
        score = 0.0
        warp = np.eye(2, 3, dtype=np.float32)
    return warp, float(score)


def project_point(point, camera):
    fx, fy, cx, cy = camera_parameters(camera)
    if point[2] <= 0:
        return None
    return int(round(fx * point[0] / point[2] + cx)), int(round(fy * point[1] / point[2] + cy))


def draw_overlay(rgb, geometry_ids, geometry_id, row, label, axis_name):
    shown = rgb.copy()
    mask = geometry_ids == geometry_id
    tint = np.zeros_like(shown)
    tint[mask] = (0, 180, 0)
    shown = cv2.addWeighted(shown, 1.0, tint, 0.35, 0)
    a = row["center_camera"] + row["axis_camera"] * row["low_m"]
    b = row["center_camera"] + row["axis_camera"] * row["high_m"]
    pa, pb = project_point(a, row["camera"]), project_point(b, row["camera"])
    if pa and pb:
        cv2.arrowedLine(shown, pa, pb, (0, 0, 255), 4, cv2.LINE_AA, tipLength=.08)
        cv2.arrowedLine(shown, pb, pa, (0, 0, 255), 4, cv2.LINE_AA, tipLength=.08)
    if "_reported_mm" in row:
        caption = (f"{label} {axis_name}: median{row['_reported_count']} {row['_reported_mm']:.1f}"
                   f" | raw {row['value_mm']:.1f} | {Path(row['frame']).stem}")
    else:
        caption = f"{label} {axis_name}: {row['value_mm']:.1f} mm | {row['frame']}"
    cv2.rectangle(shown, (10, 10), (850, 62), (255, 255, 255), -1)
    cv2.putText(shown, caption, (22, 46), cv2.FONT_HERSHEY_SIMPLEX,
                .68, (0, 0, 200), 2, cv2.LINE_AA)
    return shown


def depth_view(depth):
    valid = depth > 0
    shown = np.zeros(depth.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (2, 98))
        shown[valid] = np.clip((high - depth[valid]) / max(high - low, 1) * 255, 0, 255)
    return cv2.applyColorMap(shown, cv2.COLORMAP_TURBO)


def serializable_row(row):
    return {key: value for key, value in row.items()
            if key not in {"axis_camera", "center_camera", "camera", "_rgb", "_depth", "_geometry_ids"}}


def best_measurement(rows):
    return max(rows, key=lambda row: row["score"], default=None)


def top_measurements(rows, count):
    return sorted(sorted(rows, key=lambda row: row["score"], reverse=True)[:count],
                  key=lambda row: row["order"])


def review_files(output):
    summary = output / "part_dimensions_summary.jpg"
    pages = [summary, *sorted(output.glob("*_selected_per_axis.jpg"))]
    missing = [str(path) for path in pages if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing review images: {missing}")
    return pages


def review_page(path, index, count):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"failed to read review image: {path}")
    maximum = np.array((1450, 780), float)
    scale = min(1.0, *(maximum / np.array((image.shape[1], image.shape[0]))))
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    footer = np.full((72, shown.shape[1], 3), 245, np.uint8)
    cv2.putText(footer, f"{index + 1}/{count} {path.name}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(footer, "N/P: browse   A: APPROVE   R: REJECT   Esc/Q: pending", (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 180), 2, cv2.LINE_AA)
    return np.vstack((shown, footer))


def review_decision(status, source, pages):
    return {
        "status": status,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_source": source,
        "reviewed_files": [path.name for path in pages],
        "reason": ({"HITL_APPROVED": "human_approved_final_measurement",
                    "HITL_REJECTED": "human_rejected_final_measurement"}
                   .get(status, "human_approval_not_recorded")),
    }


def final_review(output):
    output = output.resolve()
    report_path = output / "part_dimensions_multiview_pca.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing measurement report: {report_path}")
    pages = review_files(output)
    status, source = "HITL_PENDING", "noninteractive"
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        source, index = "opencv_gui", 0
        cv2.namedWindow("MeshPCA final review", cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow("MeshPCA final review", review_page(pages[index], index, len(pages)))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("n"), ord("N"), 83):
                index = (index + 1) % len(pages)
            elif key in (ord("p"), ord("P"), 81):
                index = (index - 1) % len(pages)
            elif key in (ord("a"), ord("A")):
                status = "HITL_APPROVED"
                break
            elif key in (ord("r"), ord("R")):
                status = "HITL_REJECTED"
                break
            elif key in (27, ord("q"), ord("Q")):
                break
        cv2.destroyWindow("MeshPCA final review")
    elif sys.stdin.isatty():
        source = "terminal"
        print("Review these files before deciding:")
        print("\n".join(f"  {path}" for path in pages))
        answer = input("Approve [A], reject [R], leave pending [Enter]: ").strip().lower()
        status = {"a": "HITL_APPROVED", "r": "HITL_REJECTED"}.get(answer[:1], status)
    decision = review_decision(status, source, pages)
    report = json.loads(report_path.read_text())
    report["final_review"] = decision
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "final_review.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if status == "HITL_APPROVED" else 2


def self_check():
    rows = [{"frame": f"f{i}", "order": i, "score": 10 - abs(i - 5),
             "viewability": .9, "value_mm": float(i), "border_px": 10,
             "mask_pixels": 100} for i in range(11)]
    assert choose_center(rows, 0)["order"] == 8
    assert best_measurement(rows)["order"] == 5
    assert [row["order"] for row in top_measurements(rows, 3)] == [4, 5, 6]
    approved = review_decision("HITL_APPROVED", "test", [Path("summary.jpg")])
    rejected = review_decision("HITL_REJECTED", "test", [Path("part.jpg")])
    assert approved["status"] == "HITL_APPROVED" and approved["reviewed_files"] == ["summary.jpg"]
    assert rejected["reason"] == "human_rejected_final_measurement"
    x = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), float)
    scale, rotation, translation = fit_similarity(x, 2 * x + 3)
    assert np.isclose(scale, 2) and np.allclose(rotation, np.eye(3)) and np.allclose(translation, 3)
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = not ({"--self-check", "--review-existing"} & set(sys.argv))
    parser.add_argument("--project", type=Path, required=required,
                        help="project containing selected RGB, aligned Depth, SAM masks, and COLMAP model")
    parser.add_argument("--mesh-colmap", type=Path, required=required,
                        help="COLMAP dataset used to create the separated meshes")
    parser.add_argument("--labels", type=Path, required=required,
                        help=f"directory containing *{SUFFIX}")
    parser.add_argument("--db3", type=Path, required=required, help="original RealSense ROS2 DB3")
    parser.add_argument("--output", type=Path, required=required)
    parser.add_argument("--mesh-scale", type=float, required=required,
                        help="meters per unit in the mesh-generation COLMAP reconstruction")
    parser.add_argument("--rules", type=Path, help="optional per-label JSON rules")
    parser.add_argument("--frames-per-axis", type=int, default=3,
                        help="top original DB3 frames used for each median (default: 3)")
    parser.add_argument("--raw-radius", type=int, default=15)
    parser.add_argument("--review-final", action="store_true",
                        help="require a final human approve/reject decision")
    parser.add_argument("--review-existing", type=Path,
                        help="review an existing output without recomputing measurements")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.review_existing:
        return final_review(args.review_existing)
    if args.frames_per_axis < 3 or args.frames_per_axis % 2 == 0:
        parser.error("--frames-per-axis must be an odd number >= 3")
    if args.raw_radius < 0:
        parser.error("--raw-radius cannot be negative")
    if 2 * args.raw_radius + 1 < args.frames_per_axis:
        parser.error("--raw-radius window is smaller than --frames-per-axis")
    if args.mesh_scale <= 0:
        parser.error("--mesh-scale must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    alignment_path = args.project / "depth/native_aligned_colmap_v1/alignment.json"
    alignment = json.loads(alignment_path.read_text())
    if Path(alignment["source"]).resolve() != args.db3.resolve():
        raise RuntimeError("aligned Depth does not come from the requested DB3")

    rgb_sparse = args.project / "colmap_v4_final/sparse/0"
    mesh_sparse = args.mesh_colmap / "sparse/0"
    rgb_cameras, rgb_images = load_colmap(rgb_sparse)
    _, mesh_images = load_colmap(mesh_sparse)
    align_scale, align_rotation, align_translation, align_report = robust_similarity(
        mesh_images, rgb_images)
    metric_per_original_unit = args.mesh_scale / align_scale

    paths = sorted(args.labels.glob(f"*{SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"no *{SUFFIX} in {args.labels}")
    scene, labels, axes, geometry_to_label = build_scene(
        paths, align_scale, align_rotation, align_translation, args.mesh_scale)
    label_to_geometry = {label: geometry for geometry, label in geometry_to_label.items()}
    rules = json.loads(args.rules.read_text()) if args.rules else {}
    unknown_rules = sorted(set(rules) - set(labels))
    if unknown_rules:
        raise ValueError(f"rules reference unknown labels: {unknown_rules}")
    central_ranges = {}
    for label, rule in rules.items():
        if "central_longitudinal_range" in rule:
            start, stop = map(float, rule["central_longitudinal_range"])
            if not 0 <= start < stop <= 1:
                raise ValueError(f"invalid central_longitudinal_range for {label}")
            central_ranges[label] = (start, stop)

    rgb_root = args.project / "colmap_v4_final/images"
    depth_root = args.project / "depth/native_aligned_colmap_v1/by_source_name"
    sam_root = args.project / "masks/sam3_original_v1/masks_left_video"
    rgb_paths = sorted(rgb_root.glob("*.png"), key=lambda path: int(source_id(path.name)))
    mask_by_name = {path.name: sam_root / f"{index:06d}.png" for index, path in enumerate(rgb_paths)}
    registered = {image.name: image for image in rgb_images.values()}
    rows = {label: {axis: [] for axis in range(3)} for label in labels}

    for order, rgb_path in enumerate(rgb_paths):
        image = registered.get(rgb_path.name)
        depth_path = depth_root / rgb_path.name
        if image is None or not depth_path.is_file():
            continue
        camera = rgb_cameras[image.camera_id]
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        sam = cv2.imread(str(mask_by_name[rgb_path.name]), cv2.IMREAD_GRAYSCALE)
        if depth is None or sam is None:
            continue
        geometry_ids, t_hit = cast_frame(scene, image, camera)
        for label in labels:
            geometry_id = label_to_geometry[label]
            for axis_index in range(3):
                if label in central_ranges:
                    measured = section_measurement(
                        depth, sam > 0, geometry_ids, t_hit, geometry_id, image, camera,
                        axes[label], axis_index, metric_per_original_unit, central_ranges[label])
                else:
                    measured = frame_measurement(
                        depth, sam > 0, geometry_ids, t_hit, geometry_id, image, camera,
                        axes[label][axis_index], metric_per_original_unit, axis_index)
                if measured:
                    measured.update(frame=rgb_path.name, order=order, camera=camera)
                    rows[label][axis_index].append(measured)
        print(f"[{order + 1:03d}/{len(rgb_paths)}] {rgb_path.name}", flush=True)

    axis_names = ("long", "middle", "short")
    report = {
        "source_db3": str(args.db3),
        "units": "mm",
        "method": "best COLMAP frame per label/axis, then median of the top-scoring original DB3 frames in its local temporal window",
        "original_db3_frames_per_axis": args.frames_per_axis,
        "labels": {},
        "alignment": {**align_report,
                      "benchmark_to_original_scale": float(align_scale),
                      "meters_per_original_colmap_unit": float(metric_per_original_unit)},
    }
    csv_rows, candidate_rows, summary_lines = [], [], []
    raw = open_raw_rgbd(args.db3, next(iter(rgb_cameras.values())))
    radius = args.raw_radius
    for label in labels:
        label_report = {"rule": rules.get(label, {"mode": "full-part robust PCA extents"})}
        panels = []
        for axis_index, axis_name in enumerate(axis_names):
            center = choose_center(rows[label][axis_index], axis_index)
            if center is None:
                label_report[axis_name] = {
                    "status": "insufficient_valid_frames", "valid_frames": len(rows[label][axis_index])}
                summary_lines.append(f"{label} {axis_name}: FAILED ({len(rows[label][axis_index])} valid)")
                continue
            center_frame = center["frame"]
            center_image = registered[center_frame]
            camera = rgb_cameras[center_image.camera_id]
            center_rgb = cv2.imread(str(rgb_root / center_frame))
            center_sam = cv2.imread(str(mask_by_name[center_frame]), cv2.IMREAD_GRAYSCALE)
            center_geometry, center_t_hit = cast_frame(scene, center_image, camera)
            center_mask = (center_geometry == label_to_geometry[label]).astype(np.uint8)
            center_t = np.where(np.isfinite(center_t_hit), center_t_hit, 0).astype(np.float32)
            center_source = int(source_id(center_frame))
            candidates = []
            raw_start = max(0, center_source - radius)
            raw_stop = min(len(raw["color_rows"]), center_source + radius + 1)
            for raw_index in range(raw_start, raw_stop):
                rgb, depth, raw_meta = load_raw_frame(raw, raw_index)
                warp, ecc_score = estimate_raw_warp(center_rgb, rgb)
                size = (camera.width, camera.height)
                label_mask = cv2.warpAffine(center_mask, warp, size, flags=cv2.INTER_NEAREST) > 0
                sam = cv2.warpAffine(center_sam, warp, size, flags=cv2.INTER_NEAREST) > 0
                warped_t = cv2.warpAffine(center_t, warp, size, flags=cv2.INTER_LINEAR)
                geometry = np.full(label_mask.shape, np.iinfo(np.uint32).max, np.uint32)
                geometry[label_mask] = 0
                if label in central_ranges:
                    measured = section_measurement(
                        depth, sam, geometry, warped_t, 0, center_image, camera,
                        axes[label], axis_index, metric_per_original_unit, central_ranges[label])
                else:
                    measured = frame_measurement(
                        depth, sam, geometry, warped_t, 0, center_image, camera,
                        axes[label][axis_index], metric_per_original_unit, axis_index)
                if measured:
                    measured.update(
                        frame=f"raw_{raw_meta['source_index']:06d}.png", order=raw_meta["source_index"],
                        raw_offset=raw_meta["source_index"] - center_source,
                        center_colmap_frame=center_frame, ecc_score=ecc_score,
                        sync_dt_ms=raw_meta["sync_dt_ms"], camera=camera,
                        _rgb=rgb, _depth=depth, _geometry_ids=geometry)
                    candidates.append(measured)
            for rank, row in enumerate(
                    sorted(candidates, key=lambda item: item["score"], reverse=True), 1):
                candidate_rows.append((
                    label, axis_name, row["frame"], row["order"], row["raw_offset"],
                    row["value_mm"], row["score"], row["depth_coverage"],
                    row["mesh_depth_agreement"], row["viewability"], row["ecc_score"], rank))
            selected = top_measurements(candidates, args.frames_per_axis)
            if len(selected) != args.frames_per_axis:
                label_report[axis_name] = {
                    "status": "insufficient_raw_neighbours", "center_colmap_frame": center_frame,
                    "valid_frames": len(selected)}
                summary_lines.append(
                    f"{label} {axis_name}: FAILED ({len(selected)}/{args.frames_per_axis} raw)")
                continue
            values = np.asarray([row["value_mm"] for row in selected])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            representative = best_measurement(selected)
            label_report[axis_name] = {
                "status": "ok", "median_mm": median, "mad_mm": mad,
                "center_colmap_frame": center_frame,
                "candidate_count": len(candidates),
                "representative_frame": representative["frame"],
                "frames": [serializable_row(row) for row in selected],
            }
            representative["_reported_mm"] = median
            representative["_reported_count"] = len(selected)
            csv_rows.append((label, axis_name, median, mad, representative["frame"],
                             " ".join(row["frame"] for row in selected)))
            summary_lines.append(
                f"{label:12s} {axis_name:6s}: {median:7.2f} mm  MAD {mad:5.2f}")
            frame = representative["frame"]
            overlay = draw_overlay(
                representative["_rgb"], representative["_geometry_ids"], 0,
                representative, label, axis_name)
            out = args.output / f"{label}_{axis_name}_{Path(frame).stem}_rgb.jpg"
            cv2.imwrite(str(out), overlay)
            depth_overlay = draw_overlay(
                depth_view(representative["_depth"]), representative["_geometry_ids"], 0,
                representative, label, axis_name)
            cv2.imwrite(str(args.output / f"{label}_{axis_name}_{Path(frame).stem}_depth.jpg"),
                        depth_overlay)
            panels.append(np.hstack((cv2.resize(overlay, (318, 178)),
                                     cv2.resize(depth_overlay, (318, 178)))))
        report["labels"][label] = label_report
        if panels:
            cv2.imwrite(str(args.output / f"{label}_selected_per_axis.jpg"), np.vstack(panels))
    raw["database"].close()

    (args.output / "part_dimensions_multiview_pca.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "part_dimensions_multiview_pca.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "axis", "median_mm", "mad_mm", "representative_frame",
                         "original_db3_frames"))
        writer.writerows(csv_rows)
    with (args.output / "part_dimensions_raw_candidates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "axis", "frame", "source_index", "raw_offset", "value_mm",
                         "score", "depth_coverage", "mesh_depth_agreement", "viewability",
                         "ecc_score", "quality_rank"))
        writer.writerows(candidate_rows)
    summary = np.full((max(260, 55 + 34 * len(summary_lines)), 950, 3), 255, np.uint8)
    cv2.putText(summary, "ORIGINAL DEPTH MULTI-FRAME MEDIAN PER PCA AXIS", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 2, cv2.LINE_AA)
    for index, line in enumerate(summary_lines):
        cv2.putText(summary, line, (24, 78 + 34 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .67, (0, 0, 180), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "part_dimensions_summary.jpg"), summary)
    print(json.dumps(report["labels"], ensure_ascii=False, indent=2))
    return final_review(args.output) if args.review_final else 0


if __name__ == "__main__":
    raise SystemExit(main())
