#!/usr/bin/env python3
"""Track desk-lamp parts and report two object-axis joint angles."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from axis_geometry import axis_geometry, lamp_joint_angles, pca_axis


PARTS = ("base", "support", "head")
COLORS = {"base": (255, 100, 20), "support": (40, 210, 40),
          "head": (30, 40, 240)}
ALIGNMENT_TOP_CANDIDATES = 32


def observed_axis(mask, depth, intrinsics):
    y, x = np.where(mask & (depth > 0))
    z = depth[y, x]
    points = np.column_stack((
        (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z,
    ))
    _, vectors = np.linalg.eigh(np.cov(points.T))
    return vectors[:, -1]


def select_axis_aligned_pose(tracker, model_axis, image_axis):
    candidates = tracker.poses[:ALIGNMENT_TOP_CANDIDATES]
    outputs = (candidates @ tracker.get_tf_to_centered_mesh()).detach().cpu().numpy()
    directions = np.einsum("nij,j->ni", outputs[:, :3, :3], model_axis)
    index = int(np.argmax(np.abs(directions @ image_axis)))
    tracker.pose_last = candidates[index]
    return outputs[index]


def projected_box_mask(pose, to_origin, extents, intrinsics, shape):
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ])
    corners = np.column_stack((signs * extents / 2, np.ones(8)))
    camera = (pose @ np.linalg.inv(to_origin) @ corners.T).T[:, :3]
    pixels = ((camera[:, :2] / camera[:, 2, None]) @ intrinsics[:2, :2].T
              + intrinsics[:2, 2])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels).astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((31, 31), np.uint8)).astype(bool)


def write_status(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def configure_camera(config, serial, width, height):
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)


def capture_and_mask(args):
    for path in (args.init_rgb, args.init_depth, args.intrinsics):
        path.parent.mkdir(parents=True, exist_ok=True)

    pipeline, config = rs.pipeline(), rs.config()
    configure_camera(config, args.serial, 1280, 720)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            frames = align.process(pipeline.wait_for_frames(10000))
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        cv2.imwrite(str(args.init_rgb), np.asanyarray(color_frame.get_data()))
        np.save(args.init_depth,
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * depth_frame.get_units())
        args.intrinsics.write_text(json.dumps({
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
        }, indent=2) + "\n")
    finally:
        pipeline.stop()

    subprocess.run([
        str(args.sam3_python), str(Path(__file__).with_name("make_desk_lamp_masks.py")),
        str(args.init_rgb), str(args.masks), "--sam3-root", str(args.sam3_root),
        "--manual",
    ], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, default=Path.cwd())
    parser.add_argument("--sam3-root", type=Path)
    parser.add_argument("--sam3-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--init-rgb", type=Path, required=True)
    parser.add_argument("--init-depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--reuse-init", action="store_true")
    args = parser.parse_args()
    if not args.reuse_init and args.sam3_root is None:
        parser.error("--sam3-root is required unless --reuse-init is used")
    return args


def main():
    global cv2, rs, trimesh
    args = parse_args()

    import cv2
    import pyrealsense2 as rs
    import trimesh

    foundationpose_root = args.foundationpose_root.resolve()
    sys.path[:0] = [str(foundationpose_root),
                    str(foundationpose_root / "mycpp/build")]
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr
    from Utils import draw_posed_3d_box

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.reuse_init:
        capture_and_mask(args)

    intr = json.loads(args.intrinsics.read_text())
    intrinsics = np.float32([
        [intr["fx"], 0, intr["cx"]],
        [0, intr["fy"], intr["cy"]],
        [0, 0, 1],
    ])
    init_bgr = cv2.imread(str(args.init_rgb))
    if init_bgr is None:
        raise RuntimeError(f"cannot read {args.init_rgb}")
    init_rgb = cv2.cvtColor(init_bgr, cv2.COLOR_BGR2RGB)
    init_depth = np.load(args.init_depth).astype(np.float32)
    init_depth[(init_depth < 0.15) | (init_depth > 3.0)] = 0

    meshes = {
        part: trimesh.load_mesh(
            args.mesh_dir / f"{part}_metric_watertight.ply", process=False)
        for part in PARTS
    }
    boxes = {part: trimesh.bounds.oriented_bounds(meshes[part])
             for part in PARTS}
    masks = {
        part: cv2.imread(str(args.masks / f"{part}.png"),
                         cv2.IMREAD_GRAYSCALE) > 0
        for part in PARTS
    }
    masks["head"] &= ~masks["support"]
    masks["base"] &= ~(masks["support"] | masks["head"])
    if any((masks[part] & (init_depth > 0)).sum() <= 100 for part in PARTS):
        raise RuntimeError("not enough valid masked depth for lamp registration")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    trackers, poses = {}, {}
    for part in PARTS:
        mesh = meshes[part]
        tracker = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
            debug=0, debug_dir=str(args.output / f"{part}_debug"),
        )
        poses[part] = tracker.register(
            intrinsics, init_rgb, init_depth, masks[part], iteration=5)
        if part == "support":
            poses[part] = select_axis_aligned_pose(
                tracker, pca_axis(mesh, largest=True),
                observed_axis(masks[part], init_depth, intrinsics),
            )
        trackers[part] = tracker
        print(f"[FoundationPose] {part}=registered")

    axes = {
        "base_inward": pca_axis(meshes["base"], largest=True),
        "support": pca_axis(meshes["support"], largest=True),
        "head": pca_axis(meshes["head"], largest=True),
    }
    for part in ("support", "head"):
        axes[f"{part}_axis_center"], axes[f"{part}_half"] = axis_geometry(
            meshes[part], axes[part])
    centers = {
        part: poses[part][:3, :3] @ meshes[part].centroid + poses[part][:3, 3]
        for part in PARTS
    }
    base_normal = poses["base"][:3, :3] @ pca_axis(
        meshes["base"], largest=False)
    inward_hint = centers["base"] - centers["support"]
    inward_hint -= np.dot(inward_hint, base_normal) * base_normal
    if np.dot(poses["base"][:3, :3] @ axes["base_inward"], inward_hint) < 0:
        axes["base_inward"] *= -1

    pipeline, config = rs.pipeline(), rs.config()
    configure_camera(config, args.serial, 1280, 720)
    pipeline.start(config)
    align, last, view = rs.align(rs.stream.color), {}, init_bgr
    try:
        for _ in range(20):
            align.process(pipeline.wait_for_frames(10000))
        frame = 0
        while not args.max_frames or frame < args.max_frames:
            frames = align.process(pipeline.wait_for_frames(10000))
            bgr = np.asanyarray(frames.get_color_frame().get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_frame = frames.get_depth_frame()
            depth = (np.asanyarray(depth_frame.get_data()).astype(np.float32)
                     * depth_frame.get_units())
            depth[(depth < 0.15) | (depth > 3.0)] = 0
            tracked = {}
            for part in PARTS:
                part_depth = depth.copy()
                part_depth[~projected_box_mask(
                    poses[part], *boxes[part], intrinsics, bgr.shape)] = 0
                tracked[part] = trackers[part].track_one(
                    rgb, part_depth, intrinsics, iteration=1)
            poses = tracked
            angles = lamp_joint_angles(poses, axes)
            view = bgr.copy()
            for part in PARTS:
                to_origin, extents = boxes[part]
                view = draw_posed_3d_box(
                    intrinsics, view,
                    poses[part] @ np.linalg.inv(to_origin),
                    np.array((-extents / 2, extents / 2)), COLORS[part], 2,
                )
            cv2.rectangle(view, (12, 12), (510, 112), (0, 0, 0), -1)
            cv2.putText(view, f"base-support: {angles[0]:7.2f} deg",
                        (28, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (0, 255, 255), 2)
            cv2.putText(view, f"support-head: {angles[1]:7.2f} deg",
                        (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (0, 255, 255), 2)
            last = {
                "status": "tracking", "timestamp_s": time.time(),
                "frame": frame, "base_support_deg": angles[0],
                "support_head_deg": angles[1],
            }
            if args.calibration_path:
                last["calibration_path"] = str(args.calibration_path.resolve())
            write_status(args.output / "latest.json", last)
            if not args.no_window:
                cv2.imshow("FoundationPose Lamp Angles | Q: quit", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            frame += 1
        cv2.imwrite(str(args.output / "latest.png"), view)
        write_status(args.output / "latest.json", last)
        print(json.dumps(last))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
