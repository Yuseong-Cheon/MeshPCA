#!/usr/bin/env python3
"""Track laptop parts and report the object-axis opening angle."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from axis_geometry import laptop_opening_angle, pca_axis


PARTS = ("base", "moving_link")
COLORS = {"base": (255, 100, 20), "moving_link": (30, 40, 240)}


def configure_camera(config, serial):
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


def capture_and_mask(args):
    for path in (args.init_rgb, args.init_depth, args.intrinsics):
        path.parent.mkdir(parents=True, exist_ok=True)
    pipeline, config = rs.pipeline(), rs.config()
    configure_camera(config, args.serial)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            frames = align.process(pipeline.wait_for_frames(10000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        cv2.imwrite(str(args.init_rgb), np.asanyarray(color.get_data()))
        np.save(args.init_depth,
                np.asanyarray(depth.get_data()).astype(np.float32)
                * depth.get_units())
        args.intrinsics.write_text(json.dumps({
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
        }, indent=2) + "\n")
    finally:
        pipeline.stop()
    subprocess.run([
        str(args.sam3_python), str(Path(__file__).with_name("make_laptop_masks.py")),
        str(args.init_rgb), str(args.masks), "--sam3-root", str(args.sam3_root),
    ], check=True)


def write_status(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


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

    base_mesh = args.mesh_dir / "base_metric_watertight.ply"
    if not base_mesh.exists():
        base_mesh = args.mesh_dir / "base_metric_watertight (1).ply"
    mesh_files = {
        "base": base_mesh,
        "moving_link": args.mesh_dir / "moving_link_metric_watertight.ply",
    }
    meshes = {part: trimesh.load_mesh(mesh_files[part], process=False)
              for part in PARTS}
    masks = {part: cv2.imread(str(args.masks / f"{part}.png"), 0) > 0
             for part in PARTS}
    if any((masks[part] & (init_depth > 0)).sum() < 500 for part in PARTS):
        raise RuntimeError("not enough valid masked depth for laptop registration")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    trackers, poses = {}, {}
    for part in PARTS:
        mesh = meshes[part]
        tracker = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
            debug_dir=str(args.output / f"{part}_debug"), debug=0,
        )
        poses[part] = tracker.register(
            intrinsics, init_rgb, init_depth, masks[part], iteration=5)
        trackers[part] = tracker
        print(f"[FoundationPose] {part}=registered")

    normals = {part: pca_axis(meshes[part], largest=False) for part in PARTS}
    boxes = {}
    for part in PARTS:
        to_origin, extents = trimesh.bounds.oriented_bounds(meshes[part])
        boxes[part] = (np.linalg.inv(to_origin),
                       np.array((-extents / 2, extents / 2)))

    pipeline, config = rs.pipeline(), rs.config()
    configure_camera(config, args.serial)
    pipeline.start(config)
    align, frame, last, view = rs.align(rs.stream.color), 0, {}, init_bgr
    try:
        for _ in range(20):
            align.process(pipeline.wait_for_frames(10000))
        while not args.max_frames or frame < args.max_frames:
            frames = align.process(pipeline.wait_for_frames(10000))
            bgr = np.asanyarray(frames.get_color_frame().get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_frame = frames.get_depth_frame()
            depth = (np.asanyarray(depth_frame.get_data()).astype(np.float32)
                     * depth_frame.get_units())
            depth[(depth < 0.15) | (depth > 3.0)] = 0
            poses = {part: trackers[part].track_one(
                rgb, depth, intrinsics, iteration=1) for part in PARTS}
            angle = laptop_opening_angle(poses, normals)
            view = bgr.copy()
            for part in PARTS:
                transform, bounds = boxes[part]
                view = draw_posed_3d_box(
                    intrinsics, view, poses[part] @ transform, bounds,
                    COLORS[part], 2)
            cv2.rectangle(view, (12, 12), (430, 72), (0, 0, 0), -1)
            cv2.putText(view, f"laptop opening: {angle:7.2f} deg", (28, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            last = {"frame": frame, "opening_angle_deg": angle}
            write_status(args.output / "latest.json", last)
            if not args.no_window:
                cv2.imshow("FoundationPose Laptop Opening | Q: quit", view)
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
