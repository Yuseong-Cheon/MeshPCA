#!/usr/bin/env python3
"""Convert an EasyHeC Tc_c2b matrix to MeshPCA's hand-eye JSON format."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_matrix(path, shape):
    matrix = np.loadtxt(path, dtype=float)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{path} must contain a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def convert(transform_path, intrinsics_path, camera_model, serial):
    camera_from_base = load_matrix(transform_path, (4, 4))
    if not np.allclose(camera_from_base[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Tc_c2b is not a homogeneous transform")
    rotation = camera_from_base[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("Tc_c2b rotation is not orthonormal")
    intrinsics = load_matrix(intrinsics_path, (3, 3))
    return {
        "status": "valid",
        "method": "EasyHeC differentiable silhouette alignment",
        "camera": {"model": camera_model, "serial": serial},
        "camera_from_base": camera_from_base.tolist(),
        "base_from_camera": np.linalg.inv(camera_from_base).tolist(),
        "intrinsics": {
            "fx": float(intrinsics[0, 0]), "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]), "cy": float(intrinsics[1, 2]),
        },
        "source": str(transform_path.resolve()),
    }


def self_test():
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    assert np.allclose(np.linalg.inv(transform) @ transform, np.eye(4))
    print("self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", type=Path,
                        help="EasyHeC Tc_c2b.txt (camera-from-base)")
    parser.add_argument("--intrinsics", type=Path, help="3x3 K.txt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-model", default="Intel RealSense D456")
    parser.add_argument("--serial", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.transform, args.intrinsics, args.output)):
        parser.error("--transform, --intrinsics, and --output are required")
    result = convert(args.transform, args.intrinsics, args.camera_model, args.serial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
