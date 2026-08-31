#!/usr/bin/env python3
"""Small hardware-free check for the object-axis angle definitions."""

import numpy as np

from axis_geometry import lamp_joint_angles, laptop_opening_angle


def main():
    identity = np.eye(4)
    poses = {name: identity.copy() for name in ("base", "support", "head")}
    axes = {
        "base_inward": np.array([1.0, 0.0, 0.0]),
        "support": np.array([1.0, 0.0, 0.0]),
        "head": np.array([0.0, 1.0, 0.0]),
        "support_axis_center": np.zeros(3),
        "head_axis_center": np.array([1.0, 0.0, 0.0]),
        "support_half": 1.0,
        "head_half": 1.0,
    }
    assert np.allclose(lamp_joint_angles(poses, axes), (0.0, 90.0))

    laptop_poses = {"base": identity.copy(), "moving_link": identity.copy()}
    normals = {"base": np.array([0.0, 0.0, 1.0]),
               "moving_link": np.array([0.0, 0.0, 1.0])}
    assert np.isclose(laptop_opening_angle(laptop_poses, normals), 180.0)
    laptop_poses["moving_link"][:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    assert np.isclose(laptop_opening_angle(laptop_poses, normals), 90.0)
    print("FoundationPose axis geometry check passed")


if __name__ == "__main__":
    main()
