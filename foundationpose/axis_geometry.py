"""Object-axis geometry used by the lamp and laptop trackers."""

import numpy as np


def pca_axis(mesh, largest):
    _, vectors = np.linalg.eigh(np.cov(mesh.vertices.T))
    return vectors[:, -1 if largest else 0]


def axis_geometry(mesh, axis):
    projection = mesh.vertices @ axis
    midpoint = (projection.min() + projection.max()) / 2
    center = mesh.centroid + axis * (midpoint - np.dot(mesh.centroid, axis))
    return center, (projection.max() - projection.min()) / 2


def vector_angle(first, second):
    cosine = float(np.dot(first, second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def lamp_joint_angles(poses, axes):
    base_inward = poses["base"][:3, :3] @ axes["base_inward"]
    endpoints = {}
    for part in ("support", "head"):
        direction = poses[part][:3, :3] @ axes[part]
        center = (poses[part][:3, :3] @ axes[f"{part}_axis_center"]
                  + poses[part][:3, 3])
        endpoints[part] = np.stack((
            center - direction * axes[f"{part}_half"],
            center + direction * axes[f"{part}_half"],
        ))
    distances = np.linalg.norm(
        endpoints["support"][:, None] - endpoints["head"][None], axis=2)
    support_joint, head_joint = np.unravel_index(
        np.argmin(distances), distances.shape)
    support_to_base = (endpoints["support"][1 - support_joint]
                       - endpoints["support"][support_joint])
    head_outward = (endpoints["head"][1 - head_joint]
                    - endpoints["head"][head_joint])
    support_forward = -support_to_base
    support_to_base /= np.linalg.norm(support_to_base)
    head_outward /= np.linalg.norm(head_outward)
    support_forward /= np.linalg.norm(support_forward)
    return (vector_angle(base_inward, support_forward),
            vector_angle(support_to_base, head_outward))


def laptop_opening_angle(poses, normals):
    base = poses["base"][:3, :3] @ normals["base"]
    moving = poses["moving_link"][:3, :3] @ normals["moving_link"]
    plane_angle = np.degrees(
        np.arccos(np.clip(abs(np.dot(base, moving)), 0.0, 1.0)))
    return float(180.0 - plane_angle)
