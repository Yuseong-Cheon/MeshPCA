"""Minimal RORA GUI geometry helpers bundled for standalone delivery."""

import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=float).ravel()
    length = np.linalg.norm(vector)
    return vector * 0.0 if length < 1e-12 else vector / length


def rotmat(axis, theta):
    x, y, z = normalize(axis)
    cosine, sine = np.cos(theta), np.sin(theta)
    complement = 1.0 - cosine
    return np.array([
        [cosine + x*x*complement, x*y*complement - z*sine, x*z*complement + y*sine],
        [y*x*complement + z*sine, cosine + y*y*complement, y*z*complement - x*sine],
        [z*x*complement - y*sine, z*y*complement + x*sine, cosine + z*z*complement],
    ], dtype=float)
