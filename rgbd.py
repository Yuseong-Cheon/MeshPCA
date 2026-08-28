"""Minimal ROS2 RealSense DB3 decoding and native-depth alignment."""

import struct

import cv2
import numpy as np


def decode_image(blob):
    data = memoryview(blob)
    offset = 12
    length = struct.unpack_from("<I", data, offset)[0]
    offset = (offset + 4 + length + 3) // 4 * 4
    height, width = struct.unpack_from("<II", data, offset)
    offset += 8
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    encoding = bytes(data[offset:offset + length - 1]).decode()
    offset += length + 1
    offset = (offset + 3) // 4 * 4
    step = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    raw = np.frombuffer(data[offset:offset + length], np.uint8).copy()
    if encoding == "mono16":
        return raw.view("<u2").reshape(height, step // 2)[:, :width]
    if encoding in {"8UC1", "mono8"}:
        return raw.reshape(height, step)[:, :width]
    channels = step // width
    image = raw.reshape(height, step)[:, :width * channels].reshape(height, width, channels)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image


def decode_string(blob):
    length = struct.unpack_from("<I", blob, 4)[0]
    return bytes(blob[8:8 + length - 1]).decode()


def parse_camera_info(text):
    values = dict(field.split("=", 1) for field in text.split(";") if "=" in field)
    return {
        "width": int(values["width"]), "height": int(values["height"]),
        "fx": float(values["fx"]), "fy": float(values["fy"]),
        "ppx": float(values["ppx"]), "ppy": float(values["ppy"]),
        "model": values.get("model", ""),
        "coeffs": [float(value) for value in values.get("coeffs", "").split(",") if value],
    }


def parse_transform(text):
    values = dict(field.split("=", 1) for field in text.split(";") if "=" in field)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [float(value) for value in values["rotation"].split(",")]).reshape(3, 3)
    transform[:3, 3] = [float(value) for value in values["translation"].split(",")]
    return transform


def align_depth(depth_mm, depth_cal, color_cal, color_from_depth):
    valid_y, valid_x = np.nonzero(depth_mm)
    z = depth_mm[valid_y, valid_x].astype(np.float64) / 1000
    xyz = np.vstack(((valid_x - depth_cal["ppx"]) * z / depth_cal["fx"],
                     (valid_y - depth_cal["ppy"]) * z / depth_cal["fy"], z))
    xyz = color_from_depth[:3, :3] @ xyz + color_from_depth[:3, 3:4]
    keep = xyz[2] > 0
    xyz = xyz[:, keep]
    u = np.rint(color_cal["fx"] * xyz[0] / xyz[2] + color_cal["ppx"]).astype(int)
    v = np.rint(color_cal["fy"] * xyz[1] / xyz[2] + color_cal["ppy"]).astype(int)
    keep = ((u >= 0) & (u < color_cal["width"]) &
            (v >= 0) & (v < color_cal["height"]))
    flat = np.full(color_cal["width"] * color_cal["height"], np.inf, np.float32)
    np.minimum.at(flat, v[keep] * color_cal["width"] + u[keep], xyz[2, keep].astype(np.float32))
    aligned = flat.reshape(color_cal["height"], color_cal["width"])
    aligned[~np.isfinite(aligned)] = 0
    return aligned
