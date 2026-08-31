# FoundationPose object-axis angle feature

This optional feature does not replace or modify FoundationPose. It uses each
part pose estimated by FoundationPose together with the bundled metric-mesh
axes to report articulated-object angles.

- Desk lamp: `base`, `support`, and `head` long axes produce the two joint
  angles. For the symmetric support, the registration candidate closest to the
  observed RGB-D PCA axis is selected.
- Laptop: the smallest PCA axes are treated as the base and lid plane normals,
  producing the opening angle.

The reported angles are unsigned geometry angles. They are object-specific,
not general FoundationPose outputs.

## Prerequisites

Prepare an NVIDIA FoundationPose checkout and a SAM3 checkout separately. Run
the trackers in the FoundationPose environment; `--sam3-python` may point to a
separate SAM3 environment. FoundationPose and SAM3 are intentionally not added
to MeshPCA's main requirements.

Tested local sources:

- <https://github.com/Yuseong-Cheon/Foundation_pose_edit>
- <https://github.com/facebookresearch/sam3>

## Desk lamp

From the MeshPCA repository root:

```bash
conda activate bundlesdf
python foundationpose/run_desk_lamp_live.py \
  --foundationpose-root /path/to/FoundationPose \
  --sam3-root /path/to/sam3 \
  --sam3-python /path/to/sam3/env/bin/python \
  --mesh-dir foundationpose/assets/lamp \
  --init-rgb /tmp/lamp_live_rgb.png \
  --init-depth /tmp/lamp_live_depth_m.npy \
  --intrinsics /tmp/lamp_live_intrinsics.json \
  --masks /tmp/lamp_sam3_masks \
  --output /tmp/lamp_foundationpose_live
```

The output JSON contains `base_support_deg` and `support_head_deg`.

## Laptop

```bash
python foundationpose/run_laptop_live.py \
  --foundationpose-root /path/to/FoundationPose \
  --sam3-root /path/to/sam3 \
  --sam3-python /path/to/sam3/env/bin/python \
  --mesh-dir foundationpose/assets/laptop \
  --init-rgb /tmp/laptop_live_rgb.png \
  --init-depth /tmp/laptop_live_depth_m.npy \
  --intrinsics /tmp/laptop_live_intrinsics.json \
  --masks /tmp/laptop_sam3_masks \
  --output /tmp/laptop_foundationpose_live
```

The output JSON contains `opening_angle_deg`. Add `--serial CAMERA_SERIAL` when
more than one RealSense camera is connected. Add `--reuse-init` to reuse the
saved RGB, Depth, intrinsics, and masks without running SAM3 again.

## Hardware-free check

```bash
python foundationpose/test_axis_geometry.py
```
