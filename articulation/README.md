# Part split and joint pipeline

This directory contains only the articulated-mesh stage that surrounds MeshPCA.

## Choose one input path

- One connected watertight body: `scripts/rora_prior_split_ply.py`
- Already-separated watertight bodies in one PLY: `scripts/articulate_disconnected_ply.py`

Inputs are triangle PLY files in meters. Connected inputs also need a config copied from
`configs/connected_tree_example.json`. A single static mesh cannot determine hidden joint geometry
or motion limits, so the joint axis, direction, and limits remain human-approved.

## Connected mesh

```bash
python articulation/scripts/rora_prior_split_ply.py INPUT.ply \
  --config OBJECT.json --output output/object \
  --reselect --articulated --review-joints --review-cuts \
  --robot-name object
```

The main result is `output/object/metric_parts/*_metric_watertight.ply`. Measure those parts directly:

```bash
python meshpca.py \
  --db3 RECORDING.db3 --project RGBD_PROJECT --mesh-colmap MESH_COLMAP \
  --mesh-scale METERS_PER_UNIT --labels output/object/metric_parts \
  --output output/object/measurements
```

## Disconnected mesh

```bash
python articulation/scripts/articulate_disconnected_ply.py INPUT.ply \
  --output output/object --review-joints --robot-name object
```

## Optional resize

Copy `configs/resize_example.json` and enter the measured dimensions from MeshPCA. Confirm the
part-local length, width, and thickness axes instead of assuming world XYZ.

```bash
python articulation/scripts/resize_articulated_parts.py \
  --parts output/object/metric_parts \
  --config RESIZE.json --joints output/object/joint_selection.json \
  --output output/object/resized --dimension-profile A --robot-name object_resized
```

Add `--joint-anchored-affine` only for disconnected inputs without an exact shared interface.

## Checks

```bash
python -m py_compile articulation/scripts/*.py articulation/rora_patch/functions/lib/*.py
python articulation/scripts/articulate_disconnected_ply.py --self-test
python articulation/scripts/static_rora_metric_parts.py --self-test
python articulation/scripts/resize_articulated_parts.py --self-check
python articulation/scripts/test_final_part_split_pipeline.py
python articulation/scripts/test_compile_hitl_articulated_asset.py
python meshpca.py --self-check
```

GUI and OS setup is documented in `ENVIRONMENT.md`.
