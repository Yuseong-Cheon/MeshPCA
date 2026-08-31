# MeshPCA + Articulated Mesh Pipeline

MeshPCA measures the three principal dimensions of any number of separated mesh labels from the original RealSense RGB-D recording.

The separated meshes provide only label masks and PCA-axis directions. Metric lengths are computed from the original aligned Depth frames. For every label and axis, MeshPCA selects one suitable COLMAP frame, searches the nearby original DB3 frames, and uses the median of the three highest-quality Depth measurements.

`articulation/` adds the upstream part split and joint approval pipeline, plus optional measured-dimension resize and URDF export. Its output naming already matches MeshPCA, so no adapter is required.

`foundationpose/` is an optional object-axis angle feature for the included desk-lamp and laptop meshes. It converts per-part FoundationPose estimates into lamp joint angles or a laptop opening angle without changing the volume-estimation pipeline. See [`foundationpose/README.md`](foundationpose/README.md).

```text
watertight PLY
  -> articulation/scripts/rora_prior_split_ply.py
  -> <output>/metric_parts/*_metric_watertight.ply
  -> meshpca.py --labels <output>/metric_parts
  -> optional articulation/scripts/resize_articulated_parts.py
  -> URDF asset
```

See [`articulation/README.md`](articulation/README.md) for part split and joint usage.

When MeshPCA results must be expressed in an RB5 robot-base frame, first run the optional
camera-to-robot and table-plane calibration. See [`CALIBRATION.md`](CALIBRATION.md) for the
RealSense D456 + EasyHeC/SAM feature, transform convention, and quality checks. Metric dimension
measurement itself does not require this calibration.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inputs

The RGB-D project must contain:

```text
project/
├── colmap_v4_final/
│   ├── images/
│   └── sparse/0/
├── depth/native_aligned_colmap_v1/
│   ├── alignment.json
│   └── by_source_name/
└── masks/sam3_original_v1/masks_left_video/
```

The label directory can contain any number of meshes named:

```text
<label>_metric_watertight.ply
```

`--mesh-scale` is the metric scale used when the mesh-generation COLMAP reconstruction was converted to meters.

## Usage

```bash
python meshpca.py \
  --db3 /path/to/recording.db3 \
  --project /path/to/rgbd_project \
  --mesh-colmap /path/to/mesh_colmap_dataset \
  --mesh-scale 0.17473159013427114 \
  --labels /path/to/separated_meshes \
  --rules label_rules.example.json \
  --output /tmp/meshpca_output
```

The output contains JSON and CSV measurements, `part_dimensions_raw_candidates.csv` with every valid nearby original DB3 measurement, one representative RGB/Depth frame per axis, per-label review sheets, and a summary image. The median is the reported dimension; the highest-scoring frame is used only for visualization and HITL review. Use `--frames-per-axis 3`, `5`, or `7` for an ablation while keeping an odd median sample count.

## Optional final HITL review

Add `--review-final` to keep frame selection and measurement automatic while requiring a human to approve or reject the final result:

```bash
python meshpca.py \
  --db3 /path/to/recording.db3 \
  --project /path/to/rgbd_project \
  --mesh-colmap /path/to/mesh_colmap_dataset \
  --mesh-scale 0.17473159013427114 \
  --labels /path/to/separated_meshes \
  --output /tmp/meshpca_output \
  --review-final
```

Use `N/P` to browse part sheets, `A` to approve, and `R` to reject. Rejection preserves all measurements and records `HITL_REJECTED` in `final_review.json`; it does not select another frame. A saved result can be reviewed again without recomputing Depth:

```bash
python meshpca.py --review-existing /tmp/meshpca_output
```

## Per-label rules

Without a rule, all three axes use the full label. A label can keep its full long axis while measuring its middle and short axes only on a central longitudinal section:

```json
{
  "support": {
    "central_longitudinal_range": [0.2, 0.8]
  }
}
```

Labels are discovered from mesh filenames, so adding another mesh requires no code change.

## Check

```bash
python meshpca.py --self-check
```
