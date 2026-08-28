# MeshPCA + Articulated Mesh Pipeline

MeshPCA measures the three principal dimensions of any number of separated mesh labels from the original RealSense RGB-D recording.

The separated meshes provide only label masks and PCA-axis directions. Metric lengths are computed from the original aligned Depth frames. For every label and axis, MeshPCA selects one suitable COLMAP frame, searches the nearby raw DB3 frames, and reports the median of the best seven measurements.

`articulation/` adds the upstream part split and joint approval pipeline, plus optional measured-dimension resize and URDF export. Its output naming already matches MeshPCA, so no adapter is required.

```text
watertight PLY
  -> articulation/scripts/rora_prior_split_ply.py
  -> <output>/metric_parts/*_metric_watertight.ply
  -> meshpca.py --labels <output>/metric_parts
  -> optional articulation/scripts/resize_articulated_parts.py
  -> URDF asset
```

See [`articulation/README.md`](articulation/README.md) for part split and joint usage.

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

The output contains JSON and CSV measurements, RGB/Depth overlays for the seven selected raw frames per axis, per-label contact sheets, and a summary image.

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
