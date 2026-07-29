# Grounding DINO feature extraction

This optional tool extracts the image-level and object-level Grounding DINO
features consumed by SegDINO3D. Grounding DINO provides multi-scale image
features, object queries, boxes, and confidence scores. Segment Anything
(SAM) converts the boxes into masks, which are projected with ScanNet depth
and camera parameters to obtain 3D query centers.

The extractor supports both ScanNet20 and ScanNet200 prompts and produces the
same file layout expected by the SegDINO3D dataset classes.

## Installation

Complete the main [SegDINO3D installation](../../installation.md) first. To
isolate the optional packages without copying or reinstalling the existing
PyTorch/CUDA stack, create a virtual-environment overlay from `dino3d`:

```bash
conda activate dino3d
python -m venv --system-site-packages .venv/segdino3d-gdsam
source .venv/segdino3d-gdsam/bin/activate
pip install -r tools/grounding_dino_feature_extraction/requirements.txt
pip install --no-deps -e \
  tools/grounding_dino_feature_extraction/third_party/GroundingDINO
```

The overlay reads PyTorch 2.4, CUDA 12.1, and the main dependencies from
`dino3d`, while packages installed by these commands stay inside
`.venv/segdino3d-gdsam`. CUDA does not need to be installed or configured
again. Installing directly into `dino3d` is also supported if an overlay is
not desired.

The Grounding DINO extension is compiled against the existing PyTorch/CUDA
toolchain. The Grounding DINO and SAM source required by the extractor is
included under `third_party/`.

Build the extension on a machine with a visible CUDA GPU. When compiling on a
GPU-less login node, set `TORCH_CUDA_ARCH_LIST` for the target GPUs before the
editable install.

## Prepare ScanNet RGB-D frames

In addition to the processed point clouds described in
[`data/readme.md`](../../data/readme.md), feature extraction requires RGB
images, depth maps, camera poses, and intrinsics from the ScanNet `.sens`
files.

Use the official
[ScanNet SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python)
to extract them:

```bash
python tools/grounding_dino_feature_extraction/prepare_scannet_rgbd.py \
  --scannet-root data/scannet \
  --sens-reader /path/to/ScanNet/SensReader/python/reader.py
```

Use `--scene scene0000_00` for a single scene, `--scene-list FILE` for a
subset, or `--limit N` for a short test. The resulting layout is:

```text
data/scannet/
├── points/
│   └── scene0000_00.bin
└── ExtractedRGBD_GAP1/
    └── scene0000_00/
        ├── color/
        ├── depth/
        ├── pose/
        └── intrinsic/
            ├── intrinsic_color.txt
            └── intrinsic_depth.txt
```

The point files must be those produced by the normal SegDINO3D ScanNet
preprocessing, because output point features preserve that exact point order.

## Download checkpoints

Download the official Grounding DINO Swin-B and SAM ViT-H checkpoints:

```bash
bash tools/grounding_dino_feature_extraction/download_checkpoints.sh
```

By default they are stored in the Git-ignored directory:

```text
tools/grounding_dino_feature_extraction/checkpoints/
├── groundingdino_swinb_cogcoor.pth
└── sam_vit_h_4b8939.pth
```

Existing checkpoints can be used without copying them:

```bash
export GROUNDING_DINO_CHECKPOINT=/path/to/groundingdino_swinb_cogcoor.pth
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
```

The equivalent command-line options are `--grounding-checkpoint` and
`--sam-checkpoint`.

## Extract features

Extract one ScanNet200 scene:

```bash
python tools/grounding_dino_feature_extraction/extract_scannet_gdsam.py \
  --root-data data/scannet200 \
  --rgbd-root data/scannet/ExtractedRGBD_GAP1 \
  --output-dir data/features_2d/scannet200_gdsam \
  --class-set scannet200 \
  --scene scene0000_00
```

For ScanNet20, change the class set and output directory:

```bash
python tools/grounding_dino_feature_extraction/extract_scannet_gdsam.py \
  --root-data data/scannet \
  --output-dir data/features_2d/scannet_gdsam \
  --class-set scannet20 \
  --scene scene0000_00
```

Omit `--scene` to process all extracted scenes. Alternatively, repeat
`--scene`, pass `--scene-list FILE`, or use `--limit N`. Multiple jobs can
partition the scene list:

```text
--num-groups 8 --group-id 0
```

Launch one group per GPU and select the GPU with `CUDA_VISIBLE_DEVICES`.
Existing complete scenes are skipped; pass `--overwrite` to regenerate them.

Large mask medoids are processed in small batches. The default
`--medoid-batch-size 4` is conservative for 24 GB GPUs; increasing it may
improve throughput when more memory is available, while reducing it lowers
peak memory without changing the extraction procedure.

Each scene produces:

```text
scene0000_00.pth
scene0000_00_query_feats.pth
scene0000_00_query_3dctr.pth
scene0000_00_query_clsdists.pth
```

The first file contains four `[num_points, 256]` feature tensors. The other
files contain up to 2048 object-query features, their 3D centers, and
confidence scores.

## Use the generated features

Set `root_points_2dfeats` in the ScanNet or ScanNet200 dataset configuration
to the generated directory. Keep Grounding DINO features separate from the
provided DINO-X features: checkpoints trained with one feature source should
not be assumed to be interchangeable with the other.

Feature extraction is storage intensive because four 256-dimensional tensors
are saved for every scene point. Check the available disk space before
processing the full dataset.
