# CHMv2 Full-Image Tiled Inference

Canopy height estimation over large satellite images using [CHMv2](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head) (Meta, March 2026).

## Model

`facebook/dinov3-vitl16-chmv2-dpt-head` — DINOv3 ViT-L backbone + DPT depth head, 0.3B parameters.

## Input

- 20000×20000px Google satellite image, 0.25m/px resolution, EPSG:32555
- 4-band RGBA uint8 GeoTIFF

## Scripts

### `test_chmv2.py`
Runs inference on a single 512×512 crop from the center of the image. Good for quickly testing the environment.

### `run_chmv2_full.py`
Processes the full 20000×20000 image using overlapping tiles with Hann-window blending:
- **Tile size:** 512×512 (max that fits in GPU fp32 with ~5.6GB VRAM)
- **Overlap:** 64px each side, stride 384px → 2704 tiles
- **Blending:** 2D Hann weight mask to eliminate seam artifacts
- **Memory:** `np.memmap` accumulators to avoid RAM OOM
- **Output:** Compressed tiled GeoTIFF with original CRS and geotransform

Runtime: ~13 minutes on a GPU with 5.6GB VRAM.

## Results

| Stat | Value |
|------|-------|
| Min height | 0.01 m |
| Max height | 68.46 m |
| Mean height | 11.45 m |

## Setup

```bash
conda activate tree_height_env
pip install rasterio tqdm
pip install git+https://github.com/huggingface/transformers  # dev ≥5.3.0 required
```

## How tiling works

See [TILING_EXPLAINED.md](TILING_EXPLAINED.md) for a detailed explanation of the overlap and blending strategy.
