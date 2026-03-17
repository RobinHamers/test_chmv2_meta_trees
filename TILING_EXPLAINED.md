# Tiled Inference with Hann-Window Blending

## Why tiling?

The CHMv2 model (DINOv3 ViT-L + DPT head) requires fp32 and only fits a **512×512 tile** in ~5.6GB VRAM. The source image is 20000×20000px, so it must be processed in chunks.

---

## Step 1 — Tile grid

A 512×512 window slides across the image with a **stride of 384px**:

```
stride = tile_size - 2 × overlap = 512 - 2×64 = 384
```

This means each tile overlaps its neighbours by **64px on every side**. The last row and column are snapped to the image edge so no pixels are missed.

For a 20000×20000 image this produces a **52×52 = 2704 tile** grid.

---

## Step 2 — Per-tile inference

Each tile is:
1. Read from GCS via GDAL `/vsigs/` virtual filesystem
2. Converted to a PIL RGB image
3. Passed through `CHMv2ImageProcessorFast` → `CHMv2ForDepthEstimation`
4. Post-processed back to a 512×512 float32 depth map (meters)

---

## Step 3 — Weighted accumulation

Instead of hard-cutting tiles (which leaves visible seams), each tile's output is multiplied by a **2D Hann mask** before being summed into two accumulators:

```python
acc[r:r+512, c:c+512] += depth * hann
wgt[r:r+512, c:c+512] += hann
```

The Hann mask is a cosine taper — it is ~0 at the tile edges and ~1 at the centre:

```
hann_1d = np.hanning(512)          # shape (512,)
hann_2d = np.outer(hann_1d, hann_1d)  # shape (512, 512)
```

In overlap zones, **multiple tiles contribute** to the same pixel. The central part of each tile (highest weight) dominates; the edges (lowest weight) are down-weighted. This smoothly blends away any model discontinuities at tile boundaries.

---

## Step 4 — Normalisation

After all tiles are accumulated, the final canopy height map is:

```python
result = acc / wgt   # element-wise
```

Dividing by the summed weights converts the weighted sum back to a properly scaled height value at every pixel.

---

## Memory

The two 20000×20000 float32 accumulators would cost ~3.2GB of RAM each. Instead they are stored as **`np.memmap` files on disk**, so RAM usage stays low throughout the run.

---

## Output

A cloud-optimised GeoTIFF (`chmv2_full_output.tif`) with:
- Deflate compression + floating-point predictor
- 512×512 internal tiles
- Original CRS and geotransform preserved

**Run stats:** 2704 tiles · ~4 tiles/sec · 13 min total · height range 0.01 – 68.46m · mean 11.45m
