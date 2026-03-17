"""
CHMv2 full-image tiled inference with overlapping tiles and Hann-window blending.

Image: 20000x20000px, 0.25m res, EPSG:32555 from GCS
Tile size: 512x512 (max that fits in GPU fp32 with ~5.6GB VRAM)
Overlap: 64px each side → stride 384px → ~53x53 = ~2809 tiles
Blending: 2D Hann weight mask to eliminate edge artifacts
Memory: np.memmap for two 20000x20000 float32 accumulators (~1.6GB each)
"""

import os
import time
import math
import tempfile
import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessorFast
from tqdm import tqdm

GCS_PATH = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32555.tif"
GCS_PATH_ALT = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32655.tif"
MODEL_ID = "facebook/dinov3-vitl16-chmv2-dpt-head"
OUTPUT_DIR = "/home/robin-hamers/test_chm2_meta_trees"

TILE_SIZE = 512
OVERLAP = 64
STRIDE = TILE_SIZE - 2 * OVERLAP  # 384


def make_hann_mask(tile_size):
    """2D Hann (cosine-taper) weight mask for smooth tile blending."""
    hann_1d = np.hanning(tile_size).astype(np.float32)
    return np.outer(hann_1d, hann_1d)


def get_tile_coords(img_h, img_w, tile_size, stride):
    """Generate (row_off, col_off) for each tile, covering full image."""
    rows = list(range(0, img_h - tile_size + 1, stride))
    if rows[-1] + tile_size < img_h:
        rows.append(img_h - tile_size)

    cols = list(range(0, img_w - tile_size + 1, stride))
    if cols[-1] + tile_size < img_w:
        cols.append(img_w - tile_size)

    return [(r, c) for r in rows for c in cols]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    processor = CHMv2ImageProcessorFast.from_pretrained(MODEL_ID)
    model = CHMv2ForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("Model loaded.")

    # Open source TIF (streaming from GCS via GDAL vsigs)
    print("\nOpening source TIF from GCS...")
    src_path = GCS_PATH
    try:
        src = rasterio.open(src_path)
        _ = src.crs  # force a read to confirm it opens
    except Exception as e:
        print(f"  Primary path failed ({e}), trying alt...")
        src_path = GCS_PATH_ALT
        src = rasterio.open(src_path)

    img_h, img_w = src.height, src.width
    crs = src.crs
    transform = src.transform
    print(f"  Size: {img_w}x{img_h}, CRS: {crs}, Res: {src.res}")

    # Tile grid
    tile_coords = get_tile_coords(img_h, img_w, TILE_SIZE, STRIDE)
    n_tiles = len(tile_coords)
    n_rows = math.ceil((img_h - TILE_SIZE) / STRIDE) + 1
    n_cols = math.ceil((img_w - TILE_SIZE) / STRIDE) + 1
    print(f"  Tile grid: ~{n_rows}x{n_cols} = {n_tiles} tiles (stride={STRIDE}, overlap={OVERLAP})")

    # Hann mask
    hann = make_hann_mask(TILE_SIZE)

    # Memmap accumulators
    tmp_dir = tempfile.mkdtemp()
    acc_path = os.path.join(tmp_dir, "accumulator.dat")
    wgt_path = os.path.join(tmp_dir, "weights.dat")
    acc = np.memmap(acc_path, dtype=np.float32, mode="w+", shape=(img_h, img_w))
    wgt = np.memmap(wgt_path, dtype=np.float32, mode="w+", shape=(img_h, img_w))
    print(f"  Accumulators: 2x {img_h}x{img_w} float32 (~{2*img_h*img_w*4/1e9:.1f}GB) in {tmp_dir}")

    # Inference loop
    print(f"\nStarting inference...")
    t0 = time.time()
    skipped = 0

    for row_off, col_off in tqdm(tile_coords, unit="tile"):
        # Read tile from GCS
        from rasterio.windows import Window
        win = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
        data = src.read(window=win)  # (bands, H, W)

        # Skip blank/masked tiles
        rgb = data[:3] if data.shape[0] >= 3 else np.stack([data[0]] * 3)
        if rgb.max() == 0:
            skipped += 1
            continue

        img = Image.fromarray(np.transpose(rgb, (1, 2, 0)).astype(np.uint8))

        # Inference
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        depth = processor.post_process_depth_estimation(
            outputs, target_sizes=[(TILE_SIZE, TILE_SIZE)]
        )[0]["predicted_depth"].cpu().numpy().astype(np.float32)

        # Weighted accumulation
        acc[row_off:row_off+TILE_SIZE, col_off:col_off+TILE_SIZE] += depth * hann
        wgt[row_off:row_off+TILE_SIZE, col_off:col_off+TILE_SIZE] += hann

    src.close()
    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed/60:.1f} min. Skipped {skipped} blank tiles.")

    # Normalize
    print("Normalizing accumulator...")
    result = np.zeros((img_h, img_w), dtype=np.float32)
    valid = wgt > 0
    result[valid] = acc[valid] / wgt[valid]

    # Save GeoTIFF
    out_path = os.path.join(OUTPUT_DIR, "chmv2_full_output.tif")
    print(f"Writing GeoTIFF: {out_path}")
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=img_h,
        width=img_w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(result, 1)
        dst.update_tags(
            model=MODEL_ID,
            units="meters",
            description="CHMv2 full-image canopy height",
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            stride=STRIDE,
        )

    print(f"\nStats — min: {result[valid].min():.2f}m, max: {result[valid].max():.2f}m, mean: {result[valid].mean():.2f}m")
    print(f"Output: {out_path}")
    print("Done!")

    # Cleanup memmaps
    del acc, wgt
    os.remove(acc_path)
    os.remove(wgt_path)
    os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
