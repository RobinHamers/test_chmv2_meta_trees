"""
CHM-Meta (v1) full-image tiled inference with overlapping tiles and Hann-window blending.

Model: WEO-SAS/chm-meta (SSL ViT-H + DPT head)
Image: 20000x20000px, 0.25m res, EPSG:32555 from GCS
Tile size: 512x512
Overlap: 64px each side → stride 384px → ~2704 tiles
Normalization: mean=(0.420, 0.411, 0.296), std=(0.213, 0.156, 0.143)

Uses the same tiling/blending strategy as ../run_chmv2_full.py for direct comparison.
"""

import os
import sys
import time
import math
import tempfile
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add this directory to sys.path so model_chm.py / backbone.py / dpt_head.py can be imported
sys.path.insert(0, os.path.dirname(__file__))

GCS_PATH = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32555.tif"
GCS_PATH_ALT = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32655.tif"
MODEL_REPO = "WEO-SAS/chm-meta"
OUTPUT_DIR = os.path.dirname(__file__)

TILE_SIZE = 512
OVERLAP = 64
STRIDE = TILE_SIZE - 2 * OVERLAP  # 384

NORM_MEAN = (0.420, 0.411, 0.296)
NORM_STD = (0.213, 0.156, 0.143)


def download_model_files():
    """Download model source files and weights from HuggingFace to this directory."""
    files = ["backbone.py", "dpt_head.py", "model_chm.py", "SSLhuge_satellite.pth"]
    for fname in files:
        dest = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(dest):
            print(f"  Downloading {fname}...")
            path = hf_hub_download(MODEL_REPO, fname)
            import shutil
            shutil.copy(path, dest)
        else:
            print(f"  {fname} already present.")
    print("  All model files ready.")


def make_hann_mask(tile_size):
    hann_1d = np.hanning(tile_size).astype(np.float32)
    return np.outer(hann_1d, hann_1d)


def get_tile_coords(img_h, img_w, tile_size, stride):
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

    # Download model files
    print(f"\nDownloading model files from {MODEL_REPO}...")
    download_model_files()

    # Import model (after files are on disk and sys.path is set)
    from model_chm import SSLModule

    weights_path = os.path.join(OUTPUT_DIR, "SSLhuge_satellite.pth")
    print(f"\nLoading model from {weights_path}...")
    model = SSLModule(ssl_path="SSLhuge_satellite.pth", local_path=weights_path)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Preprocessing pipeline
    normalize = T.Normalize(mean=NORM_MEAN, std=NORM_STD)

    def preprocess(rgb_hwc: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 numpy → normalized (1, 3, H, W) float32 tensor."""
        tensor = TF.to_tensor(rgb_hwc)        # (3, H, W) float32 in [0, 1]
        tensor = normalize(tensor)
        return tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    # Open source TIF
    print("\nOpening source TIF from GCS...")
    try:
        src = rasterio.open(GCS_PATH)
        _ = src.crs
    except Exception as e:
        print(f"  Primary path failed ({e}), trying alt...")
        src = rasterio.open(GCS_PATH_ALT)

    img_h, img_w = src.height, src.width
    crs = src.crs
    transform = src.transform
    print(f"  Size: {img_w}x{img_h}, CRS: {crs}, Res: {src.res}")

    # Tile grid
    tile_coords = get_tile_coords(img_h, img_w, TILE_SIZE, STRIDE)
    n_tiles = len(tile_coords)
    print(f"  Tile grid: {n_tiles} tiles (stride={STRIDE}, overlap={OVERLAP})")

    hann = make_hann_mask(TILE_SIZE)

    # Memmap accumulators
    tmp_dir = tempfile.mkdtemp()
    acc = np.memmap(os.path.join(tmp_dir, "acc.dat"), dtype=np.float32, mode="w+", shape=(img_h, img_w))
    wgt = np.memmap(os.path.join(tmp_dir, "wgt.dat"), dtype=np.float32, mode="w+", shape=(img_h, img_w))
    print(f"  Accumulators: 2x {img_h}x{img_w} float32 (~{2*img_h*img_w*4/1e9:.1f}GB) in {tmp_dir}")

    # Inference loop
    print(f"\nStarting inference...")
    t0 = time.time()
    skipped = 0

    for row_off, col_off in tqdm(tile_coords, unit="tile"):
        win = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
        data = src.read(window=win)  # (bands, H, W)

        rgb = data[:3] if data.shape[0] >= 3 else np.stack([data[0]] * 3)
        if rgb.max() == 0:
            skipped += 1
            continue

        rgb_hwc = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
        x = preprocess(rgb_hwc)  # (1, 3, H, W)

        with torch.no_grad():
            out = model(x)  # (1, 1, H, W)

        depth = out.squeeze().cpu().numpy().astype(np.float32)  # (H, W)

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
    out_path = os.path.join(OUTPUT_DIR, "chmv1_full_output.tif")
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
            model=MODEL_REPO,
            units="meters",
            description="CHM-Meta (v1) full-image canopy height",
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            stride=STRIDE,
        )

    print(f"\nStats — min: {result[valid].min():.2f}m, max: {result[valid].max():.2f}m, mean: {result[valid].mean():.2f}m")
    print(f"Output: {out_path}")
    print("Done!")

    del acc, wgt
    os.remove(os.path.join(tmp_dir, "acc.dat"))
    os.remove(os.path.join(tmp_dir, "wgt.dat"))
    os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
