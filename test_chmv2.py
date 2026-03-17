"""
CHMv2 (Canopy Height Maps v2) inference test
Model: facebook/dinov3-vitl16-chmv2-dpt-head
Input: Google satellite TIF at 24-25cm resolution from GCS
"""

import numpy as np
import torch
from PIL import Image
from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessorFast
import rasterio
from rasterio.windows import Window
import os

GCS_PATH = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32655.tif"
# Also try the actual file name
GCS_PATH_ALT = "/vsigs/frankwin_tmp/chm_meta_au_test/inference_image/google_sat_24_25cm_32555.tif"
MODEL_ID = "facebook/dinov3-vitl16-chmv2-dpt-head"
CROP_SIZE = 512  # pixels to crop for testing
OUTPUT_DIR = "/home/robin-hamers/test_chm2_meta_trees"


def load_crop_from_gcs(path, crop_size=1024):
    """Read a crop from GCS TIF using GDAL virtual filesystem."""
    print(f"Opening: {path}")
    with rasterio.open(path) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width} x {src.height} pixels")
        print(f"  Bands: {src.count}")
        print(f"  Dtype: {src.dtypes}")
        print(f"  Resolution: {src.res}")

        # Crop from center
        col_off = max(0, (src.width - crop_size) // 2)
        row_off = max(0, (src.height - crop_size) // 2)
        w = min(crop_size, src.width)
        h = min(crop_size, src.height)

        window = Window(col_off, row_off, w, h)
        print(f"  Reading crop: {w}x{h} at offset ({col_off}, {row_off})")

        data = src.read(window=window)  # shape: (bands, H, W)

        # Save crop metadata for georeferencing output
        transform = src.window_transform(window)
        crs = src.crs

    print(f"  Raw data shape: {data.shape}, dtype: {data.dtype}")

    # Convert to RGB PIL Image
    if data.shape[0] >= 3:
        rgb = data[:3]  # Take first 3 bands (RGB)
    else:
        rgb = np.stack([data[0]] * 3, axis=0)

    # Normalize to uint8 if needed
    if rgb.dtype != np.uint8:
        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)

    img = Image.fromarray(np.transpose(rgb, (1, 2, 0)))
    print(f"  PIL image size: {img.size}")
    return img, transform, crs


def run_inference(image, model, processor, device):
    """Run CHMv2 depth estimation inference."""
    print(f"\nRunning inference on {image.size} image...")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]

    return depth.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model: {MODEL_ID}")
    processor = CHMv2ImageProcessorFast.from_pretrained(MODEL_ID)
    model = CHMv2ForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("Model loaded.")

    # Load crop from GCS
    print(f"\nLoading satellite image crop from GCS...")
    try:
        img, transform, crs = load_crop_from_gcs(GCS_PATH_ALT, crop_size=CROP_SIZE)
    except Exception as e:
        print(f"Primary path failed ({e}), trying alt path...")
        img, transform, crs = load_crop_from_gcs(GCS_PATH, crop_size=CROP_SIZE)

    # Save input crop for reference
    crop_path = os.path.join(OUTPUT_DIR, "input_crop.png")
    img.save(crop_path)
    print(f"Input crop saved: {crop_path}")

    # Run inference
    depth_map = run_inference(img, model, processor, device)
    print(f"Output shape: {depth_map.shape}")
    print(f"Canopy height stats — min: {depth_map.min():.2f}m, max: {depth_map.max():.2f}m, mean: {depth_map.mean():.2f}m")

    # Save output as GeoTIFF
    out_path = os.path.join(OUTPUT_DIR, "chmv2_output.tif")
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=depth_map.shape[0],
        width=depth_map.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(depth_map.astype(np.float32), 1)
        dst.update_tags(model=MODEL_ID, units="meters", description="CHMv2 canopy height")
    print(f"Output GeoTIFF saved: {out_path}")

    # Also save a colorized visualization
    viz_arr = depth_map.copy()
    viz_arr = np.clip(viz_arr, 0, None)
    normalized = (viz_arr / (viz_arr.max() + 1e-8) * 255).astype(np.uint8)
    viz_img = Image.fromarray(normalized, mode="L")
    viz_path = os.path.join(OUTPUT_DIR, "chmv2_visualization.png")
    viz_img.save(viz_path)
    print(f"Visualization saved: {viz_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
