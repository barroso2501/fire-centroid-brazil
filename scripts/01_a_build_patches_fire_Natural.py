"""
build_patches_fire_Natural.py

Description
-----------
Block-wise extraction of burned patches from annual burned-area rasters
for natural, exporting one CSV per year with patch-level
centroids and area.

Inputs
------
- One GeoTIFF per year in RASTER_FOLDER, e.g.:
    fogo_natural_1985.tif
    fogo_natural_1986.tif
    ...
  Burned pixels must have value 255.

Outputs
-------
- One CSV per year in OUTPUT_FOLDER (or directly in the Google Drive
  folder linked to OSF), named:
    patches_fire_{year}_filtered.csv

Each CSV has the columns:
    Ano, Patch_ID, Longitude, Latitude, NumPixels, Area_ha

Notes
-----
- Uses block-wise processing (block_size) to handle large rasters.
- Uses 4-neighbour (rook) connectivity to define patches.
- Uses a status file (blocos_processados_{ano}.txt) to resume processing.
"""

import os
import re
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Name of the group, just for readability/logging
GROUP_NAME = "Anthropogenic_use"

# Folder with input rasters (one per year)
# Example: 'E:/Process/Natural' or a Google Drive mount
RASTER_FOLDER = r"E:/Process/Natural"

# Folder where the CSVs and status files will be written
# Example: r"E:/Process" or a path inside your Centroid_OSF/Anthropogenic_use
OUTPUT_FOLDER = r"E:/Process/Anthropogenic_use_patches"

# Pixel size in metres (MapBiomas / Landsat = 30 m)
PIXEL_SIZE_METERS = 30

# Value representing burned pixels in the raster
BURN_VALUE = 255

# Block size for reading the raster (in pixels)
BLOCK_SIZE = 1000

# Filename pattern for input rasters (must capture the year)
RASTER_PATTERN = r"fogo_antropico_(\d{4})\.tif"


# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

def process_block(arr, transform, row_off, col_off, patch_id_offset=0, ano=0):
    """
    Process a raster block and extract patch centroids and sizes.

    Parameters
    ----------
    arr : 2D numpy array
        Raster block (single band).
    transform : affine.Affine
        Georeferencing transform of the full raster.
    row_off, col_off : int
        Offsets (row, col) of the block within the full raster.
    patch_id_offset : int
        Offset to make patch IDs unique across blocks.
    ano : int
        Year of the raster (for output).

    Returns
    -------
    points : list of dict
        One dict per patch, with keys:
        'Ano', 'Patch_ID', 'Longitude', 'Latitude', 'NumPixels', 'Area_ha'
    num_features : int
        Number of patches found in this block.
    """
    # Mask of burned pixels
    mask = arr == BURN_VALUE
    if not np.any(mask):
        return [], 0

    # Rook neighbourhood (4-neighbour connectivity)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    labeled_array, num_features = label(mask, structure=structure)

    # Count pixels per patch
    unique_ids, counts = np.unique(labeled_array[labeled_array > 0], return_counts=True)
    patch_sizes = dict(zip(unique_ids, counts))

    points = []
    visited_ids = set()

    rows, cols = np.where(labeled_array > 0)

    for row, col in zip(rows, cols):
        patch_local_id = labeled_array[row, col]
        if patch_local_id in visited_ids:
            continue

        visited_ids.add(patch_local_id)

        # Global position in the full raster
        global_row = row + row_off
        global_col = col + col_off

        # Georeferenced coordinates (center of pixel)
        x, y = rasterio.transform.xy(transform, global_row, global_col, offset='center')

        # Area in hectares
        n_pixels = int(patch_sizes[patch_local_id])
        area_ha = n_pixels * (PIXEL_SIZE_METERS * PIXEL_SIZE_METERS) / 10000.0

        points.append({
            "Ano": ano,
            "Patch_ID": patch_local_id + patch_id_offset,
            "Longitude": x,
            "Latitude": y,
            "NumPixels": n_pixels,
            "Area_ha": round(area_ha, 4),
        })

    return points, num_features


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------

def main():
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in sorted(os.listdir(RASTER_FOLDER)):
        match = re.match(RASTER_PATTERN, filename)
        if not match:
            continue

        ano = int(match.group(1))
        raster_path = os.path.join(RASTER_FOLDER, filename)

        # Output CSV and status file for this year
        output_csv_path = os.path.join(OUTPUT_FOLDER, f"patches_fire_{ano}_filtered.csv")
        bloco_status_path = os.path.join(OUTPUT_FOLDER, f"blocos_processados_{ano}.txt")

        print(f"\n=== Processing year {ano} for group {GROUP_NAME} ===")
        print(f"Input raster: {raster_path}")
        print(f"Output CSV:   {output_csv_path}")

        # Load status of processed blocks (if resuming)
        bloco_processados = set()
        if os.path.exists(bloco_status_path):
            with open(bloco_status_path, "r") as f:
                bloco_processados = set(f.read().splitlines())

        with rasterio.open(raster_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            patch_global_id = 0

            for row_off in range(0, height, BLOCK_SIZE):
                for col_off in range(0, width, BLOCK_SIZE):
                    bloco_id = f"{row_off}_{col_off}"
                    if bloco_id in bloco_processados:
                        print(f"> Block {bloco_id} for year {ano} already processed. Skipping.")
                        continue

                    print(f"🔄 Processing block {bloco_id} for year {ano}...")
                    win_height = min(BLOCK_SIZE, height - row_off)
                    win_width = min(BLOCK_SIZE, width - col_off)
                    window = rasterio.windows.Window(col_off, row_off, win_width, win_height)
                    arr = src.read(1, window=window)

                    try:
                        points, n_patches = process_block(
                            arr,
                            transform,
                            row_off,
                            col_off,
                            patch_id_offset=patch_global_id,
                            ano=ano,
                        )
                        if points:
                            df = pd.DataFrame(points)
                            df.to_csv(
                                output_csv_path,
                                index=False,
                                mode="a",
                                header=not os.path.exists(output_csv_path),
                            )
                            print(f"✅ Block {bloco_id} ({ano}): {len(points)} patches saved.")
                            patch_global_id += n_patches
                        else:
                            print(f"⏭ Block {bloco_id} ({ano}) contains no burned pixels (value {BURN_VALUE}).")
                    except Exception as e:
                        print(f"❌ Error in block {bloco_id} ({ano}): {e}")
                        continue

                    # Mark block as processed
                    with open(bloco_status_path, "a") as f:
                        f.write(bloco_id + "\n")

        print(f"=== Finished year {ano} for group {GROUP_NAME} ===")


if __name__ == "__main__":
    main()