"""
filter_small_patches.py

Description
-----------
Reads raw patch-level CSV files (one per year) and removes very small
patches based on a minimum number of pixels. The filtered files are
saved with the suffix '_filtered.csv' and are used in subsequent
centroid and trajectory calculations.

Inputs
------
- Input folder with raw CSV files named:
    patches_fire_{year}.csv

Each file must contain at least the columns:
    NumPixels, Area_ha, Longitude, Latitude, Ano, Patch_ID

Outputs
-------
- Output folder with filtered CSV files named:
    patches_fire_{year}_filtered.csv

Filtering rule
--------------
- Only keep patches with NumPixels >= 4
  (≈ 0.36 ha for 30 m × 30 m pixels).
"""

import os
from glob import glob
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Folder with raw patch CSV files
INPUT_FOLDER = r"E:/Process/Anthropogenic"          # adjust to your folder

# Folder to write filtered patch CSV files
OUTPUT_FOLDER = r"E:/Process/Anthropogenic/Filtered"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Minimum number of pixels per patch
MIN_PIXELS = 4

# Filename pattern for input patch files
INPUT_PATTERN = "patches_fire_*.csv"


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    csv_files = sorted(glob(os.path.join(INPUT_FOLDER, INPUT_PATTERN)))

    if not csv_files:
        print(f"No input files found in: {INPUT_FOLDER}")
        return

    for file_path in tqdm(csv_files, desc="Filtering small patches"):
        df = pd.read_csv(file_path)

        # Filter patches by size
        df_filtered = df[df["NumPixels"] >= MIN_PIXELS].copy()

        # Build output filename with _filtered suffix
        filename = os.path.basename(file_path).replace(".csv", "_filtered.csv")
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        df_filtered.to_csv(output_path, index=False)

    print(f"Done. Filtered files saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()