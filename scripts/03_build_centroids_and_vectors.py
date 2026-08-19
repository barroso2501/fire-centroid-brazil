"""
build_centroids_and_vectors.py

Description
-----------
Reads annual patch-level CSVs for Natural and Anthropogenic_use groups,
computes area-weighted centroids per year for each group, and derives
interannual displacement vectors (distance and direction).

Inputs
------
Google Drive (linked to this OSF project via the Google Drive add-on):
- Centroid_OSF/Natural/patches_fire_{year}_filtered.csv
- Centroid_OSF/Anthropogenic_use/patches_fire_{year}_filtered.csv

Each file has columns:
    Ano, Patch_ID, Longitude, Latitude, NumPixels, Area_ha

Outputs
-------
In 02_Data_derived/:
- centroids_annual.csv
    Columns:
        Year,
        longitude_nat, latitude_nat, area_total_ha_nat, num_patches_nat,
        longitude_use, latitude_use, area_total_ha_use, num_patches_use

- centroids_natural.csv
- centroids_anthropogenic.csv
- vectors_natural.csv
- vectors_anthropogenic.csv

Each vectors_* file contains:
    group, Year_from, Year_to,
    Lon_from, Lat_from, Lon_to, Lat_to,
    Dist_km, Angle_deg
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from geopy.distance import geodesic

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Base folder that contains the 'Natural' and 'Anthropogenic_use' folders
BASE_DRIVE_FOLDER = r"E:/Process/Centroid_OSF"  # adjust to your environment

# Folder where derived outputs (centroids, vectors) will be written
OUTPUT_DATA_FOLDER = r"E:/Process/02_Data_derived"  # mirror OSF 02_Data_derived

os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

GROUPS_INFO = {
    "natural": {
        "folder": os.path.join(BASE_DRIVE_FOLDER, "Natural"),
        "prefix_out": "natural",
    },
    "anthropogenic": {
        "folder": os.path.join(BASE_DRIVE_FOLDER, "Anthropogenic_use"),
        "prefix_out": "anthropogenic",
    },
}


# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------

def load_patches_for_group(group_name: str, folder: str) -> pd.DataFrame:
    """
    Load all patch CSV files for a given group (Natural or Anthropogenic_use)
    and return a single DataFrame with an extra 'group' column.

    Parameters
    ----------
    group_name : str
        Name of the group, e.g. "natural" or "anthropogenic".
    folder : str
        Path to the folder that contains patches_fire_{year}_filtered.csv files.

    Returns
    -------
    df_all : pandas.DataFrame
        Concatenated DataFrame of all patches for this group.
    """
    pattern = os.path.join(folder, "patches_fire_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No patch files found in {folder} for group '{group_name}'.")

    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["group"] = group_name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Ensure numeric types
    df_all["Ano"] = pd.to_numeric(df_all["Ano"], errors="coerce").astype("Int64")
    df_all["Area_ha"] = pd.to_numeric(df_all["Area_ha"], errors="coerce")
    df_all["Longitude"] = pd.to_numeric(df_all["Longitude"], errors="coerce")
    df_all["Latitude"] = pd.to_numeric(df_all["Latitude"], errors="coerce")

    # Drop rows with missing essential values
    df_all = df_all.dropna(subset=["Ano", "Area_ha", "Longitude", "Latitude"])

    return df_all


def compute_centroids(df_all: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Compute area-weighted annual centroids for a given group.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Patch-level data with columns Ano, Longitude, Latitude, Area_ha.
    group_name : str
        Group label, e.g. "natural" or "anthropogenic".

    Returns
    -------
    centroids_df : pandas.DataFrame
        DataFrame with columns:
            Year, longitude, latitude, area_total_ha, num_patches, group
    """
    centroids_df = (
        df_all
        .groupby("Ano")
        .apply(lambda g: pd.Series({
            "longitude": np.average(g["Longitude"], weights=g["Area_ha"]),
            "latitude": np.average(g["Latitude"], weights=g["Area_ha"]),
            "area_total_ha": g["Area_ha"].sum(),
            "num_patches": len(g),
        }))
        .reset_index()
        .rename(columns={"Ano": "Year"})
        .sort_values("Year")
    )

    centroids_df["group"] = group_name
    return centroids_df


def compute_vectors(centroids_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Compute interannual displacement vectors from a centroid time series.

    Parameters
    ----------
    centroids_df : pandas.DataFrame
        Must contain columns Year, longitude, latitude.
    group_name : str
        Group label, e.g. "natural" or "anthropogenic".

    Returns
    -------
    vectors_df : pandas.DataFrame
        DataFrame with one row per consecutive-year pair:
            group, Year_from, Year_to,
            Lon_from, Lat_from, Lon_to, Lat_to,
            Dist_km, Angle_deg
    """
    centroids_df = centroids_df.sort_values("Year").reset_index(drop=True)
    rows = []

    for i in range(1, len(centroids_df)):
        prev = centroids_df.iloc[i - 1]
        curr = centroids_df.iloc[i]

        origin = (prev["latitude"], prev["longitude"])
        destination = (curr["latitude"], curr["longitude"])

        dist_km = geodesic(origin, destination).kilometers
        delta_x = curr["longitude"] - prev["longitude"]
        delta_y = curr["latitude"] - prev["latitude"]
        angle_deg = (math.degrees(math.atan2(delta_y, delta_x)) + 360.0) % 360.0

        rows.append({
            "group": group_name,
            "Year_from": int(prev["Year"]),
            "Year_to": int(curr["Year"]),
            "Lon_from": prev["longitude"],
            "Lat_from": prev["latitude"],
            "Lon_to": curr["longitude"],
            "Lat_to": curr["latitude"],
            "Dist_km": round(dist_km, 2),
            "Angle_deg": round(angle_deg, 2),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------

def main():
    all_centroids = []
    vectors_by_group = {}

    for group_name, info in GROUPS_INFO.items():
        print(f"\n=== Processing group: {group_name} ===")
        folder = info["folder"]

        # 1. Load patch-level data for all years
        df_patches = load_patches_for_group(group_name, folder)
        print(f"  Loaded {len(df_patches)} patches for group '{group_name}'.")

        # 2. Compute annual centroids
        centroids_df = compute_centroids(df_patches, group_name)
        all_centroids.append(centroids_df)

        # 3. Compute interannual displacement vectors
        vectors_df = compute_vectors(centroids_df, group_name)
        vectors_by_group[group_name] = vectors_df

        # 4. Save centroids and vectors per group (optional but useful)
        out_centroids_group = os.path.join(
            OUTPUT_DATA_FOLDER, f"centroids_{group_name}.csv"
        )
        out_vectors_group = os.path.join(
            OUTPUT_DATA_FOLDER, f"vectors_{group_name}.csv"
        )
        centroids_df.to_csv(out_centroids_group, index=False)
        vectors_df.to_csv(out_vectors_group, index=False)

        print(f"  ✅ Saved centroids for group '{group_name}' to: {out_centroids_group}")
        print(f"  ✅ Saved vectors for group '{group_name}' to:   {out_vectors_group}")

    # 5. Combine natural + anthropogenic centroids into a single annual table
    df_all_centroids = pd.concat(all_centroids, ignore_index=True)

    df_nat = df_all_centroids[df_all_centroids["group"] == "natural"].copy()
    df_use = df_all_centroids[df_all_centroids["group"] == "anthropogenic"].copy()

    df_nat = df_nat.rename(columns={
        "longitude": "longitude_nat",
        "latitude": "latitude_nat",
        "area_total_ha": "area_total_ha_nat",
        "num_patches": "num_patches_nat",
    }).drop(columns=["group"])

    df_use = df_use.rename(columns={
        "longitude": "longitude_use",
        "latitude": "latitude_use",
        "area_total_ha": "area_total_ha_use",
        "num_patches": "num_patches_use",
    }).drop(columns=["group"])

    centroids_annual = pd.merge(df_nat, df_use, on="Year", how="outer").sort_values("Year")

    out_centroids_annual = os.path.join(OUTPUT_DATA_FOLDER, "centroids_annual.csv")
    centroids_annual.to_csv(out_centroids_annual, index=False)

    print(f"\n✅ Saved combined annual centroids (natural + anthropogenic) to: {out_centroids_annual}")


if __name__ == "__main__":
    main()