"""
04_biome_contribution_albers.py

WHAT THIS SCRIPT COMPUTES
--------------------------
Decomposes the national centroid's position and its year-to-year shifts into
per-biome contributions: leave-one-out (LOO) sensitivity, influence as
weight x distance (each biome's share of burned area times its distance from
the national centroid), and the reweighting-vs-within-biome-migration split.
Everything computed in Albers.

Reads a SINGLE consolidated file (with a 'grupo' column) produced by
03_biome_centroids_from_patches.py -- biome centroids already computed IN
Albers directly from the patches. A legacy two-files-per-context mode is
also supported for older inputs; see MODE 2 below.

The script AUTO-DETECTS the coordinate space and column names:
      - already in Albers (meters or km)  -> just normalizes to km
      - in degrees (lon/lat)              -> reprojects to Albers
so it works with both the "_albers" files and the old degrees-based CSV,
without manual edits.

A sensitivity comparison (degrees vs Albers) is still produced: when the
input is already Albers, the script also runs the INVERSE transform to
degrees and reruns the legacy convention on the SAME centroids. This
isolates exactly the effect of the METRIC SPACE (euclidean-in-degrees vs
euclidean-in-Albers), which is the change under review.

WHAT THE LOGIC DOES NOT CHANGE
-----------------------------
The weighted barycenter, influence (w x d), LOO, and the K decomposition
keep the SAME formulas as the original script. Only the metric space changes.

OUTPUTS (all in km, in OUT_DIR)
  *_brasil_reconstruido.csv            Brazil centroid per year/context
  *_influencia_anual.csv               w, dist_to_br_km, influence_mag_km, share
  *_loo_shift.csv                      LOO in km (previously: in degrees)
  *_contrib_delta.csv                  K_move / K_weight / K_proj in km
  *_sensibilidade_graus_vs_albers.csv  comparison of the two conventions (supplement)

WHAT CAN BREAK AND HOW YOU WOULD NOTICE
  * Unexpected column name -> the script PRINTS the columns it found and
    stops with an error naming which role (year/biome/area/x/y) it could not
    map. If this happens, fill in COLMAP manually in the configuration.
  * Wrong CRS detected -> the script PRINTS what it detected ("degrees" /
    "Albers in meters" / "Albers in km") and the value range. CHECK this
    line: if it says "degrees" for a file you know is in Albers, something
    is wrong in the data.
  * A year/context with a single biome -> LOO undefined (NaN); the script
    warns how many.
  * If the sensitivity comes out with a median difference well above ~3%,
    suspect the input CRS (or that the centroids aren't really biome-level).
"""

import os
import numpy as np
import pandas as pd
from pyproj import Transformer

# ============================================================
# CONFIGURATION (edit only this section)
# ============================================================

# MODE 1 (recommended now): single file produced by 03_biome_centroids_from_patches.py,
# which already carries the 'grupo' column. Centroids computed IN Albers from the patches.
# Use a path relative to the repository/Zenodo release rather than an absolute Windows path.
SINGLE_FILE = "02_Data_derived/Biome_centroids_from_patches/biome_centroids_albers.csv"

# MODE 2 (legacy): one file per context. Only used if SINGLE_FILE is None.
FILES = {
    "Natural": "centroids_nat_biome_albers.csv",
    "Use":     "centroids_use_biome_albers.csv",
}

# Consistency check (optional, but STRONGLY recommended): point this at the
# centroids_observed_albers.csv produced by 05_uncertainty_separation.py.
# A weighted mean is associative, so the national centroid reconstructed
# from the biomes MUST match the one computed directly from the patches.
# If it does not, there is an error in the pipeline.
CHECK_FILE = "02_Data_derived/Uncertainty_separation/centroids_observed_albers.csv"

OUT_DIR    = "02_Data_derived/Biome_contribution_albers"
OUT_PREFIX = "centroid_contrib"

SRC_GEO = "EPSG:4674"     # SIRGAS 2000 (source datum confirmed against the data)
ALBERS  = "ESRI:102033"   # South America Albers Equal Area Conic

KM_PER_DEG = 111.32       # used ONLY to make the legacy comparison readable

# If auto-detection fails, fill this in manually. E.g.:
# COLMAP = {"ano":"Ano","bioma":"Bioma","area":"total_area_ha","x":"X","y":"Y"}
COLMAP = None


# ============================================================
# 1. READING, COLUMN MAPPING, AND NORMALIZATION TO ALBERS-km
# ============================================================

# accepted names for each role (lowercase)
# ORDER MATTERS: the first name found wins.
# For x/y, POINT_X/POINT_Y (already Albers) are prioritized over lon/lat (degrees).
# WATCH OUT for names truncated to 10 characters by shapefile (DBF) export:
#   centroid_l = LONGITUDE   |   centroid_1 = LATITUDE  (digit ONE, not letter L!)
CANDIDATOS = {
    "ano":   ["ano", "year", "yr"],
    "bioma": ["bioma", "biome"],
    "area":  ["total_area_ha", "area_total_ha", "total_area", "area_ha",
              "weight_sum", "area", "burned_area_ha"],
    "x":     ["point_x", "x_km", "x", "centroid_x", "xc_km", "xc",
              "centroid_lon", "centroid_l", "longitude", "lon", "long"],
    "y":     ["point_y", "y_km", "y", "centroid_y", "yc_km", "yc",
              "centroid_lat", "centroid_1", "latitude", "lat"],
}


def mapear_colunas(df):
    """Figures out which column fills each role. Fails with a clear message."""
    if COLMAP:
        return COLMAP
    lower = {c.lower().strip(): c for c in df.columns}
    m = {}
    for papel, nomes in CANDIDATOS.items():
        for n in nomes:
            if n in lower:
                m[papel] = lower[n]
                break
    faltando = [p for p in CANDIDATOS if p not in m]
    if faltando:
        raise ValueError(
            f"Could not identify columns for: {faltando}.\n"
            f"Columns found in the file: {list(df.columns)}\n"
            f"Fill in COLMAP in the configuration."
        )
    return m


def detectar_espaco(x, y):
    """
    Detects which coordinate space the data is in, from the order of magnitude.
      degrees : |x| <= 180 and |y| <= 90
      km      : magnitudes in the thousands
      meters  : magnitudes in the millions
    """
    ax, ay = np.nanmax(np.abs(x)), np.nanmax(np.abs(y))
    if ax <= 180 and ay <= 90:
        return "degrees"
    if ax < 50_000 and ay < 50_000:
        return "albers_km"
    return "albers_m"


def carregar(path, regime):
    """Reads one context's file and returns a standardized DataFrame with X_km/Y_km."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Robust reading: files exported from ArcGIS come in European format
    # (';' separator, ',' decimal, with a BOM). We try the combinations in
    # order and keep the first one that yields more than one column.
    df = None
    for sep, dec in [(";", ","), (",", "."), (";", "."), ("\t", ".")]:
        try:
            cand = pd.read_csv(path, sep=sep, decimal=dec, encoding="utf-8-sig")
        except Exception:
            continue
        if cand.shape[1] > 1:
            df = cand
            print(f"  [{regime}] read with sep='{sep}' decimal='{dec}' -> {cand.shape[1]} columns")
            break
    if df is None:
        raise RuntimeError(f"Could not parse the CSV: {path}")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    m = mapear_colunas(df)
    out = pd.DataFrame({
        "Ano":   pd.to_numeric(df[m["ano"]], errors="coerce"),
        "Bioma": df[m["bioma"]].astype(str).str.strip(),
        "area":  pd.to_numeric(df[m["area"]], errors="coerce"),
        "cx":    pd.to_numeric(df[m["x"]], errors="coerce"),
        "cy":    pd.to_numeric(df[m["y"]], errors="coerce"),
    }).dropna(subset=["Ano", "area", "cx", "cy"])
    out["Ano"] = out["Ano"].astype(int)
    out["grupo"] = regime

    espaco = detectar_espaco(out["cx"].values, out["cy"].values)
    print(f"  [{regime}] columns: {m}")
    print(f"  [{regime}] detected space: {espaco} "
          f"(x: {out.cx.min():.1f}..{out.cx.max():.1f} | y: {out.cy.min():.1f}..{out.cy.max():.1f})")

    if espaco == "degrees":
        # geographic input -> reproject to Albers
        tr = Transformer.from_crs(SRC_GEO, ALBERS, always_xy=True)
        xm, ym = tr.transform(out["cx"].values, out["cy"].values)
        out["X_km"] = np.asarray(xm) / 1000.0
        out["Y_km"] = np.asarray(ym) / 1000.0
    elif espaco == "albers_m":
        out["X_km"] = out["cx"] / 1000.0
        out["Y_km"] = out["cy"] / 1000.0
    else:  # albers_km
        out["X_km"] = out["cx"]
        out["Y_km"] = out["cy"]

    # always keep lon/lat available (for maps and for the legacy comparison)
    tr_back = Transformer.from_crs(ALBERS, SRC_GEO, always_xy=True)
    lon, lat = tr_back.transform(out["X_km"].values * 1000.0, out["Y_km"].values * 1000.0)
    out["lon"] = lon
    out["lat"] = lat

    return out[["Ano", "Bioma", "grupo", "area", "X_km", "Y_km", "lon", "lat"]]


# ============================================================
# 2. CORE COMPUTATION (generic: runs on any pair of axes)
# ============================================================
# Written generically so that the SAME implementation runs in Albers and in
# degrees. This way, the sensitivity difference we observe comes from the
# CONVENTION, not from two separate implementations.

def _euclid(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_all(df, xcol, ycol, scale=1.0):
    """
    Full pipeline in a given coordinate space.
    scale: factor to convert the native unit to km
           (Albers -> 1.0; degrees -> KM_PER_DEG, for readability only).
    """
    d = df.copy()

    # ---- (1) "Brazil" centroid: area-weighted barycenter of the biomes ----
    rows = []
    for (ano, grupo), sub in d.groupby(["Ano", "grupo"], sort=True):
        w = sub["area"].to_numpy(float)
        wsum = w.sum()
        rows.append({
            "Ano": ano, "grupo": grupo,
            "br_x": np.sum(w * sub[xcol].to_numpy(float)) / wsum if wsum else np.nan,
            "br_y": np.sum(w * sub[ycol].to_numpy(float)) / wsum if wsum else np.nan,
            "br_area_ha": wsum,
        })
    br = pd.DataFrame(rows).sort_values(["grupo", "Ano"]).reset_index(drop=True)
    d = d.merge(br, on=["Ano", "grupo"], how="left")

    # each biome's relative weight in that year/context's national total
    d["w"] = d["area"] / d["br_area_ha"]

    # ---- (2) Influence: weight x distance from the national centroid ----
    d["dist_to_br_km"] = _euclid(d[xcol], d[ycol], d["br_x"], d["br_y"]) * scale
    d["influence_mag_km"] = d["w"] * d["dist_to_br_km"]
    d["influence_vec_x"] = d["w"] * (d[xcol] - d["br_x"]) * scale
    d["influence_vec_y"] = d["w"] * (d[ycol] - d["br_y"]) * scale

    den = d.groupby(["Ano", "grupo"])["influence_mag_km"].transform("sum")
    d["influence_share"] = np.where(den > 0, d["influence_mag_km"] / den, np.nan)

    # ---- (3) LOO: national centroid shift when the biome is removed ----
    loo_rows = []
    for (ano, grupo), sub in d.groupby(["Ano", "grupo"], sort=True):
        x_full, y_full = sub["br_x"].iloc[0], sub["br_y"].iloc[0]
        w = sub["area"].to_numpy(float)
        x = sub[xcol].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        sw, swx, swy = w.sum(), np.sum(w * x), np.sum(w * y)

        for i, bioma in enumerate(sub["Bioma"].tolist()):
            sw_wo = sw - w[i]
            if sw_wo <= 0:
                x_wo = y_wo = shift = np.nan   # single biome: LOO undefined
            else:
                x_wo = (swx - w[i] * x[i]) / sw_wo
                y_wo = (swy - w[i] * y[i]) / sw_wo
                shift = _euclid(x_full, y_full, x_wo, y_wo) * scale
            loo_rows.append({"Ano": ano, "grupo": grupo, "Bioma": bioma,
                             "br_x_full": x_full, "br_y_full": y_full,
                             "br_x_wo": x_wo, "br_y_wo": y_wo,
                             "loo_shift_km": shift})
    loo = pd.DataFrame(loo_rows).sort_values(["grupo", "Ano", "Bioma"]).reset_index(drop=True)

    # ---- (4) Decomposition of the year-to-year shift (the paper's "novelty") ----
    #   K = w_bar * (C_b,t - C_b,t-1)         <- fire MIGRATION within the biome
    #     + (delta w) * (C_b_bar - C_BR,t-1)  <- REWEIGHTING across biomes
    prev = d[["Ano", "grupo", "Bioma", "w", xcol, ycol]].copy()
    prev["Ano"] += 1                                   # pair t with t-1
    prev = prev.rename(columns={"w": "w_prev", xcol: "x_prev", ycol: "y_prev"})
    dd = d.merge(prev, on=["Ano", "grupo", "Bioma"], how="left")

    br_prev = br.rename(columns={"br_x": "br_x_prev", "br_y": "br_y_prev"}).copy()
    br_prev["Ano"] += 1
    dd = dd.merge(br_prev[["Ano", "grupo", "br_x_prev", "br_y_prev"]],
                  on=["Ano", "grupo"], how="left")

    dd["dbr_x"] = (dd["br_x"] - dd["br_x_prev"]) * scale
    dd["dbr_y"] = (dd["br_y"] - dd["br_y_prev"]) * scale
    dd["dbr_norm"] = np.sqrt(dd["dbr_x"] ** 2 + dd["dbr_y"] ** 2)

    dd["w_bar"] = 0.5 * (dd["w"] + dd["w_prev"])
    dd["dw"] = dd["w"] - dd["w_prev"]
    dd["dx_b"] = (dd[xcol] - dd["x_prev"]) * scale
    dd["dy_b"] = (dd[ycol] - dd["y_prev"]) * scale
    dd["x_bar"] = 0.5 * (dd[xcol] + dd["x_prev"])
    dd["y_bar"] = 0.5 * (dd[ycol] + dd["y_prev"])

    dd["K_move_x"] = dd["w_bar"] * dd["dx_b"]                              # migration
    dd["K_move_y"] = dd["w_bar"] * dd["dy_b"]
    dd["K_weight_x"] = dd["dw"] * (dd["x_bar"] - dd["br_x_prev"]) * scale  # reweighting
    dd["K_weight_y"] = dd["dw"] * (dd["y_bar"] - dd["br_y_prev"]) * scale

    dd["K_x"] = dd["K_move_x"] + dd["K_weight_x"]
    dd["K_y"] = dd["K_move_y"] + dd["K_weight_y"]

    # projection onto the direction of the national shift (positive = pushes with it)
    dd["K_proj"] = np.where(dd["dbr_norm"] > 0,
                            (dd["K_x"] * dd["dbr_x"] + dd["K_y"] * dd["dbr_y"]) / dd["dbr_norm"],
                            np.nan)

    influence = d[["Ano", "grupo", "Bioma", "area", "w", xcol, ycol,
                   "br_x", "br_y", "dist_to_br_km", "influence_mag_km",
                   "influence_share", "influence_vec_x", "influence_vec_y"]].copy()

    contrib = dd[["Ano", "grupo", "Bioma", "w", "w_prev", "dw",
                  "dbr_x", "dbr_y", "dbr_norm",
                  "K_move_x", "K_move_y", "K_weight_x", "K_weight_y",
                  "K_x", "K_y", "K_proj"]].sort_values(
                      ["grupo", "Ano", "Bioma"]).reset_index(drop=True)

    return br, influence, loo, contrib


# ============================================================
# 3. EXECUTION
# ============================================================

def carregar_unico(path):
    """Reads the consolidated file (with a 'grupo' column) from 03_biome_centroids_from_patches.py."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    m = mapear_colunas(df)
    if "grupo" not in [c.lower() for c in df.columns]:
        raise ValueError(f"Consolidated file has no 'grupo' column: {path}")
    gcol = [c for c in df.columns if c.lower() == "grupo"][0]

    out = pd.DataFrame({
        "Ano":   pd.to_numeric(df[m["ano"]], errors="coerce").astype("Int64"),
        "Bioma": df[m["bioma"]].astype(str).str.strip(),
        "grupo": df[gcol].astype(str).str.strip(),
        "area":  pd.to_numeric(df[m["area"]], errors="coerce"),
        "cx":    pd.to_numeric(df[m["x"]], errors="coerce"),
        "cy":    pd.to_numeric(df[m["y"]], errors="coerce"),
    }).dropna()
    out["Ano"] = out["Ano"].astype(int)

    espaco = detectar_espaco(out["cx"].values, out["cy"].values)
    print(f"  detected space: {espaco} | columns: {m} | contexts: {sorted(out.grupo.unique())}")
    if espaco == "degrees":
        tr = Transformer.from_crs(SRC_GEO, ALBERS, always_xy=True)
        xm, ym = tr.transform(out["cx"].values, out["cy"].values)
        out["X_km"], out["Y_km"] = np.asarray(xm)/1000.0, np.asarray(ym)/1000.0
    elif espaco == "albers_m":
        out["X_km"], out["Y_km"] = out["cx"]/1000.0, out["cy"]/1000.0
    else:
        out["X_km"], out["Y_km"] = out["cx"], out["cy"]

    tr_b = Transformer.from_crs(ALBERS, SRC_GEO, always_xy=True)
    lon, lat = tr_b.transform(out["X_km"].values*1000.0, out["Y_km"].values*1000.0)
    out["lon"], out["lat"] = lon, lat
    return out[["Ano","Bioma","grupo","area","X_km","Y_km","lon","lat"]]


def checar_consistencia(br, path):
    """
    CRITICAL CHECK: the national centroid reconstructed from the biome
    centroids must be IDENTICAL to the one computed directly from the
    patches (a weighted mean is associative). A divergence > ~1 km flags an
    error somewhere in the pipeline.
    """
    if not path or not os.path.exists(path):
        print("\n  (consistency check skipped: CHECK_FILE not found)")
        return
    chk = pd.read_csv(path)
    chk.columns = [c.strip() for c in chk.columns]
    # normalize the context label between the two files
    mapa = {"natural":"Natural","use":"Use","anthropogenic_use":"Anthropogenic_use"}
    chk["grupo"] = chk["Regime"].astype(str).str.strip().str.lower().map(lambda v: mapa.get(v, v))
    b = br.copy()
    b["gl"] = b["grupo"].astype(str).str.lower()
    chk["gl"] = chk["grupo"].astype(str).str.lower()
    # match by year; if labels don't line up, matching falls back to context order
    mg = b.merge(chk, on=["Ano","gl"], suffixes=("_bioma","_patch"))
    if mg.empty:
        print("\n  WARNING: could not match contexts between the two files -- "
              "consistency check not performed.")
        return
    d = np.hypot(mg["br_X_km"] - mg["Xc_km"], mg["br_Y_km"] - mg["Yc_km"])
    print("\n=== CONSISTENCY: national centroid (via biomes) vs (direct from patches) ===")
    print(f"  median difference: {d.median():.4f} km | max: {d.max():.4f} km  (n={len(d)})")
    if d.max() < 1.0:
        print("  OK -- both paths converge. The pipeline is coherent.")
    else:
        print("  WARNING -- divergence above 1 km. Investigate: the biome centroids")
        print("  may not be in Albers, or the weights (area) differ between the files.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if SINGLE_FILE:
        print("Reading the consolidated file (centroids computed IN Albers from the patches):")
        df = carregar_unico(SINGLE_FILE)
    else:
        print("Reading per-context files (LEGACY -- centroids averaged in degrees):")
        df = pd.concat([carregar(p, reg) for reg, p in FILES.items()], ignore_index=True)
    print(f"\nTotal: {len(df)} records | years {df.Ano.min()}-{df.Ano.max()} | "
          f"biomes: {df.Bioma.nunique()} | contexts: {sorted(df.grupo.unique())}")

    # ---- OFFICIAL VERSION: everything in Albers (km) ----
    br_a, inf_a, loo_a, con_a = compute_all(df, "X_km", "Y_km", scale=1.0)
    br_a = br_a.rename(columns={"br_x": "br_X_km", "br_y": "br_Y_km"})

    # national centroid's lon/lat, for maps
    tr_back = Transformer.from_crs(ALBERS, SRC_GEO, always_xy=True)
    lo, la = tr_back.transform(br_a["br_X_km"].values * 1000.0,
                               br_a["br_Y_km"].values * 1000.0)
    br_a["br_lon"], br_a["br_lat"] = lo, la

    n_nan = int(loo_a["loo_shift_km"].isna().sum())
    if n_nan:
        print(f"  WARNING: {n_nan} undefined LOO values (year/context with a single biome).")

    checar_consistencia(br_a, CHECK_FILE)

    # ---- LEGACY VERSION (euclidean in DEGREES), for the sensitivity comparison only ----
    # Runs the old convention on the SAME centroids -> isolates the effect of the metric.
    _, inf_d, loo_d, _ = compute_all(df, "lon", "lat", scale=KM_PER_DEG)

    comp = (loo_a[["Ano", "grupo", "Bioma", "loo_shift_km"]]
            .merge(loo_d[["Ano", "grupo", "Bioma", "loo_shift_km"]],
                   on=["Ano", "grupo", "Bioma"], suffixes=("_albers", "_graus"))
            .merge(inf_a[["Ano", "grupo", "Bioma", "influence_mag_km", "influence_share"]]
                   .merge(inf_d[["Ano", "grupo", "Bioma", "influence_mag_km", "influence_share"]],
                          on=["Ano", "grupo", "Bioma"], suffixes=("_albers", "_graus")),
                   on=["Ano", "grupo", "Bioma"]))

    comp["loo_diff_pct"] = 100 * (comp.loo_shift_km_albers - comp.loo_shift_km_graus) \
                           / comp.loo_shift_km_graus.replace(0, np.nan)
    comp["influence_diff_pct"] = 100 * (comp.influence_mag_km_albers - comp.influence_mag_km_graus) \
                                 / comp.influence_mag_km_graus.replace(0, np.nan)
    comp["share_diff_pp"] = 100 * (comp.influence_share_albers - comp.influence_share_graus)

    # ---- save ----
    p = lambda n: os.path.join(OUT_DIR, f"{OUT_PREFIX}_{n}.csv")
    br_a.to_csv(p("brasil_reconstruido"), index=False)
    inf_a.to_csv(p("influencia_anual"), index=False)
    loo_a.to_csv(p("loo_shift"), index=False)
    con_a.to_csv(p("contrib_delta"), index=False)
    comp.to_csv(p("sensibilidade_graus_vs_albers"), index=False)

    # ---- numbers that go into the response letter ----
    print("\n=== SENSITIVITY TO THE CONVENTION (euclidean in degrees -> Albers) ===")
    for col, nome in [("loo_diff_pct", "LOO (shift)"),
                      ("influence_diff_pct", "Influence (w x d)")]:
        v = comp[col].abs().dropna()
        if len(v):
            print(f"  {nome:22s}: median {v.median():.2f}% | p95 {v.quantile(.95):.2f}% | max {v.max():.2f}%")
    sh = comp["share_diff_pp"].abs().dropna()
    print(f"  {'Influence share':22s}: median {sh.median():.3f} pp | max {sh.max():.3f} pp")

    print("\n=== DOES THE BIOME RANKING CHANGE? ===")
    trocas, top1, total = 0, 0, 0
    linhas_trocadas = []
    for (ano, grupo), g in comp.groupby(["Ano", "grupo"]):
        ra = g.sort_values("influence_share_albers", ascending=False).Bioma.tolist()
        rd = g.sort_values("influence_share_graus", ascending=False).Bioma.tolist()
        total += 1
        if ra != rd:
            trocas += 1
            difs = sorted({ra[i] for i in range(len(ra)) if ra[i] != rd[i]} |
                          {rd[i] for i in range(len(rd)) if ra[i] != rd[i]})
            linhas_trocadas.append((ano, grupo, difs))
        if ra[0] != rd[0]:
            top1 += 1
    print(f"  Identical ordering in {total - trocas}/{total} year x context combinations.")
    print(f"  Top rank (dominant biome) differs in {top1}/{total}.")
    if linhas_trocadas:
        print("  Swaps (check whether the manuscript text claims something about these biomes):")
        for ano, grupo, difs in linhas_trocadas[:15]:
            print(f"    {ano} {grupo}: {difs}")
        if len(linhas_trocadas) > 15:
            print(f"    ... and {len(linhas_trocadas)-15} more")

    print(f"\nFiles saved in: {OUT_DIR}/")


if __name__ == "__main__":
    main()