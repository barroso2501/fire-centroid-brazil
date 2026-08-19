"""
06_separation_metrics_auc.py

WHAT THIS SCRIPT DOES, IN ONE SENTENCE:
  Measures how separated the two clouds of ANNUAL centroids are (40 points
  per context) using a metric that does NOT depend on smoothing (rank-based
  AUC), and keeps a 1D OVL only as a continuity number with the earlier 0.17
  -- now measured along the Cerrado-Amazon axis.

WHY WE CHANGED METRIC:
  The old OVL (0.17) came from estimating a density (KDE) over ~40 points
  per context. Density estimation with 40 points is unstable: the number
  swings a lot with the bandwidth choice. Not robust enough to defend under
  review. The separation AUC solves this: it is based on RANKING, not
  density, so it has no free parameter to swing.

INTERPRETING THE AUC:
  AUC = probability that, drawing one Anthropogenic-use centroid and one
  Natural centroid at random, the Use one lies further toward the Amazon
  along the Cerrado-Amazon axis.
  AUC = 0.5  -> indistinguishable clouds (full overlap)
  AUC = 1.0  -> perfect separation (no overlap)
  It answers the same question as OVL, just measured robustly.

RUNS IN: Jupyter. Only needs numpy and pandas (no scipy, no sklearn).
  Can run in one go (fast -- 80 points, not millions).

NOTE ON OVL 1D (kept in this script). Unlike the 2D histogram OVL that was
fully retired from 05_uncertainty_separation.py, the 1D version here is kept
deliberately as a SECONDARY continuity check against the earlier manuscript
number (0.17), explicitly not reported as the paper's separability metric.
AUC is the metric that goes in the text; OVL 1D is diagnostic only. This
script was already functionally finalized before this cleaning pass, so only
language and formatting were standardized here -- no logic was changed.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# BLOCK 1 -- CONFIGURATION (this is what you edit)
# ---------------------------------------------------------------------

# File with the national ANNUAL centroids per context, already in Albers (km).
# This is the 'centroids_observed_albers.csv' output of 05_uncertainty_separation.py.
# Expected columns: Ano, Regime, Xc_km, Yc_km   (Regime = 'Natural' / 'Use')
INPUT_FILE = "centroids_observed_albers.csv"

# CERRADO-AMAZON AXIS (EXTERNAL to the fire data -- important!)
# For the metric to be honest, the axis must NOT be defined by the fire
# centroids themselves (that would be circular: pick the direction that
# separates, then claim separation). Use the GEOGRAPHIC CENTROIDS of the
# Cerrado and Amazon biome polygons (from the IBGE 2025 mask), in Albers
# (km) -- the same anchors used for the axis projection elsewhere in the
# paper (see 07_figure_separability_axis.py).
CERRADO_ANCHOR  = (988.146, 2060.081)    # Cerrado polygon centroid, IBGE 2025, Albers km
AMAZONIA_ANCHOR = (-63.053, 2997.224)    # Amazon polygon centroid, IBGE 2025, Albers km
# If left as None, the script still runs, but falls back to an axis derived
# from the data (the direction connecting the two contexts' means). That
# axis gives the MAXIMUM possible AUC (an upper bound on separation) and is
# mildly circular -- useful for a first look, NOT for the paper's final
# number. The script warns when it is running in that fallback mode.

N_BOOT = 5000     # resamples for the CI (bootstrap over years). Cheap.
N_PERM = 9999     # permutations for the p-value (label swap within each year).
SEED   = 42       # seed for reproducibility

# Number of bins for the 1D OVL (continuity number only; AUC is the primary metric).
OVL_BINS = 30

OUT_FILE = "separation_metrics_summary.csv"

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# BLOCK 2 -- LOAD THE ANNUAL CENTROIDS
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# standardize context labels, in case they arrive under different spellings
df["Regime"] = df["Regime"].replace({"Anthropogenic_use": "Use", "Anthropogenic": "Use"})

nat = df[df.Regime == "Natural"].sort_values("Ano").reset_index(drop=True)
use = df[df.Regime == "Use"].sort_values("Ano").reset_index(drop=True)

# only use years present in BOTH contexts (needed for the within-year permutation)
anos_comuns = sorted(set(nat.Ano) & set(use.Ano))
nat = nat[nat.Ano.isin(anos_comuns)].sort_values("Ano").reset_index(drop=True)
use = use[use.Ano.isin(anos_comuns)].sort_values("Ano").reset_index(drop=True)

Pn = nat[["Xc_km", "Yc_km"]].to_numpy()   # (n_years, 2) Natural centroids
Pu = use[["Xc_km", "Yc_km"]].to_numpy()   # (n_years, 2) Use centroids
n_anos = len(anos_comuns)
print(f"Annual centroids loaded: {n_anos} years per context "
      f"({anos_comuns[0]}-{anos_comuns[-1]}).")


# ---------------------------------------------------------------------
# BLOCK 3 -- DEFINE THE AXIS AND PROJECT THE CENTROIDS ONTO IT
# ---------------------------------------------------------------------
def eixo_unitario(cerrado, amazonia, Pn, Pu):
    """
    Returns (origin, unit_vector) of the Cerrado -> Amazon axis.
    If the external anchors are given, uses them (HONEST mode).
    Otherwise, uses the direction between the two contexts' means
    (DATA-DRIVEN mode, circular, first-look only) and warns.
    """
    if cerrado is not None and amazonia is not None:
        o = np.asarray(cerrado, float)
        v = np.asarray(amazonia, float) - o
        modo = "EXTERNAL (geographic biome anchors) -- honest"
    else:
        o = Pn.mean(axis=0)
        v = Pu.mean(axis=0) - o
        modo = "DATA-DRIVEN (direction between context means) -- UPPER BOUND, circular"
    v = v / np.linalg.norm(v)
    return o, v, modo

origem, eixo, modo_eixo = eixo_unitario(CERRADO_ANCHOR, AMAZONIA_ANCHOR, Pn, Pu)
print(f"Cerrado-Amazon axis in use: {modo_eixo}")
if CERRADO_ANCHOR is None:
    print("  >> WARNING: fill in CERRADO_ANCHOR/AMAZONIA_ANCHOR with the biomes'")
    print("     geographic centroids to get the honest (non-circular) AUC.")

def projeta(P, origem, eixo):
    """Each point's scalar coordinate along the axis (Cerrado=0, +Amazon)."""
    return (P - origem) @ eixo

proj_nat = projeta(Pn, origem, eixo)
proj_use = projeta(Pu, origem, eixo)


# ---------------------------------------------------------------------
# BLOCK 4 -- THE METRICS (pure functions, easy to audit)
# ---------------------------------------------------------------------
def auc_1d(a_nat, a_use):
    """
    Separation AUC along the axis (Mann-Whitney statistic).
    = fraction of (nat, use) pairs where the Use one lies further toward the
      Amazon, counting ties as one half. No free parameter. 40x40 pairs is trivial.
    """
    diff = a_use[:, None] - a_nat[None, :]           # (n_use, n_nat)
    return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size

def auc_discriminante(Pn, Pu):
    """
    AUC along the direction of MAXIMUM separation (direction between the means).
    This is the upper bound on separability using both dimensions.
    Report it as 'at most, separation reaches this'.
    """
    v = Pu.mean(axis=0) - Pn.mean(axis=0)
    v = v / np.linalg.norm(v)
    return auc_1d(Pn @ v, Pu @ v)

def ovl_1d(a_nat, a_use, nbins):
    """
    1D OVL on the axis (histogram) -- ONLY for continuity with the earlier 0.17.
    Integral of the minimum of the two 1D densities. Depends (mildly) on nbins.
    """
    lo = min(a_nat.min(), a_use.min())
    hi = max(a_nat.max(), a_use.max())
    e = np.linspace(lo, hi, nbins + 1)
    hn, _ = np.histogram(a_nat, bins=e, density=True)
    hu, _ = np.histogram(a_use, bins=e, density=True)
    return float(np.minimum(hn, hu).sum() * (e[1] - e[0]))

def dist_media_anual(Pn, Pu):
    """Mean year-by-year distance between the two centroids (self-check: should be ~308)."""
    return float(np.mean(np.linalg.norm(Pu - Pn, axis=1)))


# ---------------------------------------------------------------------
# BLOCK 5 -- POINT ESTIMATES
# ---------------------------------------------------------------------
auc_obs      = auc_1d(proj_nat, proj_use)
auc_disc_obs = auc_discriminante(Pn, Pu)
ovl_obs      = ovl_1d(proj_nat, proj_use, OVL_BINS)
dist_obs     = dist_media_anual(Pn, Pu)

print("\n--- Point estimates ---")
print(f"AUC (axis)           : {auc_obs:.3f}   [overlap ~ {2*(1-auc_obs):.3f}]")
print(f"AUC (discriminant)   : {auc_disc_obs:.3f}   (upper bound on separation)")
print(f"OVL 1D (nbins={OVL_BINS})    : {ovl_obs:.3f}   (continuity number only)")
print(f"Mean distance        : {dist_obs:.1f} km   (self-check: should match ~308)")


# ---------------------------------------------------------------------
# BLOCK 6 -- CONFIDENCE INTERVALS (bootstrap resampling YEARS)
# ---------------------------------------------------------------------
# We resample years with replacement (each year is one paired Natural+Use
# observation) and recompute the metrics. Consistent with the rest of the
# paper and cheap. CAVEAT (same as the other script): treating years as
# independent ignores temporal autocorrelation -> the CI is a FLOOR.
def bootstrap_ic(func, *args, n=N_BOOT):
    vals = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, n_anos, size=n_anos)     # years with replacement
        vals[b] = func(idx)
    return np.percentile(vals, [2.5, 97.5])

ic_auc  = bootstrap_ic(lambda idx: auc_1d(proj_nat[idx], proj_use[idx]))
ic_ovl  = bootstrap_ic(lambda idx: ovl_1d(proj_nat[idx], proj_use[idx], OVL_BINS))
ic_dist = bootstrap_ic(lambda idx: dist_media_anual(Pn[idx], Pu[idx]))

print("\n--- 95% CI (bootstrap over years) ---")
print(f"AUC (axis)      : [{ic_auc[0]:.3f}, {ic_auc[1]:.3f}]")
print(f"OVL 1D          : [{ic_ovl[0]:.3f}, {ic_ovl[1]:.3f}]")
print(f"Mean distance   : [{ic_dist[0]:.1f}, {ic_dist[1]:.1f}] km")


# ---------------------------------------------------------------------
# BLOCK 7 -- P-VALUE (permutation: label swap WITHIN each year)
# ---------------------------------------------------------------------
# H0: within each year, it is arbitrary which centroid is 'Natural' and
# which is 'Use'. Each permutation swaps the two centroids of a year with
# probability 1/2 and recomputes the AUC. If the observed AUC is further
# from 0.5 than chance produces, the separation is real.
# NOTE: DISTANCE does not change under this swap (it is symmetric), so the
# distance p-value comes from the other script (patch-level permutation, p<0.001).
def perm_auc(n=N_PERM):
    obs_dev = abs(auc_obs - 0.5)
    count = 0
    for _ in range(n):
        troca = rng.random(n_anos) < 0.5              # which years get swapped
        a_nat = np.where(troca, proj_use, proj_nat)
        a_use = np.where(troca, proj_nat, proj_use)
        if abs(auc_1d(a_nat, a_use) - 0.5) >= obs_dev:
            count += 1
    return (count + 1) / (n + 1)                       # p with +1 correction

p_auc = perm_auc()
print("\n--- p-value (within-year permutation, AUC) ---")
print(f"p = {p_auc:.4f}   (H0: Natural/Use labels interchangeable within each year)")


# ---------------------------------------------------------------------
# BLOCK 8 -- OVL SENSITIVITY (transparency) AND SAVED SUMMARY
# ---------------------------------------------------------------------
print("\n--- 1D OVL sensitivity to bin count (why it is NOT the primary metric) ---")
for nb in (20, 30, 60, 120):
    print(f"   nbins={nb:3d} -> OVL = {ovl_1d(proj_nat, proj_use, nb):.3f}")

resumo = pd.DataFrame([
    {"metrica": "AUC_eixo",          "valor": auc_obs,      "ic_baixo": ic_auc[0],  "ic_alto": ic_auc[1],  "p": p_auc},
    {"metrica": "AUC_discriminante", "valor": auc_disc_obs, "ic_baixo": np.nan,     "ic_alto": np.nan,     "p": np.nan},
    {"metrica": "sobreposicao_2(1-AUC)", "valor": 2*(1-auc_obs), "ic_baixo": 2*(1-ic_auc[1]), "ic_alto": 2*(1-ic_auc[0]), "p": np.nan},
    {"metrica": "OVL_1D",            "valor": ovl_obs,      "ic_baixo": ic_ovl[0],  "ic_alto": ic_ovl[1],  "p": np.nan},
    {"metrica": "distancia_media_km","valor": dist_obs,     "ic_baixo": ic_dist[0], "ic_alto": ic_dist[1], "p": np.nan},
])
resumo.to_csv(OUT_FILE, index=False)
print(f"\nSummary saved to: {OUT_FILE}")
print(resumo.round(3).to_string(index=False))


# =====================================================================
# WHAT CAN BREAK (and how you would notice)
# =====================================================================
# 1. Axis anchors set to None -> falls back to the data-driven axis and WARNS
#    in the output ("DATA-DRIVEN ... circular"). How you'd notice: the axis-mode
#    line says so, and the AUC comes out equal to AUC_discriminante. For the
#    paper's number, keep CERRADO_ANCHOR/AMAZONIA_ANCHOR filled with the
#    biomes' GEOGRAPHIC centroids (from the IBGE mask), never anything
#    derived from the fire data itself.
#
# 2. Mean distance does NOT come out ~308 km -> sign that the input file is
#    not the same one behind the 308 km figure (datum, projection, or wrong
#    aggregation level). How you'd notice: the self-check prints the
#    distance; if it comes out far from 308, stop and check INPUT_FILE.
#
# 3. AUC very high with the CI pinned near 1.0 -> near-perfect separation;
#    the bootstrap CI becomes asymmetric (butts up against the ceiling of
#    1.0). This is expected with strong separation, not an error. Report
#    the AUC with that CI as is.
#
# 4. Columns under a different name (e.g. 'X_km' instead of 'Xc_km', or
#    Regime under other labels) -> KeyError while reading. How you'd notice:
#    the error points at the read_csv/column-selection line. Adjust the
#    names in BLOCK 2.
#
# 5. This script assumes the ANNUAL CENTROID level (40 points/context),
#    consistent with the 308 km figure. If the manuscript's 0.17 OVL was
#    computed at the PATCH level instead, that is a different quantity and
#    this number is not comparable to it -- flag it if so.
# =====================================================================