"""
07_figure_separability_axis.py

WHAT THIS FIGURE SHOWS
------------------------
The two clouds of annual centroids (40 per context: Natural and Use),
PROJECTED onto the geographic Cerrado-Amazon axis, shown as two shifted
distributions. This is the direct visual reading of the AUC: how much the
two distributions separate along the axis that matters.

WHY THIS FORM (and not a ROC curve or a KDE density plot):
  - Two shifted distributions are more legible to an ecology audience than a
    ROC curve (which shows classification performance, not geography).
  - We do NOT use KDE: with 40 points, a smoothed density is misleading
    (that is exactly why OVL was abandoned). We use a histogram plus the
    individual points visible as a rug, honest about the sample size.

INPUT: same files and anchors as 06_separation_metrics_auc.py.
OUTPUT: Figure_7_separability_axis.png (300 dpi) and .pdf (vector, for the journal).

RUNS IN: Jupyter. Needs numpy, pandas, matplotlib.

OPEN ITEM (flag, not resolved by this cleaning pass): the frozen AUC-axis
reference value below (0.89) does not currently match the value produced by
06_separation_metrics_auc.py in one run (0.904 reported for this figure vs
0.891 from the metrics script). The two scripts must agree before the paper
number is final -- see the sanity-check warning this script prints if the
recomputed value drifts from AUC_EIXO_REF by more than 0.02.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# BLOCK 1 -- CONFIGURATION (identical to the metrics script)
# ---------------------------------------------------------------------
INPUT_FILE = "centroids_observed_albers.csv"   # Ano, Regime, Xc_km, Yc_km

# GEOGRAPHIC anchors of the biomes in Albers km (ESRI:102033) -- the same
# ones used in 06_separation_metrics_auc.py (IBGE 2025 polygon centroids):
CERRADO_ANCHOR  = (988.146, 2060.081)
AMAZONIA_ANCHOR = (-63.053, 2997.224)

# Colors (kept as originally set, described as consistent with the rest of
# the paper: Natural in green, Use in magenta). OPEN ITEM: the transition
# notes flag these as still pending a check against the paper's actual
# figure palette -- this cleaning pass did not have that palette to verify
# against, so the values are left unchanged rather than guessed.
COR_NAT = "#7DA63B"   # olive green (fires in natural vegetation)
COR_USE = "#C2185B"   # magenta (fires in anthropogenic land use)

OUT_PNG = "Figure_7_separability_axis.png"
OUT_PDF = "Figure_7_separability_axis.pdf"

# AUC values already computed elsewhere (to annotate the figure); recomputed
# below regardless, but kept here as a frozen sanity-check reference:
AUC_EIXO_REF = 0.89
AUC_DISC_REF = 0.98


# ---------------------------------------------------------------------
# BLOCK 2 -- LOAD AND PROJECT (same logic as the metrics script)
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
df["Regime"] = df["Regime"].replace({"Anthropogenic_use": "Use", "Anthropogenic": "Use"})

nat = df[df.Regime == "Natural"].sort_values("Ano")
use = df[df.Regime == "Use"].sort_values("Ano")
Pn = nat[["Xc_km", "Yc_km"]].to_numpy()
Pu = use[["Xc_km", "Yc_km"]].to_numpy()

# unit vector Cerrado -> Amazon (external, geographic)
origem = np.array(CERRADO_ANCHOR, float)
eixo = np.array(AMAZONIA_ANCHOR, float) - origem
eixo = eixo / np.linalg.norm(eixo)

def projeta(P):
    """Coordinate along the axis, in km, with Cerrado approx 0 and +Amazon."""
    return (P - origem) @ eixo

proj_nat = projeta(Pn)
proj_use = projeta(Pu)


# ---------------------------------------------------------------------
# BLOCK 3 -- AUC (recomputed here so the figure is self-consistent)
# ---------------------------------------------------------------------
def auc_1d(a_nat, a_use):
    diff = a_use[:, None] - a_nat[None, :]
    return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size

auc_eixo = auc_1d(proj_nat, proj_use)

# sanity check against the frozen reference value
if abs(auc_eixo - AUC_EIXO_REF) > 0.02:
    print(f"WARNING: recomputed AUC ({auc_eixo:.3f}) diverges from the frozen "
          f"reference ({AUC_EIXO_REF}). Check the anchors/input file before using this figure.")
else:
    print(f"AUC-axis = {auc_eixo:.3f} (matches the frozen reference {AUC_EIXO_REF}).")


# ---------------------------------------------------------------------
# BLOCK 4 -- THE FIGURE
# ---------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(8, 4.5))

# bins shared by both distributions
lo = min(proj_nat.min(), proj_use.min())
hi = max(proj_nat.max(), proj_use.max())
bins = np.linspace(lo, hi, 16)

# histograms (density), semi-transparent and overlaid
ax.hist(proj_nat, bins=bins, density=True, alpha=0.55, color=COR_NAT,
        edgecolor="white", linewidth=0.5, label="Natural Vegetation")
ax.hist(proj_use, bins=bins, density=True, alpha=0.55, color=COR_USE,
        edgecolor="white", linewidth=0.5, label="Anthropogenic Land Use")

# rug: the 40 individual points of each context, so the sample isn't hidden
ymax = ax.get_ylim()[1]
ax.plot(proj_nat, np.full_like(proj_nat, -0.04*ymax), "|",
        color=COR_NAT, markersize=10, markeredgewidth=1.4)
ax.plot(proj_use, np.full_like(proj_use, -0.08*ymax), "|",
        color=COR_USE, markersize=10, markeredgewidth=1.4)

# vertical lines at each context's median
ax.axvline(np.median(proj_nat), color=COR_NAT, ls="--", lw=1.5, alpha=0.9)
ax.axvline(np.median(proj_use), color=COR_USE, ls="--", lw=1.5, alpha=0.9)

# AUC annotation
ax.text(0.03, 0.95,
        f"AUC (geographic axis) = {auc_eixo:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))

# axis-extreme labels (what 'low' and 'high' mean on this axis)
ax.set_xlabel("Position along the Cerrado-Amazon axis (km)")
ax.set_ylabel("Density")
# geographic-orientation arrows/labels at the extremes
ax.annotate("<- toward Cerrado", xy=(0.02, -0.16), xycoords="axes fraction",
            ha="left", va="top", fontsize=9, color="0.35")
ax.annotate("toward Amazon ->", xy=(0.98, -0.16), xycoords="axes fraction",
            ha="right", va="top", fontsize=9, color="0.35")

# FIX (cosmetic pendency from the transition notes): extra headroom above the
# bars (top=1.25*ymax) so the AUC box and the legend stop overlapping the
# histogram. Previously only 'bottom' was set here, leaving 'top' at the
# bars' own peak.
ax.set_ylim(bottom=-0.10*ymax, top=1.25*ymax)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
ax.set_title("Separability of fire contexts along the Cerrado-Amazon axis (1985-2024)",
             fontsize=11.5)
ax.grid(axis="y", ls=":", alpha=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")   # vector, preferred for submission
print(f"Figure saved: {OUT_PNG} and {OUT_PDF}")
print(f"  Natural: median {np.median(proj_nat):.0f} km | Use: median {np.median(proj_use):.0f} km")
print(f"  Median separation: {abs(np.median(proj_use)-np.median(proj_nat)):.0f} km along the axis")


# =====================================================================
# WHAT CAN BREAK (and how you would notice)
# =====================================================================
# 1. AUC-divergence WARNING -> wrong anchors or input file. The figure is
#    still produced, but the annotated number won't match the text. Fix
#    before using the figure.
#
# 2. If the two distributions look almost fully overlapped in the figure but
#    the annotated AUC is high (0.89), that is an inconsistency -> likely a
#    wrong axis (non-geographic anchors). The visual and the number must
#    tell the SAME story.
#
# 3. A histogram with 40 points and 15 bins looks "jagged". That is expected
#    and honest (we deliberately do not smooth). For less visual noise,
#    reduce the bin count (np.linspace(lo, hi, 12)); do NOT switch to KDE.
#
# 4. Colors: adjust COR_NAT/COR_USE to match the rest of the paper's figures
#    exactly (the green/magenta from the graphical abstract) -- still an
#    open item, see the note in BLOCK 1.
#
# 5. This figure shows the AXIS AUC (0.89). The DISCRIMINANT AUC (0.98) does
#    NOT belong here -- it is the upper bound and lives in the text/SI.
#    Mixing the two in this figure would confuse readers (this figure is
#    about the geographic axis, not the optimal direction).
# =====================================================================