# Analysis pipeline

End-to-end description of the workflow that produces the figures and quantitative
results in Ramos Neto & Hofmann (2026). Scripts are in `scripts/`, numbered in
execution order; derived outputs are in `data_derived/`. Inline code comments are
in Portuguese (the authors' working language); this document and each script's
header summarise the logic in English.

All spatial metrics are computed in the **South America Albers Equal-Area Conic**
projection (ESRI:102033; SIRGAS 2000, EPSG:4674). This is a deliberate choice: the
manuscript reports equal-area metrics throughout, and standardising every step to
this projection avoids the mixed degree/projected conventions of earlier drafts.

---

## Data sources (not redistributed in this repository)

- **MapBiomas Fire, Collection 4** — annual burned-area rasters, already clipped to
  the official Brazilian boundary. Land-use/land-cover class attribution uses the
  MapBiomas structure embedded in the Fire product; no separate LULC collection is
  required. Public: https://mapbiomas.org
- **Biome boundaries** — IBGE (2025), *Biomas e Sistema Costeiro-Marinho do Brasil,
  compatible with 1:250,000, first revision* (Notas metodológicas 01/2025). The
  geographic centroids of the Cerrado and Amazon biome polygons, computed from this
  layer in ESRI:102033, define the external axis used for the separability metric.

---

## Step-by-step

### 01 — Build centroids and vectors
`scripts/01_build_centroids_and_vectors.py`
Reads the burn-scar patches and computes, for each year and context (Natural
Vegetation vs Anthropogenic Land Use), the area-weighted national centroid. Patches
are reprojected to Albers **before** any aggregation, so centroids are equal-area
centers of mass rather than averages of geographic coordinates.

### 02 — Filter small patches
`scripts/02_filter_small_patches.py`
Retains burn patches composed of **≥3 rook-connected pixels**; isolated 1–2-pixel
detections are removed to suppress detection noise. Rook (4-neighbour) connectivity
is the conservative criterion. This removes 0.30–0.41% of gross burned area
(comparable across contexts; see *Validation chain*), and is documented in the
manuscript Methods as noise removal rather than data loss.

### 03 — Biome centroids from patches
`scripts/03_biome_centroids_from_patches.py`
Computes, directly from the patches and in Albers, the area-weighted centroid of
each biome × year × context. Each patch is assigned to the biome containing its
centroid, using the IBGE (2025) delimitation; patches whose centroids fall in
cartographic gaps between the national boundary and the biome polygons are assigned
to the **nearest biome**, following IBGE's directive that boundary features be
annexed to the nearest adjacent biome. Streams the patches in batches, accumulating
weighted sums per group, so memory scales with the number of groups, not patches.

### 04 — Biome contribution (decomposition)
`scripts/04_biome_contribution_albers.py`
Decomposes national centroid position and interannual shifts into biome
contributions: leave-one-out sensitivity, influence as weight × distance (each
biome's share of burned area times its distance from the national centroid), and
the reweighting vs within-biome-migration split. All computed in Albers.

### 05 — Uncertainty of the separation
`scripts/05_uncertainty_separation.py`
Quantifies the inter-context separation and its uncertainty:
- annual inter-centroid distance, its mean, 95% bootstrap confidence interval, and
  range;
- a **within-year label-permutation test** (Natural/Anthropogenic labels reassigned
  within each year, annual spatial structure held fixed) yielding the null
  distribution of the separation;
- per-centroid position uncertainty (95% radial spread) for each context.

**Note on the confidence interval.** The bootstrap resamples patches within each
year and does not model spatial autocorrelation among patches or temporal
dependence among years; the interval is therefore a **lower bound** on uncertainty.
The inference of separation rests on the within-year permutation test (robust to
between-year dependence by construction) and on the effect size, not on the interval.

### 06 — Separability metric (AUC)
`scripts/06_separation_metrics_auc.py`
Quantifies separability of the two centroid clouds with a **rank-based AUC**, which
has no free smoothing parameter (unlike the kernel overlap coefficient used in an
earlier draft, which was strongly bandwidth-dependent given only 40 annual centroids
per context). Each annual centroid is projected onto the external geographic
Cerrado–Amazon axis; the AUC is the probability that an Anthropogenic centroid lies
further toward the Amazon than a Natural one. Reported both along the geographic axis
and, as an upper bound, along the data-driven direction of maximal separation.
Confidence intervals by bootstrapping over years; significance by within-year
permutation.

### 07 — Separability figure
`scripts/07_figure_separability_axis.py`
Produces the axis-separability figure: the two contexts' annual centroids projected
onto the Cerrado–Amazon axis, shown as overlapping distributions with individual
years as tick marks and the AUC annotated. Uses histograms plus visible individual
points rather than kernel density, honest about the 40-point sample.

---

## Validation chain

The pipeline is validated end to end (reported in the manuscript Supplement):

1. **Fidelity to source** — national totals reproduce the MapBiomas Fire reference
   to within **0.001%**.
2. **Noise filter** — the ≥3-pixel rook filter removes **0.41%** of gross burned
   area in Anthropogenic Land Use and **0.30%** in Natural Vegetation; the near-equal
   proportions mean the filter does not induce the separation reported.
3. **Biome attribution** — the biome-summed burned area equals the filtered national
   total **exactly in all 40 years (0.000%)**: no orphaned patches, no double-counting.
4. **Pipeline associativity** — the national centroid reconstructed from biome-level
   aggregation matches the direct patch-level computation (residual ~66 m,
   attributable to datum handling).
5. **Projection sensitivity** — degrees vs. Albers differ by a median of ~2%, and the
   leading biome is identical in 80/80 year × context combinations.

---

## Reproducing the numbers

The scripts read from and write to `data_derived/`. With the derived files provided,
steps 05–07 regenerate the paper's separation statistics, separability metrics, and
figure without reprocessing the raw rasters. Steps 01–04 require the MapBiomas
patches (not redistributed here) and are included for full transparency of the
pipeline. Paths and the biome-centroid anchors are set at the top of each script.

---

> **Status.** This repository accompanies a manuscript under review. The archived
> Zenodo release and DOI will be created at submission, once all numbers are final.
> Until then, contents may change.
