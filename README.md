# Directional Centroid Trajectories of Fire Activity Across Brazilian Biomes (1985–2024)

Code and derived data for the analysis in:

> Ramos Neto, M. & Hofmann, G. S. (2026). *Directional Centroid Trajectories
> Reveal Shifting Fire Activity Across Brazilian Biomes.* Fire Ecology (in review).

This repository reproduces the figures and quantitative results of the paper from
derived data. It does **not** redistribute the raw MapBiomas rasters (see *Data sources*).

---

## What this repository contains

- **`scripts/`** — the analysis pipeline, numbered in execution order (01–07):
  from burn-scar patches to national centroids, biome decomposition, uncertainty
  quantification, separability metrics, and the axis-separability figure.
- **`data_derived/`** — intermediate outputs (annual centroids, biome centroids,
  bootstrap/permutation results, separation metrics) sufficient to regenerate the
  paper's numbers and figures **without** reprocessing the raw rasters.
- **`figures/`** — final figures as produced by the scripts.
- **`docs/pipeline.md`** — end-to-end description of the workflow and the
  validation chain.

## Data sources (not redistributed here)

- **MapBiomas Fire, Collection 4** — annual burned-area products, clipped to the
  official Brazilian boundary. Publicly available at https://mapbiomas.org.
  Class attribution follows the MapBiomas land-use/land-cover structure embedded
  in the Fire product.
- **Biome boundaries** — IBGE (2025), *Biomas e Sistema Costeiro-Marinho do Brasil,
  1:250,000, first revision* (Notas metodológicas 01/2025).
- Processed patches data are available in https://zenodo.org/records/22029092

## Methods summary

All spatial metrics are computed in the South America Albers Equal-Area Conic
projection (ESRI:102033; SIRGAS 2000, EPSG:4674). Burn patches are retained when
composed of ≥3 rook-connected pixels (isolated 1–2-pixel detections removed as
detection noise). Each patch is assigned to the biome containing its centroid,
using the IBGE (2025) delimitation; patches in cartographic gaps are assigned to
the nearest biome.

**Validation chain (see `docs/pipeline.md`):** national totals reproduce the
MapBiomas reference to within 0.001%; the noise filter removes 0.30–0.41% of gross
burned area; biome attribution preserves the filtered total exactly (0.000% over
40 years).

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate fire-centroids
# run scripts in order; each reads from data_derived/ and writes outputs there
python scripts/05_uncertainty_separation.py
python scripts/06_separation_metrics_auc.py
python scripts/07_figure_separability_axis.py
```

Paths and external anchors (biome geographic centroids) are set at the top of each
script — edit them to match your local data layout.

## Citing

If you use this code or data, please cite both the paper (above) and this
repository via its archived DOI (see `CITATION.cff`).

## License

Code: [MIT]. Derived data: [CC-BY-4.0]. See `LICENSE`.

---

> ⚠️ **Status:** this repository accompanies a manuscript under review. The archived
> Zenodo release and its DOI will be created at submission, once all numbers are final.
> Until then, contents may change.
