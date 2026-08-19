"""
biome_contribution_albers.py  (v2 — dois arquivos por regime, autodetecção de CRS)

MUDANÇAS DA v1
--------------
1. Lê DOIS arquivos (um por regime) em vez de um consolidado. O rótulo de
   regime ("Natural" / "Use") vem da configuração, não de uma coluna.
2. AUTODETECTA o espaço das coordenadas e os nomes das colunas:
      - se já estiverem em Albers (metros ou km)  -> apenas normaliza para km
      - se estiverem em graus (lon/lat)           -> reprojeta para Albers
   Assim o script funciona tanto com seus arquivos "_albers" quanto com o CSV
   antigo em graus, sem edição manual.
3. A comparação de sensibilidade (graus vs Albers) continua sendo produzida:
   quando a entrada já é Albers, o script faz a transformação INVERSA para
   graus e roda a convenção legada sobre os MESMOS centroides. Isso isola
   exatamente o efeito da MÉTRICA (euclidiana-em-graus vs euclidiana-em-Albers),
   que é a mudança que estamos fazendo.

O QUE A LÓGICA NÃO MUDA
-----------------------
Baricentro ponderado, influência (w x d), LOO e decomposição K continuam com as
MESMAS fórmulas do seu script original. Só o espaço métrico muda.

SAÍDAS (todas em km, em OUT_DIR)
  *_brasil_reconstruido.csv            centroide Brasil por ano/regime
  *_influencia_anual.csv               w, dist_to_br_km, influence_mag_km, share
  *_loo_shift.csv                      LOO em km (antes: em graus)
  *_contrib_delta.csv                  K_move / K_weight / K_proj em km
  *_sensibilidade_graus_vs_albers.csv  comparação das convenções (suplemento)

O QUE PODE QUEBRAR E COMO VOCÊ PERCEBERIA
  * Nome de coluna inesperado -> o script IMPRIME as colunas que encontrou e
    para com erro dizendo qual papel (ano/bioma/área/x/y) não conseguiu mapear.
    Se isso acontecer, preencha COLMAP na configuração manualmente.
  * CRS detectado errado -> o script IMPRIME o que detectou ("graus" / "Albers
    em metros" / "Albers em km") e a faixa de valores. CONFIRA essa linha: se
    disser "graus" para um arquivo que você sabe estar em Albers, algo está
    errado nos dados.
  * Ano/regime com um único bioma -> LOO indefinido (NaN); o script avisa quantos.
  * Se a sensibilidade der diferença mediana muito acima de ~3%, desconfie do
    CRS de entrada (ou de que os centroides não são realmente de bioma).
"""

import os
import numpy as np
import pandas as pd
from pyproj import Transformer

# ============================================================
# CONFIGURAÇÃO (edite só aqui)
# ============================================================

# MODO 1 (recomendado agora): arquivo ÚNICO gerado por biome_centroids_from_patches.py,
# que já traz a coluna 'grupo'. Centroides calculados EM Albers a partir dos patches.
SINGLE_FILE = r"D:/Projetos/Centroide/biomes/outputs_from_patches/biome_centroids_albers.csv"

# MODO 2 (legado): um arquivo por regime. Só é usado se SINGLE_FILE for None.
FILES = {
    "Natural": "centroids_nat_biome_albers.csv",
    "Use":     "centroids_use_biome_albers.csv",
}

# Verificação de consistência (opcional, mas MUITO recomendada):
# aponte para o centroids_observed_albers.csv gerado pelo uncertainty_separation_nat_vs_use.py.
# Média ponderada é associativa, então o centroide nacional reconstruído a partir dos
# biomas TEM de bater com o calculado direto dos patches. Se não bater, há erro no pipeline.
CHECK_FILE = r"D:/Projetos/Centroide/Uncertainty_separation/centroids_observed_albers.csv"

OUT_DIR    = r"D:/Projetos/Centroide/biomes/outputs_albers"
OUT_PREFIX = "centroid_contrib"

SRC_GEO = "EPSG:4674"     # SIRGAS 2000 (datum de origem confirmado nos dados)
ALBERS  = "ESRI:102033"   # South America Albers Equal Area Conic

KM_PER_DEG = 111.32       # usado APENAS para tornar a comparação legada legível

# Se a autodetecção falhar, preencha manualmente. Ex.:
# COLMAP = {"ano":"Ano","bioma":"Bioma","area":"total_area_ha","x":"X","y":"Y"}
COLMAP = None


# ============================================================
# 1. LEITURA, MAPEAMENTO DE COLUNAS E NORMALIZAÇÃO PARA ALBERS-km
# ============================================================

# nomes aceitos para cada papel (minúsculas)
# ORDEM IMPORTA: o primeiro nome encontrado vence.
# Para x/y, priorizamos POINT_X/POINT_Y (já em Albers) sobre lon/lat (graus).
# ATENÇÃO aos nomes truncados em 10 caracteres pelo export de shapefile (DBF):
#   centroid_l = LONGITUDE   |   centroid_1 = LATITUDE  (dígito UM, não letra L!)
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
    """Descobre qual coluna cumpre cada papel. Falha com mensagem clara."""
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
            f"Não consegui identificar as colunas para: {faltando}.\n"
            f"Colunas encontradas no arquivo: {list(df.columns)}\n"
            f"Preencha COLMAP na configuração."
        )
    return m


def detectar_espaco(x, y):
    """
    Descobre em que espaço as coordenadas estão, pela ordem de grandeza.
      graus  : |x| <= 180 e |y| <= 90
      km     : magnitudes na casa dos milhares
      metros : magnitudes na casa dos milhões
    """
    ax, ay = np.nanmax(np.abs(x)), np.nanmax(np.abs(y))
    if ax <= 180 and ay <= 90:
        return "graus"
    if ax < 50_000 and ay < 50_000:
        return "albers_km"
    return "albers_m"


def carregar(path, regime):
    """Lê um arquivo de regime e devolve DataFrame padronizado com X_km/Y_km."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Leitura robusta: os arquivos exportados do ArcGIS vêm em formato europeu
    # (separador ';', decimal ',', com BOM). Tentamos as combinações em ordem e
    # ficamos com a primeira que produzir mais de uma coluna.
    df = None
    for sep, dec in [(";", ","), (",", "."), (";", "."), ("\t", ".")]:
        try:
            cand = pd.read_csv(path, sep=sep, decimal=dec, encoding="utf-8-sig")
        except Exception:
            continue
        if cand.shape[1] > 1:
            df = cand
            print(f"  [{regime}] lido com sep='{sep}' decimal='{dec}' -> {cand.shape[1]} colunas")
            break
    if df is None:
        raise RuntimeError(f"Não consegui interpretar o CSV: {path}")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

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
    print(f"  [{regime}] colunas: {m}")
    print(f"  [{regime}] espaço detectado: {espaco} "
          f"(x: {out.cx.min():.1f}..{out.cx.max():.1f} | y: {out.cy.min():.1f}..{out.cy.max():.1f})")

    if espaco == "graus":
        # entrada geográfica -> reprojeta para Albers
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

    # sempre garante lon/lat (para mapas e para a comparação legada)
    tr_back = Transformer.from_crs(ALBERS, SRC_GEO, always_xy=True)
    lon, lat = tr_back.transform(out["X_km"].values * 1000.0, out["Y_km"].values * 1000.0)
    out["lon"] = lon
    out["lat"] = lat

    return out[["Ano", "Bioma", "grupo", "area", "X_km", "Y_km", "lon", "lat"]]


# ============================================================
# 2. NÚCLEO DE CÁLCULO (genérico: roda em qualquer par de eixos)
# ============================================================
# Escrito de forma genérica para que a MESMA implementação rode em Albers e em
# graus. Assim, a diferença observada na sensibilidade vem da CONVENÇÃO, e não
# de duas implementações distintas.

def _euclid(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_all(df, xcol, ycol, scale=1.0):
    """
    Pipeline completo num dado espaço de coordenadas.
    scale: fator para converter a unidade nativa em km
           (Albers -> 1.0; graus -> KM_PER_DEG, só para leitura).
    """
    d = df.copy()

    # ---- (1) Centroide "Brasil": baricentro dos biomas ponderado por área ----
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

    # peso relativo do bioma no total nacional daquele ano/regime
    d["w"] = d["area"] / d["br_area_ha"]

    # ---- (2) Influência: peso x afastamento do centroide nacional ----
    d["dist_to_br_km"] = _euclid(d[xcol], d[ycol], d["br_x"], d["br_y"]) * scale
    d["influence_mag_km"] = d["w"] * d["dist_to_br_km"]
    d["influence_vec_x"] = d["w"] * (d[xcol] - d["br_x"]) * scale
    d["influence_vec_y"] = d["w"] * (d[ycol] - d["br_y"]) * scale

    den = d.groupby(["Ano", "grupo"])["influence_mag_km"].transform("sum")
    d["influence_share"] = np.where(den > 0, d["influence_mag_km"] / den, np.nan)

    # ---- (3) LOO: deslocamento do centroide nacional ao remover o bioma ----
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
                x_wo = y_wo = shift = np.nan   # único bioma: LOO indefinido
            else:
                x_wo = (swx - w[i] * x[i]) / sw_wo
                y_wo = (swy - w[i] * y[i]) / sw_wo
                shift = _euclid(x_full, y_full, x_wo, y_wo) * scale
            loo_rows.append({"Ano": ano, "grupo": grupo, "Bioma": bioma,
                             "br_x_full": x_full, "br_y_full": y_full,
                             "br_x_wo": x_wo, "br_y_wo": y_wo,
                             "loo_shift_km": shift})
    loo = pd.DataFrame(loo_rows).sort_values(["grupo", "Ano", "Bioma"]).reset_index(drop=True)

    # ---- (4) Decomposição do deslocamento interanual (a "novidade" do paper) ----
    #   K = w_barra * (C_b,t - C_b,t-1)        <- MIGRAÇÃO do fogo dentro do bioma
    #     + (delta w) * (C_b_barra - C_BR,t-1) <- REPESAGEM entre biomas
    prev = d[["Ano", "grupo", "Bioma", "w", xcol, ycol]].copy()
    prev["Ano"] += 1                                   # parear t com t-1
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

    dd["K_move_x"] = dd["w_bar"] * dd["dx_b"]                              # migração
    dd["K_move_y"] = dd["w_bar"] * dd["dy_b"]
    dd["K_weight_x"] = dd["dw"] * (dd["x_bar"] - dd["br_x_prev"]) * scale  # repesagem
    dd["K_weight_y"] = dd["dw"] * (dd["y_bar"] - dd["br_y_prev"]) * scale

    dd["K_x"] = dd["K_move_x"] + dd["K_weight_x"]
    dd["K_y"] = dd["K_move_y"] + dd["K_weight_y"]

    # projeção na direção do deslocamento nacional (positivo = empurra a favor)
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
# 3. EXECUÇÃO
# ============================================================

def carregar_unico(path):
    """Lê o arquivo consolidado (com coluna 'grupo') vindo de biome_centroids_from_patches.py."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    m = mapear_colunas(df)
    if "grupo" not in [c.lower() for c in df.columns]:
        raise ValueError(f"Arquivo consolidado sem coluna 'grupo': {path}")
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
    print(f"  espaço detectado: {espaco} | colunas: {m} | regimes: {sorted(out.grupo.unique())}")
    if espaco == "graus":
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
    VERIFICAÇÃO CRÍTICA: o centroide nacional reconstruído a partir dos centroides
    de bioma deve ser IDÊNTICO ao calculado direto dos patches (média ponderada é
    associativa). Divergência > ~1 km indica erro no pipeline.
    """
    if not path or not os.path.exists(path):
        print("\n  (verificação de consistência pulada: CHECK_FILE não encontrado)")
        return
    chk = pd.read_csv(path)
    chk.columns = [c.strip() for c in chk.columns]
    # normaliza rótulo de regime entre os dois arquivos
    mapa = {"natural":"Natural","use":"Use","anthropogenic_use":"Anthropogenic_use"}
    chk["grupo"] = chk["Regime"].astype(str).str.strip().str.lower().map(lambda v: mapa.get(v, v))
    b = br.copy()
    b["gl"] = b["grupo"].astype(str).str.lower()
    chk["gl"] = chk["grupo"].astype(str).str.lower()
    # casa por ano; se os rótulos não coincidirem, casa por ordem de regime
    mg = b.merge(chk, on=["Ano","gl"], suffixes=("_bioma","_patch"))
    if mg.empty:
        print("\n  AVISO: não consegui casar os regimes entre os dois arquivos — "
              "verificação de consistência não realizada.")
        return
    d = np.hypot(mg["br_X_km"] - mg["Xc_km"], mg["br_Y_km"] - mg["Yc_km"])
    print("\n=== CONSISTÊNCIA: centroide nacional (via biomas) vs (direto dos patches) ===")
    print(f"  diferença mediana: {d.median():.4f} km | máx: {d.max():.4f} km  (n={len(d)})")
    if d.max() < 1.0:
        print("  OK — os dois caminhos convergem. O pipeline está coerente.")
    else:
        print("  ATENÇÃO — divergência acima de 1 km. Investigue: os centroides de bioma")
        print("  podem não estar em Albers, ou os pesos (área) diferem entre os arquivos.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if SINGLE_FILE:
        print("Lendo arquivo consolidado (centroides calculados EM Albers a partir dos patches):")
        df = carregar_unico(SINGLE_FILE)
    else:
        print("Lendo arquivos por regime (LEGADO — centroides promediados em graus):")
        df = pd.concat([carregar(p, reg) for reg, p in FILES.items()], ignore_index=True)
    print(f"\nTotal: {len(df)} registros | anos {df.Ano.min()}-{df.Ano.max()} | "
          f"biomas: {df.Bioma.nunique()} | regimes: {sorted(df.grupo.unique())}")

    # ---- VERSÃO OFICIAL: tudo em Albers (km) ----
    br_a, inf_a, loo_a, con_a = compute_all(df, "X_km", "Y_km", scale=1.0)
    br_a = br_a.rename(columns={"br_x": "br_X_km", "br_y": "br_Y_km"})

    # lon/lat do centroide nacional, para mapas
    tr_back = Transformer.from_crs(ALBERS, SRC_GEO, always_xy=True)
    lo, la = tr_back.transform(br_a["br_X_km"].values * 1000.0,
                               br_a["br_Y_km"].values * 1000.0)
    br_a["br_lon"], br_a["br_lat"] = lo, la

    n_nan = int(loo_a["loo_shift_km"].isna().sum())
    if n_nan:
        print(f"  AVISO: {n_nan} LOO indefinidos (ano/regime com um único bioma).")

    checar_consistencia(br_a, CHECK_FILE)

    # ---- VERSÃO LEGADA (euclidiana em GRAUS), só para a sensibilidade ----
    # Roda a convenção antiga sobre os MESMOS centroides -> isola o efeito da métrica.
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

    # ---- salva ----
    p = lambda n: os.path.join(OUT_DIR, f"{OUT_PREFIX}_{n}.csv")
    br_a.to_csv(p("brasil_reconstruido"), index=False)
    inf_a.to_csv(p("influencia_anual"), index=False)
    loo_a.to_csv(p("loo_shift"), index=False)
    con_a.to_csv(p("contrib_delta"), index=False)
    comp.to_csv(p("sensibilidade_graus_vs_albers"), index=False)

    # ---- números que vão para a carta ----
    print("\n=== SENSIBILIDADE À CONVENÇÃO (euclidiana em graus -> Albers) ===")
    for col, nome in [("loo_diff_pct", "LOO (deslocamento)"),
                      ("influence_diff_pct", "Influência (w x d)")]:
        v = comp[col].abs().dropna()
        if len(v):
            print(f"  {nome:22s}: mediana {v.median():.2f}% | p95 {v.quantile(.95):.2f}% | máx {v.max():.2f}%")
    sh = comp["share_diff_pp"].abs().dropna()
    print(f"  {'Share de influência':22s}: mediana {sh.median():.3f} pp | máx {sh.max():.3f} pp")

    print("\n=== O RANKING DE BIOMAS MUDA? ===")
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
    print(f"  Ordenação idêntica em {total - trocas}/{total} combinações ano x regime.")
    print(f"  1º lugar (bioma dominante) difere em {top1}/{total}.")
    if linhas_trocadas:
        print("  Trocas (verifique se o texto do artigo afirma algo sobre esses biomas):")
        for ano, grupo, difs in linhas_trocadas[:15]:
            print(f"    {ano} {grupo}: {difs}")
        if len(linhas_trocadas) > 15:
            print(f"    ... e mais {len(linhas_trocadas)-15}")

    print(f"\nArquivos salvos em: {OUT_DIR}/")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# NOTE -- spatial block bootstrap (more conservative option)
# ------------------------------------------------------------
# If a reviewer insists that the CI is too narrow because of autocorrelation,
# the answer is to resample spatial BLOCKS instead of individual patches:
# aggregate each year's patches into grid cells (e.g. 1 degree or 100 km) and
# resample CELLS with replacement (keeping each cell's patches together).
# This preserves local dependence and widens the CI in an honest way. The
# only change is in resample_year_arrays; the rest of the script stays the
# same. This variant can be implemented as a sensitivity analysis if useful.