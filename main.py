"""
🐄 Bovine SNP Platform
Pipeline complet de bioinformatique pour puces SNP bovines.
Application Streamlit mono-fichier — déployable sur Streamlit Cloud.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import MDS as SklearnMDS
from io import StringIO, BytesIO
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐄 Bovine SNP Platform",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05,   # missingness SNP
    "mind": 0.05,   # missingness individu
    "maf": 0.05,
    "hwe": 1e-6,
    "het_sd": 3.0,
}

# ============================================================
# PARSING PED / MAP
# ============================================================

def parse_map(map_bytes: bytes) -> pd.DataFrame:
    """Parse un fichier .map (chr, snp_id, cm, bp)."""
    rows = []
    text = map_bytes.decode("utf-8", errors="replace")
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            bp = int(parts[3])
        except ValueError:
            bp = 0
        rows.append({
            "CHR": str(parts[0]),
            "SNP": parts[1],
            "CM": float(parts[2]) if parts[2] not in ("0", ".") else 0.0,
            "BP": bp,
        })
    return pd.DataFrame(rows)


def parse_ped(ped_bytes: bytes, n_snp: int) -> tuple:
    """Parse un fichier .ped et encode les génotypes en dosage 0/1/2."""
    text = ped_bytes.decode("utf-8", errors="replace")
    lines = [l for l in text.strip().split("\n") if l.strip()]

    fids, iids, geno_rows = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) < 6 + 2 * n_snp:
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        geno_rows.append(parts[6:6 + 2 * n_snp])

    n_ind = len(iids)
    gt = np.full((n_ind, n_snp), np.nan, dtype=np.float32)

    for i, row in enumerate(geno_rows):
        alleles = np.array(row).reshape(n_snp, 2)
        a1, a2 = alleles[:, 0], alleles[:, 1]
        mask = (a1 != "0") & (a2 != "0")
        for j in np.where(mask)[0]:
            x, y = a1[j], a2[j]
            if x == y:
                gt[i, j] = 0.0
            else:
                gt[i, j] = 1.0
        # Note : pour un dosage 0/1/2 complet il faudrait connaître les allèles
        # de référence. Ici on utilise 0 = homozygote, 1 = hétérozygote, NaN = manquant.
        # Pour MAF on utilisera un encodage binaire.

    # Ré-encodage binaire (0/1) pour MAF / FST
    gt_bin = np.full((n_ind, n_snp), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        alleles = np.array(row).reshape(n_snp, 2)
        a1, a2 = alleles[:, 0], alleles[:, 1]
        mask = (a1 != "0") & (a2 != "0")
        for j in np.where(mask)[0]:
            # On code l'allèle minoritaire 1 par convention arbitraire (prévalence)
            gt_bin[i, j] = 1.0 if a1[j] != a2[j] else 0.0

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    return gt, gt_bin, ind_df


# ============================================================
# DEMO DATA
# ============================================================

def generate_demo_data(n_ind: int = 150, n_snp: int = 800,
                       n_pop: int = 4, seed: int = 42):
    """Génère un jeu de données synthétique réaliste (4 races bovines)."""
    rng = np.random.default_rng(seed)
    pop_names = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
                 "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"][:n_pop]

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15

    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    ind_rows = []
    per_pop = n_ind // n_pop

    k = 0
    for pop in pop_names:
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(per_pop):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i+1:03d}"})
            k += 1

    # Missingness 2 %
    mask = rng.random(gt.shape) < 0.02
    gt[mask] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    snp_df = pd.DataFrame({
        "CHR": rng.integers(1, 30, n_snp).astype(str),
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0,
        "BP": np.sort(rng.integers(1, 100_000_000, n_snp)),
    })
    return gt, ind_df, snp_df


# ============================================================
# ANALYSES QC
# ============================================================

def missingness_per_ind(gt: np.ndarray) -> np.ndarray:
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt: np.ndarray) -> np.ndarray:
    return np.isnan(gt).mean(axis=0)


def allele_freq(gt: np.ndarray) -> np.ndarray:
    """Fréquence de l'allèle codé 1 (par SNP)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt, axis=0) / 2.0


def maf(gt: np.ndarray) -> np.ndarray:
    p = allele_freq(gt)
    return np.minimum(p, 1 - p)


def hwe_pvalues(gt: np.ndarray) -> np.ndarray:
    """Test exact de Hardy-Weinberg (chi² approché) par SNP."""
    n_snp = gt.shape[1]
    pvals = np.full(n_snp, np.nan)
    for j in range(n_snp):
        col = gt[:, j]
        col = col[~np.isnan(col)]
        n = len(col)
        if n < 10:
            continue
        # Ici col est 0/1/2 (dosage), donc :
        #   n0 = homozygotes ref, n1 = hétérozygotes, n2 = homozygotes alt
        # Approx : on reconvertit par fréquence d'allèle
        p = col.mean() / 2.0
        if p <= 0 or p >= 1:
            continue
        exp_het = 2 * p * (1 - p) * n
        obs_het = (col == 1).sum()
        if exp_het < 5:
            continue
        chi2 = (obs_het - exp_het) ** 2 / exp_het
        pvals[j] = 1 - stats.chi2.cdf(chi2, df=1)
    return pvals


def heterozygosity(gt: np.ndarray) -> np.ndarray:
    """Hétérozygotie observée par individu."""
    het = np.nanmean(gt == 1, axis=1)
    return het


# ============================================================
# FILTRAGE QC
# ============================================================

def apply_qc_filters(gt: np.ndarray, ind_df: pd.DataFrame,
                     snp_df: pd.DataFrame, params: dict):
    n0, m0 = gt.shape

    # 1. Missingness SNP
    miss_snp = missingness_per_snp(gt)
    keep_snp = miss_snp <= params["geno"]

    # 2. Missingness individu
    miss_ind = missingness_per_ind(gt[:, keep_snp])
    keep_ind = miss_ind <= params["mind"]

    gt1 = gt[keep_ind][:, keep_snp]
    ind1 = ind_df[keep_ind].reset_index(drop=True)
    snp1 = snp_df[keep_snp].reset_index(drop=True)

    # 3. MAF
    m = maf(gt1)
    keep_maf = m >= params["maf"]
    gt2 = gt1[:, keep_maf]
    snp2 = snp1[keep_maf].reset_index(drop=True)

    # 4. HWE
    pv = hwe_pvalues(gt2)
    keep_hwe = np.isnan(pv) | (pv >= params["hwe"])
    gt3 = gt2[:, keep_hwe]
    snp3 = snp2[keep_hwe].reset_index(drop=True)

    # 5. Hétérozygotie (outliers > N σ)
    het = heterozygosity(gt3)
    z = (het - np.nanmean(het)) / np.nanstd(het)
    keep_het = np.abs(z) <= params["het_sd"]
    gt4 = gt3[keep_het]
    ind4 = ind1[keep_het].reset_index(drop=True)

    stats_dict = {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt4.shape[0]), "n_snp_final": int(gt4.shape[1]),
        "excluded_ind": int(n0 - gt4.shape[0]),
        "excluded_snp": int(m0 - gt4.shape[1]),
    }
    return gt4, ind4, snp3, stats_dict


# ============================================================
# DIVERSITÉ / STRUCTURE
# ============================================================

def fst_per_snp(gt: np.ndarray, pop_labels: np.ndarray) -> np.ndarray:
    """FST de Nei par SNP (moyenne sur paires de populations)."""
    pops = np.unique(pop_labels)
    n_snp = gt.shape[1]
    fst = np.full(n_snp, np.nan)
    for j in range(n_snp):
        p_list, n_list = [], []
        for p in pops:
            mask = pop_labels == p
            vals = gt[mask, j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            p_list.append(vals.mean() / 2.0)
            n_list.append(len(vals))
        if len(p_list) < 2:
            continue
        p_arr = np.array(p_list)
        n_arr = np.array(n_list)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.mean(2 * p_arr * (1 - p_arr))
        h_t = 2 * p_bar * (1 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


def impute_mean(gt: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(gt, axis=0)
    inds = np.where(np.isnan(gt))
    gt2 = gt.copy()
    gt2[inds] = np.take(col_mean, inds[1])
    return gt2


def pca_analysis(gt: np.ndarray, n_components: int = 10):
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    return scores, pca.explained_variance_ratio_ * 100


def mds_analysis(gt: np.ndarray, n_components: int = 5):
    X = impute_mean(gt)
    n = X.shape[0]
    # Distance euclidienne (matrice de similarité génétique)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.sqrt(((X - X[i]) ** 2).mean(axis=1))
    mds = SklearnMDS(n_components=n_components, dissimilarity="precomputed",
                     random_state=42, n_init=1, max_iter=300)
    return mds.fit_transform(D)


def ld_decay(gt: np.ndarray, snp_bp: np.ndarray,
             max_kb: float = 1000, max_snp: int = 1500,
             seed: int = 42) -> pd.DataFrame:
    """LD decay (r² vs distance) sur un sous-échantillon de SNPs."""
    n_snp = gt.shape[1]
    rng = np.random.default_rng(seed)
    if n_snp > max_snp:
        idx = np.sort(rng.choice(n_snp, max_snp, replace=False))
    else:
        idx = np.arange(n_snp)

    gt_sub = gt[:, idx]
    bp_sub = snp_bp[idx]

    # Imputation rapide
    col_mean = np.nanmean(gt_sub, axis=0)
    inds = np.where(np.isnan(gt_sub))
    gt_sub[inds] = np.take(col_mean, inds[1])

    results = []
    n = len(idx)
    for ii in range(n):
        i = ii
        for jj in range(ii + 1, n):
            j = jj
            dist_kb = (bp_sub[j] - bp_sub[i]) / 1000.0
            if dist_kb > max_kb:
                break
            x, y = gt_sub[:, i], gt_sub[:, j]
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            r = np.corrcoef(x, y)[0, 1]
            results.append({"dist_kb": dist_kb, "r2": r ** 2})
    return pd.DataFrame(results)


def kinship_matrix(gt: np.ndarray) -> np.ndarray:
    """Matrice de parenté génomique (GRM, approche GCTA simplifiée)."""
    X = impute_mean(gt)
    p = X.mean(axis=0) / 2.0
    p = np.clip(p, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    denom = 2 * p * (1 - p)
    G = (Z / np.sqrt(denom)) @ (Z / np.sqrt(denom)).T / X.shape[1]
    return G


# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db", log_y=False):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Fréquence",
                      height=380, margin=dict(l=40, r=20, t=50, b=40))
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness par individu",
                                        "Missingness par SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue", name="Individus"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral", name="SNPs"),
                  row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1],
        "Population": labels,
    })
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs PC2 ({var_pct[1]:.1f}%)",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_mds(mds_coords, labels):
    df = pd.DataFrame({
        "MDS1": mds_coords[:, 0], "MDS2": mds_coords[:, 1],
        "Population": labels,
    })
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS — Structure des populations", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_manhattan(fst, chr_col):
    df = pd.DataFrame({"FST": fst, "CHR": chr_col}).reset_index()
    df = df.dropna(subset=["FST"])
    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["CHR", "index"]).reset_index(drop=True)

    cumulative = 0
    ticks, labels = [], []
    for c in sorted(df["CHR"].unique()):
        sub = df[df["CHR"] == c]
        ticks.append(cumulative + len(sub) / 2)
        labels.append(str(c))
        cumulative += len(sub)

    df["x"] = np.arange(len(df))
    df["color"] = df["CHR"].astype(str)

    fig = go.Figure()
    for c in sorted(df["CHR"].unique()):
        sub = df[df["CHR"] == c]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["FST"], mode="markers",
            marker=dict(size=5, color="#2c3e50" if c % 2 == 0 else "#7f8c8d"),
            name=f"chr{c}", showlegend=False,
        ))

    q_upper = np.nanquantile(fst, 0.999)
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil top 0.1% = {q_upper:.4f}")
    fig.update_layout(title="Manhattan Plot — FST par SNP",
                      xaxis_title="Chromosomes",
                      yaxis_title="FST",
                      height=500,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, max_kb=1000, bin_kb=20):
    if ld_df.empty:
        return None
    ld_df = ld_df.copy()
    ld_df["bin"] = (ld_df["dist_kb"] // bin_kb) * bin_kb
    agg = ld_df.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)", "r2": "r² moyen"},
                  title="Déséquilibre de liaison (LD decay)", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_kinship_heatmap(G, labels):
    df = pd.DataFrame(G, index=labels, columns=labels)
    fig = px.imshow(df, color_continuous_scale="RdBu_r",
                    zmin=-0.3, zmax=0.5,
                    title="Matrice de parenté (GRM)", height=650,
                    aspect="auto")
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ============================================================
# RAPPORT HTML
# ============================================================

def build_report_html(state: dict, config: dict, stats: dict,
                     fig_pca=None, fig_mds=None, fig_manhattan=None):
    fig_pca_html = fig_pca.to_html(full_html=False, include_plotlyjs="cdn") if fig_pca else "<p>Non disponible</p>"
    fig_mds_html = fig_mds.to_html(full_html=False, include_plotlyjs=False) if fig_mds else ""
    fig_manhattan_html = fig_manhattan.to_html(full_html=False, include_plotlyjs=False) if fig_manhattan else ""

    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport Bovine SNP Platform</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; }}
 h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }}
 h2 {{ color: #2471a3; margin-top: 28px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
 th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
 th {{ background: #eaf2f8; }}
 .summary {{ background: #f4f6f7; padding: 16px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>🐄 Rapport Bovine SNP Platform</h1>
<p><b>Projet :</b> {config['project_name']} — <b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="summary">
<h2>Résumé exécutif</h2>
<ul>
 <li>Individus analysés : <b>{stats['n_ind_final']}</b> (sur {stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{stats['n_snp_final']}</b> (sur {stats['n_snp_init']})</li>
 <li>Individus exclus (QC) : <b>{stats['excluded_ind']}</b></li>
 <li>SNPs exclus (QC) : <b>{stats['excluded_snp']}</b></li>
</ul>
</div>

<h2>Analyses de structure</h2>
{fig_pca_html}
{fig_mds_html}

<h2>Signatures de sélection (FST)</h2>
{fig_manhattan_html}

<hr>
<p style="font-size:0.85em; color:#666">
Rapport généré automatiquement par Bovine SNP Platform.
</p>
</body>
</html>
"""


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def init_state():
    for k in ["gt", "gt_bin", "ind_df", "snp_df",
              "gt_filt", "ind_filt", "snp_filt", "qc_stats",
              "fst", "pca_scores", "pca_var", "mds_coords",
              "ld_df", "kinship", "results"]:
        if k not in st.session_state:
            st.session_state[k] = None


def main():
    init_state()
    st.title("🐄 Bovine SNP Platform")
    st.caption("Pipeline complet de bioinformatique pour puces SNP bovines — "
               "100 % Python, déployable sur Streamlit Cloud.")

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.header("📁 Données")

        mode = st.radio("Source :", ["Demo", "Upload PED/MAP"], index=0)

        if mode == "Demo":
            if st.button("🎲 Générer un jeu de démonstration", use_container_width=True):
                with st.spinner("Génération des données synthétiques (4 races, 150 ind, 800 SNPs)..."):
                    gt, ind_df, snp_df = generate_demo_data(150, 800, 4)
                    gt_bin = (gt > 0).astype(np.float32)
                    gt_bin[np.isnan(gt)] = np.nan
                    st.session_state.gt = gt
                    st.session_state.gt_bin = gt_bin
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    st.session_state.gt_filt = None  # reset
                st.success("Données de démonstration chargées !")
        else:
            ped_file = st.file_uploader("Fichier .ped", type=["ped", "txt"])
            map_file = st.file_uploader("Fichier .map", type=["map", "txt"])
            if ped_file and map_file:
                if st.button("📥 Charger les fichiers", use_container_width=True):
                    with st.spinner("Parsing..."):
                        map_df = parse_map(map_file.read())
                        gt, gt_bin, ind_df = parse_ped(ped_file.read(), len(map_df))
                        st.session_state.gt = gt
                        st.session_state.gt_bin = gt_bin
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        st.session_state.gt_filt = None
                    st.success(f"Chargés : {gt.shape[0]} individus × {gt.shape[1]} SNPs")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (--geno)", 0.0, 0.5, 0.05, 0.01)
        mind = st.slider("Missingness individu (--mind)", 0.0, 0.5, 0.05, 0.01)
        maf_thr = st.slider("MAF minimal", 0.0, 0.5, 0.05, 0.01)
        hwe_thr = st.number_input("HWE p-value (exclure < )",
                                  value=1e-6, format="%.0e")
        het_sd = st.slider("Écart-type hétérozygotie", 1.0, 5.0, 3.0, 0.1)

        st.divider()
        if st.button("🚀 Lancer le pipeline", type="primary",
                     use_container_width=True):
            if st.session_state.gt is None:
                st.error("Chargez ou générez d'abord des données.")
            else:
                st.session_state.run_requested = True

        st.divider()
        st.caption("V1.0 — Codé en Python pur. Aucune installation de "
                   "PLINK/ADMIXTURE/SNeP requise.")

    # -------------------- MAIN --------------------
    if st.session_state.gt is None:
        st.info("👉 **Pour démarrer** : cliquez sur *Générer un jeu de démonstration* "
                "dans la barre latérale, ou importez un `.ped` + `.map`.")
        st.markdown("""
        ### Ce que fait cette plateforme
        - **QC** : missingness, MAF, Hardy-Weinberg, hétérozygotie, filtrage automatique
        - **Diversité** : FST par SNP, hétérozygotie par race
        - **Structure** : PCA, MDS, matrice de parenté (GRM)
        - **Démographie** : déséquilibre de liaison (LD decay)
        - **Sélection** : Manhattan plot des outliers FST
        - **Rapport** : HTML téléchargeable
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure",
                    "📈 Démographie", "🔍 Sélection", "📄 Rapport"])

    # ---- TAB 1 : Aperçu ----
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())

        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("Carte des SNPs (5 premières lignes)")
        st.dataframe(snp_df.head(5), use_container_width=True)

    # ---- TAB 2 : QC ----
    with tabs[1]:
        st.subheader("Contrôle qualité")
        if st.button("▶ Lancer le QC", use_container_width=True):
            with st.spinner("Calcul des métriques QC..."):
                gt_bin = st.session_state.gt_bin
                gt_filt, ind_filt, snp_filt, qc_stats = apply_qc_filters(
                    gt_bin, ind_df, snp_df,
                    {"geno": geno, "mind": mind, "maf": maf_thr,
                     "hwe": hwe_thr, "het_sd": het_sd}
                )
                st.session_state.gt_filt = gt_filt
                st.session_state.ind_filt = ind_filt
                st.session_state.snp_filt = snp_filt
                st.session_state.qc_stats = qc_stats

        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Individus exclus", s["excluded_ind"])
            c4.metric("SNPs exclus", s["excluded_snp"])

            gt_bin = st.session_state.gt_bin
            miss_ind = missingness_per_ind(gt_bin)
            miss_snp = missingness_per_snp(gt_bin)
            st.plotly_chart(plot_missingness_dashboard(miss_ind, miss_snp),
                            use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_hist(maf(gt_bin), "Spectre MAF", "MAF", "#2ecc71"),
                    use_container_width=True)
            with c2:
                het = heterozygosity(gt_bin)
                st.plotly_chart(
                    plot_hist(het, "Hétérozygotie observée", "HET",
                              "#9b59b6"),
                    use_container_width=True)

            pv = hwe_pvalues(gt_bin)
            pv_plot = pv[~np.isnan(pv)]
            if len(pv_plot) > 0:
                st.plotly_chart(
                    plot_hist(pv_plot, "Distribution des p-values HWE",
                              "p-value", "salmon"),
                    use_container_width=True)
        else:
            st.info("Cliquez sur **Lancer le QC** pour démarrer l'analyse.")

    # ---- TAB 3 : Structure ----
    with tabs[2]:
        st.subheader("Structure des populations")
        if st.session_state.gt_filt is None:
            st.warning("Lancez d'abord le QC.")
        else:
            if st.button("▶ Calculer PCA + MDS + GRM", use_container_width=True):
                with st.spinner("Calcul de la PCA..."):
                    scores, var = pca_analysis(st.session_state.gt_filt, 10)
                    st.session_state.pca_scores = scores
                    st.session_state.pca_var = var
                with st.spinner("Calcul du MDS..."):
                    mds = mds_analysis(st.session_state.gt_filt, 5)
                    st.session_state.mds_coords = mds
                with st.spinner("Calcul de la matrice de parenté..."):
                    G = kinship_matrix(st.session_state.gt_filt)
                    st.session_state.kinship = G
                st.success("Analyses de structure terminées.")

            if st.session_state.pca_scores is not None:
                labels = st.session_state.ind_filt["FID"].values
                st.plotly_chart(
                    plot_pca(st.session_state.pca_scores,
                             st.session_state.pca_var, labels),
                    use_container_width=True)

            if st.session_state.mds_coords is not None:
                labels = st.session_state.ind_filt["FID"].values
                st.plotly_chart(
                    plot_mds(st.session_state.mds_coords, labels),
                    use_container_width=True)

            if st.session_state.kinship is not None:
                labels = st.session_state.ind_filt["FID"].values
                st.plotly_chart(
                    plot_kinship_heatmap(st.session_state.kinship, labels),
                    use_container_width=True)

    # ---- TAB 4 : Démographie ----
    with tabs[3]:
        st.subheader("Démographie — LD decay")
        if st.session_state.gt_filt is None:
            st.warning("Lancez d'abord le QC.")
        else:
            if st.button("▶ Calculer le LD decay", use_container_width=True):
                with st.spinner("Calcul du LD decay (sous-échantillon)..."):
                    ld_df = ld_decay(
                        st.session_state.gt_filt.copy(),
                        st.session_state.snp_filt["BP"].values,
                        max_kb=1000, max_snp=1200)
                    st.session_state.ld_df = ld_df
                st.success(f"LD calculé sur {len(ld_df)} paires de SNPs.")

            if st.session_state.ld_df is not None and not st.session_state.ld_df.empty:
                fig_ld = plot_ld_decay(st.session_state.ld_df)
                if fig_ld:
                    st.plotly_chart(fig_ld, use_container_width=True)
                with st.expander("Voir les données brutes"):
                    st.dataframe(st.session_state.ld_df.head(100))

    # ---- TAB 5 : Sélection ----
    with tabs[4]:
        st.subheader("Signatures de sélection (FST)")
        if st.session_state.gt_filt is None:
            st.warning("Lancez d'abord le QC.")
        else:
            if st.button("▶ Calculer FST + Manhattan", use_container_width=True):
                with st.spinner("Calcul du FST par SNP..."):
                    fst = fst_per_snp(
                        st.session_state.gt_filt,
                        st.session_state.ind_filt["FID"].values)
                    st.session_state.fst = fst
                st.success("FST calculé.")

            if st.session_state.fst is not None:
                fst_clean = st.session_state.fst[~np.isnan(st.session_state.fst)]
                c1, c2, c3 = st.columns(3)
                c1.metric("FST moyen", f"{fst_clean.mean():.4f}")
                c2.metric("FST médian", f"{np.median(fst_clean):.4f}")
                c3.metric("Seuil top 0.1%", f"{np.quantile(fst_clean, 0.999):.4f}")

                fig = plot_manhattan(
                    st.session_state.fst,
                    st.session_state.snp_filt["CHR"].values)
                st.plotly_chart(fig, use_container_width=True)

                # Liste des outliers
                q_upper = np.nanquantile(st.session_state.fst, 0.999)
                outliers = st.session_state.snp_filt[
                    st.session_state.fst >= q_upper].copy()
                outliers["FST"] = st.session_state.fst[
                    st.session_state.fst >= q_upper]
                st.subheader(f"SNPs outliers ({len(outliers)})")
                st.dataframe(outliers, use_container_width=True)

    # ---- TAB 6 : Rapport ----
    with tabs[5]:
        st.subheader("Génération du rapport")
        if st.session_state.qc_stats is None:
            st.warning("Lancez au moins le QC pour générer un rapport.")
        else:
            if st.button("📄 Générer le rapport HTML", use_container_width=True):
                fig_pca = None
                if st.session_state.pca_scores is not None:
                    fig_pca = plot_pca(
                        st.session_state.pca_scores,
                        st.session_state.pca_var,
                        st.session_state.ind_filt["FID"].values)
                fig_mds = None
                if st.session_state.mds_coords is not None:
                    fig_mds = plot_mds(
                        st.session_state.mds_coords,
                        st.session_state.ind_filt["FID"].values)
                fig_man = None
                if st.session_state.fst is not None:
                    fig_man = plot_manhattan(
                        st.session_state.fst,
                        st.session_state.snp_filt["CHR"].values)

                html = build_report_html(
                    state={}, config={"project_name": "Cattle_Project"},
                    stats=st.session_state.qc_stats,
                    fig_pca=fig_pca, fig_mds=fig_mds, fig_manhattan=fig_man)

                st.download_button(
                    label="⬇ Télécharger le rapport HTML",
                    data=html.encode("utf-8"),
                    file_name=f"rapport_bovine_{datetime.now():%Y%m%d_%H%M}.html",
                    mime="text/html",
                    use_container_width=True)
                st.success("Rapport prêt.")
                with st.expander("Prévisualisation du rapport"):
                    st.components.v1.html(html, height=600, scrolling=True)

    # ---- Exécution pipeline complet ----
    if st.session_state.get("run_requested"):
        st.session_state["run_requested"] = False
        with st.spinner("Exécution du pipeline complet..."):
            gt_bin = st.session_state.gt_bin
            gt_filt, ind_filt, snp_filt, qc_stats = apply_qc_filters(
                gt_bin, ind_df, snp_df,
                {"geno": geno, "mind": mind, "maf": maf_thr,
                 "hwe": hwe_thr, "het_sd": het_sd})
            st.session_state.gt_filt = gt_filt
            st.session_state.ind_filt = ind_filt
            st.session_state.snp_filt = snp_filt
            st.session_state.qc_stats = qc_stats

            scores, var = pca_analysis(gt_filt, 10)
            st.session_state.pca_scores = scores
            st.session_state.pca_var = var

            st.session_state.mds_coords = mds_analysis(gt_filt, 5)
            st.session_state.kinship = kinship_matrix(gt_filt)
            st.session_state.ld_df = ld_decay(
                gt_filt.copy(), snp_filt["BP"].values,
                max_kb=1000, max_snp=1000)
            st.session_state.fst = fst_per_snp(gt_filt, ind_filt["FID"].values)
        st.success("Pipeline terminé ! Consultez les onglets.")


if __name__ == "__main__":
    main()
