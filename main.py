"""
🐄 Bovine SNP Platform
Pipeline complet de bioinformatique pour puces SNP bovines.
Application Streamlit mono-fichier — déployable sur Streamlit Cloud.

Version 2.0 — corrections majeures :
 - Encodage 0/1/2 réel (dosage de l'allèle mineur) pour PED
 - Génération de démo alignée (gt ↔ ind_df)
 - Test exact de Hardy-Weinberg (Wigginton et al. 2005)
 - FST de Nei avec Hs pondéré
 - LD decay vectorisé (matrice de corrélation)
 - Manhattan robuste aux chromosomes non numériques
 - Garde-fous d'alignement dans le filtrage QC
 - Invalidation automatique des analyses en aval
"""

import warnings
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    "geno": 0.05,     # missingness SNP
    "mind": 0.05,     # missingness individu
    "maf": 0.05,
    "hwe": 1e-6,
    "het_sd": 3.0,
}

# Nombre max de SNPs pour le calcul HWE exact (au-delà : approximation chi²)
HWE_EXACT_MAX_SNP = 20_000


# ============================================================
# UTILITAIRES NUMÉRIQUES
# ============================================================

def impute_mean(gt: np.ndarray) -> np.ndarray:
    """Imputation par moyenne de colonne (par SNP). Retourne une copie."""
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    # SNP totalement manquant → moyenne 0 (deviendra informatif nul)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2


def safe_divide(a, b, fill=np.nan):
    """Division sûre, évite les RuntimeWarning."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.true_divide(a, b)
    r = np.where(np.isfinite(r), r, fill)
    return r


# ============================================================
# PARSING PED / MAP
# ============================================================

def parse_map(map_bytes: bytes) -> tuple:
    """
    Parse un fichier .map (chr, snp_id, cm, bp).
    Retourne (DataFrame, nombre de lignes rejetées).
    """
    text = map_bytes.decode("utf-8", errors="replace")
    rows, rejected = [], 0
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try:
            cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        try:
            bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({
            "CHR": str(parts[0]),
            "SNP": parts[1],
            "CM": cm,
            "BP": bp,
        })
    if not rows:
        raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected


def parse_ped(ped_bytes: bytes, n_snp: int) -> tuple:
    """
    Parse un fichier .ped et encode les génotypes en dosage 0/1/2 de
    l'allèle MINEUR (déterminé par comptage sur l'échantillon).

    Retourne (gt, ind_df, n_rejected).
    """
    text = ped_bytes.decode("utf-8", errors="replace")
    fids, iids, geno_rows = [], [], []
    rejected = 0
    expected_cols = 6 + 2 * n_snp

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < expected_cols:
            rejected += 1
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        geno_rows.append(parts[6:6 + 2 * n_snp])

    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError(
            f"Aucun individu valide : toutes les lignes du .ped ont moins de "
            f"{expected_cols} colonnes."
        )

    # --- Passe 1 : comptage des allèles par SNP ---
    allele_counts = [Counter() for _ in range(n_snp)]
    for row in geno_rows:
        for j in range(n_snp):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            c = allele_counts[j]
            c[a1] += 1
            c[a2] += 1

    # Détermination de l'allèle mineur (None si monomorphe)
    minor = [None] * n_snp
    for j, c in enumerate(allele_counts):
        if len(c) < 2:
            continue
        minor[j] = min(c, key=c.get)

    # --- Passe 2 : dosage 0/1/2 de l'allèle mineur ---
    gt = np.full((n_ind, n_snp), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    return gt, ind_df, rejected


# ============================================================
# DONNÉES DE DÉMONSTRATION
# ============================================================

def generate_demo_data(n_ind: int = 150, n_snp: int = 800,
                       n_pop: int = 4, seed: int = 42) -> tuple:
    """
    Génère un jeu synthétique avec structure de populations (Fst modéré).
    Le nombre d'individus par population est réparti équitablement, le reste
    est distribué aux premières populations.
    """
    rng = np.random.default_rng(seed)

    pool_names = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
                  "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool_names)))
    pop_names = pool_names[:n_pop]

    # Répartition : base + 1 pour les premières pops
    per_pop = n_ind // n_pop
    remainder = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < remainder else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15

    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []

    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i + 1:03d}"})
            k += 1

    # Missingness ~2 %
    mask = rng.random(gt.shape) < 0.02
    gt[mask] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    snp_df = pd.DataFrame({
        "CHR": rng.integers(1, 30, n_snp).astype(str),
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0,
        "BP": rng.integers(1, 100_000_000, n_snp),
    })
    # Trier par chr puis bp (ordre réaliste)
    snp_df["_chr_int"] = pd.to_numeric(snp_df["CHR"], errors="coerce").fillna(0).astype(int)
    snp_df = snp_df.sort_values(["_chr_int", "BP"]).drop(columns="_chr_int").reset_index(drop=True)

    return gt, ind_df, snp_df


# ============================================================
# QC — MÉTRIQUES
# ============================================================

def missingness_per_ind(gt: np.ndarray) -> np.ndarray:
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt: np.ndarray) -> np.ndarray:
    return np.isnan(gt).mean(axis=0)


def allele_freq(gt: np.ndarray) -> np.ndarray:
    """Fréquence de l'allèle codé '1' (dosage moyen / 2)."""
    return np.nanmean(gt, axis=0) / 2.0


def maf(gt: np.ndarray) -> np.ndarray:
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt: np.ndarray) -> np.ndarray:
    """Hétérozygotie observée par individu (nécessite dosage 0/1/2)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def hwe_exact_p(n_het: int, n_hom1: int, n_hom2: int) -> float:
    """
    Test exact de Hardy-Weinberg (Wigginton, Cutler, Abecasis 2005).
    n_hom1 et n_hom2 = comptages des deux homozygotes (ordre indifférent).
    Retourne la p-value bilatérale. np.nan si non calculable.
    """
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    # Cas triviaux : monomorphe
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0

    rare = 2 * min(n_hom1, n_hom2) + n_het
    # mid : point de départ de la récursion
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1

    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0

    # Récurrence vers le haut
    curr_hets = mid
    curr_homr = (rare - mid) // 2
    curr_homc = n - curr_hets - curr_homr
    while curr_hets <= rare - 2:
        probs[curr_hets + 2] = (
            probs[curr_hets] * 4 * curr_homr * curr_homc
            / ((curr_hets + 2) * (curr_hets + 1))
        )
        mysum += probs[curr_hets + 2]
        curr_hets += 2
        curr_homr -= 1
        curr_homc -= 1

    # Récurrence vers le bas
    curr_hets = mid
    curr_homr = (rare - mid) // 2
    curr_homc = n - curr_hets - curr_homr
    while curr_hets >= 2:
        probs[curr_hets - 2] = (
            probs[curr_hets] * curr_hets * (curr_hets - 1)
            / (4 * (curr_homr + 1) * (curr_homc + 1))
        )
        mysum += probs[curr_hets - 2]
        curr_hets -= 2
        curr_homr += 1
        curr_homc += 1

    # p-value bilatérale : somme des probas ≤ proba observée
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    p_hwe = probs[probs <= p_obs + 1e-7].sum() / mysum
    return float(min(p_hwe, 1.0))


def hwe_pvalues(gt: np.ndarray, progress_cb=None) -> np.ndarray:
    """
    p-values HWE par SNP. Test exact si peu de SNPs, sinon approximation
    chi² avec correction de continuité (rapide).
    """
    n_ind, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)

    # Comptages vectorisés 0/1/2
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0)
    n1 = ((gt == 1) & ~nan_mask).sum(axis=0)
    n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2

    use_exact = n_snp <= HWE_EXACT_MAX_SNP

    if use_exact:
        for j in range(n_snp):
            if n_valid[j] < 5:
                continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
            if progress_cb and (j % 500 == 0):
                progress_cb(j / n_snp)
    else:
        # Approximation chi² avec Yates
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


# ============================================================
# QC — FILTRAGE
# ============================================================

def _align_shapes(gt, ind_df, snp_df):
    """Garde-fou : aligne gt / ind_df / snp_df sur leurs dimensions communes."""
    n_gt, m_gt = gt.shape
    n_ind = min(n_gt, len(ind_df))
    n_snp = min(m_gt, len(snp_df))
    mismatch = (n_gt != len(ind_df)) or (m_gt != len(snp_df))
    if mismatch:
        st.warning(
            f"⚠️ Alignement corrigé — gt={gt.shape}, "
            f"ind_df={len(ind_df)}, snp_df={len(snp_df)} → "
            f"utilisation de ({n_ind}, {n_snp})"
        )
    gt = gt[:n_ind, :n_snp]
    ind_df = ind_df.iloc[:n_ind].reset_index(drop=True)
    snp_df = snp_df.iloc[:n_snp].reset_index(drop=True)
    return gt, ind_df, snp_df


def apply_qc_filters(gt: np.ndarray, ind_df: pd.DataFrame,
                     snp_df: pd.DataFrame, params: dict,
                     progress_cb=None) -> tuple:
    """
    Pipeline QC séquentiel :
      1. missingness SNP (--geno)
      2. missingness individu (--mind)
      3. MAF
      4. HWE (test exact)
      5. hétérozygotie (outliers > N σ)
    Retourne (gt_filt, ind_filt, snp_filt, stats_dict).
    """
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape

    # 1. Missingness SNP
    miss_snp = missingness_per_snp(gt)
    keep_snp = miss_snp <= params["geno"]
    gt = gt[:, keep_snp]
    snp_df = snp_df[keep_snp].reset_index(drop=True)

    # 2. Missingness individu
    miss_ind = missingness_per_ind(gt)
    keep_ind = miss_ind <= params["mind"]
    gt = gt[keep_ind]
    ind_df = ind_df[keep_ind].reset_index(drop=True)

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError(
            "Tous les SNPs ou individus ont été exclus. "
            "Assouplissez les seuils de missingness."
        )

    # 3. MAF
    m = maf(gt)
    keep_maf = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep_maf]
    snp_df = snp_df[keep_maf].reset_index(drop=True)

    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par MAF. Réduisez --maf.")

    # 4. HWE
    if progress_cb:
        progress_cb("HWE")
    pv = hwe_pvalues(gt)
    keep_hwe = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, keep_hwe]
    snp_df = snp_df[keep_hwe].reset_index(drop=True)

    # 5. Hétérozygotie
    het = heterozygosity(gt)
    if np.nanstd(het) > 1e-9:
        z = (het - np.nanmean(het)) / np.nanstd(het)
        keep_het = np.abs(z) <= params["het_sd"]
    else:
        keep_het = np.ones_like(het, dtype=bool)
    gt = gt[keep_het]
    ind_df = ind_df[keep_het].reset_index(drop=True)

    stats_dict = {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
    }
    return gt, ind_df, snp_df, stats_dict


# ============================================================
# POPULATION GENETICS
# ============================================================

def fst_per_snp(gt: np.ndarray, pop_labels: np.ndarray) -> np.ndarray:
    """
    FST de Nei par SNP (Hs pondéré par les effectifs par population).
    """
    pops = np.unique(pop_labels)
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)

    n_snp = gt.shape[1]
    fst = np.full(n_snp, np.nan, dtype=np.float64)
    masks = {p: (pop_labels == p) for p in pops}

    for j in range(n_snp):
        p_list, n_list = [], []
        for p in pops:
            vals = gt[masks[p], j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            p_list.append(vals.mean() / 2.0)
            n_list.append(len(vals))
        if len(p_list) < 2:
            continue
        p_arr = np.asarray(p_list)
        n_arr = np.asarray(n_list)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


def pca_analysis(gt: np.ndarray, n_components: int = 10) -> tuple:
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    return scores, pca.explained_variance_ratio_ * 100.0


def mds_analysis(gt: np.ndarray, n_components: int = 5) -> np.ndarray:
    """MDS sur distance d'IBS normalisée (adaptée aux données génomiques)."""
    X = impute_mean(gt)
    n = X.shape[0]
    # Distance de Hamming normalisée par le nombre de SNPs
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(axis=1) / X.shape[1]
    mds = SklearnMDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        n_init=1,
        max_iter=300,
        normalized_stress=False,
    )
    return mds.fit_transform(D)


def ld_decay(gt: np.ndarray, snp_bp: np.ndarray,
             max_kb: float = 1000, max_snp: int = 1500,
             seed: int = 42) -> pd.DataFrame:
    """LD decay vectorisé sur un sous-échantillon de SNPs."""
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])

    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))

    gt_sub = impute_mean(gt[:, idx])
    bp_sub = snp_bp[idx].astype(np.float64)

    # Standardisation par SNP
    X = gt_sub - gt_sub.mean(axis=0)
    std = gt_sub.std(axis=0)
    std[std < 1e-8] = np.nan
    X = X / std

    # Matrice de corrélation (SNPs × SNPs)
    n = X.shape[0]
    C = (X.T @ X) / n
    R2 = C ** 2

    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)

    return pd.DataFrame({
        "dist_kb": dist_kb[mask],
        "r2": R2[iu[mask], ju[mask]],
    })


def kinship_matrix(gt: np.ndarray) -> np.ndarray:
    """Matrice de parenté génomique (GRM, approche GCTA simplifiée)."""
    X = impute_mean(gt)
    p = np.clip(X.mean(axis=0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    denom = np.sqrt(2 * p * (1 - p))
    Zn = Z / denom
    G = (Zn @ Zn.T) / X.shape[1]
    return G


# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db", log_y=False):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title="Fréquence",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Missingness par individu", "Missingness par SNP"),
    )
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60, marker_color="skyblue"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60, marker_color="coral"),
                  row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1], "Population": labels,
    })
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Population",
        title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs PC2 ({var_pct[1]:.1f}%)",
        height=550,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({
        "MDS1": coords[:, 0], "MDS2": coords[:, 1], "Population": labels,
    })
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS) — Structure des populations",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _chr_sort_key(chrom: str):
    """Clé de tri robuste : numériques < sexuels < autres (alphabétique)."""
    s = str(chrom).upper()
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102, "W": 103, "Z": 104}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)


def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst), "CHR": np.asarray(chr_col)})
    df = df.dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty:
        return None

    # Tri selon chromosomes puis ordre original (position)
    order = df["CHR"].map(lambda c: _chr_sort_key(c))
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "_k2"]).reset_index(drop=True)

    cumulative = 0
    ticks, labels = [], []
    cum_positions = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        sub_idx = df.index[df["CHR"] == chrom]
        start, end = cumulative, cumulative + len(sub_idx)
        cum_positions[start:end] = np.arange(start, end)
        ticks.append((start + end) / 2)
        labels.append(str(chrom))
        cumulative = end

    df["x"] = cum_positions
    fig = go.Figure()
    for chrom in df["CHR"].unique():
        sub = df[df["CHR"] == chrom]
        # Alternance par index pair/impair du chromosome
        idx_chr = list(df["CHR"].unique()).index(chrom)
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["FST"], mode="markers",
            marker=dict(size=5, color=color),
            name=f"chr{chrom}", showlegend=False, hoverinfo="skip",
        ))

    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(
        y=q_upper, line_dash="dash", line_color="red",
        annotation_text=f"Top {100 * (1 - threshold_q):.1f}% = {q_upper:.4f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Manhattan Plot — FST par SNP",
        xaxis_title="Chromosome", yaxis_title="FST",
        height=500, margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    ld_df = ld_df.copy()
    ld_df["bin"] = (ld_df["dist_kb"] // bin_kb) * bin_kb
    agg = ld_df.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(
        agg, x="bin", y="r2",
        labels={"bin": "Distance (kb)", "r2": "r² moyen"},
        title="Déséquilibre de liaison (LD decay)", height=450,
    )
    fig.update_traces(line=dict(color="royalblue", width=3))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_kinship_heatmap(G, labels):
    df = pd.DataFrame(G, index=labels, columns=labels)
    fig = px.imshow(
        df, color_continuous_scale="RdBu_r", zmin=-0.3, zmax=0.5,
        title="Matrice de parenté (GRM)", height=650, aspect="auto",
    )
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ============================================================
# RAPPORT HTML
# ============================================================

def build_report_html(config, stats, figures=None):
    figures = figures or {}
    first = True
    fig_blocks = []
    for title, fig in figures.items():
        if fig is None:
            continue
        html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if first else False,
        )
        first = False
        fig_blocks.append(f"<h2>{title}</h2>{html}")

    figs_html = "\n".join(fig_blocks) if fig_blocks else "<p>Aucune figure.</p>"

    return f"""<!DOCTYPE html>
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
<p><b>Projet :</b> {config.get('project_name', 'N/A')} — 
   <b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="summary">
<h2>Résumé exécutif</h2>
<ul>
 <li>Individus analysés : <b>{stats['n_ind_final']}</b> (sur {stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{stats['n_snp_final']}</b> (sur {stats['n_snp_init']})</li>
 <li>Individus exclus (QC) : <b>{stats['excluded_ind']}</b></li>
 <li>SNPs exclus (QC) : <b>{stats['excluded_snp']}</b></li>
</ul>
</div>

{figs_html}

<hr>
<p style="font-size:0.85em; color:#666">
Rapport généré automatiquement par Bovine SNP Platform v2.0.
</p>
</body>
</html>
"""


# ============================================================
# STREAMLIT — ÉTAT
# ============================================================

_STATE_KEYS = [
    "gt", "ind_df", "snp_df",
    "gt_filt", "ind_filt", "snp_filt", "qc_stats",
    "pca_scores", "pca_var", "mds_coords", "kinship",
    "ld_df", "fst",
    "run_requested",
]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def invalidate_downstream():
    """À appeler après (re)exécution du QC."""
    for k in ["pca_scores", "pca_var", "mds_coords", "kinship",
              "ld_df", "fst"]:
        st.session_state[k] = None


def has_data() -> bool:
    return st.session_state.gt is not None


def has_qc() -> bool:
    return st.session_state.gt_filt is not None


# ============================================================
# STREAMLIT — INTERFACE
# ============================================================

def main():
    init_state()

    st.title("🐄 Bovine SNP Platform")
    st.caption(
        "Pipeline complet de bioinformatique pour puces SNP bovines — "
        "100 % Python, déployable sur Streamlit Cloud."
    )

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "Upload PED/MAP"], index=0)

        if mode == "Demo":
            col1, col2 = st.columns(2)
            n_ind = col1.number_input("Individus", 20, 1000, 150, 10)
            n_snp = col2.number_input("SNPs", 100, 10000, 800, 100)
            n_pop = st.slider("Populations", 2, 10, 4)
            if st.button("🎲 Générer le jeu de démo", use_container_width=True):
                with st.spinner("Génération des données synthétiques..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop)
                    )
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    st.session_state.gt_filt = None
                    invalidate_downstream()
                st.success(
                    f"✅ {gt.shape[0]} individus × {gt.shape[1]} SNPs "
                    f"({n_pop} populations)"
                )
        else:
            ped_file = st.file_uploader("Fichier .ped", type=["ped", "txt"])
            map_file = st.file_uploader("Fichier .map", type=["map", "txt"])
            if ped_file and map_file:
                if st.button("📥 Charger les fichiers", use_container_width=True):
                    try:
                        with st.spinner("Parsing du .map..."):
                            map_df, rej_map = parse_map(map_file.read())
                        with st.spinner("Parsing du .ped (2 passes)..."):
                            gt, ind_df, rej_ped = parse_ped(
                                ped_file.read(), len(map_df)
                            )
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        st.session_state.gt_filt = None
                        invalidate_downstream()
                        msg = (f"✅ {gt.shape[0]} individus × "
                               f"{gt.shape[1]} SNPs chargés")
                        if rej_map or rej_ped:
                            msg += (f" — ⚠️ lignes rejetées : "
                                    f"{rej_map} (map), {rej_ped} (ped)")
                        st.success(msg)
                    except Exception as e:
                        st.error(f"❌ Erreur de parsing : {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (--geno)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01)
        mind = st.slider("Missingness individu (--mind)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01)
        maf_thr = st.slider("MAF minimal", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01)
        hwe_thr = st.number_input("HWE p-value (exclure < )",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e")
        het_sd = st.slider("Écart-type hétérozygotie",
                           1.0, 5.0, DEFAULT_THRESHOLDS["het_sd"], 0.1)

        st.divider()
        if st.button("🚀 Lancer le pipeline complet", type="primary",
                     use_container_width=True):
            if not has_data():
                st.error("Chargez ou générez d'abord des données.")
            else:
                st.session_state.run_requested = True

        st.divider()
        st.caption("v2.0 — Python pur. Aucune dépendance PLINK/ADMIXTURE/SNeP.")

    # ---------------- MAIN ----------------
    if not has_data():
        st.info("👉 **Pour démarrer** : cliquez sur *Générer le jeu de démo* "
                "dans la barre latérale, ou importez un `.ped` + `.map`.")
        st.markdown("""
        ### Ce que fait cette plateforme
        - **QC** : missingness, MAF, HWE exact, hétérozygotie, filtrage auto
        - **Diversité** : FST par SNP (Nei pondéré), hétérozygotie par race
        - **Structure** : PCA, MDS (IBS), matrice de parenté (GRM)
        - **Démographie** : déséquilibre de liaison (LD decay vectorisé)
        - **Sélection** : Manhattan plot des outliers FST
        - **Rapport** : HTML téléchargeable avec figures Plotly
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure",
                    "📈 Démographie", "🔍 Sélection", "📄 Rapport"])

    # ---------- TAB 1 : Aperçu ----------
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())

        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("Carte des SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True)

    # ---------- TAB 2 : QC ----------
    with tabs[1]:
        st.subheader("Contrôle qualité")
        if st.button("▶ Lancer le QC", use_container_width=True):
            try:
                with st.spinner("Filtrage QC en cours..."):
                    params = {"geno": geno, "mind": mind, "maf": maf_thr,
                              "hwe": hwe_thr, "het_sd": het_sd}
                    gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                        st.session_state.gt,
                        st.session_state.ind_df,
                        st.session_state.snp_df,
                        params,
                    )
                    st.session_state.gt_filt = gt_f
                    st.session_state.ind_filt = ind_f
                    st.session_state.snp_filt = snp_f
                    st.session_state.qc_stats = qc_stats
                    invalidate_downstream()
                st.success("✅ QC terminé.")
            except Exception as e:
                st.error(f"❌ Erreur QC : {e}")

        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Individus exclus", s["excluded_ind"])
            c4.metric("SNPs exclus", s["excluded_snp"])

            gt_full = st.session_state.gt
            miss_ind = missingness_per_ind(gt_full)
            miss_snp = missingness_per_snp(gt_full)
            st.plotly_chart(plot_missingness_dashboard(miss_ind, miss_snp),
                            use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_hist(maf(gt_full), "Spectre MAF", "MAF", "#2ecc71"),
                    use_container_width=True,
                )
            with c2:
                het = heterozygosity(gt_full)
                st.plotly_chart(
                    plot_hist(het, "Hétérozygotie observée", "HET", "#9b59b6"),
                    use_container_width=True,
                )

            with st.expander("Distribution des p-values HWE (post-filtrage)"):
                pv = hwe_pvalues(st.session_state.gt_filt)
                pv_ok = pv[np.isfinite(pv)]
                if len(pv_ok) > 0:
                    st.plotly_chart(
                        plot_hist(pv_ok, "p-values HWE (exact)",
                                  "p-value", "salmon"),
                        use_container_width=True,
                    )
                else:
                    st.info("Aucune p-value HWE calculable.")
        else:
            st.info("Cliquez sur **Lancer le QC** pour démarrer l'analyse.")

    # ---------- TAB 3 : Structure ----------
    with tabs[2]:
        st.subheader("Structure des populations")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC (onglet 🧹).")
        else:
            if st.button("▶ Calculer PCA + MDS + GRM", use_container_width=True):
                try:
                    with st.spinner("PCA..."):
                        scores, var = pca_analysis(st.session_state.gt_filt, 10)
                        st.session_state.pca_scores = scores
                        st.session_state.pca_var = var
                    with st.spinner("MDS (IBS)..."):
                        st.session_state.mds_coords = mds_analysis(
                            st.session_state.gt_filt, 5)
                    with st.spinner("Matrice de parenté (GRM)..."):
                        st.session_state.kinship = kinship_matrix(
                            st.session_state.gt_filt)
                    st.success("✅ Analyses de structure terminées.")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

            labels = (st.session_state.ind_filt["FID"].values
                      if has_qc() else None)

            if st.session_state.pca_scores is not None and labels is not None:
                st.plotly_chart(
                    plot_pca(st.session_state.pca_scores,
                             st.session_state.pca_var, labels),
                    use_container_width=True,
                )

            if st.session_state.mds_coords is not None and labels is not None:
                st.plotly_chart(
                    plot_mds(st.session_state.mds_coords, labels),
                    use_container_width=True,
                )

            if st.session_state.kinship is not None and labels is not None:
                st.plotly_chart(
                    plot_kinship_heatmap(st.session_state.kinship, labels),
                    use_container_width=True,
                )

    # ---------- TAB 4 : Démographie ----------
    with tabs[3]:
        st.subheader("Démographie — LD decay")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC (onglet 🧹).")
        else:
            c1, c2 = st.columns(2)
            max_kb = c1.slider("Distance max (kb)", 100, 5000, 1000, 100)
            max_snp = c2.slider("SNPs max", 200, 5000, 1200, 100)

            if st.button("▶ Calculer le LD decay", use_container_width=True):
                try:
                    with st.spinner("Calcul du LD decay (vectorisé)..."):
                        ld_df = ld_decay(
                            st.session_state.gt_filt,
                            st.session_state.snp_filt["BP"].values,
                            max_kb=float(max_kb),
                            max_snp=int(max_snp),
                        )
                        st.session_state.ld_df = ld_df
                    st.success(f"✅ {len(ld_df):,} paires de SNPs calculées.")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

            if st.session_state.ld_df is not None and not st.session_state.ld_df.empty:
                fig_ld = plot_ld_decay(st.session_state.ld_df)
                if fig_ld:
                    st.plotly_chart(fig_ld, use_container_width=True)
                with st.expander("Voir les données brutes (100 premières)"):
                    st.dataframe(st.session_state.ld_df.head(100),
                                 use_container_width=True)

    # ---------- TAB 5 : Sélection ----------
    with tabs[4]:
        st.subheader("Signatures de sélection (FST)")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC (onglet 🧹).")
        else:
            threshold_q = st.slider("Quantile pour les outliers",
                                    0.95, 0.9999, 0.999, 0.0001,
                                    format="%.4f")
            if st.button("▶ Calculer FST + Manhattan", use_container_width=True):
                try:
                    with st.spinner("Calcul du FST par SNP..."):
                        fst = fst_per_snp(
                            st.session_state.gt_filt,
                            st.session_state.ind_filt["FID"].values,
                        )
                        st.session_state.fst = fst
                    st.success("✅ FST calculé.")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

            if st.session_state.fst is not None:
                fst_clean = st.session_state.fst[np.isfinite(st.session_state.fst)]
                if len(fst_clean) == 0:
                    st.warning("Aucun FST valide — vérifiez le nombre de populations.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FST moyen", f"{fst_clean.mean():.4f}")
                    c2.metric("FST médian", f"{np.median(fst_clean):.4f}")
                    c3.metric("Top outliers",
                              f"{np.quantile(fst_clean, threshold_q):.4f}")

                    fig = plot_manhattan(
                        st.session_state.fst,
                        st.session_state.snp_filt["CHR"].values,
                        threshold_q=threshold_q,
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                    q_upper = float(np.nanquantile(st.session_state.fst,
                                                   threshold_q))
                    mask = np.isfinite(st.session_state.fst) & (
                        st.session_state.fst >= q_upper)
                    outliers = st.session_state.snp_filt[mask].copy()
                    outliers["FST"] = st.session_state.fst[mask]
                    outliers = outliers.sort_values("FST", ascending=False)
                    st.subheader(f"SNPs outliers ({len(outliers)})")
                    st.dataframe(outliers, use_container_width=True)

    # ---------- TAB 6 : Rapport ----------
    with tabs[5]:
        st.subheader("Génération du rapport")
        if st.session_state.qc_stats is None:
            st.warning("⚠️ Lancez au moins le QC pour générer un rapport.")
        else:
            project_name = st.text_input("Nom du projet", "Cattle_Project")
            if st.button("📄 Générer le rapport HTML", use_container_width=True):
                figures = {}
                if st.session_state.pca_scores is not None:
                    figures["PCA"] = plot_pca(
                        st.session_state.pca_scores,
                        st.session_state.pca_var,
                        st.session_state.ind_filt["FID"].values,
                    )
                if st.session_state.mds_coords is not None:
                    figures["MDS (IBS)"] = plot_mds(
                        st.session_state.mds_coords,
                        st.session_state.ind_filt["FID"].values,
                    )
                if st.session_state.fst is not None:
                    figures["Manhattan FST"] = plot_manhattan(
                        st.session_state.fst,
                        st.session_state.snp_filt["CHR"].values,
                    )

                html = build_report_html(
                    config={"project_name": project_name},
                    stats=st.session_state.qc_stats,
                    figures=figures,
                )
                st.download_button(
                    label="⬇ Télécharger le rapport HTML",
                    data=html.encode("utf-8"),
                    file_name=f"rapport_bovine_{datetime.now():%Y%m%d_%H%M}.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.success("✅ Rapport prêt.")
                with st.expander("Prévisualisation"):
                    st.components.v1.html(html, height=700, scrolling=True)

    # ---------- PIPELINE COMPLET ----------
    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        try:
            with st.spinner("Exécution du pipeline complet..."):
                params = {"geno": geno, "mind": mind, "maf": maf_thr,
                          "hwe": hwe_thr, "het_sd": het_sd}
                gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                    st.session_state.gt,
                    st.session_state.ind_df,
                    st.session_state.snp_df,
                    params,
                )
                st.session_state.gt_filt = gt_f
                st.session_state.ind_filt = ind_f
                st.session_state.snp_filt = snp_f
                st.session_state.qc_stats = qc_stats
                invalidate_downstream()

                st.session_state.pca_scores, st.session_state.pca_var = \
                    pca_analysis(gt_f, 10)
                st.session_state.mds_coords = mds_analysis(gt_f, 5)
                st.session_state.kinship = kinship_matrix(gt_f)
                st.session_state.ld_df = ld_decay(
                    gt_f, snp_f["BP"].values, max_kb=1000, max_snp=1000)
                st.session_state.fst = fst_per_snp(gt_f, ind_f["FID"].values)
            st.success("✅ Pipeline terminé ! Consultez les onglets.")
        except Exception as e:
            st.error(f"❌ Erreur pipeline : {e}")


if __name__ == "__main__":
    main()
