#!/usr/bin/env python
"""
EvidenceOracle — Feature Assembly Pipeline (v2)
Loads pre-computed features from 9 sources, constructs a unified
feature matrix with ground truth labels, and outputs data/feature_matrix.csv.

Sources 1-6: MetaAudit, FragilityAtlas, EvidenceQuality, BenfordMA, MetaShift, Pairwise70
Sources 7-9: EvidenceHalfLife, BiasForensics, MES (Multiverse Evidence Synthesis)
"""

import io
import sys
import os
import json
import re
import glob
import warnings
from pathlib import Path

# UTF-8 stdout for Windows cp1252 — only when running standalone, not under pytest
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DRIVE_ROOTS = [Path("C:/"), Path("D:/")]


def _candidate_roots(*relative_roots):
    candidates = []
    for drive_root in DRIVE_ROOTS:
        for relative_root in relative_roots:
            candidates.append(drive_root / relative_root)
    return candidates


def resolve_existing_file(env_var, relative_path, *relative_roots):
    candidates = []
    env_root = os.getenv(env_var, "")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative_path)
    candidates.extend(root / relative_path for root in _candidate_roots(*relative_roots))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"{relative_path} not found. Set {env_var} or place the source under one of: "
        + ", ".join(str(root) for root in _candidate_roots(*relative_roots))
    )


def resolve_existing_dir(env_var, *relative_dirs):
    candidates = []
    env_dir = os.getenv(env_var, "")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(_candidate_roots(*relative_dirs))

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        f"Directory for {env_var} not found. Set {env_var} or place it under one of: "
        + ", ".join(str(path) for path in _candidate_roots(*relative_dirs))
    )


METAAUDIT_CSV = resolve_existing_file("METAAUDIT_ROOT", Path("results") / "audit_results.csv", "MetaAudit", "Models/MetaAudit")
FRAGILITY_CSV = resolve_existing_file(
    "FRAGILITY_ROOT",
    Path("data") / "output" / "fragility_atlas_results.csv",
    "FragilityAtlas",
    "Models/FragilityAtlas",
)
EVIDQUALITY_JSON = resolve_existing_file(
    "EVIDENCEQUALITY_ROOT",
    Path("data") / "reviews_compact.json",
    "EvidenceQuality",
    "Models/EvidenceQuality",
)
BENFORD_JSON = resolve_existing_file(
    "BENFORD_ROOT",
    Path("data") / "corpus_digits_compact.json",
    "BenfordMA",
    "Models/BenfordMA",
)
METASHIFT_JSON = resolve_existing_file(
    "METASHIFT_ROOT",
    Path("data") / "compact.json",
    "MetaShift",
    "Models/MetaShift",
)
PAIRWISE70_DIR = resolve_existing_dir("PAIRWISE70_DIR", "Projects/Pairwise70/data", "Pairwise70/data")
HALFLIFE_CSV = resolve_existing_file(
    "EVIDENCEHALFLIFE_ROOT",
    Path("data") / "output" / "half_life_results.csv",
    "EvidenceHalfLife",
    "Models/EvidenceHalfLife",
)
BIASFORENSICS_CSV = resolve_existing_file(
    "BIASFORENSICS_ROOT",
    Path("data") / "output" / "bias_forensics_results.csv",
    "BiasForensics",
    "Models/BiasForensics",
)
MES_CSV = resolve_existing_file(
    "MES_ROOT",
    Path("validation") / "results" / "batch_results.csv",
    "Models/MES",
    "MES",
)

OUTPUT_DIR = SCRIPT_DIR / "data"
OUTPUT_CSV = OUTPUT_DIR / "feature_matrix.csv"


# ── Helper: Benford MAD ──────────────────────────────────────────────────
BENFORD_D1 = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

def compute_benford_mad(digits_list):
    """Compute mean absolute deviation of first-digit distribution from Benford's law."""
    d1_values = [d["d1"] for d in digits_list if d.get("d1") and 1 <= d["d1"] <= 9]
    if len(d1_values) < 4:
        return np.nan
    counts = np.zeros(9)
    for v in d1_values:
        counts[v - 1] += 1
    observed = counts / counts.sum()
    return float(np.mean(np.abs(observed - BENFORD_D1)))


# ── Source 1: MetaAudit ──────────────────────────────────────────────────
def load_metaaudit():
    print("[1/9] Loading MetaAudit...")
    df = pd.read_csv(METAAUDIT_CSV)
    # Extract review_id from ma_id: CD000028_pub4_data__A1 -> CD000028
    df["review_id"] = df["ma_id"].str.extract(r"^(CD\d+)")
    n_reviews = df["review_id"].nunique()
    n_mas = df["ma_id"].nunique()

    # Number of unique MAs per review (for denominator)
    mas_per_review = df.groupby("review_id")["ma_id"].nunique().rename("n_mas")

    # Severity flags per review
    is_fail = df["severity"].isin(["FAIL", "CRITICAL"])
    is_warn = df["severity"] == "WARN"

    audit_n_fail = df[is_fail].groupby("review_id").size().rename("audit_n_fail")
    audit_n_warn = df[is_warn].groupby("review_id").size().rename("audit_n_warn")

    # Fraction of MAs with >= 1 FAIL
    ma_has_fail = df[is_fail].groupby(["review_id", "ma_id"]).size().reset_index()
    mas_with_fail = ma_has_fail.groupby("review_id")["ma_id"].nunique().rename("mas_with_fail")
    audit_pct_fail = (mas_with_fail / mas_per_review).rename("audit_pct_fail")

    # Specific detector FAIL flags (per review: 1 if any MA has that detector FAIL/CRITICAL)
    detector_flags = {}
    for module in ["pub_bias", "model_misspec", "prediction_gap", "excess_sig", "integrity"]:
        mask = is_fail & (df["module"] == module)
        flagged = df[mask].groupby("review_id").size() > 0
        detector_flags[f"audit_{module}_flag"] = flagged.astype(int)

    result = pd.DataFrame(index=df["review_id"].unique())
    result = result.join(audit_n_fail).fillna({"audit_n_fail": 0})
    result = result.join(audit_n_warn).fillna({"audit_n_warn": 0})
    result = result.join(audit_pct_fail).fillna({"audit_pct_fail": 0.0})
    for col, series in detector_flags.items():
        result = result.join(pd.DataFrame({col: series})).fillna({col: 0})

    result.index.name = "review_id"
    result = result.reset_index()
    # Ensure integer types
    for c in ["audit_n_fail", "audit_n_warn"]:
        result[c] = result[c].astype(int)
    for c in detector_flags:
        result[c] = result[c].astype(int)

    print(f"  -> {n_reviews} reviews, {n_mas} MAs, {int(result['audit_n_fail'].sum())} total FAIL flags")
    return result


# ── Source 2: FragilityAtlas ─────────────────────────────────────────────
def load_fragility():
    print("[2/9] Loading FragilityAtlas...")
    df = pd.read_csv(FRAGILITY_CSV)
    cols = [
        "review_id", "robustness_score", "classification",
        "frac_significant", "frac_reversed", "iqr_theta",
        "eta2_estimator", "top_dimension"
    ]
    result = df[cols].copy()
    # Binary target: fragile or unstable
    result["fragile_or_unstable"] = (
        result["classification"].isin(["Fragile", "Unstable"])
    ).astype(int)
    print(f"  -> {len(result)} reviews, {result['fragile_or_unstable'].sum()} fragile/unstable")
    return result


# ── Source 3: EvidenceQuality ────────────────────────────────────────────
def load_evidence_quality():
    print("[3/9] Loading EvidenceQuality...")
    with open(EVIDQUALITY_JSON, "r") as f:
        data = json.load(f)

    rows = []
    for r in data:
        row = {
            "review_id": r["review_id"],
            "quality_score": r.get("quality_score"),
            "quality_grade": r.get("quality_grade"),
            # Component scores
            "fragility_sub": r.get("fragility", {}).get("robustness_score") if isinstance(r.get("fragility"), dict) else r.get("fragility"),
            "bias_sub": r.get("bias", {}).get("n_detect") if isinstance(r.get("bias"), dict) else r.get("bias"),
            "prediction_sub": r.get("prediction", {}).get("pi_ci_ratio") if isinstance(r.get("prediction"), dict) else r.get("prediction"),
            "orb_sub": r.get("orb", {}).get("orb_score") if isinstance(r.get("orb"), dict) else r.get("orb"),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    # Secondary target: low quality (C or D)
    result["low_quality"] = (
        result["quality_grade"].isin(["C", "D"])
    ).astype(int)
    print(f"  -> {len(result)} reviews, {result['low_quality'].sum()} low-quality (C/D)")
    return result


# ── Source 4: BenfordMA ──────────────────────────────────────────────────
def load_benford():
    print("[4/9] Loading BenfordMA...")
    with open(BENFORD_JSON, "r") as f:
        data = json.load(f)

    reviews = data["reviews"]
    rows = []
    for r in reviews:
        mad = compute_benford_mad(r.get("digits", []))
        rows.append({
            "review_id": r["review_id"],
            "benford_mad": mad,
        })

    result = pd.DataFrame(rows)
    valid = result["benford_mad"].notna().sum()
    print(f"  -> {len(result)} reviews, {valid} with valid MAD")
    return result


# ── Source 5: MetaShift ──────────────────────────────────────────────────
def load_metashift():
    print("[5/9] Loading MetaShift...")
    with open(METASHIFT_JSON, "r") as f:
        data = json.load(f)

    rows = []

    # Reviews with trajectory data (have changepoint info)
    for r in data.get("reviews_with_trajectory", []):
        cp = r.get("changepoints", {})
        # Count how many of {effect, significance, heterogeneity} have a changepoint
        n_cp = sum(1 for k in ["effect", "significance", "heterogeneity"]
                   if cp.get(k) is not None)
        rows.append({
            "review_id": r["review_id"],
            "has_changepoint": int(n_cp > 0),
            "n_changepoints": n_cp,
            "stability_class": r.get("stability_class", "Unknown"),
        })

    # Aggregate-only reviews (no trajectory -> no changepoint detected)
    for r in data.get("reviews_aggregate_only", []):
        rows.append({
            "review_id": r["review_id"],
            "has_changepoint": 0,
            "n_changepoints": 0,
            "stability_class": "No Trajectory",
        })

    result = pd.DataFrame(rows)
    has_cp = result["has_changepoint"].sum()
    print(f"  -> {len(result)} reviews, {has_cp} with changepoints")
    return result


# ── Source 6: Pairwise70 (basic study-level features) ────────────────────
def load_pairwise70():
    print("[6/9] Loading Pairwise70 basics...")
    rda_files = glob.glob(os.path.join(PAIRWISE70_DIR, "*.rda"))
    print(f"  Found {len(rda_files)} .rda files")

    rows = []
    for fpath in rda_files:
        fname = os.path.basename(fpath)
        # Parse review_id and pub version from filename
        # e.g. CD000028_pub4_data.rda -> CD000028, 4
        m = re.match(r"(CD\d+)_pub(\d+)_data\.rda", fname)
        if not m:
            continue
        review_id = m.group(1)
        pub_version = int(m.group(2))

        try:
            import pyreadr
            rdata = pyreadr.read_r(fpath)
            df = list(rdata.values())[0]

            # Primary MA = Analysis.number == 1 (first analysis)
            primary = df[df["Analysis.number"] == 1]
            if len(primary) == 0:
                primary = df  # fallback: use all

            k = primary["Study"].nunique()

            # Determine data type
            has_binary = primary["Experimental.cases"].notna().any() and primary["Control.cases"].notna().any()
            has_continuous = primary["Experimental.mean"].notna().any() and primary["Control.mean"].notna().any()
            has_giv = primary["GIV.Mean"].notna().any()

            if has_giv:
                data_type = "GIV"
            elif has_continuous and not has_binary:
                data_type = "continuous"
            elif has_binary:
                data_type = "binary"
            else:
                data_type = "unknown"

            # Total N
            total_n = 0
            if has_binary or has_continuous:
                exp_n = primary["Experimental.N"].dropna().sum()
                ctl_n = primary["Control.N"].dropna().sum()
                total_n = int(exp_n + ctl_n)

            rows.append({
                "review_id": review_id,
                "pub_version": pub_version,
                "k": k,
                "total_n": total_n,
                "data_type": data_type,
            })
        except Exception as e:
            # Skip problematic files
            pass

    result = pd.DataFrame(rows)
    # If multiple pub versions, keep the latest
    result = result.sort_values("pub_version", ascending=False).drop_duplicates("review_id", keep="first")
    print(f"  -> {len(result)} reviews loaded, data types: {result['data_type'].value_counts().to_dict()}")
    return result


# ── Source 7: EvidenceHalfLife ──────────────────────────────────────────
def load_halflife():
    print("[7/9] Loading EvidenceHalfLife...")
    df = pd.read_csv(HALFLIFE_CSV)

    # Features: half_life_k, half_life_year (may be NaN if never stabilized),
    # volatility, n_flips, final_class (Robust/Moderate/Fragile/Unstable),
    # stabilizes (Yes/No), early_stabilizer (Yes/No), half_life_pct, year_span
    result = pd.DataFrame()
    result["review_id"] = df["review_id"]
    result["hl_half_life_k"] = pd.to_numeric(df["half_life_k"], errors="coerce")
    result["hl_volatility"] = pd.to_numeric(df["volatility"], errors="coerce")
    result["hl_n_flips"] = pd.to_numeric(df["n_flips"], errors="coerce")
    result["hl_half_life_pct"] = pd.to_numeric(df["half_life_pct"], errors="coerce")
    result["hl_year_span"] = pd.to_numeric(df["year_span"], errors="coerce")
    result["hl_final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

    # Binary: stabilizes, early_stabilizer
    result["hl_stabilizes"] = (df["stabilizes"] == "Yes").astype(int)
    result["hl_early_stabilizer"] = (df["early_stabilizer"] == "Yes").astype(int)

    # One-hot final_class (temporal stability class — NOT the same as FragilityAtlas)
    result["hl_class_unstable"] = (df["final_class"] == "Unstable").astype(int)
    result["hl_class_robust"] = (df["final_class"] == "Robust").astype(int)

    print(f"  -> {len(result)} reviews, {result['hl_stabilizes'].sum()} stabilized, "
          f"{result['hl_class_unstable'].sum()} temporally unstable")
    return result


# ── Source 8: BiasForensics ─────────────────────────────────────────────
def load_bias_forensics():
    print("[8/9] Loading BiasForensics...")
    df = pd.read_csv(BIASFORENSICS_CSV)

    result = pd.DataFrame()
    result["review_id"] = df["review_id"]

    # Core bias detection metrics
    result["bf_egger_p"] = pd.to_numeric(df["egger_p"], errors="coerce")
    result["bf_egger_sig"] = pd.to_numeric(df["egger_sig"], errors="coerce")
    result["bf_begg_tau"] = pd.to_numeric(df["begg_tau"], errors="coerce")
    result["bf_begg_sig"] = pd.to_numeric(df["begg_sig"], errors="coerce")

    # P-curve / z-curve evidence strength
    result["bf_pcurve_evidential"] = pd.to_numeric(df["pcurve_evidential"], errors="coerce")
    result["bf_pcurve_inadequate"] = pd.to_numeric(df["pcurve_inadequate"], errors="coerce")
    result["bf_zcurve_edr"] = pd.to_numeric(df["zcurve_edr"], errors="coerce")

    # Trim-and-fill, selection model
    result["bf_tf_k0"] = pd.to_numeric(df["tf_k0"], errors="coerce")
    result["bf_n_detect"] = pd.to_numeric(df["n_detect"], errors="coerce")
    result["bf_max_shift"] = pd.to_numeric(df["max_shift"], errors="coerce")
    result["bf_mean_shift"] = pd.to_numeric(df["mean_shift"], errors="coerce")
    result["bf_concordance"] = pd.to_numeric(df["concordance"], errors="coerce")

    # Bias classification one-hot (Confirmed/Suspected/Clean/Discordant)
    result["bf_class_confirmed"] = (df["bias_class"] == "Confirmed").astype(int)
    result["bf_class_suspected"] = (df["bias_class"] == "Suspected").astype(int)

    print(f"  -> {len(result)} reviews, "
          f"{result['bf_class_confirmed'].sum()} confirmed bias, "
          f"{result['bf_class_suspected'].sum()} suspected bias")
    return result


# ── Source 9: MES (Multiverse Evidence Synthesis) ───────────────────────
def load_mes():
    print("[9/9] Loading MES...")
    df = pd.read_csv(MES_CSV)

    # Filter to status == OK (skip SKIP rows with no data)
    df = df[df["status"] == "OK"].copy()

    # Keep only one row per review_id (take first if duplicates)
    df = df.drop_duplicates(subset="review_id", keep="first")

    result = pd.DataFrame()
    result["review_id"] = df["review_id"].values

    # c_sig: concordance significance (fraction of multiverse agreeing with primary)
    result["mes_c_sig"] = pd.to_numeric(df["c_sig"].values, errors="coerce")

    # pi_null_rate: proportion of multiverse specs yielding null result
    result["mes_pi_null_rate"] = pd.to_numeric(df["pi_null_rate"].values, errors="coerce")

    # dominant_eta2: effect size of the dominant analytical dimension
    result["mes_dominant_eta2"] = pd.to_numeric(df["dominant_eta2"].values, errors="coerce")

    # dominant_dim one-hot (almost all bias_correction, but encode it)
    result["mes_dim_bias_correction"] = (df["dominant_dim"].values == "bias_correction").astype(int)

    # NOTE: overall_class, cond_rct_class, cond_low_rob_class are EXCLUDED
    # because they encode fragility-like classifications.
    # certification is EXCLUDED because it's constant (all PASS).

    print(f"  -> {len(result)} reviews, "
          f"mean c_sig={result['mes_c_sig'].mean():.3f}, "
          f"mean pi_null_rate={result['mes_pi_null_rate'].mean():.3f}")
    return result


# ── Assembly ─────────────────────────────────────────────────────────────
def assemble():
    print("=" * 60)
    print("EvidenceOracle: Feature Assembly Pipeline (v2 — 9 sources)")
    print("=" * 60)
    print()

    # Load all sources
    audit_df     = load_metaaudit()
    fragility_df = load_fragility()
    quality_df   = load_evidence_quality()
    benford_df   = load_benford()
    metashift_df = load_metashift()
    pairwise_df  = load_pairwise70()
    halflife_df  = load_halflife()
    biasfor_df   = load_bias_forensics()
    mes_df       = load_mes()

    print()
    print("-" * 60)
    print("Joining on review_id (intersection)...")

    # Start with fragility (defines the primary target)
    merged = fragility_df.copy()
    merged = merged.merge(audit_df, on="review_id", how="left")
    merged = merged.merge(quality_df, on="review_id", how="left")
    merged = merged.merge(benford_df, on="review_id", how="left")
    merged = merged.merge(metashift_df, on="review_id", how="left")
    merged = merged.merge(pairwise_df, on="review_id", how="left")
    merged = merged.merge(halflife_df, on="review_id", how="left")
    merged = merged.merge(biasfor_df, on="review_id", how="left")
    merged = merged.merge(mes_df, on="review_id", how="left")

    print(f"  Rows after merge: {len(merged)}")
    print(f"  Columns: {len(merged.columns)}")

    # One-hot encode categorical features
    # data_type -> binary columns
    if "data_type" in merged.columns:
        for dt in ["binary", "continuous", "GIV"]:
            merged[f"data_type_{dt}"] = (merged["data_type"] == dt).astype(int)

    # top_dimension -> binary columns
    if "top_dimension" in merged.columns:
        top_dims = merged["top_dimension"].dropna().unique()
        for dim in top_dims:
            col_name = "topdim_" + re.sub(r"\W+", "_", str(dim)).strip("_")
            merged[col_name] = (merged["top_dimension"] == dim).astype(int)

    # stability_class -> binary
    if "stability_class" in merged.columns:
        merged["never_stabilized"] = (merged["stability_class"] == "Never Stabilized").astype(int)

    # Drop raw categorical columns (keep encoded versions)
    drop_cols = ["classification", "quality_grade", "data_type", "top_dimension",
                 "stability_class", "pub_version"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Reorder: review_id first, targets last
    target_cols = ["fragile_or_unstable", "low_quality"]
    other_cols = [c for c in merged.columns if c not in ["review_id"] + target_cols]
    merged = merged[["review_id"] + other_cols + target_cols]

    # Summary statistics
    print()
    print("=" * 60)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape: {merged.shape[0]} reviews x {merged.shape[1]} columns")
    print(f"  Features: {merged.shape[1] - 3} (excluding review_id + 2 targets)")
    print()

    # Missing data report
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    missing = merged[numeric_cols].isna().sum()
    if missing.any():
        print("Missing values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} ({100*missing[col]/len(merged):.1f}%)")
    else:
        print("No missing values in numeric features.")

    print()
    print("Target distribution:")
    for t in target_cols:
        if t in merged.columns:
            pos = merged[t].sum()
            total = merged[t].notna().sum()
            print(f"  {t}: {pos}/{total} positive ({100*pos/total:.1f}%)")

    print()
    print("Feature statistics:")
    feat_cols = [c for c in merged.columns if c not in ["review_id"] + target_cols]
    for c in feat_cols[:20]:
        if merged[c].dtype in [np.float64, np.int64, float, int]:
            print(f"  {c}: mean={merged[c].mean():.3f}, std={merged[c].std():.3f}, "
                  f"min={merged[c].min():.3f}, max={merged[c].max():.3f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print()
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    assemble()
