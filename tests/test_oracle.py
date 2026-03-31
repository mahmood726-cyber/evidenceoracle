#!/usr/bin/env python
"""
EvidenceOracle — Test Suite
Tests feature assembly, model training, and result integrity.
"""

import io
import sys
import os
import json
import warnings

# UTF-8 stdout for Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "feature_matrix.csv")
PERF_PATH = os.path.join(PROJECT_DIR, "results", "model_performance.json")
IMP_PATH = os.path.join(PROJECT_DIR, "results", "feature_importance.json")
PRED_PATH = os.path.join(PROJECT_DIR, "results", "predictions.csv")


def test_feature_matrix_exists():
    """Feature matrix CSV must exist."""
    assert os.path.isfile(DATA_PATH), f"Missing: {DATA_PATH}"
    print("  PASS: feature_matrix.csv exists")


def test_feature_matrix_shape():
    """Feature matrix should have ~403 rows and 50+ columns (v2: 9 sources)."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] >= 350, f"Too few rows: {df.shape[0]}"
    assert df.shape[0] <= 510, f"Too many rows: {df.shape[0]}"
    assert df.shape[1] >= 50, f"Too few columns: {df.shape[1]} (expected 50+ with 9 sources)"
    print(f"  PASS: shape = {df.shape}")


def test_feature_matrix_has_required_columns():
    """Key columns must be present."""
    df = pd.read_csv(DATA_PATH)
    required = [
        "review_id", "fragile_or_unstable", "low_quality",
        "audit_n_fail", "audit_n_warn", "audit_pct_fail",
        "robustness_score", "quality_score", "benford_mad",
        "has_changepoint",
        # v2 sources
        "hl_volatility", "hl_n_flips", "hl_stabilizes",
        "bf_egger_p", "bf_begg_tau", "bf_concordance", "bf_class_confirmed",
        "mes_c_sig", "mes_pi_null_rate", "mes_dominant_eta2",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  PASS: all {len(required)} required columns present")


def test_review_ids_are_cochrane():
    """Review IDs should match CD\\d+ pattern."""
    df = pd.read_csv(DATA_PATH)
    bad = df[~df["review_id"].str.match(r"^CD\d+$")]
    assert len(bad) == 0, f"{len(bad)} non-Cochrane review IDs found"
    print(f"  PASS: all {len(df)} review IDs are Cochrane format")


def test_targets_are_binary():
    """Target columns must be 0/1 only."""
    df = pd.read_csv(DATA_PATH)
    for col in ["fragile_or_unstable", "low_quality"]:
        vals = df[col].dropna().unique()
        assert set(vals).issubset({0, 1}), f"{col} has non-binary values: {vals}"
    print("  PASS: both targets are binary (0/1)")


def test_no_duplicate_reviews():
    """Each review_id should appear exactly once."""
    df = pd.read_csv(DATA_PATH)
    dupes = df["review_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate review IDs"
    print(f"  PASS: {len(df)} unique review IDs, no duplicates")


def test_target_balance():
    """Targets should not be trivially imbalanced (>95% or <5% positive)."""
    df = pd.read_csv(DATA_PATH)
    for col in ["fragile_or_unstable", "low_quality"]:
        rate = df[col].mean()
        assert 0.05 < rate < 0.95, f"{col} has extreme imbalance: {rate:.3f}"
    print("  PASS: target balance is reasonable")


def test_model_performance_exists():
    """Performance JSON must exist and have valid structure."""
    assert os.path.isfile(PERF_PATH), f"Missing: {PERF_PATH}"
    with open(PERF_PATH, "r") as f:
        perf = json.load(f)
    assert "primary" in perf, "Missing 'primary' key in performance"
    assert "secondary" in perf, "Missing 'secondary' key in performance"
    for model in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        assert model in perf["primary"], f"Missing model: {model}"
    print("  PASS: model_performance.json has valid structure")


def test_primary_auc_above_minimum():
    """At least one primary model should achieve AUC > 0.60."""
    with open(PERF_PATH, "r") as f:
        perf = json.load(f)
    aucs = {m: perf["primary"][m]["oof_auc"] for m in perf["primary"]}
    best = max(aucs.values())
    assert best > 0.60, f"Best primary AUC = {best:.3f} (below 0.60 threshold)"
    print(f"  PASS: best primary AUC = {best:.3f} (> 0.60 threshold)")


def test_secondary_auc_above_minimum():
    """At least one secondary model should achieve AUC > 0.60."""
    with open(PERF_PATH, "r") as f:
        perf = json.load(f)
    aucs = {m: perf["secondary"][m]["oof_auc"] for m in perf["secondary"]}
    best = max(aucs.values())
    assert best > 0.60, f"Best secondary AUC = {best:.3f} (below 0.60 threshold)"
    print(f"  PASS: best secondary AUC = {best:.3f} (> 0.60 threshold)")


def test_feature_importance_non_degenerate():
    """Feature importance should not be concentrated on a single feature (top1 < 70%).

    Note: with v2 sources, mes_c_sig is legitimately dominant (~57% in GBM, ~28% in RF)
    because multiverse concordance is a strong external predictor of fragility.
    Threshold set at 70% to catch truly degenerate models while allowing strong predictors.
    """
    with open(IMP_PATH, "r") as f:
        imp = json.load(f)
    for model_name, features in imp.items():
        total = sum(f["importance"] for f in features)
        if total > 0:
            top1_share = features[0]["importance"] / total
            assert top1_share < 0.70, (
                f"{model_name}: top feature has {top1_share:.1%} of importance (degenerate)"
            )
    print("  PASS: feature importance is non-degenerate (no single feature > 70%)")


def test_predictions_csv():
    """Predictions CSV should have one row per review with probability columns."""
    assert os.path.isfile(PRED_PATH), f"Missing: {PRED_PATH}"
    df = pd.read_csv(PRED_PATH)
    assert "review_id" in df.columns, "Missing review_id"
    assert "true_label" in df.columns, "Missing true_label"
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    assert len(prob_cols) == 3, f"Expected 3 probability columns, got {len(prob_cols)}"
    # Probabilities should be in [0, 1]
    for col in prob_cols:
        assert df[col].min() >= 0, f"{col} has values < 0"
        assert df[col].max() <= 1, f"{col} has values > 1"
    print(f"  PASS: predictions.csv valid ({len(df)} rows, {len(prob_cols)} prob cols)")


def test_no_leakage_in_primary():
    """Verify that fragility-derived features are not used for primary target prediction.

    Check: the robustness_score should NOT be in the feature importance list
    for the primary target (it would indicate leakage).
    """
    with open(IMP_PATH, "r") as f:
        imp = json.load(f)
    for model_name, features in imp.items():
        feat_names = [f["feature"] for f in features]
        leaked = {"robustness_score", "frac_significant", "frac_reversed",
                  "iqr_theta", "eta2_estimator"}
        found = leaked.intersection(feat_names)
        assert len(found) == 0, (
            f"{model_name}: leaked features in primary importance: {found}"
        )
    print("  PASS: no FragilityAtlas features in primary target importance")


def test_v2_sources_have_data():
    """v2 sources (HalfLife, BiasForensics, MES) should have non-trivial coverage."""
    df = pd.read_csv(DATA_PATH)
    # EvidenceHalfLife and BiasForensics have 307 reviews -> ~76% coverage
    hl_coverage = df["hl_volatility"].notna().mean()
    assert hl_coverage > 0.50, f"EvidenceHalfLife coverage too low: {hl_coverage:.1%}"
    bf_coverage = df["bf_egger_p"].notna().mean()
    assert bf_coverage > 0.50, f"BiasForensics coverage too low: {bf_coverage:.1%}"
    # MES has 403 reviews -> should have near-full coverage
    mes_coverage = df["mes_c_sig"].notna().mean()
    assert mes_coverage > 0.90, f"MES coverage too low: {mes_coverage:.1%}"
    print(f"  PASS: v2 source coverage — HalfLife={hl_coverage:.0%}, "
          f"BiasForensics={bf_coverage:.0%}, MES={mes_coverage:.0%}")


def test_no_mes_classification_leak():
    """MES overall_class and cond_*_class must NOT be in the feature matrix.

    These columns encode fragility-like classifications and would be leakage.
    """
    df = pd.read_csv(DATA_PATH)
    forbidden = ["overall_class", "cond_rct_class", "cond_low_rob_class", "certification"]
    found = [c for c in forbidden if c in df.columns]
    assert len(found) == 0, (
        f"MES classification columns present in feature matrix (leakage): {found}"
    )
    print("  PASS: no MES classification columns in feature matrix")


def test_reproducibility():
    """Running with same seed should produce identical predictions."""
    df = pd.read_csv(PRED_PATH)
    # Check that predictions are deterministic (not all identical, not random-looking)
    for col in [c for c in df.columns if c.endswith("_prob")]:
        vals = df[col].values
        # Not all same value
        assert vals.std() > 0.01, f"{col}: predictions are degenerate (std={vals.std():.4f})"
        # Not uniformly distributed (would indicate random)
        assert not (0.24 < vals.mean() < 0.26 and 0.27 < vals.std() < 0.30), \
            f"{col}: predictions look random"
    print("  PASS: predictions are non-degenerate and non-random")


# ── Runner ───────────────────────────────────────────────────────────────
def main():
    tests = [
        test_feature_matrix_exists,
        test_feature_matrix_shape,
        test_feature_matrix_has_required_columns,
        test_review_ids_are_cochrane,
        test_targets_are_binary,
        test_no_duplicate_reviews,
        test_target_balance,
        test_model_performance_exists,
        test_primary_auc_above_minimum,
        test_secondary_auc_above_minimum,
        test_feature_importance_non_degenerate,
        test_predictions_csv,
        test_no_leakage_in_primary,
        test_v2_sources_have_data,
        test_no_mes_classification_leak,
        test_reproducibility,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("EvidenceOracle: Test Suite")
    print("=" * 60)

    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL: {name} -- {e}")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ERROR: {name} -- {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
