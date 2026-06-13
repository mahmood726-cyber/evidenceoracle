#!/usr/bin/env python
"""
EvidenceOracle — Model Training Pipeline
Trains Logistic Regression, Random Forest, and Gradient Boosting to predict
which meta-analysis conclusions are fragile/unstable (likely to be overturned).
"""

import io
import sys
import os
import json
import warnings

# UTF-8 stdout for Windows cp1252 — only when running standalone, not under pytest
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, brier_score_loss,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ── Config ───────────────────────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "feature_matrix.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Features to ALWAYS exclude (identifiers, targets, redundant)
EXCLUDE_COLS = [
    "review_id",
    "fragile_or_unstable",   # primary target
    "low_quality",           # secondary target
    "fragility_sub",         # redundant with robustness_score (identical values)
]

# LEAKAGE EXCLUSIONS — features derived from the same source as the target
# FragilityAtlas columns leak into fragile_or_unstable (they define the target)
# quality_score also leaks because it includes robustness_score as a component
FRAGILITY_LEAK_COLS = [
    "robustness_score", "frac_significant", "frac_reversed", "iqr_theta",
    "eta2_estimator",
    "quality_score",  # composite includes robustness_score
    # top_dimension one-hots
    "topdim_Bias_Correction", "topdim_Leave_One_Out", "topdim_Estimator",
    "topdim_CI_Method",
]
# EvidenceQuality columns leak into low_quality (quality_score defines the grade)
QUALITY_LEAK_COLS = [
    "quality_score",
    # quality_score embeds fragility_sub (already excluded) and other sub-scores
]


# ── Data Loading ─────────────────────────────────────────────────────────
def load_data(target_col="fragile_or_unstable", extra_exclude=None):
    """Load feature matrix and separate features / target.

    extra_exclude: list of column names to drop (leakage prevention).
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Feature columns — remove identifiers, targets, redundant, AND leaked
    drop_set = set(EXCLUDE_COLS)
    if extra_exclude:
        drop_set.update(extra_exclude)
    feature_cols = [c for c in df.columns if c not in drop_set]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    print(f"Features: {len(feature_cols)}")
    print(f"Target '{target_col}': {y.sum()}/{len(y)} positive ({100*y.mean():.1f}%)")

    return X, y, feature_cols, df["review_id"].values


# ── Models ───────────────────────────────────────────────────────────────
def get_models():
    """Return dict of model name -> (model, needs_scaling)."""
    return {
        "LogisticRegression": (
            LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs",
                random_state=SEED, class_weight="balanced"
            ),
            True  # needs feature scaling
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=5,
                random_state=SEED, class_weight="balanced", n_jobs=-1
            ),
            False
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8,
                random_state=SEED
            ),
            False
        ),
    }


# ── Cross-Validation ────────────────────────────────────────────────────
def evaluate_models(X, y, feature_cols, review_ids, target_name):
    """Run stratified K-fold CV for all models. Return performance dict and predictions."""
    models = get_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_performance = {}
    all_predictions = pd.DataFrame({
        "review_id": review_ids,
        "true_label": y,
    })
    all_importances = {}

    for model_name, (model_template, needs_scaling) in models.items():
        print(f"\n  Training {model_name}...")
        fold_metrics = {"auc": [], "accuracy": [], "sensitivity": [], "specificity": [], "brier": []}
        oof_probs = np.zeros(len(y))
        fold_importances = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale if needed
            if needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            # Clone model (fresh instance per fold)
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train, y_train)

            # Predict
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            oof_probs[val_idx] = probs

            # Metrics
            auc = roc_auc_score(y_val, probs)
            acc = accuracy_score(y_val, preds)
            sens = recall_score(y_val, preds, pos_label=1)
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            brier = brier_score_loss(y_val, probs)

            fold_metrics["auc"].append(auc)
            fold_metrics["accuracy"].append(acc)
            fold_metrics["sensitivity"].append(sens)
            fold_metrics["specificity"].append(spec)
            fold_metrics["brier"].append(brier)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                fold_importances.append(model.feature_importances_)
            elif hasattr(model, "coef_"):
                fold_importances.append(np.abs(model.coef_[0]))

        # Aggregate metrics
        perf = {}
        for metric, values in fold_metrics.items():
            perf[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "folds": [float(v) for v in values],
            }

        # Overall OOF AUC
        perf["oof_auc"] = float(roc_auc_score(y, oof_probs))
        all_performance[model_name] = perf

        # Store predictions
        all_predictions[f"{model_name}_prob"] = oof_probs

        # Average feature importance
        if fold_importances:
            avg_imp = np.mean(fold_importances, axis=0)
            imp_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": avg_imp,
            }).sort_values("importance", ascending=False)
            all_importances[model_name] = imp_df

        print(f"    AUC: {perf['auc']['mean']:.3f} +/- {perf['auc']['std']:.3f}")
        print(f"    Accuracy: {perf['accuracy']['mean']:.3f}")
        print(f"    Sensitivity: {perf['sensitivity']['mean']:.3f}")
        print(f"    Specificity: {perf['specificity']['mean']:.3f}")
        print(f"    Brier: {perf['brier']['mean']:.3f}")

    return all_performance, all_predictions, all_importances


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EvidenceOracle: Model Training Pipeline")
    print("=" * 60)

    # ── Primary target: fragile_or_unstable ──
    print("\n" + "=" * 60)
    print("PRIMARY TARGET: fragile_or_unstable")
    print("=" * 60)

    X, y, feature_cols, review_ids = load_data(
        "fragile_or_unstable", extra_exclude=FRAGILITY_LEAK_COLS
    )
    perf_primary, preds_primary, imps_primary = evaluate_models(
        X, y, feature_cols, review_ids, "fragile_or_unstable"
    )

    # ── Secondary target: low_quality ──
    print("\n" + "=" * 60)
    print("SECONDARY TARGET: low_quality")
    print("=" * 60)

    X2, y2, feat2, rids2 = load_data(
        "low_quality", extra_exclude=QUALITY_LEAK_COLS
    )
    perf_secondary, preds_secondary, imps_secondary = evaluate_models(
        X2, y2, feat2, rids2, "low_quality"
    )

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Model performance
    combined_perf = {
        "primary_target": "fragile_or_unstable",
        "secondary_target": "low_quality",
        "n_reviews": int(len(y)),
        "n_features": len(feature_cols),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "primary": perf_primary,
        "secondary": perf_secondary,
    }
    perf_path = os.path.join(RESULTS_DIR, "model_performance.json")
    with open(perf_path, "w") as f:
        json.dump(combined_perf, f, indent=2)
    print(f"\nSaved performance: {perf_path}")

    # Feature importance (top 15 from RF and GBM for primary target)
    importance_data = {}
    for model_name in ["RandomForest", "GradientBoosting"]:
        if model_name in imps_primary:
            top15 = imps_primary[model_name].head(15)
            importance_data[model_name] = [
                {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
                for _, row in top15.iterrows()
            ]
    imp_path = os.path.join(RESULTS_DIR, "feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    print(f"Saved importance: {imp_path}")

    # Predictions
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    preds_primary.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    # ── Headline Results ──
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)

    print("\n--- Primary Target: fragile_or_unstable ---")
    best_model = max(perf_primary, key=lambda m: perf_primary[m]["oof_auc"])
    for mname in perf_primary:
        p = perf_primary[mname]
        marker = " <-- BEST" if mname == best_model else ""
        print(f"  {mname:25s}  AUC={p['oof_auc']:.3f}  "
              f"Acc={p['accuracy']['mean']:.3f}  "
              f"Sens={p['sensitivity']['mean']:.3f}  "
              f"Spec={p['specificity']['mean']:.3f}  "
              f"Brier={p['brier']['mean']:.3f}{marker}")

    print("\n--- Secondary Target: low_quality ---")
    best_model2 = max(perf_secondary, key=lambda m: perf_secondary[m]["oof_auc"])
    for mname in perf_secondary:
        p = perf_secondary[mname]
        marker = " <-- BEST" if mname == best_model2 else ""
        print(f"  {mname:25s}  AUC={p['oof_auc']:.3f}  "
              f"Acc={p['accuracy']['mean']:.3f}  "
              f"Sens={p['sensitivity']['mean']:.3f}  "
              f"Spec={p['specificity']['mean']:.3f}  "
              f"Brier={p['brier']['mean']:.3f}{marker}")

    print("\n--- Top 5 Features (GradientBoosting, primary target) ---")
    if "GradientBoosting" in imps_primary:
        top5 = imps_primary["GradientBoosting"].head(5)
        for _, row in top5.iterrows():
            print(f"  {row['feature']:30s}  importance={row['importance']:.4f}")

    print("\n--- Top 5 Features (RandomForest, primary target) ---")
    if "RandomForest" in imps_primary:
        top5 = imps_primary["RandomForest"].head(5)
        for _, row in top5.iterrows():
            print(f"  {row['feature']:30s}  importance={row['importance']:.4f}")

    # Cross-target agreement
    print("\n--- Cross-Target Agreement ---")
    primary_pred = (preds_primary[f"{best_model}_prob"] >= 0.5).astype(int)
    secondary_pred = (preds_secondary[f"{best_model2}_prob"] >= 0.5).astype(int)
    agreement = (primary_pred == secondary_pred).mean()
    print(f"  Agreement between primary ({best_model}) and secondary ({best_model2}): {agreement:.3f}")

    print("\nDone.")
    return combined_perf


if __name__ == "__main__":
    main()
