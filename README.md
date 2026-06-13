# EvidenceOracle

Machine-learning pipeline that predicts which Cochrane meta-analytic conclusions
are fragile or unstable (likely to be overturned by new evidence).

## What this repo does

- **`assemble_features.py`** — assembles a unified feature matrix
  (`data/feature_matrix.csv`) from pre-computed features across 9 upstream
  sources (MetaAudit, FragilityAtlas, EvidenceQuality, BenfordMA, MetaShift,
  Pairwise70, EvidenceHalfLife, BiasForensics, MES), with ground-truth labels.
- **`train_model.py`** — trains Logistic Regression, Random Forest, and Gradient
  Boosting classifiers with 5-fold stratified cross-validation. Predicts two
  targets (`fragile_or_unstable`, `low_quality`), applies leakage-exclusion
  column sets, and writes out-of-fold AUC / accuracy / sensitivity / specificity
  / Brier metrics, feature importances, and predictions to `results/`.
- **`dashboard.html` / `index.html`** — offline single-file dashboard of the
  results (no external CDN dependencies).

## Run

```bash
pip install -r requirements.txt
python assemble_features.py    # build data/feature_matrix.csv
python train_model.py          # train and write results/
python -m pytest -q            # 16 tests
```

## Notes

- Models are trained on the Cochrane feature matrix in `data/`; accuracy may
  differ for non-Cochrane systematic reviews.
- See `E156-PROTOCOL.md` for the study protocol and headline estimand.

## License

MIT — see `LICENSE`.
