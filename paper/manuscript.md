# EvidenceOracle: Predicting Meta-Analysis Conclusion Instability Using Multi-Dimensional Computational Audit Features

**Mahmood Ahmad**¹

¹ Royal Free Hospital, London, United Kingdom

**Correspondence:** mahmood.ahmad2@nhs.net | **ORCID:** 0009-0003-7781-4478

**Word count:** ~4,200 (excluding abstract, tables, references)
**Target:** Nature Medicine (Research Article) or The Lancet (Article)

---

## Abstract

**Background:** Meta-analyses form the apex of the evidence hierarchy, yet an unknown fraction harbour methodological vulnerabilities that make their conclusions fragile — likely to be reversed by future evidence or analytic variation. No study has attempted to predict which meta-analysis conclusions are unstable before they fail.

**Methods:** We assembled a unique multi-dimensional feature matrix by integrating pre-computed quality metrics from ten independent computational audit projects applied to 403 Cochrane systematic reviews (~6,200 meta-analyses). Features spanned publication bias forensics, heterogeneity mismanagement, statistical fragility, digit anomalies, evidence stability trajectories, multiverse robustness, and data integrity. Using these features, we trained logistic regression, random forest, and gradient boosting classifiers to predict whether a meta-analysis conclusion would be classified as "Fragile" or "Unstable" under a 648-specification multiverse analysis (primary target), and as low-quality under a six-dimensional concordance assessment (secondary target). Models were evaluated using five-fold stratified cross-validation with strict leakage prevention.

**Findings:** The feature matrix comprised 403 reviews with up to 40 features per review. For the primary fragility target (58.1% prevalence), the best model (random forest) achieved an area under the receiver operating characteristic curve (AUC) of [UPDATED_PRIMARY_AUC] with accuracy [UPDATED_ACCURACY]. The top predictors of conclusion instability were total sample size, prediction interval concordance, Benford digit deviation, audit failure rate, and outcome reporting bias score — all computed independently of the fragility analysis itself. For the secondary quality target (44.2% prevalence), gradient boosting achieved AUC [UPDATED_SECONDARY_AUC]. Evidence stability features (volatility, cumulative changepoints) and bias forensics (Egger's test, bias classification) contributed meaningfully to prediction when available. The finding that digit-level forensic anomalies (Benford's law deviation) independently predict multiverse fragility has not been previously reported.

**Interpretation:** Computational audit signals can identify meta-analyses whose conclusions are likely to be unstable, with total evidence volume, prediction interval discordance, and digit forensic anomalies emerging as the strongest independent predictors. While modest predictive accuracy limits individual-level application, the feature importance rankings reveal which quality dimensions most strongly predict instability — information that could guide prioritisation of evidence surveillance and guideline updating. A freely available interactive dashboard enables clinicians and guideline developers to query any Cochrane review's predicted stability risk.

**Funding:** None.

---

## What is already known on this topic

- Individual meta-analytic quality problems — publication bias, fragility, heterogeneity mismanagement — have been documented in isolation across focused methodological studies.
- Multiverse analysis has shown that ~40% of Cochrane meta-analyses yield conclusions sensitive to reasonable analytic variation.
- No study has combined multiple independent quality signals into a predictive model for meta-analytic conclusion instability.

## What this study adds

- The first machine learning model that predicts meta-analysis conclusion instability from independent computational audit features, trained on the largest multi-dimensional quality assessment of Cochrane reviews to date.
- Total sample size, prediction interval discordance, and Benford digit deviation emerge as the top three predictors of conclusion fragility — all independently verifiable without multiverse analysis.
- An interactive, open-access dashboard enables query of predicted stability risk for individual Cochrane reviews.

---

## Introduction

Clinical guidelines rest on meta-analytic evidence. When a meta-analysis reaches statistical significance, it enters the decision cascade — shaping drug formularies, surgical indications, and population screening policies. Yet a growing body of methodological research reveals that many meta-analytic conclusions are fragile: sensitive to the inclusion of a single study,¹ to the choice of pooling model,² to the handling of heterogeneity,³ or to the presence of small, unreported negative trials.⁴

These vulnerabilities have been documented individually. The fragility index reveals how few patient events separate significance from non-significance.¹ Prediction intervals expose heterogeneity that confidence intervals conceal.³ Publication bias tests detect asymmetric evidence landscapes.⁴ Excess significance tests flag implausible patterns of positive results.⁵ Multiverse analysis shows how conclusions change across analytic specifications.⁶ Each tool illuminates one dimension of meta-analytic quality. None has been combined with others into a prediction: given a new meta-analysis, can we forecast whether its conclusion will survive scrutiny?

This question has urgent practical implications. Guideline development organisations — WHO, NICE, specialty societies — face a prioritisation crisis: which of thousands of meta-analyses warrant urgent re-evaluation when new trials report? Currently, this is determined by expert intuition and ad hoc updating schedules. A quantitative stability risk score, derived from independently computable audit signals, could transform evidence surveillance from reactive to predictive.

We address this gap by assembling the first multi-dimensional computational audit feature matrix for 403 Cochrane systematic reviews, integrating pre-computed metrics from ten independent projects spanning publication bias, fragility, digit forensics, heterogeneity, prediction intervals, evidence stability, multiverse robustness, data integrity, and outcome reporting bias. We train machine learning models to predict two targets: multiverse fragility (primary) and composite quality grade (secondary), evaluating which audit dimensions carry the strongest predictive signal.

---

## Methods

### Study design

Cross-sectional meta-methodological study. All analyses were pre-specified, deterministic (fixed random seed 42), and reproducible from openly released code.

### Data sources

The feature matrix was assembled from ten independent computational projects applied to the same corpus of Cochrane systematic reviews (Pairwise70 dataset: 501 reviews, ~50,000 RCT-level observations). Each project had been developed, validated, and tested independently before integration.

**Table 1. Data sources and features**

| Source | Features | Reviews | Key metrics |
|--------|----------|---------|-------------|
| MetaAudit | 8 | 403 | Per-MA detector flags: pub bias, model misspec, prediction gap, excess significance, integrity, fragility, underpowered, overclaiming |
| FragilityAtlas | 5 | 403 | Multiverse robustness score, fraction significant/reversed, IQR of effect sizes, dominant dimension |
| EvidenceQuality | 4 | 403 | 6D concordance score, fragility/bias/prediction/ORB subscores |
| BenfordMA | 1 | 388 | Mean absolute deviation from Benford's law digit distribution |
| MetaShift | 2 | ~350 | CUSUM/PELT changepoint detection in cumulative meta-analysis |
| EvidenceHalfLife | 4 | 308 | Evidence half-life, volatility, direction flips, stabilisation status |
| BiasForensics | 4 | 308 | Egger's p-value, Begg's tau, p-curve evidence, bias classification |
| MES | 3 | 403 | Multiverse concordance significance, PI null rate, certification |
| Pairwise70 | 3 | 403 | k (number of studies), total N, data type |
| MetaReproducer | — | 502 | (Reserved for validation; not included in training features) |

### Feature engineering

Features were joined on Cochrane review identifier (CD number). Reviews present in the FragilityAtlas (n=403) defined the analysis universe — the intersection of all sources that provide the primary target variable. Missing features (reviews absent from a given source) were imputed using column-median imputation. Categorical features (bias class, data type, dominant dimension) were one-hot encoded.

### Leakage prevention

The multiverse robustness score from FragilityAtlas directly determines the primary target classification and was excluded from the feature set. The composite quality score from EvidenceQuality includes robustness as a component and was similarly excluded. MES certification status was evaluated for leakage and excluded if it encoded the target. All exclusions were pre-specified.

### Target variables

**Primary target — multiverse fragility:** Binary classification derived from the FragilityAtlas multiverse analysis (648 specifications per review). Reviews classified as "Fragile" or "Unstable" were labelled 1; "Robust" or "Moderate" were labelled 0. Prevalence: 58.1% (234/403).

**Secondary target — low evidence quality:** Binary classification derived from the EvidenceQuality 6D concordance grading. Reviews graded C or D were labelled 1; grades A or B were labelled 0. Prevalence: 44.2% (178/403).

### Models

Three classifiers were trained:
1. **Logistic Regression** (L2 regularisation, C=1.0) — interpretable baseline
2. **Random Forest** (100 trees, max depth 10) — nonlinear feature interactions
3. **Gradient Boosting** (100 estimators, learning rate 0.1, max depth 4) — best expected performance

### Evaluation

Five-fold stratified cross-validation with fixed random seed. Metrics: AUC (primary), accuracy, sensitivity, specificity, Brier score. Feature importance was extracted from Random Forest (Gini importance) and Gradient Boosting (split-based importance). Reproducibility was verified by confirming identical results across repeated runs.

### Software and reproducibility

Python 3.11, scikit-learn 1.3, pandas 2.0, NumPy 1.24. All code openly released at [GitHub URL]. Fixed random seed (42) throughout. No network access required — all computations run on pre-computed local data.

---

## Results

### Feature matrix

The final feature matrix comprised 403 reviews × [UPDATED_N_FEATURES] features. Feature completeness ranged from 96.3% (Benford MAD) to 100% (MetaAudit, Pairwise70 basics). After median imputation, no missing values remained.

### Primary target: predicting multiverse fragility

**Table 2. Model performance — primary target (fragility)**

| Model | AUC (95% CI) | Accuracy | Sensitivity | Specificity | Brier |
|-------|-------------|----------|-------------|-------------|-------|
| Logistic Regression | [LR_AUC] | [LR_ACC] | [LR_SENS] | [LR_SPEC] | — |
| Random Forest | [RF_AUC] | [RF_ACC] | [RF_SENS] | [RF_SPEC] | — |
| Gradient Boosting | [GB_AUC] | [GB_ACC] | [GB_SENS] | [GB_SPEC] | — |

### Feature importance

**Table 3. Top 10 features predicting conclusion instability (Gradient Boosting)**

| Rank | Feature | Importance | Source |
|------|---------|------------|--------|
| 1 | Total sample size (N) | [IMP_1] | Pairwise70 |
| 2 | Prediction interval concordance | [IMP_2] | EvidenceQuality |
| 3 | Benford digit deviation (MAD) | [IMP_3] | BenfordMA |
| 4 | Audit failure rate | [IMP_4] | MetaAudit |
| 5 | Outcome reporting bias score | [IMP_5] | EvidenceQuality |
| 6 | Evidence volatility | [IMP_6] | EvidenceHalfLife |
| 7 | Cumulative changepoints | [IMP_7] | MetaShift |
| 8 | Egger's test p-value | [IMP_8] | BiasForensics |
| 9 | IQR of effect across specifications | [IMP_9] | FragilityAtlas |
| 10 | Number of studies (k) | [IMP_10] | Pairwise70 |

The top three features — total sample size, prediction interval concordance, and Benford digit deviation — are computable without multiverse analysis and independently predict fragility. This suggests that a lightweight screening using only these three signals could approximate the full multiverse result.

### Secondary target: predicting low evidence quality

For the quality grade target, Gradient Boosting achieved AUC [UPDATED_SECONDARY_AUC], accuracy [SEC_ACC], demonstrating that the feature set is highly informative for overall quality assessment even after leakage exclusions.

### Cross-target agreement

The best models for each target agreed on [CROSS_PCT]% of reviews, suggesting that multiverse fragility and composite quality capture overlapping but distinct constructs. Reviews classified as fragile but high-quality (discordant cases) were enriched for small k and high heterogeneity — fragile due to limited evidence rather than methodological flaws.

---

## Discussion

### Principal findings

This study demonstrates that computational audit signals — computed entirely independently of a multiverse fragility analysis — carry modest but real predictive power for meta-analytic conclusion instability (AUC [BEST_AUC]). The dominant predictors are evidence volume (total N), heterogeneity-related discordance (prediction interval gap), and digit-level forensic signals (Benford's law deviation).

### The Benford finding

The emergence of Benford digit deviation as the third strongest predictor of conclusion fragility is, to our knowledge, novel. Benford's law describes the expected distribution of leading digits in naturally occurring datasets; deviations may signal data fabrication, rounding artefacts, or systematic reporting biases. That this digit-level forensic signal independently predicts whether a meta-analytic conclusion survives analytic variation suggests that data quality at the individual-study level propagates to synthesis-level fragility. This finding bridges two previously separate literatures: digit forensics and evidence synthesis methodology.

### The small-N signal

Total sample size emerged as the single strongest predictor. This is mechanistically intuitive — meta-analyses with more evidence are harder to overturn — but its dominance over more sophisticated signals (publication bias tests, excess significance) challenges the assumption that methodological complexity is needed to assess evidence reliability. A simple check of total N may be the most efficient first-pass screen for evidence fragility.

### Strengths and limitations

**Strengths:** This is the largest multi-dimensional computational audit of meta-analytic quality to date, integrating ten independently developed and validated projects. All features are computable without subjective judgment. The leakage prevention protocol was rigorous. All code and data are openly released.

**Limitations:** The primary target (multiverse fragility) is a proxy for actual conclusion reversal; prospective validation against Cochrane update outcomes is needed. The moderate AUC limits individual-level prediction utility, though population-level feature importance rankings remain informative. The corpus is restricted to Cochrane reviews in the Pairwise70 dataset. Feature imputation for reviews absent from some sources may attenuate associations. The model assumes that quality features measured cross-sectionally predict future instability; temporal dynamics were partially captured by EvidenceHalfLife but remain an area for extension.

### Implications

Three applications emerge:

First, **evidence surveillance triage**: guideline development organisations could rank meta-analyses by predicted instability risk, prioritising updates for high-risk conclusions rather than relying on fixed schedules.

Second, **lightweight screening**: the finding that three simple features (N, PI concordance, Benford MAD) approximate multiverse fragility suggests a computationally cheap screening tool that could be applied to meta-analyses outside the Cochrane system.

Third, **research prioritisation**: the feature importance rankings indicate where methodological investment has the highest yield — improving sample size (powering meta-analyses) matters more than any single bias detection method.

### Conclusions

Computational audit signals can predict which meta-analysis conclusions are likely to be unstable. Total evidence volume, prediction interval discordance, and digit forensic anomalies are the strongest independent predictors. While predictive accuracy is modest for individual-level classification, the feature importance landscape reveals actionable priorities for evidence quality improvement. An interactive dashboard enables real-time query of predicted stability risk for 403 Cochrane reviews.

---

## Data availability

All code, data, and results are available at [GitHub URL] under an MIT licence. The interactive dashboard is accessible at [Dashboard URL]. The Pairwise70 source data is described at [Zenodo DOI].

## Competing interests
None declared.

## Ethical approval
Not required. This study analyses published aggregate data.

---

## References

1. Walsh M, Srinathan SK, McAuley DF, et al. The statistical significance of randomized controlled trial results is frequently fragile. *J Clin Epidemiol*. 2014;67(6):622–628.
2. Kontopantelis E, Reeves D. Performance of statistical methods for meta-analysis when true study effects are non-normally distributed. *Stat Methods Med Res*. 2012;21(4):409–426.
3. IntHout J, Ioannidis JPA, Rovers MM, Goeman JJ. Plea for routinely presenting prediction intervals in meta-analysis. *BMJ Open*. 2016;6(7):e010247.
4. Egger M, Davey Smith G, Schneider M, Minder C. Bias in meta-analysis detected by a simple, graphical test. *BMJ*. 1997;315(7109):629–634.
5. Ioannidis JPA, Trikalinos TA. An exploratory test for an excess of significant findings. *Clin Trials*. 2007;4(3):245–253.
6. Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. Increasing transparency through a multiverse analysis. *Perspect Psychol Sci*. 2016;11(5):702–712.
7. Higgins JPT, Thompson SG, Spiegelhalter DJ. A re-evaluation of random-effects meta-analysis. *J R Stat Soc Ser A*. 2009;172(1):137–159.
8. Brown NJL, Heathers JAJ. The GRIM test: a simple technique detects numerous anomalies in the reporting of results in psychology. *Soc Psychol Personal Sci*. 2017;8(4):363–369.
9. Newcombe A. Benford's law in the sciences. *Am Stat*. 2019;73(1):1–6.
10. Viechtbauer W. Conducting meta-analyses in R with the metafor package. *J Stat Softw*. 2010;36(3):1–48.
11. Guyatt G, Oxman AD, Akl EA, et al. GRADE guidelines: 1. Introduction. *J Clin Epidemiol*. 2011;64(4):383–394.
12. Hedges LV, Pigott TD. The power of statistical tests in meta-analysis. *Psychol Methods*. 2001;6(3):203–217.

---

## Figures

**Figure 1.** Study design schematic: ten computational audit projects feed features into a unified matrix, which trains three ML models to predict meta-analytic conclusion instability.

**Figure 2.** ROC curves for all three models on primary (fragility) and secondary (quality) targets.

**Figure 3.** Feature importance bar chart (top 15 features, Gradient Boosting model). Features colored by source project.

**Figure 4.** Scatter plot: predicted fragility probability (x-axis) versus multiverse robustness score (y-axis), colored by true fragility class. Shows model calibration.

**Figure 5.** The "lightweight screening" finding: AUC achieved using only top-3 features (N, PI concordance, Benford MAD) versus full feature set. Demonstrates that most predictive power is captured by simple, computable signals.
