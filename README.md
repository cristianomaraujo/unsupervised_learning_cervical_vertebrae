# Reproducible morphometric analysis

This repository implements an unsupervised analysis of six dimensionless C3/C4
vertebral ratios, followed by external characterization using maturation stage,
chronological age, and sex. These clinical variables are never used to fit the
standardization, PCA, or clustering models, or to select a clustering solution.

## Input

Supply an Excel workbook with one observation per row and the following headers:

| Columns | Meaning |
| --- | --- |
| `C3=ho`, `C4=ho` | Horizontal dimensions |
| `C3=vert1`, `C3=vert2`, `C3=vert3` | C3 anterior, central, and posterior vertical dimensions |
| `c4 vert1`, `c4 vert2`, `c4 vert3` | Corresponding C4 vertical dimensions |
| `h=1 m=2` | Sex: 1 = male, 2 = female |
| `age in years` | Chronological age in years |
| `maturação Bacetti 1,2,3,4,5` | Five-stage maturation classification |

The stage column is recognized by its unique `Bacetti 1,2,3,4,5` suffix to tolerate
encoding differences in the preceding Portuguese word. The first worksheet is
used unless a worksheet name is supplied. Retained linear measurements must be positive;
sex and stage codes must match the schema. Missing or nonfinite values stop the
analysis because the study methodology does not specify an imputation procedure.
The input workbook is supplied separately and is not embedded in the code.

## Methodology

1. **Quality control and representation.** Audit exact repeated rows without
   automatically treating them as repeated participants. On each of the eight
   original linear dimensions, calculate the robust z-score
   `0.67448975 × (value − median) / MAD`. Exclude observations with an absolute
   score greater than 8 in any dimension, without modifying the measurements.
   The threshold and application to raw dimensions follow the original analysis
   implementation; the manuscript describes the MAD procedure without giving
   these details. Divide each vertical dimension by the horizontal dimension of
   the same vertebra to obtain six ratios. Compute their Spearman correlations.
2. **Standardization and PCA.** Standardize ratios with the feature means and
   population standard deviations (`StandardScaler`). Fit full-SVD PCA and
   retain the smallest number of components reaching at least 95% cumulative
   explained variance. Report eigenvector coefficients as loadings. Orient PC1
   so that its loadings sum to a positive value, using morphology alone.
   Apply Hartigan's dip test to PC1 using the package's tabulated p-value method.
3. **Cluster tendency and number.** Use retained PCA scores throughout discovery.
   Compute Hopkins statistics for 200 repetitions, sampling 10% of observations
   without replacement (rounded up). Generate uniform reference points within
   the PCA-coordinate bounding box. Preserve the original Hopkins implementation,
   which uses unpowered Euclidean nearest-neighbor distances and excludes each
   sampled observation from its own neighbor search. For Gap statistics, evaluate
   K-means with `k=1,…,10` against 100 uniform reference datasets per k. Use
   log inertia and reference uncertainty `SD × sqrt(1 + 1/100)`; select the
   smallest k satisfying the one-standard-error rule. If none satisfies the
   rule within the search interval, use the maximum Gap, as in the source code.
4. **Clustering and validation.** Fit K-means and Ward hierarchical clustering
   for `k=2,…,10`. Use 100 K-means initializations. Fit Gaussian mixture models
   for one through ten components with full, tied, diagonal, and spherical
   covariance matrices, 20 initializations, and covariance regularization
   `1e-6`. Select the GMM using minimum BIC, with AIC as a tie-breaker and
   complementary output. Evaluate Silhouette, Calinski–Harabasz, and
   Davies–Bouldin indices. Assess stability through 200 subsamples containing
   80% of observations, rounded down, without replacement. Refit clustering
   only; keep full-sample preprocessing fixed. Compare subsample labels with
   the corresponding full-sample labels using ARI and report its mean, median,
   and 2.5th/97.5th percentiles. These percentiles describe the stability
   distribution, not a confidence interval for its mean.
5. **Selection and cross-algorithm comparison.** Operationalize the manuscript's
   joint consideration of separation and stability using the original code's
   equal-weight composite: min–max scaled Silhouette, `log(1 + CH)`, negative
   Davies–Bouldin, and median stability ARI. Normalize over all K-means and Ward
   candidates and the GMM candidates at the BIC-selected covariance structure,
   excluding k=1. Select K-means and Ward candidates by this composite, breaking
   ties by median stability and Silhouette. GMM selection remains based on BIC,
   rather than the composite. Compare the three selected partitions using
   pairwise ARI. Choose the primary partition among these selected solutions
   using the composite, without consulting clinical variables. Order its labels
   by median PC1 for interpretation. This numeric ranking rule comes from the
   original implementation; the manuscript does not state its weights.
6. **Post-hoc characterization.** Describe sample and cluster age, sex, ratios,
   and stage distributions. Compare primary cluster labels with maturation
   stages using ARI, NMI, AMI, V-measure, and bias-corrected Cramér's V with a
   chi-square association test. Calculate Spearman correlations of PC1 with
   age and stage. Tabulate stage-by-cluster counts and within-stage-and-sex
   percentages. Display PCA scores by cluster, stage, age, and sex, PC1 by
   stage, and an alluvial stage-to-cluster diagram.
7. **Post-hoc logistic model.** For a binary primary solution, fit a maximum
   likelihood logit model with age, sex (female reference), and their interaction
   to the probability of membership in the higher-PC1 cluster. Estimate each
   sex's age at probability 0.5 by solving the linear predictor's zero
   analytically. Reject crossings outside the pooled observed age range.
   Calculate percentile 95% intervals for both ages and the male-minus-female
   difference using 5,000 bootstrap resamples stratified by sex. Keep discovered
   cluster labels fixed during bootstrap. Record all iterations and failures;
   use replicates only when the model converges and both crossings are valid.
   Plot probability curves with pointwise 95% intervals obtained on the logit
   scale and transformed to probabilities. A nonbinary primary solution stops
   this step instead of imposing two clusters.

Randomized operations use seed 40 and the original deterministic per-iteration
offsets. Computational library thread pools are limited to one thread.
Convergence warnings are not suppressed; failed clustering fits stop execution.
No absolute-size/combined-feature sensitivity analyses, alternative scalers,
HDBSCAN, or additional regression models are run.

## Generated files

- `tables/`: quality-control audit, preprocessing parameters, PCA variance and
  loadings, correlations, dip test, Hopkins and Gap repetitions, GMM information
  criteria, clustering validation, stability repetitions, selected-model and
  pairwise-ARI tables (Table 1 content), stage/sex/cluster table (Table 2 content),
  sample and cluster descriptions, external associations, logistic coefficients,
  bootstrap repetitions and intervals, and numerical figure source data.
- `figures/`: statistical Figures 2–4 as 300-dpi PNG and vector PDF.
- `run_metadata.json`: input and script SHA-256 hashes, random seed, interpreter,
  platform, package versions, and execution summary.
- `run.log`: progress and completion record.

The anatomical measurement illustration (Figure 1) requires its original
radiographic artwork and is not generated from tabular measurements. Examiner
ICC and weighted kappa require repeated examiner measurements and ratings;
they cannot be recalculated from a workbook with one assessment per participant.
