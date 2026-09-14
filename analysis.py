#!/usr/bin/env python3
"""Reproducible C3/C4 morphometric analysis; see README.md for methodology.

Clinical variables are accessed only after unsupervised model selection.
No notebook state, remote services, or participant data are embedded here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, v_measure_score,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
import statsmodels.api as sm
from diptest import diptest
from threadpoolctl import threadpool_limits

SEED = 40
RAW = ["C3=ho", "C3=vert1", "C3=vert2", "C3=vert3",
       "C4=ho", "c4 vert1", "c4 vert2", "c4 vert3"]
FEATURES = [f"C{v}_V{i}/H" for v in (3, 4) for i in (1, 2, 3)]
LOG = logging.getLogger(__name__)


def save_table(frame, folder, name, index=False):
    """Save full-precision numerical tables as UTF-8 CSV."""
    frame.to_csv(folder / f"{name}.csv", index=index, encoding="utf-8")


def fit_model(name, x, k, seed=SEED, covariance="full"):
    """Fit a partition using the original initialization settings."""
    if name == "KMeans":
        model = KMeans(n_clusters=k, n_init=100, random_state=seed)
    elif name == "Ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        model = GaussianMixture(n_components=k, covariance_type=covariance,
                                n_init=20, reg_covar=1e-6, random_state=seed)
    labels = model.fit_predict(x)
    if name == "GMM" and not model.converged_:
        raise RuntimeError(f"GMM did not converge: k={k}, covariance={covariance}")
    return model, labels


def internal_metrics(x, labels):
    if len(np.unique(labels)) < 2:
        return dict(Silhouette=np.nan, Calinski_Harabasz=np.nan, Davies_Bouldin=np.nan)
    return dict(Silhouette=silhouette_score(x, labels),
                Calinski_Harabasz=calinski_harabasz_score(x, labels),
                Davies_Bouldin=davies_bouldin_score(x, labels))


def stability(x, name, k, labels, covariance):
    """200 subsamples, without replacement; preprocessing stays fixed."""
    rng = np.random.default_rng(SEED)
    values = []
    for i in range(200):
        idx = np.sort(rng.choice(len(x), int(np.floor(.8 * len(x))), replace=False))
        _, sub = fit_model(name, x[idx], k, SEED + i + 1, covariance)
        values.append(adjusted_rand_score(labels[idx], sub))
    return np.asarray(values)


def hopkins(x):
    """Original distance-based Hopkins convention (unpowered distances)."""
    values = []
    for i in range(200):
        rng = np.random.default_rng(SEED + i)
        m = min(max(5, int(np.ceil(.1 * len(x)))), len(x) - 1)
        idx = rng.choice(len(x), m, replace=False)
        artificial = rng.uniform(x.min(0), x.max(0), size=(m, x.shape[1]))
        dist = cdist(x[idx], x)
        dist[np.arange(m), idx] = np.inf
        w = dist.min(1).sum()
        u = cdist(artificial, x).min(1).sum()
        values.append(u / (u + w))
    return np.asarray(values)


def gap_statistic(x):
    """Uniform bounding-box references and the smallest-k one-SE rule."""
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(1, 11):
        LOG.info("Gap statistic: k=%d/10", k)
        model, _ = fit_model("KMeans", x, k)
        logs = []
        for b in range(100):
            ref = rng.uniform(x.min(0), x.max(0), size=x.shape)
            km = KMeans(n_clusters=k, n_init=20, random_state=SEED + b + 1000 * k).fit(ref)
            logs.append(np.log(max(km.inertia_, np.finfo(float).eps)))
        rows.append(dict(k=k, Gap=np.mean(logs) - np.log(model.inertia_),
                         SE=np.std(logs, ddof=1) * np.sqrt(1.01)))
    frame = pd.DataFrame(rows)
    for i in range(9):
        if frame.Gap.iloc[i] >= frame.Gap.iloc[i + 1] - frame.SE.iloc[i + 1]:
            return frame, int(frame.k.iloc[i])
    return frame, int(frame.loc[frame.Gap.idxmax(), "k"])


def prepare(path, sheet, tables):
    """Validate the supplied schema, audit duplicates and flag raw outliers.

    Missing data are rejected: imputation is not specified by the manuscript.
    Exact repeated rows are audited, not presumed to be duplicate participants.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.astype(str).str.strip()
    stage_cols = [c for c in df if c.endswith("Bacetti 1,2,3,4,5")]
    required = RAW + ["h=1 m=2", "age in years"]
    if len(stage_cols) != 1 or any(c not in df for c in required):
        raise ValueError("Required measurement, sex, age, or Baccetti columns are absent/ambiguous.")
    df = df[required + stage_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing/nonfinite values: resolve input data; no imputation is specified.")
    if not df['h=1 m=2'].isin([1, 2]).all() or not df[stage_cols[0]].isin(range(1, 6)).all():
        raise ValueError("Unexpected sex or stage coding.")
    if not df['age in years'].between(0, 17, inclusive="left").all():
        raise ValueError("Age outside the study eligibility range.")
    raw = df[RAW]
    med = raw.median()
    mad = (raw - med).abs().median()
    if (mad == 0).any():
        raise ValueError("Zero MAD: the original robust-z rule is undefined.")
    rz = .67448975 * (raw - med) / mad
    excluded = (rz.abs() > 8).any(axis=1)
    audit = pd.DataFrame({'excel_row': df.index + 2, 'excluded': excluded,
                          'exact_duplicate_row': df.duplicated(keep=False)})
    audit = pd.concat([audit, rz.add_prefix("robust_z_")], axis=1)
    save_table(audit, tables, "quality_control")
    retained = df.loc[~excluded].copy()
    if (retained[RAW] <= 0).any().any():
        raise ValueError("Retained linear dimensions must be positive after the outlier audit.")
    retained.insert(0, 'excel_row', retained.index + 2)
    ratios = pd.DataFrame(index=retained.index)
    for offset, v in ((0, 3), (4, 4)):
        for i in (1, 2, 3):
            ratios[f'C{v}_V{i}/H'] = retained[RAW[offset+i]] / retained[RAW[offset]]
    return retained, ratios, stage_cols[0], len(df), int(excluded.sum())


def select_models(x, tables):
    """Rank separation/stability; select GMM strictly by BIC, then AIC."""
    gmm_rows, gmm_store = [], {}
    for cov in ('full', 'tied', 'diag', 'spherical'):
        LOG.info("GMM information criteria: %s", cov)
        for k in range(1, 11):
            model, labels = fit_model('GMM', x, k, covariance=cov)
            gmm_store[cov, k] = labels
            gmm_rows.append(dict(Model='GMM', k=k, Covariance=cov,
                                 BIC=model.bic(x), AIC=model.aic(x), **internal_metrics(x, labels)))
    gmm = pd.DataFrame(gmm_rows).sort_values(['BIC', 'AIC'])
    save_table(gmm, tables, 'gmm_information_criteria')
    best_gmm = gmm.iloc[0]
    covariance = best_gmm.Covariance
    records, labels_store, stability_rows = [], {}, []
    # Keep GMM candidates at the BIC-selected covariance to retain the original
    # common normalization domain of the separation/stability composite score.
    for name in ('KMeans', 'Ward', 'GMM'):
        ks = range(1 if name == 'GMM' else 2, 11)
        for k in ks:
            LOG.info("Stability: %s k=%d (200 subsamples)", name, k)
            if name == 'GMM':
                labels = gmm_store[covariance, k]
            else:
                _, labels = fit_model(name, x, k)
            scores = stability(x, name, k, labels, covariance)
            labels_store[name, k] = labels
            records.append(dict(Model=name, k=k, **internal_metrics(x, labels),
                                ARI_mean=scores.mean(), ARI_median=np.median(scores),
                                ARI_p025=np.percentile(scores, 2.5),
                                ARI_p975=np.percentile(scores, 97.5)))
            stability_rows.extend(dict(Model=name, k=k, iteration=i+1, ARI=s)
                                  for i, s in enumerate(scores))
    results = pd.DataFrame(records)
    score_inputs = results[['Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin', 'ARI_median']].copy()
    score_inputs['Calinski_Harabasz'] = np.log1p(score_inputs.Calinski_Harabasz)
    score_inputs['Davies_Bouldin'] *= -1
    # k=1 has no internal partition indices and must not influence normalization.
    valid = results.k >= 2
    scores = score_inputs.loc[valid]
    span = scores.max() - scores.min()
    scaled = (scores - scores.min()) / span.replace(0, np.nan)
    scaled.loc[:, span == 0] = .5
    results.loc[valid, 'Composite'] = scaled.mean(axis=1)
    selected = []
    for name in ('KMeans', 'Ward'):
        selected.append(results[results.Model == name].sort_values(
            ['Composite', 'ARI_median', 'Silhouette'], ascending=False).iloc[0])
    selected.append(results[(results.Model == 'GMM') & (results.k == best_gmm.k)].iloc[0])
    selected = pd.DataFrame(selected).reset_index(drop=True)
    selected['Covariance'] = ['not applicable', 'not applicable', covariance]
    selected['BIC'] = [np.nan, np.nan, best_gmm.BIC]
    selected['AIC'] = [np.nan, np.nan, best_gmm.AIC]
    save_table(results, tables, 'cluster_candidates')
    save_table(pd.DataFrame(stability_rows), tables, 'stability_iterations')
    save_table(selected, tables, 'table_1_selected_models')
    pairs = []
    for a, b in combinations(selected.itertuples(), 2):
        pairs.append(dict(Model_A=a.Model, k_A=a.k, Model_B=b.Model, k_B=b.k,
                          ARI=adjusted_rand_score(labels_store[a.Model, a.k], labels_store[b.Model, b.k])))
    save_table(pd.DataFrame(pairs), tables, 'table_1_pairwise_ari')
    primary = selected.sort_values(['Composite', 'ARI_median', 'Silhouette'], ascending=False).iloc[0]
    return primary, labels_store[primary.Model, primary.k], selected


def logistic(data, tables):
    """Binomial logit with female reference; stratified percentile bootstrap.

    The zero of the linear predictor gives P=0.5 analytically, avoiding the
    original age-grid approximation. Both crossings must lie in the observed
    pooled age range for a bootstrap replicate to contribute to intervals.
    """
    age = data.Age.to_numpy()
    male = (data.Sex == 'Male').to_numpy().astype(float)
    design = np.column_stack([np.ones(len(data)), age, male, age * male])
    outcome = (data.Cluster == 2).astype(int).to_numpy()
    model = sm.Logit(outcome, design).fit(disp=False)
    if not model.mle_retvals['converged']:
        raise RuntimeError('Primary logistic regression did not converge.')
    ci = model.conf_int()
    save_table(pd.DataFrame(dict(term=['Intercept', 'Age', 'Male', 'Age:Male'],
                                coefficient=model.params, OR=np.exp(model.params),
                                CI_low=np.exp(ci[:, 0]), CI_high=np.exp(ci[:, 1]),
                                p_value=model.pvalues)), tables, 'logistic_coefficients')

    def crossing(params):
        intercepts = np.array([params[0], params[0] + params[2]])
        slopes = np.array([params[1], params[1] + params[3]])
        roots = np.divide(-intercepts, slopes, out=np.full(2, np.nan), where=slopes != 0)
        roots[(roots < age.min()) | (roots > age.max())] = np.nan
        return roots

    point = crossing(model.params)
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(male == flag) for flag in (0, 1)]
    rows = []
    for b in range(5000):
        if b % 500 == 0:
            LOG.info('Sex-stratified bootstrap: %d/5000', b)
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        try:
            fit = sm.Logit(outcome[idx], design[idx]).fit(disp=False)
            roots = crossing(fit.params)
            status = 'valid' if fit.mle_retvals['converged'] and np.isfinite(roots).all() else 'nonconverged_or_no_crossing'
        except (np.linalg.LinAlgError, ValueError) as exc:
            roots, status = [np.nan, np.nan], type(exc).__name__
        rows.append(dict(iteration=b+1, Female=roots[0], Male=roots[1], status=status))
    boot = pd.DataFrame(rows)
    save_table(boot, tables, 'bootstrap_iterations')
    valid = boot.loc[boot.status == 'valid', ['Female', 'Male']].copy()
    if len(valid) < 2:
        raise RuntimeError('Insufficient valid bootstrap replicates.')
    valid['Male_minus_Female'] = valid.Male - valid.Female
    summary = pd.DataFrame(dict(Estimate=valid.columns,
        Point_estimate=[*point, point[1]-point[0]], CI_low=valid.quantile(.025).values,
        CI_high=valid.quantile(.975).values, valid_replicates=len(valid), requested_replicates=5000))
    save_table(summary, tables, 'age50_bootstrap_ci')
    age_grid = np.linspace(age.min(), age.max(), 300)
    curves = []
    for flag, sex in enumerate(['Female', 'Male']):
        new = np.column_stack([np.ones(300), age_grid, np.full(300, flag), age_grid * flag])
        eta = new @ model.params
        se = np.sqrt(np.einsum('ij,jk,ik->i', new, model.cov_params(), new))
        from scipy.special import expit
        curves.append(pd.DataFrame(dict(Sex=sex, Age=age_grid, probability=expit(eta),
                                        CI_low=expit(eta-1.959963984540054*se),
                                        CI_high=expit(eta+1.959963984540054*se))))
    curves = pd.concat(curves, ignore_index=True)
    save_table(curves, tables, 'figure_4_source')
    return summary, curves


def characterize(df, ratios, stage_col, pc, labels, tables):
    """Post-hoc clinical associations; never fed back into discovery."""
    data = pd.DataFrame(dict(excel_row=df.excel_row, Age=df['age in years'],
                             Sex=df['h=1 m=2'].map({1:'Male', 2:'Female'}),
                             Baccetti=df[stage_col].astype(int), Cluster=labels,
                             PC1=pc[:, 0], PC2=pc[:, 1]))
    save_table(pd.concat([data, ratios], axis=1), tables, 'figure_2_3_source')
    save_table(data.Age.describe().to_frame('Age'), tables, 'sample_age', index=True)
    save_table(data.groupby('Sex').size().rename('n').reset_index(), tables, 'sample_sex')
    save_table(data.groupby('Baccetti').size().rename('n').reset_index(), tables, 'sample_stages')
    save_table(pd.concat([data, ratios], axis=1).groupby('Cluster')[['Age']+FEATURES].agg(['count','mean','std']),
               tables, 'cluster_profiles', index=True)
    save_table(pd.crosstab(data.Cluster, data.Sex), tables, 'cluster_sex', index=True)
    contingency = pd.crosstab(data.Baccetti, data.Cluster)
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    phi = max(0, chi2/n - (k-1)*(r-1)/(n-1))
    denom = min(r-1-(r-1)**2/(n-1), k-1-(k-1)**2/(n-1))
    metrics = {'ARI':adjusted_rand_score(data.Baccetti, labels),
               'NMI':normalized_mutual_info_score(data.Baccetti, labels),
               'AMI':adjusted_mutual_info_score(data.Baccetti, labels),
               'V_measure':v_measure_score(data.Baccetti, labels),
               'Cramer_V_corrected':np.sqrt(phi/denom), 'chi_square_p':p}
    for col in ['Age', 'Baccetti']:
        rho, p = stats.spearmanr(data[col], data.PC1)
        metrics[f'{col}_PC1_rho'], metrics[f'{col}_PC1_p'] = rho, p
    save_table(pd.DataFrame([metrics]), tables, 'external_associations')
    table2 = []
    for stage in sorted(data.Baccetti.unique()):
        row = {'Baccetti_stage':stage}
        for sex in ['Female', 'Male']:
            subset = data[(data.Baccetti == stage) & (data.Sex == sex)]
            for cluster in sorted(data.Cluster.unique()):
                count = int((subset.Cluster == cluster).sum())
                row[f'{sex}_Cluster_{cluster}_n'] = count
                row[f'{sex}_Cluster_{cluster}_percent'] = 100*count/len(subset) if len(subset) else np.nan
        table2.append(row)
    save_table(pd.DataFrame(table2), tables, 'table_2_stage_cluster_by_sex')
    return data, metrics, contingency


def figures(data, variance, contingency, curves, folder):
    """Regenerate statistical Figures 2–4 in PNG and vector PDF."""
    def save(fig, name):
        fig.savefig(folder / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    colors = ['#2678a8', '#da7944']
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    for ax, col, letter in zip(axes.flat, ['Cluster','Baccetti','Age','Sex'], 'ABCD'):
        if col == 'Age':
            im = ax.scatter(data.PC1, data.PC2, c=data.Age, cmap='viridis', s=22, alpha=.8)
            fig.colorbar(im, ax=ax, label='Age (years)')
        else:
            categories = sorted(data[col].unique())
            palette = plt.get_cmap('viridis')(np.linspace(.05,.9,len(categories))) if col == 'Baccetti' else colors
            for val, color in zip(categories, palette):
                subset = data[data[col] == val]
                ax.scatter(subset.PC1, subset.PC2, c=[color], label=str(val), s=22, alpha=.75)
            ax.legend(title=col, frameon=False)
        ax.set(title=f'{letter}  {col}', xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})')
    save(fig, 'figure_2_pca')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    stages = sorted(data.Baccetti.unique())
    axes[0].boxplot([data.loc[data.Baccetti == s, 'PC1'] for s in stages], tick_labels=stages,
                    showfliers=False)
    rng = np.random.default_rng(SEED)
    for i, s in enumerate(stages, 1):
        vals = data.loc[data.Baccetti == s, 'PC1']
        axes[0].scatter(i+rng.uniform(-.16,.16,len(vals)), vals, s=9, alpha=.35, color=colors[0])
    axes[0].set(title='A  PC1 by Baccetti stage', xlabel='Baccetti stage', ylabel='PC1')
    ax = axes[1]
    n = len(data)
    left = 0
    right = {c:sum(contingency.sum(axis=0).loc[:c].iloc[:-1]) for c in contingency.columns}
    for stage, counts in contingency.iterrows():
        stage_start = left
        for c, count in counts.items():
            y0, y1, z0, z1 = left/n, (left+count)/n, right[c]/n, (right[c]+count)/n
            verts = [(0,y0),(.4,y0),(.6,z0),(1,z0),(1,z1),(.6,z1),(.4,y1),(0,y1),(0,y0)]
            codes = [MplPath.MOVETO,*([MplPath.CURVE4]*3),MplPath.LINETO,*([MplPath.CURVE4]*3),MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts,codes), facecolor=colors[(c-1)%2], alpha=.55, edgecolor='white', lw=.5))
            left += count
            right[c] += count
        ax.text(-.03,(stage_start+left)/2/n,f'Stage {stage}',ha='right',va='center')
    start = 0
    for c, count in contingency.sum(axis=0).items():
        ax.text(1.03,(start+count/2)/n,f'Cluster {c}\n(n={count})',va='center')
        start += count
    ax.set(xlim=(-.22,1.3),ylim=(-.02,1.02),title='B  Stage-to-cluster correspondence')
    ax.axis('off')
    save(fig, 'figure_3_stage_correspondence')
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for (sex, group), color in zip(curves.groupby('Sex', sort=True), colors):
        ax.plot(group.Age, group.probability, label=sex, color=color, lw=2)
        ax.fill_between(group.Age, group.CI_low, group.CI_high, color=color, alpha=.18)
    ax.axhline(.5, color='gray', ls='--', lw=1)
    ax.set(xlabel='Chronological age (years)', ylabel='Predicted probability of Cluster 2', ylim=(0,1))
    ax.legend(title='Sex', frameon=False)
    save(fig, 'figure_4_age_sex_probability')


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty to prevent mixing separate runs.')
    tables, plots = out/'tables', out/'figures'
    tables.mkdir(); plots.mkdir()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out/'run.log', encoding='utf-8')])
    df, ratios, stage_col, initial_n, excluded_n = prepare(args.data, args.sheet, tables)
    LOG.info('Input: %d; excluded: %d; retained: %d', initial_n, excluded_n, len(df))
    save_table(ratios.corr(method='spearman'), tables, 'ratio_spearman_correlations', index=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(ratios)
    pca = PCA(svd_solver='full')
    pc = pca.fit_transform(z)
    # Fix the arbitrary PC1 sign using morphology alone, never clinical labels.
    if pca.components_[0].sum() < 0:
        pca.components_[0] *= -1
        pc[:, 0] *= -1
    variance = pca.explained_variance_ratio_
    retained = int(np.searchsorted(variance.cumsum(), .95)+1)
    x = pc[:, :retained]
    save_table(pd.DataFrame(dict(PC=np.arange(1,7), variance=variance, cumulative=variance.cumsum())), tables, 'pca_variance')
    save_table(pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f'PC{i}' for i in range(1,7)]), tables, 'pca_loadings', index=True)
    save_table(pd.DataFrame(dict(feature=FEATURES, mean=scaler.mean_, scale=scaler.scale_)), tables, 'standardization')
    dip, dip_p = diptest(pc[:, 0])
    save_table(pd.DataFrame([dict(dip=dip, p_value=dip_p)]), tables, 'pc1_dip_test')
    h = hopkins(x)
    save_table(pd.DataFrame(dict(iteration=np.arange(1,201), Hopkins=h)), tables, 'hopkins_iterations')
    gap, gap_k = gap_statistic(x)
    save_table(gap, tables, 'gap_statistic')
    primary, labels, selected = select_models(x, tables)
    order = sorted(np.unique(labels), key=lambda label:np.median(pc[labels == label,0]))
    labels = np.array([{old:i+1 for i,old in enumerate(order)}[label] for label in labels])
    data, metrics, contingency = characterize(df, ratios, stage_col, pc, labels, tables)
    if len(order) != 2:
        raise RuntimeError('Selected primary solution is not binary; manuscript logistic analysis is inapplicable.')
    age50, curves = logistic(data, tables)
    figures(data, variance, contingency, curves, plots)
    packages = ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl','statsmodels','diptest','threadpoolctl']
    metadata = dict(input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_name=args.data.name, sheet=args.sheet, seed=SEED, python=platform.python_version(),
        platform=platform.platform(), versions={p:importlib.metadata.version(p) for p in packages},
        initial_n=initial_n, excluded_n=excluded_n, final_n=len(df), retained_PCs=retained,
        hopkins_mean=float(h.mean()), hopkins_median=float(np.median(h)), gap_selected_k=gap_k,
        primary_model=primary.Model, primary_k=int(primary.k), dip=float(dip), dip_p=float(dip_p),
        bootstrap_valid=int(age50.valid_replicates.iloc[0]), threads=1)
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    LOG.info('Completed: %s, k=%d. Outputs: %s', primary.Model, primary.k, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Path to the input .xlsx workbook')
    parser.add_argument('--sheet', default=0, help='Worksheet name; defaults to the first worksheet')
    parser.add_argument('--output', type=Path, default=Path('results'), help='New or empty results directory')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
