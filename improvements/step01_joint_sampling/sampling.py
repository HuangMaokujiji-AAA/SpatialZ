"""Step 01: original candidate/weight rule, gene versus whole-cell sampling."""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity


def build_candidates(reference_xy, reference_types, reference_mender,
                     query_xy, query_types, query_mender, k_sam=50, beta=1):
    """Keep SpatialZ's GLOBAL search followed by type filtering, including k cap."""
    tree = cKDTree(reference_xy)
    similarities = cosine_similarity(reference_mender, query_mender)
    pools, probabilities = [], []
    for i, cell_type in enumerate(query_types):
        same_type = reference_types == cell_type
        k = min(k_sam, int(same_type.sum()))
        indices = np.atleast_1d(tree.query(query_xy[i], k=k)[1]) if k else np.array([], dtype=int)
        indices = indices[same_type[indices]]
        weights = np.exp(beta * similarities[indices, i])
        if indices.size:
            weights = weights / weights.sum()
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Invalid donor probabilities")
        pools.append(indices)
        probabilities.append(weights)
    return pools, probabilities


def sample_expression(reference_x, pools, probabilities, unit, seed):
    """Both arms use the same cached pools/probabilities and zero-vector fallback.

    The ONLY algorithmic change in 'cell' is drawing one donor for all genes.
    A fresh RandomState isolates expression draws from earlier MENDER RNG use.
    """
    if unit not in {"gene", "cell"}:
        raise ValueError("unit must be 'gene' or 'cell'")
    if len(pools) != len(probabilities):
        raise ValueError("One probability vector is required per pool")
    rng = np.random.RandomState(seed)
    result = np.zeros((len(pools), reference_x.shape[1]), dtype=reference_x.dtype)
    donor_indices = np.full(len(pools), -1, dtype=int)
    for i, (indices, weights) in enumerate(zip(pools, probabilities)):
        if not len(indices):
            continue  # Original behavior; no new fallback donor is introduced.
        if unit == "gene":
            for j in range(reference_x.shape[1]):
                choice = rng.choice(np.arange(len(indices)), p=weights)
                result[i, j] = reference_x[indices[choice], j]
        else:
            choice = rng.choice(np.arange(len(indices)), p=weights)
            donor_indices[i] = indices[choice]
            result[i] = reference_x[donor_indices[i]]
    return result, donor_indices


def expression_metrics(prediction, real, min_covariance_cells=20):
    """Unpaired distribution diagnostics, using population moments (ddof=0)."""
    totals, true_totals = prediction.sum(axis=1), real.sum(axis=1)
    mean = float(totals.mean()) if len(totals) else None
    true_mean = float(true_totals.mean()) if len(true_totals) else None
    variance = float(totals.var()) if len(totals) > 1 else None
    true_variance = float(true_totals.var()) if len(true_totals) > 1 else None
    result = dict(generated_n=len(prediction), real_n=len(real),
                  total_expression_mean=mean, real_total_expression_mean=true_mean,
                  total_expression_std=float(totals.std()) if len(totals) > 1 else None,
                  real_total_expression_std=float(true_totals.std()) if len(true_totals) > 1 else None,
                  total_mean_relative_bias=(mean / true_mean - 1) if mean is not None and true_mean else None,
                  total_variance_ratio=(variance / true_variance)
                  if variance is not None and true_variance else None,
                  covariance_eligible=min(len(prediction), len(real)) >= min_covariance_cells)
    if not result["covariance_eligible"]:
        return result
    x = prediction.astype(np.float64) - prediction.mean(axis=0, dtype=np.float64)
    y = real.astype(np.float64) - real.mean(axis=0, dtype=np.float64)
    cov_x, cov_y = x.T @ x / len(x), y.T @ y / len(y)
    off_diagonal = ~np.eye(prediction.shape[1], dtype=bool)
    denominator = np.linalg.norm(cov_y[off_diagonal])
    result["offdiag_covariance_relative_error"] = (
        float(np.linalg.norm((cov_x - cov_y)[off_diagonal]) / denominator) if denominator > 0 else None)
    var_x, var_y = np.diag(cov_x), np.diag(cov_y)
    true_variable = var_y > 0
    both_variable = true_variable & (var_x > 0)
    valid = np.triu(both_variable[:, None] & both_variable[None, :], k=1)
    n_true = int(true_variable.sum())
    result.update(real_variable_genes=n_true,
                  collapsed_real_variable_genes=int((true_variable & (var_x == 0)).sum()),
                  correlation_pairs=int(valid.sum()),
                  real_variable_gene_pairs=n_true * (n_true - 1) // 2)
    if valid.any():
        scale_x, scale_y = np.sqrt(np.outer(var_x, var_x)), np.sqrt(np.outer(var_y, var_y))
        result["offdiag_correlation_mae"] = float(np.mean(np.abs(
            cov_x[valid] / scale_x[valid] - cov_y[valid] / scale_y[valid])))
    else:
        result["offdiag_correlation_mae"] = None
    return result
