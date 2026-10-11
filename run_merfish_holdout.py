"""Generate MERFISH 0.09 from 0.04/0.14, then evaluate against held-out 0.09."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import time

from run_merfish_demo import positive_int


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(adata, expected_slice=None):
    """Copy evaluation inputs before SpatialZ changes input annotations in place."""
    import numpy as np
    from scipy import sparse

    if not adata.n_obs or not adata.n_vars:
        raise ValueError("Empty slice")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    for key in ("cell_class", "Region"):
        if key not in adata.obs or adata.obs[key].isna().any():
            raise ValueError(f"Missing {key} annotations")
    if expected_slice is not None:
        if set(adata.obs["slice_id"].astype(str)) != {expected_slice}:
            raise ValueError(f"Unexpected slice label; expected {expected_slice}")
    x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    xy = np.asarray(adata.obsm["spatial"])
    if xy.shape != (adata.n_obs, 2):
        raise ValueError("Expected n_cells x 2 spatial coordinates")
    if not np.isfinite(x).all() or not np.isfinite(xy).all() or (x < 0).any():
        raise ValueError("Expression must be finite/nonnegative; coordinates must be finite")
    return dict(x=x.astype(np.float64, copy=True), xy=xy.astype(np.float64, copy=True),
                genes=adata.var_names.astype(str).to_numpy(copy=True),
                cell_class=adata.obs["cell_class"].astype(str).to_numpy(copy=True),
                Region=adata.obs["Region"].astype(str).to_numpy(copy=True))


def correlation(a, b, rank=False):
    import numpy as np
    from scipy.stats import rankdata

    a, b = np.asarray(a), np.asarray(b)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if rank:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def composition(a, b):
    import numpy as np
    import pandas as pd

    categories = sorted(set(a) | set(b))
    counts_a = np.array([np.count_nonzero(a == c) for c in categories])
    counts_b = np.array([np.count_nonzero(b == c) for c in categories])
    p, q = counts_a / len(a), counts_b / len(b)
    table = pd.DataFrame(dict(category=categories, candidate_count=counts_a,
                              real_count=counts_b, candidate_fraction=p,
                              real_fraction=q, difference_pp=100 * (p - q)))
    metrics = dict(total_variation=float(np.abs(p - q).sum() / 2),
                   max_absolute_difference_pp=float(100 * np.abs(p - q).max()))
    return metrics, table


def qc(data):
    import numpy as np

    totals = data["x"].sum(axis=1)
    return dict(n_cells=len(totals), n_genes=data["x"].shape[1],
                zero_expression_cells=int(np.count_nonzero(totals == 0)),
                zero_expression_genes=int(np.count_nonzero(data["x"].sum(axis=0) == 0)),
                duplicate_coordinates=int(len(totals) - len(np.unique(data["xy"], axis=0))),
                total_expression_mean=float(totals.mean()),
                total_expression_std=float(totals.std()),
                total_expression_quantiles=np.quantile(totals, [0, .05, .5, .95, 1]).tolist())


def mean_metrics(prediction, real_mean):
    import numpy as np

    difference = prediction - real_mean
    return dict(gene_mean_pearson=correlation(prediction, real_mean),
                gene_mean_spearman=correlation(prediction, real_mean, rank=True),
                gene_mean_mae=float(np.abs(difference).mean()),
                gene_mean_rmse=float(np.sqrt(np.mean(difference ** 2))))


def compare(candidate, real, directions):
    """Unpaired distribution metrics; no registration or cell matching is fitted."""
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.stats import wasserstein_distance

    if not np.array_equal(candidate["genes"], real["genes"]):
        raise ValueError("Gene names and order must match across all slices")
    x, y = candidate["x"], real["x"]
    xy, uv = candidate["xy"], real["xy"]
    forward = cKDTree(uv).query(xy, k=1)[0]
    backward = cKDTree(xy).query(uv, k=1)[0]
    projected_x, projected_y = xy @ directions.T, uv @ directions.T
    metrics = mean_metrics(x.mean(axis=0), y.mean(axis=0))
    metrics.update(
        n_cells=len(x), real_n_cells=len(y), cell_count_difference=len(x) - len(y),
        cell_count_relative_error=float((len(x) - len(y)) / len(y)),
        gene_variance_pearson=correlation(x.var(axis=0), y.var(axis=0)),
        gene_detection_mae_pp=float(100 * np.abs((x > 0).mean(axis=0) - (y > 0).mean(axis=0)).mean()),
        total_expression_wasserstein=float(wasserstein_distance(x.sum(axis=1), y.sum(axis=1))),
        spatial_sliced_wasserstein_1=float(np.mean([
            wasserstein_distance(projected_x[:, i], projected_y[:, i])
            for i in range(len(directions))])),
        spatial_chamfer_mean=float((forward.mean() + backward.mean()) / 2),
        candidate_to_real_nearest_mean=float(forward.mean()),
        real_to_candidate_nearest_mean=float(backward.mean()),
        spatial_centroid_distance=float(np.linalg.norm(xy.mean(axis=0) - uv.mean(axis=0))),
    )
    for key in ("cell_class", "Region"):
        metrics[key], _ = composition(candidate[key], real[key])
    return metrics


def evaluate(sources, generated, real, output, seed):
    import numpy as np
    import pandas as pd

    # A separate RNG does not advance or depend on the generator's RNG state.
    directions = np.random.default_rng(seed).normal(size=(128, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    candidates = {"generated_0.09": generated, "input_0.04": sources[0], "input_0.14": sources[1]}
    metrics = {
        "protocol": {
            "generation_inputs": ["0.04", "0.14"], "held_out": "0.09", "alpha": .5,
            "evaluation_projection_seed": seed, "evaluation_projections": len(directions),
            "spatial_metric": "Mean 1D Wasserstein-1 over shared unit directions; empirical probability mass normalized per slice",
            "spatial_units": "Provided coordinate units; no registration fitted to the held-out slice",
            "expression_units": "Provided expression values, without additional normalization or log transform",
            "limitations": [
                "Unpaired distributions; no cell-wise accuracy or reconstructed cell identity is asserted",
                "Inputs are supplied processed/aligned h5ad files; upstream preprocessing is not re-estimated",
                "One held-out slice and one seed; this is not a complete paper benchmark",
            ],
        },
        "qc": {name: qc(data) for name, data in {**candidates, "real_0.09": real}.items()},
        "comparisons": {name: compare(data, real, directions) for name, data in candidates.items()},
        "endpoint_mean_expression_baseline": mean_metrics(
            .5 * sources[0]["x"].mean(axis=0) + .5 * sources[1]["x"].mean(axis=0),
            real["x"].mean(axis=0)),
    }
    donor_types = set(sources[0]["cell_class"]) | set(sources[1]["cell_class"])
    metrics["real_cell_types_absent_from_inputs"] = sorted(set(real["cell_class"]) - donor_types)
    gene_table = pd.DataFrame({"gene": real["genes"]})
    for name, data in {**candidates, "real_0.09": real}.items():
        gene_table[f"{name}_mean"] = data["x"].mean(axis=0)
        gene_table[f"{name}_variance"] = data["x"].var(axis=0)
        gene_table[f"{name}_detection_fraction"] = (data["x"] > 0).mean(axis=0)
    gene_table["endpoint_mean_baseline"] = .5 * (gene_table["input_0.04_mean"] + gene_table["input_0.14_mean"])
    gene_table.to_csv(output / "gene_comparison.csv", index=False)
    for key in ("cell_class", "Region"):
        _, table = composition(generated[key], real[key])
        for label, source in zip(("0.04", "0.14"), sources):
            table[f"input_{label}_fraction"] = [np.mean(source[key] == c) for c in table["category"]]
        table.to_csv(output / f"{key}_comparison.csv", index=False)
    rows = []
    for category in sorted(set(generated["cell_class"]) | set(real["cell_class"])):
        a = generated["x"][generated["cell_class"] == category]
        b = real["x"][real["cell_class"] == category]
        row = dict(cell_class=category, generated_n=len(a), real_n=len(b),
                   small_group=len(a) < 20 or len(b) < 20)
        if len(a) and len(b):
            row.update(mean_metrics(a.mean(axis=0), b.mean(axis=0)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "expression_by_cell_class.csv", index=False)
    pd.json_normalize([dict(comparison=name, **values)
                       for name, values in metrics["comparisons"].items()]).to_csv(
                           output / "comparison_summary.csv", index=False)
    write_json(output / "metrics.json", metrics)
    plot_comparison(sources, generated, real, output)
    return metrics


def plot_comparison(sources, generated, real, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Sans"
    data = [sources[0], generated, real, sources[1]]
    titles = ["Input 0.04", "Generated 0.09", "Real 0.09 (held out)", "Input 0.14"]
    categories = sorted(set().union(*(set(d["cell_class"]) for d in data)))
    colors = {c: plt.get_cmap("tab20")(i % 20) for i, c in enumerate(categories)}
    all_xy = np.concatenate([d["xy"] for d in data])
    low, high = all_xy.min(axis=0), all_xy.max(axis=0)
    padding = np.maximum((high - low) * .03, 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(17, 7), sharex=True, sharey=True)
    for axis, d, title in zip(axes, data, titles):
        for category in categories:
            points = d["xy"][d["cell_class"] == category]
            axis.scatter(points[:, 0], points[:, 1], s=2, color=colors[category], label=category, rasterized=True)
        axis.set(title=f"{title}\nn={len(d['x']):,}", xlabel="x (provided units)",
                 xlim=(low[0] - padding[0], high[0] + padding[0]),
                 ylim=(low[1] - padding[1], high[1] + padding[1]))
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y (provided units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, markerscale=4, frameon=False)
    fig.tight_layout(rect=(0, .14, 1, .96))
    fig.savefig(output / "spatial_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    a, b = real["x"].mean(axis=0), generated["x"].mean(axis=0)
    axes[0].scatter(a, b, s=14, alpha=.7)
    limit = max(float(a.max()), float(b.max()))
    axes[0].plot([0, limit], [0, limit], color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Real gene mean", ylabel="Generated gene mean", title="Mean expression per gene")
    _, table = composition(generated["cell_class"], real["cell_class"])
    positions = np.arange(len(table))
    axes[1].bar(positions - .2, table["real_fraction"] * 100, width=.4, label="Real")
    axes[1].bar(positions + .2, table["candidate_fraction"] * 100, width=.4, label="Generated")
    axes[1].set_xticks(positions, table["category"], rotation=90)
    axes[1].set(ylabel="Cells (%)", title="Cell class composition")
    axes[1].legend(frameon=False)
    for d, label in ((real, "Real"), (generated, "Generated")):
        values = np.sort(d["x"].sum(axis=1))
        axes[2].plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    axes[2].set(xlabel="Total expression per cell", ylabel="Cumulative fraction", title="Expression distribution")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "expression_comparison.png", dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=positive_int, default=1000)
    parser.add_argument("--mender-jobs", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=root / "output" / "merfish_holdout_009")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(args.mender_jobs))

    import numpy as np
    import scanpy as sc
    import torch
    from SpatialZ import Generate_spatialz

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.mender_jobs)
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    paths = [root / "data" / "merfish" / f"merfish_{i}_paste.h5ad" for i in (4, 14)]
    held_out_path = root / "data" / "merfish" / "merfish_9_paste.h5ad"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "environment.json", {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]})
    report = dict(status="running", stage="loading_inputs", inputs=[str(p) for p in paths],
                  held_out=str(held_out_path), seed=args.seed,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_cuda_runtime=torch.version.cuda)
    start = time.perf_counter()
    try:
        adatas = [sc.read_h5ad(p) for p in paths]
        sources = [snapshot(a, label) for a, label in zip(adatas, ("0.04", "0.14"))]
        if not np.array_equal(sources[0]["genes"], sources[1]["genes"]):
            raise ValueError("Input gene names and order must match")
        n_cell = int(.5 * adatas[0].n_obs + .5 * adatas[1].n_obs)
        parameters = dict(adata1_id="0.04", adata2_id="0.14", alpha=.5, n_cell=n_cell,
                          k_neighbors=1, n_mag=1.0, micro_env_key="mender", verbose=True,
                          device=device, lr=1e5, nb_iter_max=args.iterations, seed=args.seed,
                          num_projections=80, cell_type_key="cell_class", syn_mode="default",
                          k_sam=50, Beta=1, add_obs_list=["neuron_class", "domain", "Region", "annotation"],
                          mender_n_jobs=args.mender_jobs)
        report.update(stage="generating", parameters=parameters,
                      input_shapes=[list(a.shape) for a in adatas], input_sha256=[sha256(p) for p in paths])
        write_json(output / "run.json", report)
        print(f"Generating 0.09 from 0.04/0.14: {n_cell} cells, device={device}", flush=True)
        # The held-out file has not been read. No target expression, coordinates,
        # labels, or cell count enters SpatialZ or its MENDER computation.
        generated_adata = Generate_spatialz(adatas[0], adatas[1], **parameters)
        generated = snapshot(generated_adata)
        if len(generated["x"]) != n_cell or not np.array_equal(generated["genes"], sources[0]["genes"]):
            raise ValueError("Generated cell count or gene order changed")
        generated_adata.obs["slice_id"] = "0.09"
        generated_adata.obs["data_type"] = "synthetic"
        generated_adata.write_h5ad(output / "generated_009.h5ad")
        report.update(stage="evaluating", generation_seconds=time.perf_counter() - start,
                      output_shape=list(generated_adata.shape))
        write_json(output / "run.json", report)
        print("Generation saved. Reading held-out 0.09 for evaluation only.", flush=True)
        real = snapshot(sc.read_h5ad(held_out_path), "0.09")
        report["held_out_sha256"] = sha256(held_out_path)
        metrics = evaluate(sources, generated, real, output, args.seed)
        report.update(status="complete", stage="complete", real_shape=list(real["x"].shape))
        print(json.dumps(metrics["comparisons"]["generated_0.09"], indent=2), flush=True)
        print(f"Saved results: {output}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "run.json", report)


if __name__ == "__main__":
    main()
