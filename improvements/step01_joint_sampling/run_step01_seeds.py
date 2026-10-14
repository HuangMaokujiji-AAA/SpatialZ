"""Repeat expression seeds on one frozen slice and one saved candidate cache."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from run_step01 import ROOT, positive_int, sha256, write_json


# Compare distance from the target, so a variance ratio above 1 is not rewarded.
TARGETS = dict(total_mean_relative_bias=0., total_variance_ratio=1.,
               offdiag_covariance_relative_error=0., offdiag_correlation_mae=0.,
               shared_gene_correlation_mae=0.,
               gene_mean_mae=0., gene_mean_rmse=0., gene_mean_pearson=1.)


def summarize(table, grouping):
    """Descriptive seed variation and paired error reduction (positive = better)."""
    import numpy as np
    import pandas as pd

    rows = []
    groups = table.groupby(grouping, sort=True) if grouping else [(None, table)]
    for category, data in groups:
        for metric, target in TARGETS.items():
            if metric not in data:
                continue
            pair = data.pivot(index="seed", columns="arm", values=metric).reindex(columns=["gene", "cell"])
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if pair.empty:
                continue
            reduction = (pair.gene - target).abs() - (pair.cell - target).abs()
            row = dict(metric=metric, target=target, n_seeds=len(pair),
                       gene_mean=float(pair.gene.mean()), gene_std=float(pair.gene.std()) if len(pair) > 1 else None,
                       cell_mean=float(pair.cell.mean()), cell_std=float(pair.cell.std()) if len(pair) > 1 else None,
                       paired_error_reduction_mean=float(reduction.mean()),
                       paired_error_reduction_std=float(reduction.std()) if len(pair) > 1 else None,
                       paired_error_reduction_min=float(reduction.min()),
                       paired_error_reduction_max=float(reduction.max()),
                       improved_seeds=int((reduction > 0).sum()))
            if grouping:
                row[grouping] = category
            rows.append(row)
    return pd.DataFrame(rows)


def shared_gene_correlations(seed_dirs, target_path):
    """Use identical variable genes in truth and EVERY arm/seed for correlations."""
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from run_step01 import snapshot

    real = snapshot(sc.read_h5ad(target_path), "0.09")
    generated = {(seed, arm): snapshot(sc.read_h5ad(directory / arm / "generated_009.h5ad"), "0.09")
                 for seed, directory in seed_dirs.items() for arm in ("gene", "cell")}
    rows = []
    categories = sorted(set(real["cell_class"]) | set(next(iter(generated.values()))["cell_class"]))
    for category in [None, *categories]:
        truth = real["x"] if category is None else real["x"][real["cell_class"] == category]
        groups = {key: data["x"] if category is None else data["x"][data["cell_class"] == category]
                  for key, data in generated.items()}
        eligible = min([len(truth), *[len(x) for x in groups.values()]]) >= 20
        common = truth.var(axis=0) > 0 if eligible else np.zeros(real["x"].shape[1], dtype=bool)
        if eligible:
            for x in groups.values():
                common &= x.var(axis=0) > 0
        pairs = np.triu(np.ones((int(common.sum()), int(common.sum())), dtype=bool), k=1)
        truth_corr = np.corrcoef(truth[:, common], rowvar=False) if pairs.any() else None
        for (seed, arm), x in groups.items():
            error = float(np.abs(np.corrcoef(x[:, common], rowvar=False)[pairs]
                                 - truth_corr[pairs]).mean()) if pairs.any() else None
            rows.append(dict(seed=seed, arm=arm, cell_class=category,
                             shared_gene_correlation_mae=error,
                             shared_variable_genes=int(common.sum()), shared_gene_pairs=int(pairs.sum())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--reference-run", type=Path, default=ROOT / "output/step01_joint_sampling_seed42")
    parser.add_argument("--geometry-dir", type=Path, default=ROOT / "output/merfish_holdout_009")
    parser.add_argument("--output", type=Path, default=ROOT / "output/step01_joint_sampling_multiseed")
    parser.add_argument("--mender-jobs", type=positive_int, default=2)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2 ** 32 for s in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    if len(args.seeds) < 2:
        parser.error("Provide at least two seeds")

    import pandas as pd
    reference = args.reference_run.resolve()
    old = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    if old.get("status") != "complete":
        raise ValueError("Reference run must be complete")
    geometry_hash = sha256(args.geometry_dir.resolve() / "generated_009.h5ad")
    input_hashes = [sha256(ROOT / f"data/merfish/merfish_{i}_paste.h5ad") for i in (4, 14)]
    target_hash = sha256(ROOT / "data/merfish/merfish_9_paste.h5ad")
    if (old.get("geometry_sha256") != geometry_hash or old.get("input_sha256") != input_hashes
            or old.get("held_out_sha256") != target_hash):
        raise ValueError("Current geometry, inputs or evaluation target differ from the reference run")
    cache_hash = sha256(reference / "candidate_cache.npz")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", expression_seeds=args.seeds, reference_run=str(reference),
                  geometry_sha256=geometry_hash, candidate_cache_sha256=cache_hash,
                  completed_seeds=[], protocol="Expression seed repeats; geometry and donor probabilities fixed",
                  variability="Descriptive standard deviation across seeds (ddof=1); not independent slice replicates")
    summaries, by_type = [], []
    try:
        write_json(output / "run.json", report)
        for seed in args.seeds:
            seed_dir = output / f"seed{seed}"
            if seed == old["expression_seed"]:
                print(f"Seed {seed}: copying completed reference results", flush=True)
                shutil.copytree(reference, seed_dir)
            else:
                print(f"Seed {seed}: resampling with frozen candidate cache", flush=True)
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("run_step01.py")),
                                "--seed", str(seed), "--geometry-dir", str(args.geometry_dir.resolve()),
                                "--candidate-cache-dir", str(reference), "--output", str(seed_dir),
                                "--mender-jobs", str(args.mender_jobs)], check=True, cwd=ROOT)
            current = json.loads((seed_dir / "run.json").read_text(encoding="utf-8"))
            if (current.get("status") != "complete" or current.get("expression_seed") != seed
                    or sha256(seed_dir / "candidate_cache.npz") != cache_hash
                    or any(current.get(k) != old.get(k) for k in (
                        "geometry_sha256", "input_sha256", "held_out_sha256", "parameters",
                        "frozen_shape", "evaluation_projection_seed", "min_covariance_cells"))):
                raise ValueError(f"Seed {seed} did not preserve the reference protocol")
            summaries.append(pd.read_csv(seed_dir / "step01_comparison.csv").assign(seed=seed))
            by_type.append(pd.read_csv(seed_dir / "step01_by_cell_class.csv").assign(seed=seed))
            report["completed_seeds"].append(seed)
            write_json(output / "run.json", report)
        overall, types = pd.concat(summaries, ignore_index=True), pd.concat(by_type, ignore_index=True)
        shared = shared_gene_correlations({s: output / f"seed{s}" for s in args.seeds},
                                          ROOT / "data/merfish/merfish_9_paste.h5ad")
        overall = overall.merge(shared[shared.cell_class.isna()].drop(columns="cell_class"),
                                on=["seed", "arm"], validate="one_to_one")
        types = types.merge(shared[shared.cell_class.notna()], on=["seed", "arm", "cell_class"],
                            validate="one_to_one")
        overall.to_csv(output / "all_seed_comparison.csv", index=False)
        types.to_csv(output / "all_seed_by_cell_class.csv", index=False)
        summary = summarize(overall, grouping=None)
        summary.to_csv(output / "stability_overall.csv", index=False)
        summarize(types, grouping="cell_class").to_csv(output / "stability_by_cell_class.csv", index=False)
        report["status"] = "complete"
        print(summary.to_string(index=False), flush=True)
        print(f"Results: {output}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "run.json", report)


if __name__ == "__main__":
    main()
