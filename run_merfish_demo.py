"""Generate one midpoint slice from the first two bundled MERFISH sections."""

import argparse
import json
import os
from importlib.metadata import distributions
from pathlib import Path
import time


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=positive_int, default=1000)
    parser.add_argument("--mender-jobs", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=root / "output" / "merfish_first_pair")
    args = parser.parse_args()

    # Set these before importing libraries that initialize CPU thread pools.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(args.mender_jobs))

    import numpy as np
    import scanpy as sc
    from scipy import sparse
    import torch
    from SpatialZ import Generate_spatialz

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.mender_jobs)
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    slice_ids = ["0.04", "0.09"]
    paths = [root / "data" / "merfish" / f"merfish_{i}_paste.h5ad" for i in (4, 9)]
    adatas = [sc.read_h5ad(path) for path in paths]
    annotations = ["neuron_class", "domain", "Region", "annotation"]
    for path, label, adata in zip(paths, slice_ids, adatas):
        if not adata.n_obs or not adata.n_vars:
            raise ValueError(f"Empty input: {path}")
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError(f"Duplicate cell or gene identifiers: {path}")
        for key in ["cell_class", "slice_id", *annotations]:
            if key not in adata.obs:
                raise ValueError(f"Missing obs[{key!r}]: {path}")
        if adata.obs["cell_class"].isna().any():
            raise ValueError(f"Missing cell types: {path}")
        if set(adata.obs["slice_id"].astype(str)) != {label}:
            raise ValueError(f"Unexpected slice labels: {path}")
        if "spatial" not in adata.obsm or adata.obsm["spatial"].shape != (adata.n_obs, 2):
            raise ValueError(f"Expected n_cells x 2 spatial coordinates: {path}")
        if not np.isfinite(adata.obsm["spatial"]).all():
            raise ValueError(f"Non-finite spatial coordinates: {path}")
        print(f"Input: {path.name}, shape={adata.shape}", flush=True)
    if not adatas[0].var_names.equals(adatas[1].var_names):
        raise ValueError("Input gene names and order must match.")

    parameters = dict(
        adata1_id=slice_ids[0], adata2_id=slice_ids[1],
        alpha=0.5, device=device, lr=1e5,
        nb_iter_max=args.iterations, seed=args.seed, num_projections=80,
        cell_type_key="cell_class", syn_mode="default", k_sam=50,
        Beta=1, add_obs_list=annotations, mender_n_jobs=args.mender_jobs,
    )
    # Refuse to overwrite an earlier run; use --output for a fresh directory.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    report = {
        "status": "running", "inputs": [str(path) for path in paths],
        "input_shapes": [list(a.shape) for a in adatas], "parameters": parameters,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_runtime": torch.version.cuda,
    }
    report_path = output / "run.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Device: {device}; MENDER processes: {args.mender_jobs}", flush=True)
    start = time.perf_counter()
    try:
        generated = Generate_spatialz(adatas[0], adatas[1], **parameters)
        expression = generated.X
        values = expression.data if sparse.issparse(expression) else np.asarray(expression)
        if not np.isfinite(values).all() or not np.isfinite(generated.obsm["spatial"]).all():
            raise ValueError("Generated expression or coordinates contain non-finite values.")
        if generated.n_obs != int((adatas[0].n_obs + adatas[1].n_obs) / 2):
            raise ValueError("Unexpected number of generated cells.")
        if not generated.var_names.equals(adatas[0].var_names):
            raise ValueError("Generated gene names or order changed.")
        generated.obs["slice_id"] = "0.04-0.09-1"
        generated.obs["data_type"] = "synthetic"
        filename = output / "midpoint.h5ad"
        generated.write_h5ad(filename)
        nonzero = expression.getnnz(axis=1) if sparse.issparse(expression) else np.count_nonzero(expression, axis=1)
        zero_cells = int(np.count_nonzero(nonzero == 0))
        report.update(status="complete", output_shape=list(generated.shape),
                      all_zero_expression_cells=zero_cells,
                      elapsed_seconds=time.perf_counter() - start)
        print(f"Saved: {filename}", flush=True)
        print(f"Output shape: {generated.shape}; all-zero expression cells: {zero_cells}", flush=True)
        if zero_cells:
            print("Review all-zero expression cells before using this result for evaluation.", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
