"""Compare gene and whole-cell donor sampling on frozen MERFISH geometry."""

import argparse
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_merfish_holdout import evaluate, positive_int, sha256, snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, default=ROOT / "output/merfish_holdout_009")
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--seed", type=int, default=42, help="Expression sampling seed; geometry stays frozen")
    parser.add_argument("--mender-jobs", type=positive_int, default=2)
    parser.add_argument("--candidate-cache-dir", type=Path,
                        help="Completed step01 run whose donor pools/probabilities are reused without MENDER")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(args.mender_jobs))

    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse
    from sampling import build_candidates, expression_metrics, sample_expression

    geometry_dir = args.geometry_dir.resolve()
    geometry_path = geometry_dir / "generated_009.h5ad"
    old_run = json.loads((geometry_dir / "run.json").read_text(encoding="utf-8"))
    parameters = old_run["parameters"]
    expected = dict(adata1_id="0.04", adata2_id="0.14", alpha=.5, cell_type_key="cell_class",
                    micro_env_key="mender", syn_mode="default", k_sam=50, Beta=1)
    if old_run.get("status") != "complete" or any(parameters.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen run must be the completed 0.04/0.14 default holdout run (k_sam=50, Beta=1)")
    source_paths = [ROOT / f"data/merfish/merfish_{i}_paste.h5ad" for i in (4, 14)]
    input_hashes = [sha256(p) for p in source_paths]
    if input_hashes != old_run.get("input_sha256"):
        raise ValueError("Source files differ from those used to generate the frozen geometry")
    output = (args.output or ROOT / f"output/step01_joint_sampling_seed{args.seed}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "environment.json", {d.metadata["Name"]: d.version
               for d in distributions() if d.metadata["Name"]})
    report = dict(status="running", stage="loading", geometry=str(geometry_path),
                  geometry_sha256=sha256(geometry_path), geometry_seed=old_run["seed"],
                  expression_seed=args.seed, input_sha256=input_hashes, parameters=expected,
                  mender=dict(n_scales=6, radius=15, n_jobs=args.mender_jobs),
                  comparison="E0 fresh gene sampling vs E1 whole-cell sampling; same pools and weights",
                  evaluation_projection_seed=42, min_covariance_cells=20)
    start = time.perf_counter()
    try:
        write_json(output / "run.json", report)
        adatas = [sc.read_h5ad(p) for p in source_paths]
        sources = [snapshot(a, label) for a, label in zip(adatas, ("0.04", "0.14"))]
        geometry = sc.read_h5ad(geometry_path)
        frozen = snapshot(geometry, "0.09")
        frozen_obs = geometry.obs.copy(deep=True)
        if list(geometry.shape) != old_run["output_shape"]:
            raise ValueError("Frozen geometry shape differs from its manifest")
        if any(not np.array_equal(s["genes"], frozen["genes"]) for s in sources):
            raise ValueError("All gene identifiers and their order must match")
        # Old expression is not used. Geometry and all cell annotations remain fixed.
        geometry.X = np.zeros(geometry.shape, dtype=adatas[0].X.dtype)
        combined_x = np.concatenate([a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
                                     for a in adatas])
        if args.candidate_cache_dir:
            cache_dir = args.candidate_cache_dir.resolve()
            cached_run = json.loads((cache_dir / "run.json").read_text(encoding="utf-8"))
            if cached_run.get("status") != "complete" or any(
                    cached_run.get(k) != report[k] for k in ("geometry_sha256", "input_sha256", "parameters")):
                raise ValueError("Candidate cache must use identical inputs, frozen geometry and sampling parameters")
            print(f"Reusing frozen candidate probabilities: {cache_dir}", flush=True)
            with np.load(cache_dir / "candidate_cache.npz", allow_pickle=False) as cache:
                offsets, indices, weights = (cache[k].copy() for k in ("offsets", "indices", "probabilities"))
            if (offsets.shape != (geometry.n_obs + 1,) or offsets[0] != 0
                    or not np.issubdtype(offsets.dtype, np.integer) or (np.diff(offsets) < 0).any()
                    or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or offsets[-1] != len(indices) or weights.shape != indices.shape
                    or (indices < 0).any() or (indices >= len(combined_x)).any()
                    or not np.isfinite(weights).all() or (weights < 0).any()):
                raise ValueError("Invalid candidate cache arrays")
            pools = [indices[offsets[i]:offsets[i + 1]] for i in range(geometry.n_obs)]
            probabilities = [weights[offsets[i]:offsets[i + 1]] for i in range(geometry.n_obs)]
            reference_types = np.concatenate([s["cell_class"] for s in sources])
            for i, (pool, probability) in enumerate(zip(pools, probabilities)):
                if len(pool) and (not np.isclose(probability.sum(), 1)
                                  or not np.all(reference_types[pool] == frozen["cell_class"][i])):
                    raise ValueError(f"Invalid candidate types or probability sum at row {i}")
            cached_geometry = sc.read_h5ad(cache_dir / "gene/generated_009.h5ad")
            cached = snapshot(cached_geometry, "0.09")
            for key in ("xy", "genes", "cell_class", "Region"):
                np.testing.assert_array_equal(cached[key], frozen[key])
            np.testing.assert_array_equal(cached_geometry.obs_names, geometry.obs_names)
            geometry.obsm["mender"] = cached_geometry.obsm["mender"].copy()
            report.update(candidate_cache_source=str(cache_dir),
                          mender={**cached_run["mender"], "recomputed": False})
            shutil.copyfile(cache_dir / "candidate_cache.npz", output / "candidate_cache.npz")
        else:
            from SpatialZ import perform_microenv_measure
            report["stage"] = "mender"
            write_json(output / "run.json", report)
            print("Computing shared MENDER features using only 0.04, 0.14 and frozen synthetic geometry...", flush=True)
            adatas[0], adatas[1], geometry = perform_microenv_measure(
                adatas[0], adatas[1], geometry, "cell_class", scale=6, radius=15,
                micro_env_key="mender", n_jobs=args.mender_jobs)
            pd.testing.assert_frame_equal(geometry.obs, frozen_obs)
            np.testing.assert_array_equal(geometry.obsm["spatial"], frozen["xy"])
            pools, probabilities = build_candidates(
                np.concatenate([a.obsm["spatial"] for a in adatas]),
                np.concatenate([s["cell_class"] for s in sources]),
                np.concatenate([a.obsm["mender"] for a in adatas]),
                geometry.obsm["spatial"], frozen["cell_class"], geometry.obsm["mender"],
                k_sam=50, beta=1)
            lengths = np.array([len(p) for p in pools])
            np.savez_compressed(output / "candidate_cache.npz", offsets=np.r_[0, np.cumsum(lengths)],
                                indices=np.concatenate(pools), probabilities=np.concatenate(probabilities))
        lengths = np.array([len(p) for p in pools])
        report["candidate_cache_sha256"] = sha256(output / "candidate_cache.npz")
        pd.DataFrame(dict(cell_id=geometry.obs_names.astype(str),
                          cell_class=frozen["cell_class"], candidate_count=lengths)).to_csv(
                              output / "candidate_counts.csv", index=False)
        report.update(stage="sampling", empty_candidate_cells=int((lengths == 0).sum()),
                      frozen_shape=list(geometry.shape))
        write_json(output / "run.json", report)
        generated = {}
        source_ids = np.concatenate([a.obs_names.astype(str).to_numpy() for a in adatas])
        source_slices = np.repeat(["0.04", "0.14"], [a.n_obs for a in adatas])
        for unit in ("gene", "cell"):
            print(f"Sampling arm: {unit}", flush=True)
            result, donors = sample_expression(combined_x, pools, probabilities, unit, args.seed)
            arm = geometry.copy()
            arm.X = result
            arm.obs["sampling_unit"] = unit
            pd.testing.assert_frame_equal(arm.obs.drop(columns="sampling_unit"), frozen_obs)
            np.testing.assert_array_equal(arm.obsm["spatial"], frozen["xy"])
            arm.uns["step01"] = dict(sampling_unit=unit, expression_seed=args.seed,
                                    k_sam=50, Beta=1, frozen_geometry_sha256=report["geometry_sha256"])
            arm_dir = output / unit
            arm_dir.mkdir()
            arm.write_h5ad(arm_dir / "generated_009.h5ad")
            generated[unit] = snapshot(arm, "0.09")
            if unit == "cell":
                valid = donors >= 0
                donor_ids = np.full(len(donors), "", dtype=object)
                donor_slices = np.full(len(donors), "", dtype=object)
                donor_ids[valid], donor_slices[valid] = source_ids[donors[valid]], source_slices[donors[valid]]
                pd.DataFrame(dict(cell_id=arm.obs_names.astype(str), donor_index=donors,
                                  donor_cell_id=donor_ids, donor_slice=donor_slices)).to_csv(
                                      arm_dir / "donors.csv", index=False)
        # Target expression/coordinates/types/count are first read after BOTH arms are saved.
        report["stage"] = "evaluating"
        write_json(output / "run.json", report)
        print("Both generated slices saved. Reading real 0.09 for evaluation only.", flush=True)
        held_out_path = ROOT / "data/merfish/merfish_9_paste.h5ad"
        real = snapshot(sc.read_h5ad(held_out_path), "0.09")
        if not np.array_equal(real["genes"], frozen["genes"]):
            raise ValueError("Held-out gene identifiers and order must match")
        report["held_out_sha256"] = sha256(held_out_path)
        summaries, by_type = [], []
        for unit, data in generated.items():
            base = evaluate(sources, data, real, output / unit, seed=42)
            values = expression_metrics(data["x"], real["x"])
            summaries.append(dict(arm=unit, **values,
                                  **{k: v for k, v in base["comparisons"]["generated_0.09"].items()
                                     if k.startswith("gene_mean_")}))
            for category in sorted(set(data["cell_class"]) | set(real["cell_class"])):
                by_type.append(dict(arm=unit, cell_class=category, **expression_metrics(
                    data["x"][data["cell_class"] == category],
                    real["x"][real["cell_class"] == category])))
        pd.DataFrame(summaries).to_csv(output / "step01_comparison.csv", index=False)
        pd.DataFrame(by_type).to_csv(output / "step01_by_cell_class.csv", index=False)
        write_json(output / "step01_metrics.json", dict(overall=summaries, by_cell_class=by_type))
        report.update(status="complete", stage="complete", real_shape=list(real["x"].shape))
        print(pd.DataFrame(summaries).to_string(index=False), flush=True)
        print(f"Results: {output}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "run.json", report)


if __name__ == "__main__":
    main()
