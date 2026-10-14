"""Engineering checks, not MERFISH performance evidence."""

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_step01
from sampling import build_candidates, expression_metrics, sample_expression


class FixtureSlice:
    def __init__(self, n, label, events):
        self.X = np.column_stack((np.arange(n) + 1., np.ones(n), np.arange(n) % 2))
        self.obsm = {"spatial": np.column_stack((np.arange(n), np.zeros(n))),
                     "mender": np.column_stack((np.ones(n), np.arange(n) % 2))}
        self.obs = pd.DataFrame(dict(cell_class=["A"] * n, Region=["R"] * n,
                                     slice_id=[label] * n), index=[f"{label}_{i}" for i in range(n)])
        self.var_names = pd.Index(["g1", "g2", "g3"])
        self.uns = {}
        self.events = events

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def n_obs(self):
        return len(self.obs)

    @property
    def n_vars(self):
        return len(self.var_names)

    @property
    def shape(self):
        return self.X.shape

    def copy(self):
        result = copy.deepcopy(self)
        result.events = self.events
        return result

    def concatenate(self, others, batch_key):
        return self

    def __getitem__(self, mask):
        result = self.copy()
        result.X = self.X[np.asarray(mask)]
        result.obs = self.obs.loc[mask].copy()
        return result

    def write_h5ad(self, path):
        unit = self.obs["sampling_unit"].iloc[0]
        self.events.append(f"save_{unit}")
        path.write_text("TEST PLACEHOLDER, not scientific data", encoding="utf-8")


class Step01Tests(unittest.TestCase):
    def test_candidate_rule_and_gene_draws_match_original_function(self):
        # Extract just the original function, avoiding unavailable server imports.
        tree = ast.parse((run_step01.ROOT / "SpatialZ.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "synthesize_gene_expression")
        namespace = dict(np=np, cKDTree=cKDTree, cosine_similarity=cosine_similarity)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "SpatialZ.py", "exec"), namespace)
        reference, query = FixtureSlice(6, "source", []), FixtureSlice(4, "0.09", [])
        reference.obs["cell_class"] = ["B", "A", "B", "A", "B", "A"]
        query.X[:] = 0
        pools, weights = build_candidates(reference.obsm["spatial"],
            reference.obs["cell_class"].to_numpy(), reference.obsm["mender"],
            query.obsm["spatial"], query.obs["cell_class"].to_numpy(), query.obsm["mender"], 2, 1)
        result, _ = sample_expression(reference.X, pools, weights, "gene", 42)
        # Patch only RNG choice so this test does not alter process-global RNG state.
        rng = np.random.RandomState(42)
        fake_np = SimpleNamespace(atleast_1d=np.atleast_1d, exp=np.exp, zeros=np.zeros,
                                  arange=np.arange, random=SimpleNamespace(choice=rng.choice))
        namespace["np"] = fake_np
        original = namespace[function.name]([reference], query, "cell_class", 2, "mender", 1)
        np.testing.assert_array_equal(result, original.X)
        # The global nearest-two rule is deliberately retained, rather than class-only KNN.
        np.testing.assert_array_equal(pools[0], [1])

    def test_whole_cell_is_a_complete_donor_vector_with_shared_pool(self):
        x = np.array([[0., 100.], [100., 0.]])
        pools = [np.array([0, 1])] * 1000
        weights = [np.array([.5, .5])] * 1000
        cell, donors = sample_expression(x, pools, weights, "cell", 42)
        gene, _ = sample_expression(x, pools, weights, "gene", 42)
        np.testing.assert_array_equal(cell, x[donors])
        np.testing.assert_array_equal(cell.sum(axis=1), np.full(1000, 100.))
        self.assertGreater(gene.sum(axis=1).var(), 0)
        # This negative-covariance example also rejects a guaranteed variance increase.
        self.assertGreater(np.count_nonzero(donors == 0), 400)
        self.assertGreater(np.count_nonzero(donors == 1), 400)
        repeated, _ = sample_expression(x, pools, weights, "cell", 42)
        np.testing.assert_array_equal(repeated, cell)

    def test_empty_candidates_keep_original_zero_policy_in_both_arms(self):
        for unit in ("gene", "cell"):
            x, donors = sample_expression(np.array([[1., 2.]]),
                                         [np.array([], dtype=int)], [np.array([])], unit, 42)
            np.testing.assert_array_equal(x, [[0., 0.]])
            np.testing.assert_array_equal(donors, [-1])

    def test_covariance_metrics_exclude_diagonal_and_report_collapsed_genes(self):
        real = np.tile([[0., 0.], [2., 2.]], (20, 1))
        broken = np.tile([[0., 0.], [0., 2.], [2., 0.], [2., 2.]], (10, 1))
        metrics = expression_metrics(broken, real)
        self.assertAlmostEqual(metrics["offdiag_covariance_relative_error"], 1.)
        self.assertAlmostEqual(metrics["offdiag_correlation_mae"], 1.)
        self.assertAlmostEqual(metrics["total_variance_ratio"], .5)
        collapsed = expression_metrics(np.ones_like(real), real)
        self.assertEqual(collapsed["collapsed_real_variable_genes"], 2)
        self.assertEqual(collapsed["correlation_pairs"], 0)
        self.assertIsNone(collapsed["offdiag_correlation_mae"])
        self.assertFalse(expression_metrics(real[:3], real)["covariance_eligible"])
        self.assertIsNone(expression_metrics(real[:0], real)["total_expression_mean"])

    def test_run_saves_both_arms_before_reading_target_and_preserves_geometry(self):
        events = []
        fixtures = {"merfish_4_paste.h5ad": FixtureSlice(2, "0.04", events),
                    "merfish_14_paste.h5ad": FixtureSlice(4, "0.14", events),
                    "generated_009.h5ad": FixtureSlice(3, "0.09", events),
                    "merfish_9_paste.h5ad": FixtureSlice(5, "0.09", events)}
        def read(path):
            events.append(f"read_{path.name}")
            return fixtures[path.name].copy()
        def mender(a, b, geometry, **kwargs):
            events.append("mender")
            self.assertNotIn("read_merfish_9_paste.h5ad", events)
            np.testing.assert_array_equal(geometry.X, np.zeros((3, 3)))
            return a, b, geometry
        # Accept the positional cell-type argument used by the existing helper.
        def mender_call(a, b, geometry, cell_type_key, **kwargs):
            self.assertEqual(cell_type_key, "cell_class")
            return mender(a, b, geometry, **kwargs)
        modules = {"scanpy": SimpleNamespace(read_h5ad=read),
                   "SpatialZ": SimpleNamespace(perform_microenv_measure=mender_call)}
        test_root = run_step01.ROOT / "output"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="step01-test-", dir=test_root) as temp:
            directory = Path(temp)
            geometry_dir = directory / "geometry"
            geometry_dir.mkdir()
            manifest = dict(status="complete", seed=42, output_shape=[3, 3],
                input_sha256=["fixture", "fixture"], parameters=dict(
                    adata1_id="0.04", adata2_id="0.14", alpha=.5, cell_type_key="cell_class",
                    micro_env_key="mender", syn_mode="default", k_sam=50, Beta=1))
            (geometry_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = directory / "result"
            argv = ["run_step01.py", "--geometry-dir", str(geometry_dir), "--output", str(output)]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                    patch.object(run_step01, "sha256", return_value="fixture"):
                run_step01.main()
                # Existing results must be protected from accidental reruns.
                with self.assertRaises(FileExistsError):
                    run_step01.main()
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["frozen_shape"], [3, 3])
            self.assertEqual(report["real_shape"], [5, 3])
            summary = pd.read_csv(output / "step01_comparison.csv")
            self.assertEqual(summary["generated_n"].tolist(), [3, 3])
            for name in ("candidate_cache.npz", "candidate_counts.csv", "step01_metrics.json",
                         "step01_by_cell_class.csv", "gene/generated_009.h5ad", "cell/donors.csv",
                         "gene/metrics.json", "cell/expression_comparison.png"):
                self.assertTrue((output / name).is_file(), name)
        self.assertEqual(events, ["read_merfish_4_paste.h5ad", "read_merfish_14_paste.h5ad",
            "read_generated_009.h5ad", "mender", "save_gene", "save_cell", "read_merfish_9_paste.h5ad"])


if __name__ == "__main__":
    unittest.main()
