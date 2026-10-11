"""Engineering checks for holdout isolation and unpaired metric behavior."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse  # Load before replacing server-only modules below.

import run_merfish_holdout as holdout


class FixtureSlice:
    def __init__(self, n, label, events):
        self.X = np.column_stack((np.arange(n) + 1., np.ones(n), np.arange(n) % 2))
        self.obsm = {"spatial": np.column_stack((np.arange(n), np.zeros(n)))}
        self.obs = pd.DataFrame({"cell_class": ["A"] * n, "Region": ["R"] * n,
                                 "slice_id": [label] * n}, index=[f"cell_{i}" for i in range(n)])
        self.obs_names = self.obs.index
        self.var_names = pd.Index(["g1", "g2", "g3"])
        self.n_obs, self.n_vars = n, 3
        self.shape = (n, 3)
        self.events = events

    def write_h5ad(self, path):
        self.events.append("save_generated")
        # A test placeholder, not a generated scientific artifact.
        path.write_text("mock generation", encoding="utf-8")


class HoldoutTests(unittest.TestCase):
    def test_normalized_distributions_with_unequal_cell_counts(self):
        real = holdout.snapshot(FixtureSlice(3, "0.09", []))
        repeated = {key: (value.copy() if key == "genes" else np.repeat(value, 2, axis=0))
                    for key, value in real.items()}
        result = holdout.compare(repeated, real, np.array([[1., 0.], [0., 1.]]))
        for key in ("gene_mean_mae", "gene_detection_mae_pp", "total_expression_wasserstein",
                    "spatial_sliced_wasserstein_1", "spatial_chamfer_mean"):
            self.assertAlmostEqual(result[key], 0.)
        self.assertEqual(result["cell_class"]["total_variation"], 0.)
        self.assertEqual(result["cell_count_difference"], 3)

    def test_complete_command_reads_target_only_after_generation_is_saved(self):
        events = []
        fixtures = {"merfish_4_paste.h5ad": FixtureSlice(2, "0.04", events),
                    "merfish_14_paste.h5ad": FixtureSlice(4, "0.14", events),
                    "merfish_9_paste.h5ad": FixtureSlice(5, "0.09", events)}
        fixtures["merfish_9_paste.h5ad"].obs.loc["cell_0", "cell_class"] = "unseen"

        def read(path):
            events.append(f"read_{path.name}")
            return fixtures[path.name]

        def generate(a, b, **parameters):
            events.append("generate")
            self.assertNotIn("read_merfish_9_paste.h5ad", events)
            self.assertEqual(parameters["n_cell"], 3)  # Input mean, not target count 5.
            self.assertEqual(parameters["alpha"], .5)
            # Ensure evaluation retains the original endpoint values even if the
            # generation implementation mutates AnnData inputs in place.
            a.X[:] = 999.
            return FixtureSlice(parameters["n_cell"], "synthetic", events)

        fake_torch = SimpleNamespace(manual_seed=lambda seed: None, set_num_threads=lambda n: None,
                                     cuda=SimpleNamespace(is_available=lambda: False),
                                     version=SimpleNamespace(cuda=None))
        modules = {"scanpy": SimpleNamespace(read_h5ad=read), "torch": fake_torch,
                   "SpatialZ": SimpleNamespace(Generate_spatialz=generate)}
        # Keep temporary artifacts inside the repository's ignored output tree.
        test_root = Path(holdout.__file__).resolve().parent / "output"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="holdout-test-", dir=test_root) as temp:
            output = Path(temp) / "result"
            argv = ["run_merfish_holdout.py", "--device", "cpu", "--output", str(output)]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                    patch.object(holdout, "sha256", return_value="fixture-only"):
                holdout.main()
            report = json.loads((output / "run.json").read_text())
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["output_shape"], [3, 3])
            self.assertEqual(report["real_shape"], [5, 3])
            self.assertEqual(metrics["real_cell_types_absent_from_inputs"], ["unseen"])
            self.assertEqual(metrics["qc"]["input_0.04"]["total_expression_mean"], 3.)
            # Means are averaged with equal slice weights, not pooled by cell count.
            expected_mean = .5 * np.array([1.5, 1., .5]) + .5 * np.array([2.5, 1., .5])
            real_mean = fixtures["merfish_9_paste.h5ad"].X.mean(axis=0)
            self.assertAlmostEqual(metrics["endpoint_mean_expression_baseline"]["gene_mean_mae"],
                                   np.abs(expected_mean - real_mean).mean())
            for name in ("generated_009.h5ad", "spatial_comparison.png", "expression_comparison.png",
                         "gene_comparison.csv", "cell_class_comparison.csv", "Region_comparison.csv",
                         "expression_by_cell_class.csv", "comparison_summary.csv", "environment.json"):
                self.assertTrue((output / name).is_file(), name)
        self.assertEqual(events, ["read_merfish_4_paste.h5ad", "read_merfish_14_paste.h5ad",
                                  "generate", "save_generated", "read_merfish_9_paste.h5ad"])


if __name__ == "__main__":
    unittest.main()
