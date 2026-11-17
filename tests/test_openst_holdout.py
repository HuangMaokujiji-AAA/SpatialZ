"""Protocol, sparse I/O, leakage isolation and stitching checks; no GPU needed."""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

import run_openst_holdout as runner

try:
    import anndata as ad
except ImportError:
    ad = None


class ProtocolTests(unittest.TestCase):
    def test_whole_mode_does_not_read_training_xy_to_make_a_grid(self):
        data = SimpleNamespace(path=Path(runner.__file__), section_key='n_section',
                               annotation_key='annotation', spatial_key='spatial_3d_aligned',
                               rows={s: [s] for s in runner.SECTIONS}, z={s: s * 30. for s in runner.SECTIONS})
        data.xy = lambda s: self.fail('Whole mode must not build a training grid')
        p = runner.make_protocol(data, SimpleNamespace(spatial_mode='whole', core_um=1500.,
                                 halo_um=150., xy_scale=1., z_scale=1.))
        self.assertEqual(p['spatial_mode'], 'whole')
        self.assertEqual(p['tiles'], [])
        self.assertIsNone(p['grid_origin_um'])

    def test_physical_splits_weights_and_training_only_grid(self):
        reads = []
        data = SimpleNamespace(path=Path(runner.__file__), section_key='n_section',
                               annotation_key='annotation', spatial_key='spatial_3d_aligned',
                               rows={s: [s] for s in runner.SECTIONS}, z={s: s * 30. for s in runner.SECTIONS})

        def xy(s):
            reads.append(s)
            self.assertIn(s, runner.TRAIN)
            return np.array([[-1., -1.], [11., 11.]])

        data.xy = xy
        args = SimpleNamespace(core_um=10., halo_um=2., xy_scale=1., z_scale=1.)
        p = runner.make_protocol(data, args)
        self.assertEqual(set(reads), set(runner.TRAIN))
        self.assertEqual(len(runner.TRAIN), 13)
        self.assertEqual(len(p['training_pairs']), 12)
        self.assertEqual(len(p['pseudo_holdouts']), 11)
        self.assertEqual({t['target'] for t in p['test']}, {4, 9, 18, 24})
        self.assertEqual({t['target'] for t in p['validation']}, {6, 34})
        self.assertAlmostEqual(p['validation'][1]['alpha_left'], 2 / 3)
        self.assertTrue(any((t['left'], t['target'], t['right']) == (2, 3, 5)
                            for t in p['pseudo_holdouts']))
        self.assertTrue(any((t['left'], t['target'], t['right']) == (23, 25, 26)
                            for t in p['pseudo_holdouts']))
        self.assertEqual(p['grid_origin_um'], [-10., -10.])

    def test_core_ownership_is_unique_including_boundaries(self):
        xy = np.array([[0., 0.], [10., 0.], [10., 10.], [19.99, 19.99], [5., 10.]])
        ownership = np.zeros(len(xy), dtype=int)
        for x in (0., 10.):
            for y in (0., 10.):
                ownership += runner.in_box(xy, [x, y, x + 10, y + 10])
        np.testing.assert_array_equal(ownership, 1)

    def test_coarse_annotations_are_merged_before_generation(self):
        labels = np.array(['Tumor_Keratin_Pearl', 'Tumor', 'Plasma_IgA',
                           'Germinal_Center_Plasma_IgM_B_cell', 'Plasma_IgG', 'T_cell'])
        np.testing.assert_array_equal(runner.track_labels(labels, 1), labels)
        np.testing.assert_array_equal(runner.track_labels(labels, 2),
                                      ['Tumor', 'Tumor', 'Plasma cell', 'Plasma cell', 'Plasma cell', 'T_cell'])

    def test_csr_reader_supports_noncontiguous_rows(self):
        matrix = sparse.csr_matrix(np.arange(20).reshape(5, 4))
        with tempfile.TemporaryDirectory() as temp, h5py.File(Path(temp) / 'data.h5', 'w') as f:
            g = f.create_group('X')
            g.attrs['encoding-type'] = 'csr_matrix'
            g.attrs['shape'] = matrix.shape
            for key in ('data', 'indices', 'indptr'):
                g[key] = getattr(matrix, key)
            np.testing.assert_array_equal(runner.read_matrix(g, np.array([0, 2, 3])).toarray(),
                                          matrix[[0, 2, 3]].toarray())

    def test_constant_correlation_and_sparse_metrics(self):
        self.assertIsNone(runner.correlation(np.ones(3), np.ones(3)))
        a = SimpleNamespace(X=sparse.csr_matrix([[1, 0], [3, 4]]))
        m, v, d = runner.expression_stats(a)
        np.testing.assert_array_equal(m, [2, 2])
        np.testing.assert_array_equal(v, [1, 4])
        np.testing.assert_array_equal(d, [1, .5])


@unittest.skipIf(ad is None, 'anndata is needed for fixture H5AD integration')
class IntegrationTests(unittest.TestCase):
    def fixture(self, path):
        n = len(runner.SECTIONS) * 8
        labels = ['Tumor_Keratin_Pearl', 'Tumor', 'Plasma_IgA', 'Plasma_IgG',
                  'Germinal_Center_Plasma_IgM_B_cell', 'T_cell', 'Tumor', 'Tumor']
        a = ad.AnnData(X=sparse.csr_matrix(np.arange(n * 4).reshape(n, 4) % 7, dtype=np.float32),
                      obs=pd.DataFrame({'n_section': np.repeat(runner.SECTIONS, 8),
                                        'annotation': pd.Categorical(labels * len(runner.SECTIONS))},
                                       index=[f'cell{i}' for i in range(n)]),
                      var=pd.DataFrame(index=['g1', 'g2', 'g3', 'g4']))
        xy = np.tile(np.array([[0., 0.], [5., 1.], [9., 2.], [10., 1.],
                               [15., 2.], [19., 3.], [9.5, 1.], [10.5, 2.]]), (19, 1))
        a.obsm['spatial_3d_aligned'] = np.column_stack((xy, np.repeat(runner.SECTIONS, 8) * 30.))
        # Deliberately invalid unregistered coordinates: the runner must ignore these.
        a.obsm['spatial'] = np.full_like(xy, np.nan)
        a.write_h5ad(path)

    def test_full_command_locks_validation_then_reads_saved_test_predictions(self):
        events = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'input.h5ad', root / 'results'
            self.fixture(source)
            candidates = root / 'candidates.json'
            candidates.write_text(json.dumps([{'iterations': 2}, {'iterations': 3}]))
            original = runner.OpenST.section

            def read(data, section, track):
                events.append(('read', section, track))
                if section in runner.TEST:
                    self.assertTrue((output / 'selection.json').is_file())
                    self.assertTrue((output / f'track{track}/test/section_{section}/generated.h5ad').is_file())
                elif section in runner.VALIDATION:
                    base = output / f'track{track}/validation'
                    self.assertTrue(any(base.glob(f'candidate_*/section_{section}/generated.h5ad')))
                return original(data, section, track)

            def generate(a, b, **p):
                events.append(('generate', p['nb_iter_max']))
                self.assertEqual(p['cell_type_key'], 'cell_type')
                self.assertEqual(set(a.obs.columns), {'cell_type'})
                self.assertEqual(set(b.obs.columns), {'cell_type'})
                combined = ad.concat([a, b])
                g = combined[np.linspace(0, combined.n_obs - 1, p['n_cell'], dtype=int)].copy()
                # Exercise cropping on a core edge: a point at x=10 must only
                # survive in the right tile even when seen in both halo inputs.
                g.obs_names = [f'g{i}' for i in range(g.n_obs)]
                return g

            @contextmanager
            def backend(device, jobs):
                yield generate, SimpleNamespace(manual_seed=lambda s: None), 'cpu'

            argv = ['--stage', 'all', '--input', str(source), '--output', str(output), '--core-um', '10',
                    '--spatial-mode', 'tiled', '--halo-um', '2', '--candidates', str(candidates), '--eval-projections', '4']
            with patch.object(runner, 'generation_backend', backend), patch.object(runner.OpenST, 'section', read):
                runner.main(argv)
                selection = json.loads((output / 'selection.json').read_text())
                self.assertEqual(set(selection['tracks']), {'1', '2'})
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 24)
                self.assertEqual(len(pd.read_csv(output / 'test_macro.csv')), 6)
                for track in (1, 2):
                    for section in runner.TEST:
                        folder = output / f'track{track}/test/section_{section}'
                        g = ad.read_h5ad(folder / 'generated.h5ad')
                        self.assertTrue(sparse.issparse(g.X))
                        self.assertTrue(g.obs_names.is_unique)
                        self.assertEqual(set(g.obs['section']), {section})
                        tiles = pd.read_csv(folder / 'tiles.csv')
                        for _, tile in tiles.iterrows():
                            bounds = json.loads(tile['bounds'])
                            mask = g.obs['tile_id'].to_numpy() == tile['tile']
                            self.assertTrue(runner.in_box(g.obsm['spatial'][mask], bounds).all())
                        if track == 2:
                            self.assertFalse(set(g.obs['cell_type']) & set(runner.COARSE))
                # Completed task checkpoints are reusable without target reads or generation.
                events.clear()
                runner.main(argv)
                self.assertEqual(events, [])
                other = root / 'changed_candidates.json'
                other.write_text(json.dumps([{'iterations': 9}]))
                with self.assertRaisesRegex(ValueError, 'already frozen'):
                    runner.main(argv + ['--candidates', str(other)])
                self.assertEqual(events, [])
                with self.assertRaisesRegex(ValueError, 'frozen validation'):
                    runner.main(['--stage', 'selected-test', '--input', str(source),
                                 '--output', str(output), '--core-um', '10', '--halo-um', '2',
                                 '--spatial-mode', 'tiled', '--eval-projections', '4', '--seed', '43'])
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    runner.main(argv + ['--halo-um', '3'])

    def test_default_directly_reconstructs_test_slices_without_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'input.h5ad', Path(temp) / 'direct'
            self.fixture(source)
            # Target #4 has seven cells; its endpoints each have eight. Prediction
            # must request eight from endpoints, without consulting the target.
            fixture = ad.read_h5ad(source)
            keep = np.ones(fixture.n_obs, dtype=bool)
            keep[np.flatnonzero(fixture.obs['n_section'].to_numpy() == 4)[-1]] = False
            fixture[keep].copy().write_h5ad(source)
            original = runner.OpenST.section
            reads = []
            generations = []

            def read(data, section, track):
                self.assertNotIn(section, runner.VALIDATION)
                reads.append((section, track))
                if section in runner.TEST:
                    self.assertTrue((output / f'track{track}/test/section_{section}/generated.h5ad').is_file())
                return original(data, section, track)

            def generate(a, b, **p):
                generations.append(p)
                self.assertEqual(p['nb_iter_max'], runner.DEFAULT_PARAMETERS['iterations'])
                self.assertEqual(p['Beta'], runner.DEFAULT_PARAMETERS['beta'])
                self.assertEqual(a.n_obs, 8)
                self.assertEqual(b.n_obs, 8)
                self.assertEqual(p['n_cell'], 8)
                combined = ad.concat([a, b])
                return combined[np.linspace(0, combined.n_obs - 1, p['n_cell'], dtype=int)].copy()

            @contextmanager
            def backend(device, jobs):
                yield generate, SimpleNamespace(manual_seed=lambda s: None), 'cpu'

            argv = ['--input', str(source), '--output', str(output), '--core-um', '10',
                    '--halo-um', '2', '--eval-projections', '4']
            with patch.object(runner, 'generation_backend', backend), \
                    patch.object(runner.OpenST, 'section', read), \
                    patch.object(runner, 'run_validation', side_effect=AssertionError('No validation allowed')):
                # An existing selection file must not affect the fixed baseline.
                output.mkdir()
                (output / 'selection.json').write_text('not used by direct mode')
                runner.main(argv)
                self.assertEqual({s for s, _ in reads if s in runner.TEST}, set(runner.TEST))
                self.assertEqual(sum(s in runner.TEST for s, _ in reads), 8)
                self.assertEqual(len(generations), 8)  # Exactly one call per target and track.
                self.assertEqual(json.loads((output / 'protocol.json').read_text())['spatial_mode'], 'whole')
                settings = json.loads((output / 'test_parameters.json').read_text())
                self.assertEqual(settings['parameter_source'], 'fixed_defaults')
                self.assertFalse((output / 'track1/validation').exists())
                self.assertFalse((output / 'validation_scores.csv').exists())
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 24)
                metric = json.loads((output / 'track1/test/section_4/metrics.json').read_text())
                self.assertEqual(metric['comparisons']['generated']['n_cells'], 8)
                self.assertEqual(metric['comparisons']['generated']['real_n_cells'], 7)
                reads.clear()
                runner.main(argv)
                self.assertEqual(reads, [])
                self.assertEqual(len(generations), 8)
                runner.main(argv + ['--targets', '4', '--tracks', '1'])
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 3)

    def test_empty_tiles_and_single_endpoint_fallback_resume(self):
        def endpoint(x):
            a = ad.AnnData(X=sparse.csr_matrix([[1., 2.], [3., 4.]]),
                           obs=pd.DataFrame({'cell_type': ['A', 'A']}, index=['a', 'b']),
                           var=pd.DataFrame(index=['g1', 'g2']))
            a.obsm['spatial'] = np.array([[x, 1.], [x + 1., 2.]])
            return a

        data = SimpleNamespace(genes=np.array(['g1', 'g2']), z={4: 120.},
                               section=lambda s, t: endpoint(1. if s == 3 else 21.))
        protocol = dict(halo_um=0., tiles=[dict(id=i, bounds=[i * 10., 0., (i + 1) * 10., 10.])
                                           for i in range(3)])
        args = SimpleNamespace(seed=42, mender_jobs=2)
        task = dict(left=3, target=4, right=5, alpha_left=.5)

        def unused(*a, **k):
            self.fail('SpatialZ must not be called with an empty endpoint')

        backend = (unused, SimpleNamespace(manual_seed=lambda s: None), 'cpu')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            _, g, records = runner.generate_tiles(data, task, 1, protocol,
                                                    runner.DEFAULT_PARAMETERS, args, path, backend)
            self.assertEqual(g.n_obs, 2)
            self.assertEqual([r['method'] for r in records],
                             ['single_endpoint_resampling', 'empty', 'single_endpoint_resampling'])
            self.assertEqual([r['kept_cells'] for r in records], [1, 0, 1])
            # Restart after generation, before evaluation: reuse every tile checkpoint.
            _, resumed, again = runner.generate_tiles(data, task, 1, protocol,
                                                        runner.DEFAULT_PARAMETERS, args, path, backend)
            np.testing.assert_array_equal(resumed.X.toarray(), g.X.toarray())
            self.assertEqual(records, again)

    def test_chunked_expression_sampling_matches_scalar_reference_for_dense_and_sparse(self):
        # Load just the function, so this test does not require torch/MENDER/scanpy.
        source = (Path(runner.__file__).parent / 'SpatialZ.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'synthesize_gene_expression')
        calls = []

        def cosine(a, b):
            calls.append((a.shape[0], b.shape[0]))
            return cosine_similarity(a, b)

        namespace = dict(np=np, pd=pd, cKDTree=cKDTree, cosine_similarity=cosine,
                         isspmatrix=sparse.isspmatrix)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<sampling>', 'exec'), namespace)
        for use_sparse in (False, True):
            x = np.arange(30, dtype=float).reshape(6, 5)
            xy = np.column_stack((np.arange(6), np.zeros(6)))
            obs = pd.DataFrame({'ct': ['A', 'B', 'A', 'B', 'A', 'B']}, index=list('abcdef'))
            ref = ad.AnnData(X=sparse.csr_matrix(x) if use_sparse else x, obs=obs)
            ref.obsm['spatial'] = xy
            ref.obsm['env'] = np.array([[1., 0.], [0., 1.]] * 3)
            query = ad.AnnData(X=np.zeros((3, 5)), obs=pd.DataFrame({'ct': ['A', 'B', 'A']}, index=list('xyz')))
            query.obsm['spatial'] = np.array([[.1, 0.], [2.9, 0.], [4.1, 0.]])
            query.obsm['env'] = np.array([[1., .5], [.5, 1.], [1., .5]])
            np.random.seed(42)
            expected = np.zeros((3, 5))
            all_cos = cosine_similarity(ref.obsm['env'], query.obsm['env'])
            for i, label in enumerate(query.obs['ct']):
                # Original k=min(k_sam, same_type_count), global neighborhood then filter.
                indices = np.atleast_1d(cKDTree(xy).query(query.obsm['spatial'][i], k=3)[1])
                valid = indices[obs['ct'].to_numpy()[indices] == label]
                weights = np.exp(all_cos[valid, i]); weights /= weights.sum()
                for gene in range(5):
                    chosen = np.random.choice(np.arange(len(valid)), p=weights)
                    expected[i, gene] = x[valid[chosen], gene]
            np.random.seed(42)
            result = namespace['synthesize_gene_expression']([ref[:3].copy(), ref[3:].copy()],
                          query, 'ct', 4, 'env', 1., chunk_size=2)
            np.testing.assert_array_equal(result.X, expected)
            self.assertEqual(len(calls), 3 * (1 + int(use_sparse)))
            self.assertTrue(all(n_ref <= 4 and n_query == 1 for n_ref, n_query in calls))


if __name__ == '__main__':
    unittest.main()
