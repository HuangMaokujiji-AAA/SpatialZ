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
    import scanpy as sc
except ImportError:
    ad = None


class ProtocolTests(unittest.TestCase):
    def test_whole_mode_does_not_read_training_xy_to_make_a_grid(self):
        data = SimpleNamespace(path=Path(runner.__file__), section_key='n_section',
                               annotation_key='annotation', spatial_key='spatial_3d_aligned',
                               rows={s: [s] for s in runner.SECTIONS}, z={s: s * 30. for s in runner.SECTIONS})
        data.hvg_info = {'ordered_genes_sha256': 'fixture'}
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

        data.hvg_info = {'ordered_genes_sha256': 'fixture'}
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

    def test_production_panel_is_exactly_2000(self):
        self.assertEqual(runner.N_HVG, 2000)

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
    def setUp(self):
        # Small fixtures exercise exactly the same selection and dimension checks.
        self.panel_size = patch.object(runner, 'N_HVG', 2)
        self.panel_size.start()
        self.addCleanup(self.panel_size.stop)

    def fixture(self, path):
        n = len(runner.SECTIONS) * 8
        labels = ['Tumor_Keratin_Pearl', 'Tumor', 'Plasma_IgA', 'Plasma_IgG',
                  'Germinal_Center_Plasma_IgM_B_cell', 'T_cell', 'Tumor', 'Tumor']
        a = ad.AnnData(X=sparse.csr_matrix(np.log1p(np.arange(n * 4).reshape(n, 4) % 7), dtype=np.float32),
                      obs=pd.DataFrame({'n_section': np.repeat(runner.SECTIONS, 8),
                                        'annotation': pd.Categorical(labels * len(runner.SECTIONS))},
                                       index=[f'cell{i}' for i in range(n)]),
                      var=pd.DataFrame(index=['g1', 'g2', 'g3', 'g4']))
        xy = np.tile(np.array([[0., 0.], [5., 1.], [9., 2.], [10., 1.],
                               [15., 2.], [19., 3.], [9.5, 1.], [10.5, 2.]]), (19, 1))
        a.obsm['spatial_3d_aligned'] = np.column_stack((xy, np.repeat(runner.SECTIONS, 8) * 30.))
        a.obsm['spatial'] = np.full_like(xy, np.nan)
        a.uns['log1p'] = {'base': None}
        a.write_h5ad(path)

    def test_training_only_hvg_matches_scanpy_and_ignores_hidden_expression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'input.h5ad'
            self.fixture(path)
            output = root / 'panel'
            output.mkdir()
            data = runner.OpenST(path)
            try:
                original = data.expression
                reads = []

                def read(section):
                    self.assertIn(section, runner.TRAIN)
                    reads.append(section)
                    return original(section)

                ref = ad.AnnData(sparse.vstack([original(s) for s in runner.TRAIN], format='csr'))
                ref.uns['log1p'] = {'base': None}
                sc.pp.highly_variable_genes(ref, flavor='seurat', n_top_genes=2, n_bins=20)
                values = ref.var['dispersions_norm'].to_numpy()
                candidates = np.flatnonzero(ref.var['highly_variable'].to_numpy() & np.isfinite(values))
                expected = np.sort(candidates[np.lexsort((candidates, -values[candidates]))][:2])
                with patch.object(data, 'expression', read):
                    info = runner.prepare_hvgs(data, output)
                self.assertEqual(reads, list(runner.TRAIN))
                np.testing.assert_array_equal(data.gene_indices, expected)
                self.assertEqual(info['fit']['flavor'], 'seurat')
                self.assertEqual(info['n_fit_cells'], 13 * 8)
                with patch.object(data, 'expression', side_effect=AssertionError('Frozen panel must be reused')):
                    self.assertEqual(runner.prepare_hvgs(data, output), info)
                a = data.section(3)
                self.assertEqual(a.n_vars, 2)
                self.assertIn('Tumor_Keratin_Pearl', set(a.obs['cell_type']))
                self.assertIn('Plasma_IgA', set(a.obs['cell_type']))
                self.assertIn('Germinal_Center_Plasma_IgM_B_cell', set(a.obs['cell_type']))
            finally:
                data.close()
            changed = ad.read_h5ad(path)
            hidden = ~changed.obs['n_section'].isin(runner.TRAIN).to_numpy()
            x = changed.X.toarray()
            x[hidden] = np.random.default_rng(19).uniform(5, 20, (hidden.sum(), changed.n_vars))
            changed.X = sparse.csr_matrix(x)
            changed_path = root / 'changed_hidden.h5ad'
            changed.write_h5ad(changed_path)
            other = root / 'other_panel'
            other.mkdir()
            data = runner.OpenST(changed_path)
            try:
                self.assertEqual(runner.prepare_hvgs(data, other)['genes'], info['genes'])
            finally:
                data.close()

    def test_optional_validation_and_tiled_stitching(self):
        events = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'input.h5ad', root / 'results'
            self.fixture(source)
            candidates = root / 'candidates.json'
            candidates.write_text(json.dumps([{'iterations': 2}, {'iterations': 3}]))
            original = runner.OpenST.section

            def read(data, section):
                events.append(('read', section))
                if section in runner.TEST:
                    self.assertTrue((output / 'selection.json').is_file())
                    self.assertTrue((output / f'test/section_{section}/generated.h5ad').is_file())
                elif section in runner.VALIDATION:
                    self.assertTrue(any((output / 'validation').glob(f'candidate_*/section_{section}/generated.h5ad')))
                return original(data, section)

            def generate(a, b, **p):
                events.append(('generate', p['nb_iter_max']))
                self.assertEqual(a.n_vars, 2)
                self.assertEqual(b.n_vars, 2)
                self.assertEqual(set(a.obs.columns), {'cell_type'})
                combined = ad.concat([a, b])
                return combined[np.linspace(0, combined.n_obs - 1, p['n_cell'], dtype=int)].copy()

            @contextmanager
            def backend(device, jobs):
                yield generate, SimpleNamespace(manual_seed=lambda s: None), 'cpu'

            argv = ['--stage', 'all', '--input', str(source), '--output', str(output), '--core-um', '10',
                    '--spatial-mode', 'tiled', '--halo-um', '2', '--candidates', str(candidates), '--eval-projections', '4']
            with patch.object(runner, 'generation_backend', backend), patch.object(runner.OpenST, 'section', read):
                runner.main(argv)
                self.assertIn('parameters', json.loads((output / 'selection.json').read_text()))
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 12)
                self.assertEqual(len(pd.read_csv(output / 'test_macro.csv')), 3)
                for section in runner.TEST:
                    folder = output / f'test/section_{section}'
                    g = ad.read_h5ad(folder / 'generated.h5ad')
                    self.assertEqual(g.n_vars, 2)
                    self.assertNotIn('track', g.obs.columns)
                    self.assertTrue(g.obs_names.is_unique)
                    self.assertEqual(set(g.obs['section']), {section})
                    for _, tile in pd.read_csv(folder / 'tiles.csv').iterrows():
                        mask = g.obs['tile_id'].to_numpy() == tile['tile']
                        self.assertTrue(runner.in_box(g.obsm['spatial'][mask], json.loads(tile['bounds'])).all())
                events.clear()
                runner.main(argv)
                self.assertEqual(events, [])
                other = root / 'changed_candidates.json'
                other.write_text(json.dumps([{'iterations': 9}]))
                with self.assertRaisesRegex(ValueError, 'already frozen'):
                    runner.main(argv + ['--candidates', str(other)])
                with self.assertRaisesRegex(ValueError, 'frozen validation'):
                    runner.main(['--stage', 'selected-test', '--input', str(source),
                                 '--output', str(output), '--core-um', '10', '--halo-um', '2',
                                 '--spatial-mode', 'tiled', '--eval-projections', '4', '--seed', '43'])
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    runner.main(argv + ['--halo-um', '3'])

    def test_default_single_fine_annotation_hvg_run(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'input.h5ad', Path(temp) / 'direct'
            self.fixture(source)
            fixture = ad.read_h5ad(source)
            keep = np.ones(fixture.n_obs, dtype=bool)
            keep[np.flatnonzero(fixture.obs['n_section'].to_numpy() == 4)[-1]] = False
            fixture[keep].copy().write_h5ad(source)
            original = runner.OpenST.section
            reads, generations = [], []

            def read(data, section):
                self.assertNotIn(section, runner.VALIDATION)
                reads.append(section)
                if section in runner.TEST:
                    self.assertTrue((output / f'test/section_{section}/generated.h5ad').is_file())
                return original(data, section)

            def generate(a, b, **p):
                generations.append(p)
                self.assertEqual(p['nb_iter_max'], runner.DEFAULT_PARAMETERS['iterations'])
                self.assertEqual(a.shape, (8, 2))
                self.assertEqual(b.shape, (8, 2))
                self.assertEqual(p['n_cell'], 8)
                self.assertIn('Tumor_Keratin_Pearl', set(a.obs['cell_type']))
                self.assertIn('Plasma_IgA', set(a.obs['cell_type']))
                np.testing.assert_array_equal(a.var_names, b.var_names)
                combined = ad.concat([a, b])
                return combined[np.linspace(0, combined.n_obs - 1, p['n_cell'], dtype=int)].copy()

            @contextmanager
            def backend(device, jobs):
                yield generate, SimpleNamespace(manual_seed=lambda s: None), 'cpu'

            argv = ['--input', str(source), '--output', str(output), '--eval-projections', '4']
            with patch.object(runner, 'generation_backend', backend),                     patch.object(runner.OpenST, 'section', read),                     patch.object(runner, 'run_validation', side_effect=AssertionError('No validation allowed')):
                output.mkdir()
                (output / 'selection.json').write_text('not used by direct mode')
                runner.main(argv)
                self.assertEqual(set(s for s in reads if s in runner.TEST), set(runner.TEST))
                self.assertEqual(sum(s in runner.TEST for s in reads), 4)
                self.assertEqual(len(generations), 4)
                panel = json.loads((output / 'hvg_selection.json').read_text())
                for section in runner.TEST:
                    g = ad.read_h5ad(output / f'test/section_{section}/generated.h5ad')
                    self.assertEqual(g.shape, (8, 2))
                    self.assertEqual(g.var_names.tolist(), panel['genes'])
                    self.assertNotIn('track', g.obs.columns)
                self.assertFalse((output / 'track1').exists())
                self.assertFalse((output / 'validation_scores.csv').exists())
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 12)
                metric = json.loads((output / 'test/section_4/metrics.json').read_text())
                self.assertEqual(metric['comparisons']['generated']['n_cells'], 8)
                self.assertEqual(metric['comparisons']['generated']['real_n_cells'], 7)
                reads.clear()
                runner.main(argv)
                self.assertEqual(reads, [])
                self.assertEqual(len(generations), 4)
                runner.main(argv + ['--targets', '4'])
                self.assertEqual(len(pd.read_csv(output / 'test_summary.csv')), 3)

    def test_old_full_transcriptome_results_are_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'input.h5ad', Path(temp) / 'old'
            self.fixture(source)
            output.mkdir()
            (output / 'protocol.json').write_text(json.dumps({'schema': 1}))
            with self.assertRaisesRegex(ValueError, 'cannot be reused'):
                runner.main(['--input', str(source), '--output', str(output), '--stage', 'prepare'])

    def test_empty_tiles_and_single_endpoint_fallback_resume(self):
        def endpoint(x):
            a = ad.AnnData(X=sparse.csr_matrix([[1., 2.], [3., 4.]]),
                           obs=pd.DataFrame({'cell_type': ['A', 'A']}, index=['a', 'b']),
                           var=pd.DataFrame(index=['g1', 'g2']))
            a.obsm['spatial'] = np.array([[x, 1.], [x + 1., 2.]])
            return a

        data = SimpleNamespace(genes=np.array(['g1', 'g2']), z={4: 120.},
                               hvg_info={'ordered_genes_sha256': 'fixture'},
                               section=lambda s: endpoint(1. if s == 3 else 21.))
        protocol = dict(halo_um=0., tiles=[dict(id=i, bounds=[i * 10., 0., (i + 1) * 10., 10.])
                                           for i in range(3)])
        args = SimpleNamespace(seed=42, mender_jobs=2)
        task = dict(left=3, target=4, right=5, alpha_left=.5)

        def unused(*a, **k):
            self.fail('SpatialZ must not be called with an empty endpoint')

        backend = (unused, SimpleNamespace(manual_seed=lambda s: None), 'cpu')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            _, g, records = runner.generate_tiles(data, task, protocol,
                                                    runner.DEFAULT_PARAMETERS, args, path, backend)
            self.assertEqual(g.n_obs, 2)
            self.assertEqual([r['method'] for r in records],
                             ['single_endpoint_resampling', 'empty', 'single_endpoint_resampling'])
            _, resumed, again = runner.generate_tiles(data, task, protocol,
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
