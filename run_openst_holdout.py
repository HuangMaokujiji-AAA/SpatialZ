"""Open-ST: fine annotations, training-only Seurat HVG2000, four held-out slices.

No flow-matching or RARS model is implemented by SpatialZ. Their observation
pairs and pseudo-holdouts are exported as protocol metadata, not trained here.
Validation-based parameter selection is optional (--stage all).
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from run_merfish_holdout import composition, correlation, mean_metrics


SECTIONS = (2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36)
TEST = {4: (3, 5), 18: (17, 19), 24: (23, 25), 9: (7, 11)}
VALIDATION = {6: (5, 7), 34: (33, 36)}
TRAIN = tuple(s for s in SECTIONS if s not in TEST and s not in VALIDATION)
DEFAULT_INPUT = (Path(__file__).resolve().parent / 'data' / 'Open-ST'
                 / 'GSE251926_metastatic_lymph_node_3d.h5ad')
N_HVG = 2000
DEFAULT_PARAMETERS = dict(iterations=1000, lr=1e5, projections=80,
                          k_sam=50, beta=1., syn_mode='default')


def write_json(path, value):
    """Atomic reports/checkpoints; interrupted writes cannot look complete."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def read_column(node, rows):
    if isinstance(node, h5py.Group):
        codes = node['codes'][rows]
        if (codes < 0).any():
            raise ValueError(f'Missing categorical values in {node.name}')
        return strings(node['categories'][:])[codes]
    return strings(node[rows])


def read_matrix(node, rows):
    """Read only the requested rows, without loading the 8.3 GB file or raw layer."""
    if isinstance(node, h5py.Dataset):
        return sparse.csr_matrix(node[rows])
    if node.attrs.get('encoding-type') != 'csr_matrix':
        raise ValueError('Expression must be dense or CSR encoded')
    shape = tuple(node.attrs['shape'])
    pieces = []
    # Section rows are contiguous in the real file; also support interleaved rows.
    for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
        if not len(run):
            continue
        pointers = node['indptr'][int(run[0]):int(run[-1]) + 2]
        lo, hi = int(pointers[0]), int(pointers[-1])
        pieces.append(sparse.csr_matrix((node['data'][lo:hi], node['indices'][lo:hi],
                                         pointers - lo), shape=(len(run), shape[1])))
    return sparse.vstack(pieces, format='csr')


class OpenST:
    """Metadata access is separate from expression/XY/annotation access."""

    def __init__(self, path, section_key='n_section', annotation_key='annotation',
                 spatial_key='spatial_3d_aligned', xy_scale=1., z_scale=1.):
        self.path = Path(path).resolve()
        self.handle = h5py.File(self.path, 'r')
        self.section_key, self.annotation_key = section_key, annotation_key
        self.spatial_key, self.xy_scale, self.z_scale = spatial_key, xy_scale, z_scale
        try:
            f = self.handle
            sections = read_column(f['obs'][section_key], slice(None))
            self.sections = np.asarray(sections, dtype=int)
            if set(self.sections) != set(SECTIONS):
                raise ValueError(f'Expected physical section IDs {SECTIONS}; found {np.unique(self.sections)}')
            self.rows = {s: np.flatnonzero(self.sections == s) for s in SECTIONS}
            coords = f['obsm'][spatial_key]
            if coords.shape != (len(sections), 3):
                raise ValueError('Use registered n_cells x 3 coordinates, including physical Z')
            # Z is design metadata. Target XY, expression and labels are not read here.
            self.z = {}
            for s, rows in self.rows.items():
                zs = coords[rows, 2] * z_scale
                if not np.isfinite(zs).all() or np.ptp(zs) > 1e-6:
                    raise ValueError(f'Section #{s} must have a constant, finite Z')
                self.z[s] = float(zs[0])
            if any(self.z[b] <= self.z[a] for a, b in zip(SECTIONS, SECTIONS[1:])):
                raise ValueError('Physical Z must increase with section number')
            var = f['var']
            self.all_genes = read_column(var[var.attrs.get('_index', '_index')], slice(None))
            if len(set(self.all_genes)) != len(self.all_genes):
                raise ValueError('Gene identifiers must be unique')
            if 'log1p' not in f['uns']:
                raise ValueError("Seurat HVG expects log1p-normalized X with uns['log1p'] metadata")
            base = f['uns/log1p/base'] if 'base' in f['uns/log1p'] else None
            self.log_base = (None if base is None or base.attrs.get('encoding-type') == 'null'
                             else float(base[()]))
            if self.log_base is not None and (not np.isfinite(self.log_base)
                                              or self.log_base <= 0 or self.log_base == 1):
                raise ValueError('Invalid log1p base')
            self.gene_indices = None
            self.hvg_info = None
        except Exception:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()

    def xy(self, section):
        xy = self.handle['obsm'][self.spatial_key][self.rows[section], :2] * self.xy_scale
        if not np.isfinite(xy).all():
            raise ValueError(f'Nonfinite XY in section #{section}')
        return xy

    def expression(self, section):
        x = read_matrix(self.handle['X'], self.rows[section])
        x.sum_duplicates()
        x.eliminate_zeros()
        if not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError('Expected finite, nonnegative expression in X')
        return x

    def section(self, section):
        from anndata import AnnData

        if self.gene_indices is None:
            raise RuntimeError('Select and freeze training-only HVGs before loading model inputs')
        rows, f = self.rows[section], self.handle
        x = self.expression(section)[:, self.gene_indices].tocsr()
        labels = read_column(f['obs'][self.annotation_key], rows)
        obs = pd.DataFrame({'cell_type': pd.Categorical(labels)},
                           index=[f's{section}_r{r}' for r in rows])
        adata = AnnData(X=x, obs=obs, var=pd.DataFrame(index=self.genes))
        adata.obsm['spatial'] = self.xy(section)
        return adata


def prepare_hvgs(data, output):
    """Run Scanpy's Seurat selector once on pooled visible training cells only."""
    stat = data.path.stat()
    identity = dict(input=dict(path=str(data.path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                    training_sections=list(TRAIN), expression='X', flavor='seurat',
                    n_top_genes=N_HVG, n_bins=20, batch_key=None, log1p_base=data.log_base,
                    n_input_genes=len(data.all_genes))
    path = output / 'hvg_selection.json'
    if path.exists():
        info = json.loads(path.read_text())
        if info['fit'] != identity:
            raise ValueError('HVG input or settings changed; use a new output directory')
    else:
        import scanpy as sc
        from anndata import AnnData

        if len(data.all_genes) < N_HVG:
            raise ValueError(f'Need at least {N_HVG} input genes')
        matrices = []
        for section in TRAIN:
            print(f'HVG fit: reading visible training section #{section}', flush=True)
            matrices.append(data.expression(section))
        x = sparse.vstack(matrices, format='csr')
        del matrices
        training = AnnData(X=x, var=pd.DataFrame(index=data.all_genes))
        del x
        training.uns['log1p'] = {'base': data.log_base}
        sc.pp.highly_variable_genes(training, flavor='seurat', n_top_genes=N_HVG,
                                    n_bins=20, batch_key=None, subset=False, inplace=True)
        table = training.var.copy()
        n_fit_cells = training.n_obs
        del training
        scores = table['dispersions_norm'].to_numpy(dtype=float)
        eligible = np.flatnonzero(table['highly_variable'].to_numpy() & np.isfinite(scores))
        if len(eligible) < N_HVG:
            raise ValueError(f'Seurat returned only {len(eligible)} finite HVG candidates; need {N_HVG}')
        # Scanpy can include extra genes tied at its cutoff. Resolve ties by input
        # gene index to enforce exactly 2000, then preserve original column order.
        ranked = eligible[np.lexsort((eligible, -scores[eligible]))][:N_HVG]
        indices = np.sort(ranked)
        table['highly_variable'] = False
        table.iloc[indices, table.columns.get_loc('highly_variable')] = True
        ranks = np.full(len(table), np.nan)
        ranks[ranked] = np.arange(1, N_HVG + 1)
        table['hvg_rank'] = ranks
        table.index.name = 'gene'
        temporary = output / 'hvg_statistics.csv.tmp'
        table.to_csv(temporary)
        temporary.replace(output / 'hvg_statistics.csv')
        genes = data.all_genes[indices].tolist()
        digest = hashlib.sha256(json.dumps(genes, ensure_ascii=False).encode()).hexdigest()
        info = dict(schema=1, fit=identity, scanpy_version=version('scanpy'), n_fit_cells=n_fit_cells,
                    gene_indices=indices.tolist(), genes=genes, ordered_genes_sha256=digest,
                    ordering='original input columns; cutoff ties broken by original gene index')
        write_json(path, info)
    indices = np.asarray(info['gene_indices'], dtype=int)
    if (len(indices) != N_HVG or len(np.unique(indices)) != N_HVG
            or (indices < 0).any() or (indices >= len(data.all_genes)).any()
            or not np.array_equal(data.all_genes[indices], info['genes'])
            or hashlib.sha256(json.dumps(info['genes'], ensure_ascii=False).encode()).hexdigest()
               != info['ordered_genes_sha256']):
        raise ValueError('Invalid frozen HVG panel')
    data.gene_indices, data.genes, data.hvg_info = indices, data.all_genes[indices], info
    (output / 'hvg_genes.txt').write_text('\n'.join(info['genes']) + '\n', encoding='utf-8')
    print(f'Frozen {N_HVG} Seurat HVGs from {info["n_fit_cells"]:,} training cells.', flush=True)
    return info


def triplet(left, target, right, z):
    if not z[left] < z[target] < z[right]:
        raise ValueError('Target Z must lie strictly between the two endpoints')
    # SpatialZ alpha is the LEFT endpoint weight, not distance from the left.
    return dict(left=left, target=target, right=right,
                alpha_left=(z[right] - z[target]) / (z[right] - z[left]))


def in_box(xy, bounds, halo=0.):
    x0, y0, x1, y1 = bounds
    return ((xy[:, 0] >= x0 - halo) & (xy[:, 0] < x1 + halo)
            & (xy[:, 1] >= y0 - halo) & (xy[:, 1] < y1 + halo))


def make_protocol(data, args):
    mode = getattr(args, 'spatial_mode', 'tiled')
    origin, shape = None, None
    tiles = []
    if mode == 'tiled':
        # The optional global column grid uses visible training observations only.
        low, high = np.full(2, np.inf), np.full(2, -np.inf)
        for s in TRAIN:
            xy = data.xy(s)
            low, high = np.minimum(low, xy.min(0)), np.maximum(high, xy.max(0))
        origin = np.floor(low / args.core_um) * args.core_um
        shape = (np.floor((high - origin) / args.core_um).astype(int) + 1).tolist()
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                x0, y0 = origin + args.core_um * np.array([ix, iy])
                tiles.append(dict(id=len(tiles), bounds=[float(x0), float(y0),
                                                        float(x0 + args.core_um), float(y0 + args.core_um)]))
    stat = data.path.stat()
    return dict(schema=2, method='SpatialZ baseline (no flow matching or RARS)',
                input=dict(path=str(data.path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                fields=dict(section=data.section_key, annotation=data.annotation_key,
                            spatial=data.spatial_key, expression='X'),
                coordinate_units='micrometers', xy_scale=args.xy_scale, z_scale=args.z_scale,
                section_counts={str(s): len(data.rows[s]) for s in SECTIONS},
                z_um={str(s): data.z[s] for s in SECTIONS}, train=list(TRAIN),
                validation=[triplet(a, t, b, data.z) for t, (a, b) in VALIDATION.items()],
                test=[triplet(a, t, b, data.z) for t, (a, b) in TEST.items()],
                training_pairs=[dict(left=a, right=b, dz_um=data.z[b] - data.z[a])
                                for a, b in zip(TRAIN, TRAIN[1:])],
                pseudo_holdouts=[triplet(a, t, b, data.z)
                                 for a, t, b in zip(TRAIN, TRAIN[1:], TRAIN[2:])],
                spatial_mode=mode, core_um=args.core_um, halo_um=args.halo_um,
                grid_origin_um=None if origin is None else origin.tolist(),
                grid_shape=shape, tiles=tiles, annotation_mode='fine (former Track 1)',
                hvg=data.hvg_info,
                upstream_note='The supplied globally aligned/processed file is used as-is; '
                              'upstream registration/normalization leakage is not re-estimated.')


def parameters_from_file(path):
    values = [DEFAULT_PARAMETERS] if path is None else json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError('Candidates must be a nonempty JSON list')
    candidates = []
    for value in values:
        if not isinstance(value, dict) or set(value) - set(DEFAULT_PARAMETERS):
            raise ValueError(f'Allowed candidate keys: {list(DEFAULT_PARAMETERS)}')
        p = {**DEFAULT_PARAMETERS, **value}
        for key in ('iterations', 'projections', 'k_sam'):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if (not np.isfinite(p['lr']) or p['lr'] <= 0 or not np.isfinite(p['beta'])
                or p['beta'] < 0 or p['syn_mode'] not in ('default', 'fast')):
            raise ValueError('Invalid lr, beta or syn_mode')
        candidates.append(p)
    return candidates


@contextmanager
def generation_backend(device, jobs):
    import torch
    from SpatialZ import Generate_spatialz

    torch.set_num_threads(jobs)
    chosen = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    yield Generate_spatialz, torch, chosen


def generate_tiles(data, task, protocol, parameters, args, directory, backend):
    """No held-out data access: only the two endpoints enter this function."""
    import anndata as ad

    generate, torch, device = backend
    left, right = task['left'], task['right']
    a, b = data.section(left), data.section(right)
    alpha = task['alpha_left']
    pieces, records = [], []
    for tile in protocol['tiles']:
        tile_path = directory / f"tile_{tile['id']:04d}.h5ad"
        record_path = tile_path.with_suffix('.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['kept_cells']:
                pieces.append(tile_path)
            records.append(record)
            continue
        bounds = tile['bounds']
        masks = [in_box(s.obsm['spatial'], bounds, protocol['halo_um']) for s in (a, b)]
        na, nb = (int(m.sum()) for m in masks)
        n = int(alpha * na + (1 - alpha) * nb)
        seed = (args.seed + task['target'] * 1009 + tile['id']) % (2**32)
        record = dict(tile=tile['id'], bounds=bounds, left_cells=na, right_cells=nb,
                      requested_halo_cells=n, kept_cells=0, seed=seed, method='empty')
        if n:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if na and nb:
                record['method'] = 'SpatialZ'
                print(f"Section #{task['target']} tile {tile['id']}: {na}/{nb} -> {n}", flush=True)
                g = generate(a[masks[0]].copy(), b[masks[1]].copy(),
                             adata1_id=f'left_s{left}', adata2_id=f'right_s{right}',
                             alpha=alpha, device=device, n_cell=n, seed=seed,
                             nb_iter_max=parameters['iterations'], lr=parameters['lr'],
                             num_projections=parameters['projections'], k_sam=parameters['k_sam'],
                             Beta=parameters['beta'], syn_mode=parameters['syn_mode'],
                             cell_type_key='cell_type', add_obs_list=None,
                             mender_n_jobs=args.mender_jobs, verbose=True)
            else:
                # SpatialZ requires two nonempty endpoints. Explicit zero-mass
                # endpoint fallback; the output records these cells separately.
                record['method'] = 'single_endpoint_resampling'
                donor = a[masks[0]] if na else b[masks[1]]
                chosen = np.random.choice(donor.n_obs, n, replace=n > donor.n_obs)
                g = donor[chosen].copy()
            if g.n_obs != n or not np.array_equal(g.var_names, data.genes):
                raise ValueError('Generator changed requested count or gene order')
            xy = np.asarray(g.obsm['spatial'])
            if xy.shape != (n, 2) or not np.isfinite(xy).all():
                raise ValueError('Generator returned invalid coordinates')
            g = g[in_box(xy, bounds)].copy()
            record['kept_cells'] = g.n_obs
            if g.n_obs:
                g.X = sparse.csr_matrix(g.X)
                g.X.eliminate_zeros()
                if not np.isfinite(g.X.data).all() or (g.X.data < 0).any():
                    raise ValueError('Generator returned invalid expression')
                # Drop MENDER representations and retain fine cell annotations.
                g = ad.AnnData(X=g.X, obs=pd.DataFrame({
                    'cell_type': pd.Categorical(g.obs['cell_type'].astype(str).to_numpy()),
                    'tile_id': tile['id'], 'generation_method': record['method'],
                    'section': task['target'], 'data_type': 'synthetic'},
                    index=[f"s{task['target']}_tile{tile['id']}_c{i}" for i in range(g.n_obs)]),
                    var=pd.DataFrame(index=data.genes), obsm={'spatial': g.obsm['spatial'].copy()})
                xyz = np.column_stack((g.obsm['spatial'], np.full(g.n_obs, data.z[task['target']])))
                g.obsm['spatial_3d_aligned'] = xyz
                temporary = tile_path.with_suffix('.tmp.h5ad')
                g.write_h5ad(temporary)
                temporary.replace(tile_path)
                pieces.append(tile_path)
            del g
        write_json(record_path, record)
        records.append(record)
    if not pieces:
        raise ValueError('No generated core cells; inspect endpoint support and tile parameters')
    # CSR concatenation keeps whole-slice HVG evaluation sparse.
    generated = ad.concat([ad.read_h5ad(p) for p in pieces], join='inner', merge='same')
    generated.uns['openst_task'] = {**task, 'parameters': parameters,
                                   'hvg_sha256': data.hvg_info['ordered_genes_sha256']}
    temporary = directory / 'generated.tmp.h5ad'
    generated.write_h5ad(temporary)
    temporary.replace(directory / 'generated.h5ad')
    pd.DataFrame(records).to_csv(directory / 'tiles.csv', index=False)
    return (a, b), generated, records


def generate_whole(data, task, protocol, parameters, args, directory, backend):
    """One call on the two full endpoints, matching the MERFISH runner's workflow."""
    import anndata as ad

    generate, torch, device = backend
    a, b = data.section(task['left']), data.section(task['right'])
    path = directory / 'generated.h5ad'
    if path.exists():
        return (a, b), ad.read_h5ad(path), []
    alpha = task['alpha_left']
    n_cell = int(alpha * a.n_obs + (1 - alpha) * b.n_obs)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Section #{task['target']} whole slice: "
          f"{a.n_obs}/{b.n_obs} -> {n_cell} cells, {a.n_vars} genes", flush=True)
    start = time.perf_counter()
    g = generate(a.copy(), b.copy(), adata1_id=f"left_s{task['left']}",
                 adata2_id=f"right_s{task['right']}", alpha=alpha, n_cell=n_cell,
                 device=device, seed=args.seed, nb_iter_max=parameters['iterations'],
                 lr=parameters['lr'], num_projections=parameters['projections'],
                 cell_type_key='cell_type', syn_mode=parameters['syn_mode'],
                 k_sam=parameters['k_sam'], Beta=parameters['beta'],
                 add_obs_list=None, mender_n_jobs=args.mender_jobs, verbose=True)
    if g.n_obs != n_cell or not np.array_equal(g.var_names, data.genes):
        raise ValueError('Generator changed requested count or gene order')
    xy = np.asarray(g.obsm['spatial'])
    if xy.shape != (n_cell, 2) or not np.isfinite(xy).all():
        raise ValueError('Generator returned invalid coordinates')
    # Convert before constructing the saved object; avoid a second dense slice copy.
    g.X = sparse.csr_matrix(g.X)
    g.X.eliminate_zeros()
    if not np.isfinite(g.X.data).all() or (g.X.data < 0).any():
        raise ValueError('Generator returned invalid expression')
    generated = ad.AnnData(X=g.X, var=pd.DataFrame(index=data.genes),
        obs=pd.DataFrame({'cell_type': pd.Categorical(g.obs['cell_type'].astype(str).to_numpy()),
                          'section': task['target'], 'data_type': 'synthetic',
                          'generation_method': 'SpatialZ_whole'},
                         index=[f"s{task['target']}_c{i}" for i in range(n_cell)]),
        obsm={'spatial': xy.copy(), 'spatial_3d_aligned':
              np.column_stack((xy, np.full(n_cell, data.z[task['target']])))})
    del g
    generated.uns['openst_task'] = {**task, 'parameters': parameters,
                                   'spatial_mode': 'whole',
                                   'hvg_sha256': data.hvg_info['ordered_genes_sha256']}
    temporary = directory / 'generated.tmp.h5ad'
    generated.write_h5ad(temporary)
    temporary.replace(path)
    print(f"Whole-slice prediction saved in {time.perf_counter() - start:.1f}s; "
          'reading target for evaluation next.', flush=True)
    return (a, b), generated, []


def expression_stats(adata):
    x = sparse.csr_matrix(adata.X, dtype=np.float64)
    x.sum_duplicates()
    x.eliminate_zeros()
    mean = np.asarray(x.mean(0)).ravel()
    variance = np.maximum(0, np.asarray(x.power(2).mean(0)).ravel() - mean**2)
    detection = np.asarray((x > 0).mean(0)).ravel()
    return mean, variance, detection


def compare(candidate, real, directions):
    if not candidate.var_names.equals(real.var_names):
        raise ValueError('Evaluation gene order differs')
    x, xv, xd = expression_stats(candidate)
    y, yv, yd = expression_stats(real)
    xy, uv = candidate.obsm['spatial'], real.obsm['spatial']
    result = mean_metrics(x, y)
    result.update(n_cells=candidate.n_obs, real_n_cells=real.n_obs,
                  cell_count_relative_error=float((candidate.n_obs - real.n_obs) / real.n_obs),
                  gene_variance_pearson=correlation(xv, yv),
                  gene_detection_mae_pp=float(100 * np.abs(xd - yd).mean()),
                  spatial_sliced_wasserstein_1_um=float(np.mean([
                      wasserstein_distance(xy @ v, uv @ v) for v in directions])),
                  spatial_chamfer_mean_um=float((cKDTree(uv).query(xy)[0].mean()
                                                 + cKDTree(xy).query(uv)[0].mean()) / 2),
                  zero_expression_cells=int(np.count_nonzero(np.asarray(candidate.X.sum(1)).ravel() == 0)))
    result['cell_type'], _ = composition(candidate.obs['cell_type'].astype(str).to_numpy(),
                                         real.obs['cell_type'].astype(str).to_numpy())
    return result


def evaluate(sources, generated, real, records, directory, args):
    directions = np.random.default_rng(args.seed).normal(size=(args.eval_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    candidates = {'generated': generated, 'left_endpoint': sources[0], 'right_endpoint': sources[1]}
    comparisons = {name: compare(value, real, directions) for name, value in candidates.items()}
    alpha = generated.uns['openst_task']['alpha_left']
    baseline = alpha * expression_stats(sources[0])[0] + (1 - alpha) * expression_stats(sources[1])[0]
    truth_mean = expression_stats(real)[0]
    gene_table = pd.DataFrame({'gene': real.var_names, 'real_mean': truth_mean,
                               'endpoint_weighted_mean': baseline})
    gene_table['generated_mean'] = expression_stats(generated)[0]
    gene_table.to_csv(directory / 'gene_comparison.csv', index=False)
    _, table = composition(generated.obs['cell_type'].astype(str).to_numpy(),
                           real.obs['cell_type'].astype(str).to_numpy())
    table.to_csv(directory / 'cell_type_comparison.csv', index=False)
    rows = []
    for ct in table['category']:
        a = generated[generated.obs['cell_type'].astype(str).to_numpy() == ct]
        b = real[real.obs['cell_type'].astype(str).to_numpy() == ct]
        row = dict(cell_type=ct, generated_n=a.n_obs, real_n=b.n_obs,
                   small_group=a.n_obs < 20 or b.n_obs < 20)
        if a.n_obs and b.n_obs:
            row.update(compare(a, b, directions))
        rows.append(row)
    pd.json_normalize(rows).to_csv(directory / 'metrics_by_cell_type.csv', index=False)
    score = comparisons['generated']
    # Dimensionless, equal-component selection rule, fixed before any test access.
    selection_score = (score['spatial_sliced_wasserstein_1_um'] / args.core_um
                       + score['cell_type']['total_variation']
                       + score['gene_mean_mae'] / max(float(truth_mean.mean()), 1e-12))
    metrics = dict(comparisons=comparisons, selection_score=float(selection_score),
                   endpoint_mean_baseline=mean_metrics(baseline, truth_mean),
                   target_types_absent_from_endpoints=sorted(set(real.obs['cell_type'].astype(str))
                       - set(sources[0].obs['cell_type'].astype(str))
                       - set(sources[1].obs['cell_type'].astype(str))),
                   fallback_cells=sum(r['kept_cells'] for r in records
                                      if r['method'] == 'single_endpoint_resampling'),
                   empty_tiles=sum(r['requested_halo_cells'] == 0 for r in records),
                   notes=['Unpaired distribution metrics; no cell identity correspondence.',
                          'Spatial distances are in micrometers; probability mass is normalized per slice.',
                          'Expression is supplied log1p X restricted to training-only Seurat HVGs.',
                          'Fine cell annotations are retained; expression metrics use the frozen HVG panel.'])
    return metrics


def run_task(data, task, protocol, parameters, args, directory, backend):
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(task=task, parameters=parameters, seed=args.seed,
                    hvg_sha256=data.hvg_info['ordered_genes_sha256'],
                    device=backend[2], mender_jobs=args.mender_jobs,
                    eval_projections=args.eval_projections, spatial_mode=args.spatial_mode)
    identity_path = directory / 'task.json'
    if identity_path.exists():
        saved = json.loads(identity_path.read_text())
        saved.setdefault('spatial_mode', 'tiled')  # Earlier runner versions only tiled.
        if saved != identity:
            raise ValueError(f'Task settings changed: use a new output directory ({directory})')
    write_json(identity_path, identity)
    metrics_path = directory / 'metrics.json'
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    start = time.perf_counter()
    report = dict(status='running', **identity)
    try:
        generator = generate_whole if args.spatial_mode == 'whole' else generate_tiles
        sources, generated, records = generator(data, task, protocol, parameters,
                                                args, directory, backend)
        # Prediction has been persisted before the held-out expression/XY/labels are read.
        report.update(stage='evaluating')
        write_json(directory / 'run.json', report)
        real = data.section(task['target'])
        metrics = evaluate(sources, generated, real, records, directory, args)
        write_json(metrics_path, metrics)
        report.update(status='complete')
        return metrics
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        write_json(directory / 'run.json', report)


def run_validation(data, protocol, candidates, args, backend):
    path = args.output / 'selection.json'
    frozen_settings = dict(seed=args.seed, candidates=candidates,
                           eval_projections=args.eval_projections, mender_jobs=args.mender_jobs,
                           hvg_sha256=data.hvg_info['ordered_genes_sha256'])
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen['settings'] != frozen_settings:
            raise ValueError('Selection is already frozen; use a new output directory')
        return frozen
    rows, scores = [], []
    for i, parameters in enumerate(candidates):
        values = []
        for task in protocol['validation']:
            directory = args.output / 'validation' / f'candidate_{i:03d}' / f"section_{task['target']}"
            metrics = run_task(data, task, protocol, parameters, args, directory, backend)
            values.append(metrics['selection_score'])
            rows.append(dict(candidate=i, target=task['target'], score=values[-1]))
        scores.append(float(np.mean(values)))
    best = int(np.argmin(scores))
    pd.DataFrame(rows).to_csv(args.output / 'validation_scores.csv', index=False)
    selection = dict(schema=2, settings=frozen_settings, candidate=best,
                     parameters=candidates[best], validation_score=scores[best],
                     selection_rule='mean over #6/#34 of spatial_SW1_um/core_um + '
                                    'cell_type_TV + gene_mean_MAE/real_global_mean',
                     early_stopping='SpatialZ has no learned epochs; iteration budget is a candidate hyperparameter.')
    write_json(path, selection)
    return selection


def run_test(data, protocol, selection, args, backend):
    if selection is not None:
        frozen = selection['settings']
        if (frozen['seed'] != args.seed or frozen['eval_projections'] != args.eval_projections
                or frozen['mender_jobs'] != args.mender_jobs
                or frozen['hvg_sha256'] != data.hvg_info['ordered_genes_sha256']):
            raise ValueError('Test settings must match frozen validation')
    parameters = DEFAULT_PARAMETERS if selection is None else selection['parameters']
    settings = dict(parameter_source='fixed_defaults' if selection is None else 'validation_selection',
                    parameters=parameters, seed=args.seed, eval_projections=args.eval_projections,
                    mender_jobs=args.mender_jobs, hvg_sha256=data.hvg_info['ordered_genes_sha256'])
    path = args.output / 'test_parameters.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Test parameters or their source changed; use a new output directory')
    write_json(path, settings)
    rows = []
    for task in protocol['test']:
        if task['target'] not in args.targets:
            continue
        directory = args.output / 'test' / f"section_{task['target']}"
        metrics = run_task(data, task, protocol, parameters, args, directory, backend)
        for name, values in metrics['comparisons'].items():
            rows.append(dict(target=task['target'], comparison=name,
                             fallback_cells=metrics['fallback_cells'], **values))
    table = pd.json_normalize(rows)
    table.to_csv(args.output / 'test_summary.csv', index=False)
    write_json(args.output / 'test_scope.json', dict(targets=args.targets))
    numeric = [c for c in table.select_dtypes(include='number') if c != 'target']
    table.groupby('comparison')[numeric].mean().to_csv(args.output / 'test_macro.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Default: data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad relative to this script')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'output/openst_hvg2000')
    parser.add_argument('--stage', choices=['prepare', 'validate', 'test', 'selected-test', 'all'],
                        default='test', help='test (default): fixed parameters, no validation; '
                        'all: validate then test; selected-test: reuse frozen validation selection')
    parser.add_argument('--candidates', type=Path, help='JSON list of validation-only hyperparameter candidates')
    parser.add_argument('--targets', type=int, nargs='+', choices=list(TEST), default=list(TEST),
                        help='Test sections to reconstruct (default: all four); e.g. --targets 4')
    parser.add_argument('--spatial-mode', choices=['whole', 'tiled'], default='whole',
                        help='whole: one Generate_spatialz call per target, like MERFISH; '
                        'tiled: original Core/Halo protocol')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mender-jobs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--core-um', type=float, default=1500.)
    parser.add_argument('--halo-um', type=float, default=150., help='Halo width in micrometers (confirmed: 150)')
    parser.add_argument('--xy-scale', type=float, default=1., help='Convert registered XY to micrometers')
    parser.add_argument('--z-scale', type=float, default=1., help='Convert Z to micrometers')
    parser.add_argument('--section-key', default='n_section')
    parser.add_argument('--annotation-key', default='annotation')
    parser.add_argument('--spatial-key', default='spatial_3d_aligned')
    parser.add_argument('--eval-projections', type=int, default=128)
    args = parser.parse_args(argv)
    if (args.seed < 0 or args.seed >= 2**32 or args.mender_jobs < 1 or args.eval_projections < 1
            or not all(np.isfinite(v) and v > 0 for v in
                       (args.core_um, args.xy_scale, args.z_scale))
            or not np.isfinite(args.halo_um) or args.halo_um < 0):
        parser.error('Invalid seed, worker count, projection count, coordinate scale or tile dimensions')
    if len(set(args.targets)) != len(args.targets):
        parser.error('Do not repeat targets')
    if args.candidates is not None and args.stage in ('test', 'selected-test'):
        parser.error('--candidates is for --stage validate or --stage all; '
                     'omit it for direct fixed-parameter testing')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, str(args.mender_jobs))
    candidates = parameters_from_file(args.candidates)
    args.output = args.output.resolve()
    data = OpenST(args.input, args.section_key, args.annotation_key, args.spatial_key,
                  args.xy_scale, args.z_scale)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / 'protocol.json'
        if path.exists() and json.loads(path.read_text()).get('schema') != 2:
            raise ValueError('Old full-transcriptome/Track results cannot be reused; use a new output directory')
        prepare_hvgs(data, args.output)
        protocol = make_protocol(data, args)
        if path.exists():
            saved = json.loads(path.read_text())
            saved.setdefault('spatial_mode', 'tiled')
            if saved != protocol:
                raise ValueError('Input or protocol changed: use a new output directory')
        write_json(path, protocol)
        for key in ('training_pairs', 'pseudo_holdouts', 'validation', 'test'):
            pd.DataFrame(protocol[key]).to_csv(args.output / f'{key}.csv', index=False)
        print(f"Prepared {len(SECTIONS)} sections, {len(data.sections):,} cells, "
              f"mode={args.spatial_mode}, {len(protocol['tiles'])} columns. Output: {args.output}", flush=True)
        if args.stage == 'prepare':
            return
        if args.stage == 'selected-test' and not (args.output / 'selection.json').exists():
            raise ValueError('Run --stage validate first; selected-test requires frozen selection.json')
        with generation_backend(args.device, args.mender_jobs) as backend:
            environment = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
            write_json(args.output / 'environment.json', environment)
            if args.stage == 'test':
                selection = None
                print(f'Direct SpatialZ baseline: fixed parameters, targets={args.targets}, '
                      f'{N_HVG} HVGs, fine annotations; no validation selection or model training.', flush=True)
            elif args.stage in ('validate', 'all'):
                selection = run_validation(data, protocol, candidates, args, backend)
            else:
                selection = json.loads((args.output / 'selection.json').read_text())
            if args.stage in ('test', 'selected-test', 'all'):
                run_test(data, protocol, selection, args, backend)
    finally:
        data.close()


if __name__ == '__main__':
    main()
