"""Check real imports, POT optimization and MENDER on tiny synthetic data."""

from importlib.metadata import version
from pathlib import Path
import sys

# Running `python scripts/...` should import this checkout's SpatialZ module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import torch
from SpatialZ import Generate_spatialz


def main():
    for package in ('torch', 'numpy', 'pandas', 'scipy', 'anndata', 'scanpy',
                    'POT', 'SOMENDER', 'squidpy', 'numba', 'scikit-image', 'xarray', 'zarr'):
        print(f'{package}: {version(package)}', flush=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Smoke check device: {device}', flush=True)
    np.random.seed(42)
    torch.manual_seed(42)
    a = AnnData(X=sparse.csr_matrix(np.arange(24, dtype=np.float32).reshape(8, 3) + 1),
                obs=pd.DataFrame({'cell_type': pd.Categorical(['A', 'B'] * 4)},
                                 index=[f'c{i}' for i in range(8)]),
                var=pd.DataFrame(index=['g1', 'g2', 'g3']))
    a.obsm['spatial'] = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
    b = a.copy()
    b.obsm['spatial'] += .2
    generated = Generate_spatialz(a, b, n_cell=8, alpha=.5, device=device,
                                 cell_type_key='cell_type', nb_iter_max=2, lr=1.,
                                 num_projections=4, syn_mode='default', k_sam=3,
                                 Beta=1., mender_n_jobs=1, seed=42, verbose=False)
    x = generated.X.toarray() if sparse.issparse(generated.X) else np.asarray(generated.X)
    if (generated.shape != (8, 3) or not np.isfinite(x).all() or (x < 0).any()
            or not np.isfinite(generated.obsm['spatial']).all() or (x.sum(1) == 0).any()):
        raise RuntimeError('SpatialZ dependency smoke check returned invalid synthetic data')
    print('PASS: real SpatialZ + POT + MENDER generation (synthetic fixture only).', flush=True)


if __name__ == '__main__':
    main()
