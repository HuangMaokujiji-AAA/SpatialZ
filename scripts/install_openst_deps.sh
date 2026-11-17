#!/usr/bin/env bash
# Supplement the supplied server environment; preserve installed versions.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON:-python}"

# Fail before installation if this is not the prepared server Python environment.
"$python_bin" - <<'PY'
import sys
import anndata, h5py, numpy, pandas, scipy, scanpy, sklearn, torch, ot, seaborn, shapely
print('Python:', sys.executable, flush=True)
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), flush=True)
PY

mkdir -p output/openst_dependency_setup
snapshot_dir="$(mktemp -d "$project_dir/output/openst_dependency_setup/run-XXXXXXXX")"
"$python_bin" -m pip list --format=freeze > "$snapshot_dir/installed-constraints.txt"
echo "Existing package snapshot: $snapshot_dir/installed-constraints.txt"

# Normal resolution supplies dependencies of these missing runtime imports.
# Constraints make version changes to existing packages an explicit error.
"$python_bin" -m pip install \
  --constraint "$snapshot_dir/installed-constraints.txt" \
  --requirement requirements-openst-extra.txt

# These packages' historical metadata pins conflict with the existing numerical
# stack. Install their code only; runtime dependencies were supplied above.
"$python_bin" -m pip install --no-deps \
  --constraint "$snapshot_dir/installed-constraints.txt" \
  'SOMENDER==1.1' 'squidpy==1.2.3'

"$python_bin" scripts/check_openst_dependencies.py
"$python_bin" -m pip list --format=freeze > "$snapshot_dir/installed-after.txt"
echo "Dependency installation and SpatialZ smoke check passed. Records: $snapshot_dir"
