
The implementations of **SpatialZ**, as well as tutorials, can be found at [https://spatialz-tutorial.readthedocs.io/en/latest/](https://spatialz-tutorial.readthedocs.io/en/latest/).


# 1 Installation Guide
This document provides detailed instructions for installing and setting up the SpatialZ project. 
Follow the steps below to configure your environment and install dependencies.

## Step 1: Create a new virtual environment called spatialz with Python 3.9.19
```bash
conda create -n spatialz python=3.9.19 -y
```
## Step 2: Activate the spatialz1 environment
```bash
conda activate spatialz
```
## Step 3: Install PyTorch with CUDA 11.7 support
```bash
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## Step 4: Clone the SpatialZ repository and navigate to the project directory
```bash
git clone [https://github.com/senlin-lin/SpatialZ.git] /path/to/your/SpatialZ_code
cd /path/to/your/SpatialZ_code
```

## Step 5: Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

# 2 Deploying a Docker Image on a New Server

We also provide a Docker image that encapsulates our code and demo data, making it easier for users to directly download and use the provided resources. This image ensures a consistent and reproducible environment, allowing users to seamlessly run the code and explore the demo data without needing to configure dependencies or environments manually.

## Step 1: Install Docker on the New Server

The following commands illustrate the basic steps to install Docker on Ubuntu system (Ubuntu system required):

```bash
sudo apt-get update
sudo apt-get install docker-ce
```

## Step 2: Pull the Image from Docker Hub

To download the Docker image, execute the following command:

```bash
sudo docker pull linsenlin/spatialz:latest
```

## Step 3: Launch the Docker Container on the New Server

Once the image is pulled, users can start the Docker container on the new server. The following command will run the container and map port 8888 of the server to port 8888 of the container:
 
```bash
sudo docker run --gpus all -p 8888:8888 linsenlin/spatialz:latest
```

## Step 4: Access Jupyter Notebook

After launching the Docker container, users can access Jupyter Notebook by navigating to port 8888 on the server. If the server's IP address is 'server_ip', simply enter the following URL in a web browser:

`http://server_ip:8888`


# 3 First MERFISH interpolation

After preparing the dependencies, run one midpoint interpolation between the
bundled sections `0.04` and `0.09`:

```bash
python -u run_merfish_demo.py --device cuda:0 --mender-jobs 2
```

The script uses both complete sections, 1,000 coordinate optimization iterations,
`lr=1e5`, 80 projections, and the default microenvironment-based expression
synthesis with `cell_type_key='cell_class'`, `k_sam=50`, and `Beta=1`.
It seeds NumPy and PyTorch with 42 and keeps the input expression values and
spatial coordinates as provided.

The default output directory is `output/merfish_first_pair/`. It contains
`midpoint.h5ad` (expected shape: 5,522 cells by 155 genes), `run.json` with
parameters and completion status, and `environment.json` with installed package
versions. The run report also counts cells with all-zero generated expression
for inspection before evaluation. Use `--output` with a new directory to rerun;
existing run directories are not overwritten.

The generation functions accept `mender_n_jobs`, defaulting to 2 CPU processes.
This replaces the original fixed count of 200 without changing the
microenvironment scales or radius. In server environments where existing
package versions must be preserved, do not install `requirements.txt` directly;
add missing dependencies and verify the relevant computations first.

# 4 MERFISH held-out slice validation

Generate the middle section using **only `0.04` and `0.14`**, then compare it
with the real held-out section `0.09`:

```bash
cd /3250608056/code/SpatialZ
git pull --ff-only origin master
CUDA_VISIBLE_DEVICES=0 python -u run_merfish_holdout.py --device cuda:0 --mender-jobs 2
```

Use the existing server Python environment. This command does not install
packages or change their versions; Conda is not required.

The script uses `alpha=0.5`, seed 42, 1,000 optimization iterations, `lr=1e5`,
80 optimization projections, `k_sam=50`, `Beta=1`, and two MENDER processes,
matching the first-pair runner settings. The number of generated cells is the
integer average of the two **input** counts: 5,707 for the bundled files. The
real target's 5,557 cells do not determine generation. Target expression,
coordinates, labels, and cell count are read only after generation has been
saved. Input expression and coordinates are used as supplied, with no further
normalization or target-based registration. These are already processed/aligned
files; this procedure does not establish whether upstream preprocessing used
information from other sections.

Results are saved in `output/merfish_holdout_009/`:

| File | Content |
| --- | --- |
| `generated_009.h5ad` | Synthesized section, marked `data_type=synthetic` |
| `run.json` | Completion status, generation parameters, file hashes, shapes, runtime |
| `environment.json` | Installed package versions |
| `metrics.json` | QC, comparisons against real 0.09, metric definitions and limitations |
| `comparison_summary.csv` | Generated and endpoint comparison metrics |
| `gene_comparison.csv` | Gene means, variances, detection fractions and the endpoint-mean baseline |
| `cell_class_comparison.csv`, `Region_comparison.csv` | Annotation counts and proportions |
| `expression_by_cell_class.csv` | Expression comparison per cell class; groups below 20 cells flagged |
| `spatial_comparison.png` | Four sections with identical coordinate limits and cell-class colors |
| `expression_comparison.png` | Gene means, class proportions and total-expression distributions |

There is no one-to-one cell correspondence. Metrics compare unpaired
distributions: gene-mean Pearson/Spearman (higher is better), expression MAE/RMSE,
detection MAE in percentage points, cell-class/region total variation, and
spatial sliced Wasserstein-1 and symmetric mean nearest-neighbor distance
(lower is better). Spatial Wasserstein uses 128 shared unit directions with a
separate seeded RNG and normalizes each slice to unit probability mass. Spatial
distances retain the supplied coordinate units. Nearest-neighbor distances also
depend on sampling density and are descriptive, not cell-matching accuracy.
Constant-vector correlations are saved as JSON `null` rather than misleading
numbers. Total expression uses the supplied values, not assumed raw UMI counts.

The two endpoints are each compared to real 0.09 as reference distributions.
The simple expression baseline averages their per-gene means with equal slice
weights. A high generated gene-mean correlation alone does not establish an
improvement over this baseline. Inspect spatial error, proportions, per-class
expression and all-zero cells together. This is one held-out section with one
seed using the specified runner settings, rather than the complete published
benchmark.

Existing output directories are never overwritten. To repeat with another seed:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_merfish_holdout.py --device cuda:0 --mender-jobs 2 --seed 43 --output output/merfish_holdout_009_seed43
```

Engineering checks (no GPU or generation dependencies needed):

```bash
python -m unittest discover -s tests -v
```

These checks test held-out isolation, metric probability normalization and
artifact creation with a mocked generator; they do not measure method accuracy.

**Reference:** Lin, S., Wang, Z., Cui, Y., Zou, Q., Han, C., Yan, R., … & Yuan, Z. (2024). *Bridging the Dimensional Gap from Planar Spatial Transcriptomics to 3D Cell Atlases*. *bioRxiv*, 2024-12.

 
