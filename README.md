
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

**Reference:** Lin, S., Wang, Z., Cui, Y., Zou, Q., Han, C., Yan, R., … & Yuan, Z. (2024). *Bridging the Dimensional Gap from Planar Spatial Transcriptomics to 3D Cell Atlases*. *bioRxiv*, 2024-12.

 
