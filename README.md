
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

# 5 Open-ST 淋巴结留出验证：精细注释 + Seurat HVG2000

`run_openst_holdout.py` 现在只使用原 Track 1 的精细注释，不再划分两个 Track。
保留 `Tumor_Keratin_Pearl`、`Plasma_IgA`、`Plasma_IgG` 和
`Germinal_Center_Plasma_IgM_B_cell` 等原始标签，不合并为粗类。

输入为项目目录下的 `data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad`。
使用 `obs['n_section']` 物理编号、`obs['annotation']` 注释、
`obsm['spatial_3d_aligned']` 的已配准 XYZ 和现有 `X` 表达值。
`X` 已经过 log1p 处理，脚本要求保留 `uns['log1p']` 元数据，不重复归一化或取对数，
不读取 `layers['raw']`，不使用未配准的 `obsm['spatial']`。

| 用途 | 物理切片 | 重建端点 |
| --- | --- | --- |
| 测试 | #4、#18、#24、#9 | #3/#5、#17/#19、#23/#25、#7/#11 |
| 验证（可选选参） | #6、#34 | #5/#7、#33/#36 |
| 可见训练观测及 HVG 拟合 | #2、#3、#5、#7、#11、#17、#19、#23、#25、#26、#28、#33、#36 | 仅这 13 片 |

第一次运行会合并上述 13 张训练片的稀疏 `X`，按 Scanpy 的 Seurat 方法选基因：

```python
sc.pp.highly_variable_genes(
    training, flavor="seurat", n_top_genes=2000,
    n_bins=20, batch_key=None, subset=False, inplace=True,
)
```

这是对全部可见训练细胞的合并筛选，不是每片分别选基因，也不是
`flavor='seurat_v3'`。验证片、测试片的表达、XY 和标签不参与 HVG 筛选。
Scanpy 可能因阈值并列返回超过 2000 个基因；脚本按标准化离散度降序、原始基因
索引升序打破并列，确保**恰好 2000 个 HVG**。选定后保持原输入列顺序，所有端点、
预测和真实留出片统一使用这套顺序。输出 `generated.h5ad` 的 `X` 为
`n_generated_cells × 2000`，不再生成或保存全转录组预测。

HVG 记录保存在输出根目录：

- `hvg_selection.json`：输入指纹、13 张拟合切片、Scanpy 版本、HVG 索引/名称及顺序哈希。
- `hvg_genes.txt`：按模型实际输入/输出顺序排列的 2000 个基因，可供其他对照方法复用。
- `hvg_statistics.csv`：全部候选基因的均值、离散度、标准化离散度、选择标记和 HVG 排名。

后续运行复用冻结清单，不重新读取训练表达来拟合 HVG。HVG 拟合本身仍需读取
训练片的全基因稀疏表达；生成阶段只把选中的 2000 列传入 SpatialZ。
评分也仅针对相同的 2000 个基因，不能将其称为全转录组表达指标。

SpatialZ 无需训练。默认固定 `iterations=1000`、`lr=1e5`、80 个投影、
`k_sam=50`、`Beta=1`、`syn_mode=default`，对每张目标片使用两张完整端点，
调用一次 `Generate_spatialz`。默认只重建四张测试片，不重建 #6、#34。
目标细胞数取端点的物理距离加权计数，不使用真实目标片计数；预测保存后才读取
该留出片的表达、XY 和注释进行评分。按实际 Z 计算左端点权重 `alpha`，
#6 为 1/2，#34 为 2/3。论文中涉及的流匹配/RARS 模型不在本仓库内；
脚本仍导出 12 组观测对、11 组伪留出三元组清单，但不执行这些模型的训练。

服务器依赖已安装后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_openst_holdout.py \
  --stage test --spatial-mode whole --device cuda:0 \
  --output output/openst_hvg2000
```

仅先准备 HVG 和划分、或只运行一张测试片：

```bash
python run_openst_holdout.py --stage prepare --output output/openst_hvg2000
CUDA_VISIBLE_DEVICES=0 python -u run_openst_holdout.py \
  --device cuda:0 --targets 4 --output output/openst_hvg2000
```

旧的 `--tracks` 参数已删除。请使用新的输出目录；旧双 Track/全转录组目录不能续跑
为 HVG2000 实验。相同新目录支持逐片恢复，输入文件、参数和 HVG 清单必须一致。
完成单片后再次运行完整命令，会复用该片结果。每次 `test_summary.csv` 与
`test_macro.csv` 汇总当次请求的目标，`test_scope.json` 记录汇总范围。

单次约 5 万细胞的稠密表达中间矩阵现在约 0.4 GB；坐标优化、微环境和 I/O
仍有额外开销。输入与最终输出保持 CSR。内存受限时，可使用原分块方案
`--spatial-mode tiled`：1500 × 1500 μm Core、150 μm Halo，以训练片确定网格，
生成后裁掉 Halo，单端点非空窗口单独标记重采样来源。
整片和分块使用不同结果目录。

可选选参仍只使用 #6、#34，且 HVG 继续固定为 13 张训练片拟合的清单：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_openst_holdout.py \
  --stage all --device cuda:0 --candidates configs/openst_candidates.json \
  --output output/openst_hvg2000_tuned
```

也可先 `--stage validate`，再 `--stage selected-test`；默认 `--stage test`
不要求也不读取 `selection.json`。可选选型分数为两张验证片的等权平均：
`spatial_SW1_um/core_um + cell_type_TV + gene_mean_MAE/real_global_mean`。

生成文件位于 `test/section_4/generated.h5ad` 等目录，附有 `metrics.json`、
逐 HVG/细胞类型比较 CSV、配置及运行状态；根目录输出 `test_summary.csv` 和
按测试片等权平均的 `test_macro.csv`。现有指标涵盖空间 SW1、双向最近邻距离、
细胞类型总变差、计数误差、HVG 表达均值相关性/误差、方差相关性、检测率误差
和全零表达细胞数，并报告端点与端点加权表达均值基线。

缺少 MENDER 时，使用已有服务器环境的补包脚本：

```bash
bash scripts/install_openst_deps.sh
```

它通过已安装版本快照约束补包，再使用 `--no-deps` 安装 SOMENDER/Squidpy，
并运行真实小样本生成检查，避免自动替换已有 NVIDIA PyTorch/数值库。
详见 `requirements-openst-extra.txt`，记录保存在 `output/openst_dependency_setup/`。

工程检查（不代表完整生物学验证完成）：

```bash
python -m unittest discover -s tests -v
```

提供文件已经全局处理和配准；本次流程不使用隐藏表达拟合 HVG，但不重新估计
上游预处理，因此不能据此证明原始配准与归一化完全无泄漏。

**Reference:** Lin, S., Wang, Z., Cui, Y., Zou, Q., Han, C., Yan, R., … & Yuan, Z. (2024). *Bridging the Dimensional Gap from Planar Spatial Transcriptomics to 3D Cell Atlases*. *bioRxiv*, 2024-12.

 
