
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

# 5 Open-ST 淋巴结留出验证（GSE251926）

`run_openst_holdout.py` 使用本地完整 H5AD 的 `obs['n_section']` 物理编号、
`obs['annotation']` 注释、`obsm['spatial_3d_aligned']` 的已配准 XYZ，以及
现有 `X` 表达值。**不使用未配准的 `obsm['spatial']`，不重新归一化，不读取
`layers['raw']`，不使用留出片进行配准或基因选择。**

| 用途 | 物理切片 | 生成时可见的端点 |
| --- | --- | --- |
| 测试 | #4、#18、#24、#9 | #3/#5、#17/#19、#23/#25、#7/#11 |
| 验证 | #6、#34 | #5/#7、#33/#36 |
| 可见训练观测 | #2、#3、#5、#7、#11、#17、#19、#23、#25、#26、#28、#33、#36 | 仅以上 13 片 |

SpatialZ 是逐对切片求 Wasserstein 重心的基线，仓库没有流匹配网络或 RARS 门控。
脚本导出 12 组相邻**可见观测**对（不代表物理编号连续）和 11 组可见观测的
相邻三元组，包含 `#2/#3/#5`、`#23/#25/#26`，供后续模型使用，
**不声称执行了流匹配训练或 RARS 训练**。

按真实 Z 距离计算 SpatialZ 的左端点权重 `alpha`：#6 为 1/2，#34 为 2/3。
XY 使用 1500 × 1500 μm 核心窗口、150 μm Halo；网格原点和范围仅由 13 片
训练观测确定。每个窗口共用相同 XY 范围贯穿各片，生成时使用扩展窗口，拼接时
按左闭右开的核心边界裁剪。目标细胞数取端点窗口的加权计数，不读取真实目标片
计数。如果窗口只有一个非空端点，使用该端点按加权计数重采样，并在
`generation_method`、`tiles.csv` 和 `fallback_cells` 中明确记录；两端都空则跳过。
因此裁剪后的整片数量并不强制等于端点加权计数。

Track 1 保留完整细注释。Track 2 **在生成前**执行以下映射，且不会把细标签
作为额外字段传入模型：

| 文件中的真实标签 | Track 2 输入/输出标签 |
| --- | --- |
| `Tumor_Keratin_Pearl`、`Tumor` | `Tumor` |
| `Germinal_Center_Plasma_IgM_B_cell`、`Plasma_IgA`、`Plasma_IgG` | `Plasma cell` |
| 其余标签（含 `unknown`） | 原值 |

这里将现有 IgG 主类也统一命名为 `Plasma cell`，以真正形成一个粗粒度 Plasma
类。Track 2 按粗类评分；从粗标签预测不能宣称恢复了 Keratin pearl 或 IgM/IgA
亚型身份。Track 1 的各类型指标可以直接检查这些稀有类型。

先检查数据和导出划分（只需 NumPy、Pandas、SciPy、h5py、scikit-learn）：

```bash
python run_openst_holdout.py --stage prepare
```

默认输入为脚本所在项目目录下的
`data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad`，与服务器目录结构一致，
无需传 `--input`，也不受启动命令所在目录影响。其他数据位置仍可用 `--input` 指定。
正式运行需要项目原有的 PyTorch、POT、AnnData、Scanpy、SOMENDER 等依赖。
以下命令对两个轨道、两个验证片运行三组候选参数，再锁定每轨道最优配置，
最后生成和评估四个测试片：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_openst_holdout.py \
  --device cuda:0 --mender-jobs 2 \
  --candidates configs/openst_candidates.json \
  --output output/openst_holdout
```

也可分为 `--stage validate` 与 `--stage test` 两次运行，保持相同路径和配置。
`test` 必须读取已经保存的 `selection.json`；只有 #6、#34 参与选型。
不传 `--candidates` 时仅评估默认的一组参数，不构成超参数搜索。
SpatialZ 没有跨数据集训练的 epoch，不能实现神经网络式早停；这里把优化迭代
预算作为验证超参数，选择标准预先固定为两张验证片等权平均的：

```text
空间 SW1(μm) / core_um + 细胞类型总变差 + 基因均值 MAE / 真实片全局表达均值
```

输出包括 `protocol.json`、四份划分 CSV、`validation_scores.csv`、冻结的
`selection.json`、`test_summary.csv` 和四个测试片等权的 `test_macro.csv`。
各轨道各切片目录保存 `generated.h5ad`、逐 Tile 检查点、`tiles.csv`、
`metrics.json`、逐基因与逐细胞类型 CSV、任务配置和运行状态。
指标覆盖空间 SW1/双向最近邻距离、类型比例总变差、细胞数误差、表达均值相关性
和误差、方差相关性、检测率误差及全零表达细胞数，并同时报告两端点与表达加权
均值基线。每张预测完整保存后才读取该留出片的表达、XY 和注释。

同一命令可恢复已完成的任务和 Tile。输入文件大小/修改时间、坐标字段、缩放和
网格配置必须与 `protocol.json` 相同；每个已运行任务的参数、种子和设备也必须
匹配，否则使用新的 `--output`。正式测试后不要在同一结果目录更换候选参数。
完整数据保持 CSR 存储；表达合成按块计算余弦相似度、向量化独立逐基因采样，
保留原有候选邻域规则。稀疏/稠密输入的采样回归检查与完整命令的模拟生成检查：

```bash
python -m unittest discover -s tests -v
```

工程检查不代表真实生物学验证完成。提供文件已经全局处理和配准；此流程隔离
本次预测的数据访问，但不重新估计上游处理，因此不能证明原始预处理完全无泄漏。

**Reference:** Lin, S., Wang, Z., Cui, Y., Zou, Q., Han, C., Yan, R., … & Yuan, Z. (2024). *Bridging the Dimensional Gap from Planar Spatial Transcriptomics to 3D Cell Atlases*. *bioRxiv*, 2024-12.

 
