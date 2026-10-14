# 第一次改进：整细胞联合供体采样

对应改进方案的模块 1A，以及 E0 / E1 配对实验。所有新增代码均在本文件夹内；原有 `SpatialZ.py`、运行脚本、数据和已有结果不作修改。依赖沿用当前服务器环境，不需要安装或更改包版本。

## 本次只改哪一处

原方法对每个生成细胞的每个基因独立抽一个供体；改进方法只抽一次供体，复制该供体的完整表达向量。

| 条件 | E0：gene | E1：cell |
| --- | --- | --- |
| 坐标、细胞数、细胞类型、Region 等注释 | 复用原生成切片 | 完全相同 |
| MENDER 特征 | 共同计算一次 | 共用同一结果 |
| 候选规则 | 合并两张输入，全局找近邻后筛同类型；保留原来的 k 上限 | 共用同一候选缓存 |
| 权重 | `exp(Beta * cosine_similarity)`，`Beta=1` | 共用同一权重缓存 |
| 抽样单元 | 每个基因独立抽供体 | 全部基因共享一个供体 |

本次没有引入类内 KD-tree、距离权重、细胞配额、残差组合、均值收缩或额外归一化。没有候选时，两组均保留原方法的零向量行为，数量写入 `run.json` 和 `candidate_counts.csv`。

固定候选和权重时，两组每个基因的条件边际分布与期望均值相同；E1 保留该局部供体分布的基因联合关系。它是否更接近真实 0.09 需要看结果，不能保证总表达均值提高或方差增加。由于细胞类型固定，漏检类型和各类型数量也不会在这一步改善。

## 输入与对照

需要保留第一次实验的两个文件：

```text
output/merfish_holdout_009/generated_009.h5ad
output/merfish_holdout_009/run.json
```

脚本读取其中的坐标和注释，并校验原实验参数与两张输入的 SHA256。原生成表达被清零，随后仅使用输入 0.04、0.14 和冻结的合成坐标/类型重算 MENDER。两组都保存完后，才读取真实 0.09 进行评估。

E0 是本轮重新运行的原采样规则，不要求逐值复现历史文件的表达：历史随机数状态和 MENDER 执行状态可能不同。因此，本次应比较同一输出目录内的 `gene` 与 `cell`。两组分别使用相同初始表达种子，实际抽样消耗的随机数数量不同；使用多个种子时，是在同一冻结切片上重复表达采样，不是重新独立生成整张切片。

0.09 已在前期分析中使用，本轮属于开发验证。最终研究效果还需在独立留出位置验证。

## 同步到 GitHub

本地 PowerShell，在仓库根目录执行，只提交这个新增文件夹：

```powershell
cd D:\Code\SpatialZ
git add improvements/step01_joint_sampling
git commit -m "Add step 01 paired whole-cell sampling experiment"
git push origin master
```

## 服务器运行

先在服务器的仓库根目录拉取新代码：

```bash
cd /3250608056/code/SpatialZ
git pull --ff-only origin master
python -u improvements/step01_joint_sampling/run_step01.py --mender-jobs 2
```

无需 `--device` 或 GPU 参数：本轮复用原有坐标，不再运行 GPU 坐标优化。MENDER 和表达采样在 CPU 上运行。命令默认使用表达种子 42，结果保存到：

```text
output/step01_joint_sampling_seed42/
```

如果原结果存放在其他位置，可指定 `--geometry-dir /实际路径/merfish_holdout_009`。默认以脚本路径定位仓库，避免当前位于 `output/` 时找不到输入；上面的运行命令仍应在仓库根目录执行。

脚本拒绝覆盖已有输出。如需重复同一种子，改用新目录：

```bash
python -u improvements/step01_joint_sampling/run_step01.py --mender-jobs 2 --output output/step01_joint_sampling_seed42_repeat
```

先检查种子 42 的结果，再决定是否重复其他种子。例如：

```bash
python -u improvements/step01_joint_sampling/run_step01.py --mender-jobs 2 --seed 43
```

## 看哪些结果

| 文件 | 用途 |
| --- | --- |
| `run.json` | 是否完成、输入/冻结切片哈希、空候选数量和参数 |
| `step01_comparison.csv` / `step01_metrics.json` | 两组的新增表达指标及逐基因均值指标 |
| `step01_by_cell_class.csv` | 分类型的方差、协方差和相关结构；避免仅看整体混合分布 |
| `candidate_cache.npz` / `candidate_counts.csv` | 两组共享的候选、概率及每细胞候选数量 |
| `gene/generated_009.h5ad` | 重新抽样的原规则结果 E0 |
| `cell/generated_009.h5ad` | 第一次改进结果 E1 |
| `cell/donors.csv` | 每个合成细胞的供体索引、输入切片及供体 ID；无供体为 -1 |
| 两组各自的 `metrics.json`、CSV、PNG | 复用原留出实验评估函数生成的详细结果 |

主要指标按输入文件提供的表达单位计算，不做额外变换，均值、方差和协方差均使用总体矩（`ddof=0`）：

- `total_mean_relative_bias = 合成每细胞总表达均值 / 真实均值 - 1`：越接近 0 越好；乘 100 为百分比。
- `total_variance_ratio = 合成每细胞总表达方差 / 真实方差`：越接近 1 越好。这个指标是方差比，不是标准差比。
- `offdiag_covariance_relative_error = ||Off(C_syn - C_real)||_F / ||Off(C_real)||_F`：仅比较非对角协方差，越小越好；真实分母为 0 时留空。
- `offdiag_correlation_mae`：基因对相关系数的绝对误差均值，越小越好。只统计两组表达矩阵内均非恒定的基因对，结合 `correlation_pairs`、`real_variable_gene_pairs` 和 `collapsed_real_variable_genes` 看覆盖情况；不能只凭较小 MAE 判断改善。
- `gene_mean_mae`、`gene_mean_rmse`、`gene_mean_pearson`：检查联合结构变化是否伴随边际均值误差变化。

分类型指标以该类型的真实分布为参照。任一侧不足 20 个细胞时不计算协方差/相关误差，`covariance_eligible=False`；20 是描述性报告阈值，不意味着估计已经稳定。缺失类型保留计数，其无法定义的表达指标留空。

空间和类型比例指标在 E0 / E1 间应完全一致。首轮重点判断表达联合结构是否改善，并检查总表达偏差和逐基因均值误差；不预设改进结果。

## 工程检查

本地或服务器均可运行，不需导入 Scanpy / MENDER：

```bash
python -m unittest discover -s improvements/step01_joint_sampling -p "test_*.py" -v
```

测试验证原函数采样规则一致性、整细胞供体一致性、空候选行为、协方差指标和两组保存后才读取真实切片的顺序。端到端检查使用模拟 AnnData 和 MENDER；它不是服务器上的真实 MERFISH 实验。

运行成功后，将整个 `output/step01_joint_sampling_seed42/` 下载到本地即可比较两组结果。
