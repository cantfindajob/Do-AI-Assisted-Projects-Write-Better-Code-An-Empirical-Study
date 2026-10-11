# AI 辅助开发的项目能写出更好的代码吗？

本文是论文《AI 辅助开发的项目能写出更好的代码吗？》的**补充材料**。

本研究基于 SonarQube 静态分析，对 **AI 辅助开发**与**人工编写**的开源项目进行可维护性问题密度和严重性对比，涵盖 **4 种编程语言**（Java、JavaScript、TypeScript、Python）、**4 种项目规模**（小型、中小型、中型、大型）、**16 个可维护性类别**（C1–C16）以及 **5 个严重性级别**（Blocker、Critical、Major、Minor、Info）。

## 仓库结构

```text
Do-AI-Assisted-Projects-Write-Better-Code/
├── dataset/
│   ├── filtered_merged_repository.xlsx          # 经 PDP 方法过滤后的最终仓库清单
│   └── pdp过滤/
│       ├── 爬取特征脚本.py                        # 爬取仓库的 106 列特征用于训练
│       ├── train.py                              # PDP 模型训练代码
│       └── pre.py                                # PDP 模型预测代码
│
├── RQ1 experimental results/
│   ├── [规模]/[规模]-[语言] result/             # 共 16 个文件夹（4 规模 × 4 语言）
│   │   ├── AI_*Maintainability issue density.xlsx
│   │   ├── Human_*Maintainability issue density.xlsx
│   │   └── 2022_Human_*Maintainability issue density.xlsx   # 2022 年人工基线
│   ├── 4633个Maintainability issues corresponding to each project_filtered.xlsx
│   ├── Academic_Mann_Whitney1.png                # AI 与人工对比的 Cliff's Delta 值
│   ├── maintainability_heatmap.png               # 可维护性密度热力图
│   ├── project_all_rules_.xlsx                   # 4 种语言用到的 SonarQube 规则
│   ├── sonarqube_Default_maintainability_rules.xlsx  # 详细规则名称
│   └── *.xlsx                                    # 中间合并数据表
│
├── RQ2 experimental results/
│   ├── [规模]/[规模]-[语言] result/             # 共 16 个文件夹（4 规模 × 4 语言）
│   │   ├── AI_Issue Density by Severity.xlsx     # 每个项目的 5 个严重性级别
│   │   └── Human_Issue Density by Severity.xlsx
│   ├── remediation_time.xlsx                     # 每个仓库的预估修复时间
│   ├── rq2总表.xlsx                               # 4 规模 × 4 语言的平均修复时间
│   └── *.png                                     # 严重性 / 修复时间图表
│
├── RQ3 experimental results/
│   ├── rq3实验代码/                               # RQ3 实验代码
│   │   ├── RQ3.1用pls预测c指标.py
│   │   ├── RQ3.2去除修复时间因变量.py
│   │   └── RQ3.2用pls预测严重性密度和修复时间.py
│   ├── rq3.1_filtered_by_key.xlsx                # RQ3.1 实验数据
│   ├── rq3.2_filtered_by_key.xlsx                # RQ3.2 实验数据
│   ├── Maintainability issue density pls-results pls_results_10fold.xlsx
│   └── Severity issue density pls_results_10fold2.xlsx
│
└── README.md
```

## 数据集说明

位于 `dataset/` 目录下。

| 文件 | 说明 |
|---|---|
| `filtered_merged_repository.xlsx` | 经 PDP（公共开发项目，Public Development Projects）方法过滤后得到的最终仓库清单。 |
| `pdp过滤/爬取特征脚本.py` | 爬取脚本，用于采集模型训练所需的 **106 列仓库特征**。 |
| `pdp过滤/train.py` | PDP 分类模型的训练代码。 |
| `pdp过滤/pre.py` | 预测代码，应用训练好的 PDP 模型识别公共开发项目。 |

## RQ1 —— 可维护性问题密度

位于 `RQ1 experimental results/` 目录下。RQ1 对比 AI 辅助开发与人工编写项目在 16 个可维护性类别（C1–C16）上的问题密度。

| 文件 | 说明 |
|---|---|
| `[规模]/[语言] result/` | **共 16 个文件夹（4 规模 × 4 语言）**；每个文件夹包含 AI、人工以及 2022 年人工基线三组的各项目可维护性问题密度表。 |
| `4633个Maintainability issues corresponding to each project_filtered.xlsx` | 全部 **4633 个可维护性问题**及其所属仓库，并标注每个问题对应的可维护性类别（C1–C16）。 |
| `Academic_Mann_Whitney1.png` | AI 与人工在各可维护性类别上对比的 **Cliff's Delta** 效应量。 |
| `maintainability_heatmap.png` | AI 与人工在各项目规模、各语言下可维护性问题密度的热力图。 |
| `project_all_rules_.xlsx` | **Java、JavaScript、TypeScript、Python** 四种语言用到的 SonarQube 可维护性规则，并映射到 C1–C16 类别。 |
| `sonarqube_Default_maintainability_rules.xlsx` | SonarQube 默认可维护性规则的详细名称与说明。 |
| `mann_whitney_16_files_merged.xlsx` | 16 个类别文件的 Mann–Whitney U 检验合并结果。 |
| `各组平均可维护性问题类别密度.xlsx` | 各对比组的平均可维护性类别密度。 |
| `merged_repository_with_ncloc.xlsx` | 仓库元数据与 NLOC（代码行数）的合并表。 |
| `merged_repository_with_sonar_keys_filtered.xlsx` | 过滤后仓库与其 SonarQube 项目 Key 的合并表。 |

## RQ2 —— 问题严重性与修复时间

位于 `RQ2 experimental results/` 目录下。RQ2 对比两组项目的可维护性问题严重性密度及预估修复时间。

| 文件 | 说明 |
|---|---|
| `[规模]/[语言] result/` | **共 16 个文件夹（4 规模 × 4 语言）**；每个文件夹包含按 **5 个严重性级别**划分的 AI 与人工问题密度表。 |
| `remediation_time.xlsx` | 每个仓库的预估修复（补救）时间。 |
| `rq2总表.xlsx` | **4 规模 × 4 语言平均修复时间**的汇总表。 |
| `matrix_plot_bold.png` | 各严重性级别问题密度的矩阵图。 |
| `repair_time_comparison_academic.png` | AI 与人工两组修复时间的对比图。 |

## RQ3 —— PLS 预测分析

位于 `RQ3 experimental results/` 目录下。RQ3 采用偏最小二乘（PLS）回归结合 10 折交叉验证，预测可维护性密度、严重性密度与修复时间。

| 文件 | 说明 |
|---|---|
| `rq3实验代码/RQ3.1用pls预测c指标.py` | RQ3.1 代码：用 PLS 回归预测可维护性类别（C1–C16）指标。 |
| `rq3实验代码/RQ3.2去除修复时间因变量.py` | RQ3.2 代码：去除修复时间因变量后的 PLS 分析。 |
| `rq3实验代码/RQ3.2用pls预测严重性密度和修复时间.py` | RQ3.2 代码：用 PLS 回归预测严重性密度与修复时间。 |
| `rq3.1_filtered_by_key.xlsx` | 按项目 Key 整理的 RQ3.1 实验数据。 |
| `rq3.2_filtered_by_key.xlsx` | 按项目 Key 整理的 RQ3.2 实验数据。 |
| `Maintainability issue density pls-results pls_results_10fold.xlsx` | RQ3.1 可维护性问题密度的 PLS **10 折交叉验证**结果。 |
| `Severity issue density pls_results_10fold2.xlsx` | RQ3.2 严重性问题密度的 PLS **10 折交叉验证**结果。 |

