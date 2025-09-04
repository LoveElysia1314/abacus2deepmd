# abacus2deepmd

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**语言** | 中文 | [English](README.md)

`abacus2deepmd` 是一款专门用于处理 ABACUS 分子动力学轨迹的高级分析工具，支持智能构象采样与 DeepMD 数据导出。该项目由原 ABACUS-STRU-Analyser 重构而来，集成了自动化体系发现、多维度构象分析、并行计算框架及高效数据导出等功能，特别注重算法的数学严谨性与物理意义的解释。

## 快速开始

### 安装

```bash
# 推荐安装（包含所有依赖）
pip install .

# 开发模式安装
pip install -e .

# 包含开发依赖
pip install -e .[dev]
```

### 基本使用

```bash
# 基本分析（自动发现当前目录下的体系）
abacus2deepmd

# 指定搜索路径和输出目录
abacus2deepmd --search-paths "/path/to/systems1/*" "/path/to/systems2/*" --output-dir results

# 自定义采样参数
abacus2deepmd --sample-ratio 0.1 --power-p -0.5 --pca-variance-ratio 0.95

# 控制并行计算
abacus2deepmd --workers 8 --scheduler process
```

### 验证安装

```bash
# 检查安装是否成功
abacus2deepmd --help

# 运行示例
abacus2deepmd --search-paths ./example_data
```

### 输入文件结构

工具期望 ABACUS MD 输出按照特定的目录结构组织。以下为目录组织格式：

#### 单个系统结构

```
struct_mol_1028_conf_0_T400K
├── INPUT                          # ABACUS MD 配置文件（包含 md_dumpfreq、md_nstep）
├── running_md.log                 # ABACUS MD 输出日志（包含能量数据）
└── OUT.ABACUS
    └── STRU
        ├── STRU_MD_0              # 第 0 帧
        ├── STRU_MD_5              # 第 5 帧（帧间隔依赖于 md_dumpfreq）
        ├── STRU_MD_10             # 第 10 帧
        └── STRU_MD_N              # 第 N 帧（最后一帧）
```

#### 多个系统结构

```
目标目录
├── struct_mol_1028_conf_0_T400K/
│   ├── INPUT
│   ├── running_md.log
│   └── OUT.ABACUS/STRU/STRU_MD_*
├── struct_mol_1028_conf_1_T400K/
│   ├── INPUT
│   ├── running_md.log
│   └── OUT.ABACUS/STRU/STRU_MD_*
└── struct_mol_1029_conf_0_T300K/
    ├── INPUT
    ├── running_md.log
    └── OUT.ABACUS/STRU/STRU_MD_*
```

#### 系统命名规范

系统文件夹命名格式为：**`struct_mol_{mol_id}_conf_{conf}_T{temperature}K`**

示例：
- `struct_mol_1028_conf_0_T400K` - 分子 1028，构象 0，温度 400K
- `struct_mol_1029_conf_1_T300K` - 分子 1029，构象 1，温度 300K

#### 自定义命名规范

如果您的 MD 输出采用不同的文件夹命名约定，可以修改解析规则：
- **文件位置**：`src/abacus2deepmd/io/file_utils.py`
- **相关函数**：`lightweight_discover_systems()` 和 `parse_system_name()`

这些函数从文件夹名中提取 `mol_id`、`conf` 和 `temperature`。调整正则表达式以匹配您的命名方案。

#### 必需文件

| 文件/目录 | 用途 | 必需 |
|---|---|---|
| `INPUT` | ABACUS 配置文件，包含 `md_dumpfreq` 和 `md_nstep` 参数 | 是 |
| `running_md.log` | ABACUS MD 日志，包含每帧的能量数据 | 否* |
| `OUT.ABACUS/STRU/STRU_MD_*` | MD 轨迹的结构文件（帧间隔依赖于 `md_dumpfreq` 参数） | 是 |

*能量数据是可选的；如果不可用，分析仍会继续进行。

##### 帧文件命名与采样说明

- **帧文件格式**：`STRU_MD_{i}`，其中 `{i}` 为帧索引（如 `STRU_MD_0`、`STRU_MD_5`、`STRU_MD_10`）
- **帧间隔**：帧索引由 MD 模拟的 `md_dumpfreq` 参数决定，可能不连续（例如 `md_dumpfreq=10` 时为 0、10、20、30...）
- **采样依据**：进行轨迹降采样时，采样比例针对**可用帧的总数量**进行计算，而非帧号的最大值。例如：
  - 若轨迹包含 100 个 STRU 文件，帧索引为 [0, 10, 20, ..., 990]
  - 设置 `sample_ratio=0.1` 将从这 100 个帧中约选取 10 帧
  - 帧索引的具体数值是任意的，完全依赖于 MD 模拟的参数设置

---

## 核心功能

### 🎯 自动化体系发现与处理
- **智能路径扫描**：递归搜索指定目录，识别标准 ABACUS 输出结构
- **结构去重机制**：基于创建时间和内容哈希自动去除重复体系
- **轨迹帧过滤**：根据 `md_dumpfreq` 参数智能选择 STRU 文件

### 📊 高级轨迹解析
- **多格式支持**：解析 `STRU_MD_*` 文件格式，提取原子坐标、能量与力信息
- **物理量提取**：从 `running_md.log` 中解析能量和力数据
- **参数校验**：自动读取 `INPUT` 文件中的 MD 参数进行帧索引转换和校验

### 🔬 系统级构象分析
- **结构对齐**：迭代 Kabsch 算法计算平均结构及每帧 RMSD
- **波动分析**：计算残基/原子的 RMSF（均方根波动）
- **降维处理**：PCA 主成分分析，保留指定方差比例的特征维度
- **多样性度量**：ANND（平均最近邻距离）、MPD（平均成对距离）、覆盖度等多维度指标

### 🎲 智能采样算法
- **Power-mean 采样**：基于幂平均距离最大化的贪心采样策略
- **多策略支持**：随机采样（固定种子 42）、均匀采样、最远点采样
- **优化机制**：局部交换优化提升采样质量

### ⚡ 高效并行框架
- **多级并行**：支持进程池/线程池/顺序执行三种模式
- **资源优化**：自动控制 BLAS/OMP 线程数，避免超线程竞争
- **断点续算**：智能进度跟踪与恢复机制

### 🔄 DeepMD 数据导出
- **格式转换**：使用 `dpdata` 库将采样帧转换为 DeepMD 兼容的 npy 格式
- **灵活导出**：支持按体系单独导出或批量合并导出
- **元数据管理**：自动处理帧索引与 MD step 的转换关系

### 📈 采样效果评估
- **多方法对比**：智能采样 vs 随机采样 vs 均匀采样
- **量化评估**：JS 散度、覆盖度、多样性指标等多维度评估
- **相对值分析**：支持绝对值和相对值（以智能采样为基准）双重统计

### 🔧 Power参数测试
- **参数优化**：自动测试不同power_p参数的采样效果
- **多体系对比**：并行分析多个体系，计算均值和标准误
- **误差棒图表**：生成带误差棒的综合对比图表，突出参数影响

---

## 详细使用指南

### 命令参数详解

#### 核心分析参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-r`, `--sample_ratio` | float | 0.1 | 采样比例（0-1） |
| `-p`, `--power_p` | float | -0.5 | Power-mean 指数参数 |
| `-v`, `--pca_variance_ratio` | float | 0.90 | PCA 降维累计方差贡献率 |

#### 系统配置参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-w`, `--workers` | int | -1 | 并行工作进程数（-1=自动） |
| `-o`, `--output_dir` | str | `analysis_results` | 输出根目录 |
| `-s`, `--search_path` | str[] | 当前目录父目录 | 递归搜索路径（支持通配符） |
| `-i`, `--include_project` | bool | False | 允许搜索项目自身目录 |
| `-f`, `--force_recompute` | bool | False | 强制重新计算所有体系 |

#### 流程控制参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--steps` | str | [1,2,3] | 执行步骤，支持格式：单个数字(1)、列表([1,2,4])、范围([1,3-4])。1=采样，2=DeepMD导出，3=采样对比，4=Power参数测试 |
| `--max_systems` | int | 64 | Power参数测试时最多使用的体系数量 |

### 高级使用示例

#### 大规模体系处理
```bash
# 使用进程池并行处理大规模数据
abacus2deepmd --search-paths "/large_data/*" --workers 16 --scheduler process --sample-ratio 0.05
```

#### 高精度构象分析
```bash
# 高精度设置（更多主成分，更细采样）
abacus2deepmd --pca-variance-ratio 0.95 --sample-ratio 0.15 --power-p -0.3
```

#### Power参数测试
```bash
# 执行Power参数测试
abacus2deepmd --steps 4 --max-systems 5 --sample-ratio 0.1

# 执行多个步骤（采样 + Power参数测试）
abacus2deepmd --steps "[1,4]" --max-systems 5 --sample-ratio 0.1
```

### 输出文件结构

```
output_dir/
├── analysis_targets.json          # 体系元数据和采样信息
├── sampling_methods_comparison.csv # 采样方法对比汇总
├── single_analysis/               # 单体系分析结果
├── sampling_comparison/           # 采样对比数据
├── power_analysis_plots/          # Power参数测试图表
└── deepmd_npy_per_system/         # DeepMD 格式数据
```

---

## 算法理论基础

### 构象表示：物理一致且刚体不变

使用所有原子对的欧氏距离向量作为基础表示：
- **刚体不变**：对平移/旋转不敏感
- **物理一致**：几何距离直接反映构象差异
- **信息充足**：在一般位置条件下可确定构象至刚体变换

### PCA 降维：信息保留与噪声过滤

通过主成分分析实现：
- **噪声过滤**：过滤热噪声或局部小波动
- **计算降维**：压缩为低维"主要变化模式"
- **平衡点**：默认方差比例 0.90 在信息保留与噪声抑制间取得平衡

### 能量–结构融合

加权拼接结构特征与标准化能量：
```
F = [√w·Ez , √(1-w)·Ṽ], w∈[0,1]
```
- 能量采用 z-score 标准化
- 默认 w=0.5 获得平衡的构象代表性与势能合理性

### Power-mean 采样

最大化幂平均距离：
```
Oₚ(S) = (1/|P(S)| · Σ dᵢⱼᵖ)¹/ᵖ
```

统一多种经典准则：
- `p→-∞`：最大化最小距离（覆盖/均匀）
- `p=0`：几何平均（均衡扩展）
- `p≥1`：强调大间距（发现边界/离群）

默认 `p=-0.5` 在覆盖与多样性间取得良好平衡。

---

## 最佳实践

### 参数选择指南

1. **采样比例** (`sample-ratio`)
   - 小体系（<1000 帧）：0.05-0.1
   - 中体系（1000-10000 帧）：0.02-0.05
   - 大体系（>10000 帧）：0.01-0.02

2. **Power-mean 指数** (`power-p`)
   - 多样性优先：-0.3 到 -0.7
   - 覆盖度优先：-0.7 到 -1.0
   - 边界探测：0.5 到 1.0

3. **PCA 方差比例** (`pca-variance-ratio`)
   - 快速预览：0.88
   - 标准分析：0.91
   - 高精度：0.94

### 性能优化建议

- **计算加速**：设置合理的工作进程数（保持默认即为 CPU 核心数）
- **质量控制**：定期检查日志文件，验证输出完整性

---

## 常见问题解答

### 采样结果不理想
- **调整 Power-mean 参数**：更负的值增加覆盖度，更正的值增加多样性
- **增加采样比例**：提高采样密度
- **提高 PCA 精度**：保留更多特征

### 内存不足错误
- **降低采样比例**
- **减少并行进程**
- **使用顺序执行**

### DeepMD 导出失败
- **检查依赖安装**：`pip install dpdata`
- **验证数据完整性**：确保能量和力数据存在

### 中断计算处理
程序具备完善的断点续算机制：
- **自动恢复**：直接重新运行相同命令即可恢复
- **进度保存**：中断时自动保存分析进度
- **强制重算**：使用 `--force-recompute` 重新计算

---

## 技术细节

### 评估指标

- **ANND**：平均最近邻距离，反映采样点分布的稀疏程度
- **MPD**：平均成对距离，衡量采样点间的整体差异
- **覆盖率**：采样点在特征空间中的方差覆盖程度
- **JS 散度**：衡量采样分布与原始分布的相似性
- **RMSD 均值**：采样点与平均结构的偏差

### 代码结构

```
src/abacus2deepmd/
├── main.py              # 程序入口和工作流控制
├── core/               # 核心算法模块
│   ├── analysis_orchestrator.py  # 分析流程编排器
│   ├── system_analyser.py    # 系统分析器
│   ├── sampler.py           # 采样算法实现
│   └── metrics.py           # 评估指标工具
├── io/                  # 输入输出模块
│   ├── stru_parser.py       # 结构文件解析器
│   └── path_manager.py      # 路径管理器
└── analysis/            # 分析模块
    ├── sampling_comparison_analyser.py  # 采样效果比较
    └── power_parameter_tester.py        # Power参数测试器
```

---

## 支持与贡献

### 许可证
本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

### 联系方式
- **项目主页**: https://github.com/LoveElysia1314/abacus2deepmd
- **问题反馈**: [GitHub Issues](https://github.com/LoveElysia1314/abacus2deepmd/issues)
- **邮箱**: love_elysia1314@outlook.com

### 贡献指南
欢迎提交 Issue 和 Pull Request！

**为计算化学和分子模拟社区贡献力量，推动科学计算工具的开源发展！** 🚀