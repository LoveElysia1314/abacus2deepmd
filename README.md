# abacus2deepmd

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Language** | [中文](README_CN.md) | English

`abacus2deepmd` is an advanced analysis tool specifically designed for processing ABACUS molecular dynamics trajectories, supporting intelligent conformation sampling and DeepMD data export. This project is refactored from the original ABACUS-STRU-Analyser, integrating automated system discovery, multi-dimensional conformation analysis, parallel computing framework, and efficient data export functions, with particular emphasis on the mathematical rigor of algorithms and the physical interpretation of results.

## Quick Start

### Installation

```bash
# Recommended installation (includes all dependencies)
pip install .

# Development mode installation
pip install -e .

# Include development dependencies
pip install -e .[dev]
```

### Basic Usage

```bash
# Basic analysis (automatically discover systems in current directory)
abacus2deepmd

# Specify search paths and output directory
abacus2deepmd --search-paths "/path/to/systems1/*" "/path/to/systems2/*" --output-dir results

# Custom sampling parameters
abacus2deepmd --sample-ratio 0.1 --power-p -0.5 --pca-variance-ratio 0.95

# Control parallel computation
abacus2deepmd --workers 8 --scheduler process
```

### Verify Installation

```bash
# Check if installation is successful
abacus2deepmd --help

# Run example
abacus2deepmd --search-paths ./example_data
```

### Input File Structure

The tool expects ABACUS MD output organized in a specific directory structure. Below are the directory organization formats:

#### Single System Structure

```
struct_mol_1028_conf_0_T400K
├── INPUT                          # ABACUS MD configuration file (contains md_dumpfreq, md_nstep)
├── running_md.log                 # ABACUS MD output log (contains energy data)
└── OUT.ABACUS
    └── STRU
        ├── STRU_MD_0              # Frame 0
        ├── STRU_MD_5              # Frame 5 (frame intervals depend on md_dumpfreq)
        ├── STRU_MD_10             # Frame 10
        └── STRU_MD_N              # Frame N (last frame)
```

#### Multiple Systems Structure

```
Target_Directory
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

#### System Naming Convention

The system folder naming format is: **`struct_mol_{mol_id}_conf_{conf}_T{temperature}K`**

Examples:
- `struct_mol_1028_conf_0_T400K` - Molecule 1028, Configuration 0, Temperature 400K
- `struct_mol_1029_conf_1_T300K` - Molecule 1029, Configuration 1, Temperature 300K

#### Customizing the Naming Convention

If your MD output uses a different folder naming convention, you can modify the parsing rules in:
- **File**: `src/abacus2deepmd/io/file_utils.py`
- **Function**: `lightweight_discover_systems()` and `parse_system_name()`

These functions extract `mol_id`, `conf`, and `temperature` from folder names. Adjust the regular expressions to match your naming scheme.

#### Required Files

| File/Directory | Purpose | Required |
|---|---|---|
| `INPUT` | ABACUS configuration file, contains `md_dumpfreq` and `md_nstep` parameters | Yes |
| `running_md.log` | ABACUS MD log, contains energy data for each frame | No* |
| `OUT.ABACUS/STRU/STRU_MD_*` | Structure files for MD trajectory frames (frame intervals depend on `md_dumpfreq`) | Yes |

*Energy data is optional; analysis proceeds without it if unavailable.

##### Frame File Naming and Sampling

- **Frame File Format**: `STRU_MD_{i}` where `{i}` is the frame index (e.g., `STRU_MD_0`, `STRU_MD_5`, `STRU_MD_10`)
- **Frame Intervals**: Frame indices are determined by the MD simulation's `md_dumpfreq` parameter; they may not be consecutive (e.g., 0, 10, 20, 30... if `md_dumpfreq=10`)
- **Sampling Basis**: When downsampling trajectories, the sampling ratio applies to the **total number of available frames**, not the maximum frame index. For example:
  - If a trajectory contains 100 STRU files with indices [0, 10, 20, ..., 990]
  - Setting `sample_ratio=0.1` will select approximately 10 frames from these 100 frames
  - The frame indices are arbitrary and depend entirely on the MD simulation parameters

---

## Core Features

### 🎯 Automated System Discovery and Processing
- **Intelligent Path Scanning**: Recursively search specified directories, identify standard ABACUS output structures
- **Structure Deduplication Mechanism**: Automatically remove duplicate systems based on creation time and content hash
- **Trajectory Frame Filtering**: Intelligently select STRU files based on `md_dumpfreq` parameter

### 📊 Advanced Trajectory Parsing
- **Multi-format Support**: Parse `STRU_MD_*` file formats, extract atomic coordinates, energy and force information
- **Physical Quantity Extraction**: Parse energy and force data from `running_md.log`
- **Parameter Validation**: Automatically read MD parameters from `INPUT` file for frame index conversion and validation

### 🔬 System-level Conformation Analysis
- **Structure Alignment**: Iterative Kabsch algorithm to calculate average structure and per-frame RMSD
- **Fluctuation Analysis**: Calculate RMSF (root mean square fluctuation) for residues/atoms
- **Dimensionality Reduction**: PCA principal component analysis, retain feature dimensions with specified variance ratio
- **Diversity Metrics**: Multi-dimensional metrics including ANND (average nearest neighbor distance), MPD (mean pairwise distance), coverage, etc.

### 🎲 Intelligent Sampling Algorithm
- **Power-mean Sampling**: Greedy sampling strategy based on maximizing power-mean distance
- **Multi-strategy Support**: Random sampling (fixed seed 42), uniform sampling, farthest point sampling
- **Optimization Mechanism**: Local exchange optimization to improve sampling quality

### ⚡ Efficient Parallel Framework
- **Multi-level Parallelism**: Support three modes: process pool/thread pool/sequential execution
- **Resource Optimization**: Automatically control BLAS/OMP thread count, avoid hyper-threading competition
- **Checkpoint Resume**: Intelligent progress tracking and recovery mechanism

### 🔄 DeepMD Data Export
- **Format Conversion**: Use `dpdata` library to convert sampled frames to DeepMD-compatible npy format
- **Flexible Export**: Support per-system separate export or batch merged export
- **Metadata Management**: Automatically handle frame index and MD step conversion relationships

### 📈 Sampling Effect Evaluation
- **Multi-method Comparison**: Intelligent sampling vs random sampling vs uniform sampling
- **Quantitative Evaluation**: Multi-dimensional evaluation including JS divergence, coverage, diversity metrics, etc.
- **Relative Value Analysis**: Support both absolute and relative values (with intelligent sampling as baseline) dual statistics

### 🔧 Power Parameter Testing
- **Parameter Optimization**: Automatically test sampling effects of different power_p parameters
- **Multi-system Comparison**: Parallel analysis of multiple systems, calculate mean and standard error
- **Error Bar Charts**: Generate comprehensive comparison charts with error bars, highlighting parameter effects

---

## Detailed Usage Guide

### Command Parameter Details

#### Core Analysis Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-r`, `--sample_ratio` | float | 0.1 | Sampling ratio (0-1) |
| `-p`, `--power_p` | float | -0.5 | Power-mean exponent parameter |
| `-v`, `--pca_variance_ratio` | float | 0.90 | PCA dimensionality reduction cumulative variance contribution rate |

#### System Configuration Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-w`, `--workers` | int | -1 | Number of parallel worker processes (-1=auto) |
| `-o`, `--output_dir` | str | `analysis_results` | Output root directory |
| `-s`, `--search_path` | str[] | Parent directory of current | Recursive search paths (support wildcards) |
| `-i`, `--include_project` | bool | False | Allow searching project own directory |
| `-f`, `--force_recompute` | bool | False | Force recompute all systems |

#### Process Control Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--steps` | str | [1,2,3] | Execution steps, support formats: single number(1), list([1,2,4]), range([1,3-4]). 1=Sampling, 2=DeepMD export, 3=Sampling comparison, 4=Power parameter testing |
| `--max_systems` | int | 64 | Maximum number of systems used in Power parameter testing |

### Advanced Usage Examples

#### Large-scale System Processing
```bash
# Use process pool for parallel processing of large-scale data
abacus2deepmd --search-paths "/large_data/*" --workers 16 --scheduler process --sample-ratio 0.05
```

#### High-precision Conformation Analysis
```bash
# High-precision settings (more principal components, finer sampling)
abacus2deepmd --pca-variance-ratio 0.95 --sample-ratio 0.15 --power-p -0.3
```

#### Power Parameter Testing
```bash
# Execute Power parameter testing
abacus2deepmd --steps 4 --max-systems 5 --sample-ratio 0.1

# Execute multiple steps (sampling + Power parameter testing)
abacus2deepmd --steps "[1,4]" --max-systems 5 --sample-ratio 0.1
```

### Output File Structure

```
output_dir/
├── analysis_targets.json          # System metadata and sampling information
├── sampling_methods_comparison.csv # Sampling method comparison summary
├── single_analysis/               # Single system analysis results
├── sampling_comparison/           # Sampling comparison data
├── power_analysis_plots/          # Power parameter testing charts
└── deepmd_npy_per_system/         # DeepMD format data
```

---

## Algorithm Theoretical Foundation

### Conformation Representation: Physically Consistent and Rigid Body Invariant

Use Euclidean distance vectors of all atom pairs as the base representation:
- **Rigid Body Invariant**: Insensitive to translation/rotation
- **Physically Consistent**: Geometric distances directly reflect conformation differences
- **Information Sufficient**: Can determine conformation up to rigid body transformation under general position conditions

### PCA Dimensionality Reduction: Information Preservation and Noise Filtering

Achieved through principal component analysis:
- **Noise Filtering**: Filter thermal noise or local small fluctuations
- **Computational Reduction**: Compress to low-dimensional "main variation modes"
- **Balance Point**: Default variance ratio 0.90 achieves balance between information preservation and noise suppression

### Energy-Structure Fusion

Weighted concatenation of structural features and standardized energy:
```
F = [√w·Ez , √(1-w)·Ṽ], w∈[0,1]
```
- Energy uses z-score standardization
- Default w=0.5 achieves balanced conformation representativeness and potential energy rationality

### Power-mean Sampling

Maximize power-mean distance:
```
Oₚ(S) = (1/|P(S)| · Σ dᵢⱼᵖ)¹/ᵖ
```

Unify multiple classical criteria:
- `p→-∞`: Maximize minimum distance (coverage/uniform)
- `p=0`: Geometric mean (balanced expansion)
- `p≥1`: Emphasize large distances (boundary/outlier discovery)

Default `p=-0.5` achieves good balance between coverage and diversity.

---

## Best Practices

### Parameter Selection Guide

1. **Sampling Ratio** (`sample-ratio`)
   - Small systems (<1000 frames): 0.05-0.1
   - Medium systems (1000-10000 frames): 0.02-0.05
   - Large systems (>10000 frames): 0.01-0.02

2. **Power-mean Exponent** (`power-p`)
   - Diversity priority: -0.3 to -0.7
   - Coverage priority: -0.7 to -1.0
   - Boundary detection: 0.5 to 1.0

3. **PCA Variance Ratio** (`pca-variance-ratio`)
   - Quick preview: 0.88
   - Standard analysis: 0.91
   - High precision: 0.94

### Performance Optimization Suggestions

- **Computation Acceleration**: Set reasonable number of worker processes (keep default = CPU cores)
- **Quality Control**: Regularly check log files, verify output integrity

---

## Frequently Asked Questions

### Sampling Results Not Ideal
- **Adjust Power-mean Parameters**: More negative values increase coverage, more positive values increase diversity
- **Increase Sampling Ratio**: Improve sampling density
- **Improve PCA Precision**: Retain more features

### Out of Memory Error
- **Reduce Sampling Ratio**
- **Reduce Parallel Processes**
- **Use Sequential Execution**

### DeepMD Export Failure
- **Check Dependency Installation**: `pip install dpdata`
- **Verify Data Integrity**: Ensure energy and force data exist

### Interrupted Computation Handling
The program has a complete checkpoint resume mechanism:
- **Automatic Recovery**: Directly rerun the same command to recover
- **Progress Saving**: Automatically save analysis progress when interrupted
- **Force Recalculation**: Use `--force-recompute` to recalculate

---

## Technical Details

### Evaluation Metrics

- **ANND**: Average nearest neighbor distance, reflects the sparsity of sampling point distribution
- **MPD**: Mean pairwise distance, measures the overall difference between sampling points
- **Coverage**: Variance coverage of sampling points in feature space
- **JS Divergence**: Measures the similarity between sampling distribution and original distribution
- **RMSD Mean**: Deviation of sampling points from average structure

### Code Structure

```
src/abacus2deepmd/
├── main.py              # Program entry and workflow control
├── core/               # Core algorithm modules
│   ├── analysis_orchestrator.py  # Analysis process orchestrator
│   ├── system_analyser.py    # System analyzer
│   ├── sampler.py           # Sampling algorithm implementation
│   └── metrics.py           # Evaluation metrics tools
├── io/                  # Input/output modules
│   ├── stru_parser.py       # Structure file parser
│   └── path_manager.py      # Path manager
└── analysis/            # Analysis modules
    ├── sampling_comparison_analyser.py  # Sampling effect comparison
    └── power_parameter_tester.py        # Power parameter tester
```

---

## Support and Contribution

### License
This project uses the MIT License - see [LICENSE](LICENSE) file for details

### Contact Information
- **Project Homepage**: https://github.com/LoveElysia1314/abacus2deepmd
- **Issue Reporting**: [GitHub Issues](https://github.com/LoveElysia1314/abacus2deepmd/issues)
- **Email**: love_elysia1314@outlook.com

### Contribution Guide
Welcome to submit Issues and Pull Requests!

**Contribute to the computational chemistry and molecular simulation community, promote the open-source development of scientific computing tools!** 🚀