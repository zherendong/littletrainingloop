[Automatically generated Readme. Likely garbage.]

# Scaling Law Analysis Scripts

This directory contains scripts for analyzing scaling law experiments from Neptune.

## Overview

The analysis workflow is split into two separate commands:

1. **`prepare_scaling_data.py`**: Downloads and prepares the data
   - Extracts the final evaluation loss for each experiment
   - Filters experiments by Neptune tag
   - Aggregates data across all matching experiments
   - Saves prepared data to CSV

2. **`fit_scaling_law.py`**: Fits scaling law curves and generates plots
   - Fits both power law and shifted power law models
   - Generates publication-quality plots
   - Saves fit parameters

## Usage

### Step 1: Prepare the Data

Download and prepare scaling law data from Neptune experiments with a specific tag:

```bash
python scaling_analysis/prepare_scaling_data.py --tag <your-tag>
```

**Arguments:**
- `--tag`: (Required) Neptune tag to filter experiments by
- `--output`: (Optional) Output CSV file path (default: `scaling_analysis/scaling_data_<tag>.csv`)
- `--run_ids`: (Optional) Specific run IDs to process instead of filtering by tag

**Example:**
```bash
# Download data for all experiments tagged "scaling-experiment-v1"
python scaling_analysis/prepare_scaling_data.py --tag scaling-experiment-v1

# Specify custom output location
python scaling_analysis/prepare_scaling_data.py --tag scaling-experiment-v1 --output my_data.csv

# Process specific run IDs
python scaling_analysis/prepare_scaling_data.py --tag my-tag --run_ids TRAIN-107 TRAIN-108 TRAIN-109
```

**Output:**
- CSV file with columns: `run_id`, `pflops`, `final_loss`
- Console output showing progress and summary statistics

### Step 2: Fit Scaling Laws

Fit shifted power law curves to the prepared data and generate plots. Supports multiple datasets for comparison:

```bash
python scaling_analysis/fit_scaling_law.py --data <data-file> [<data-file2> ...]
```

**Arguments:**
- `--data`: (Required) Path(s) to CSV file(s) with prepared scaling data. Multiple files can be specified for comparison.
- `--output`: (Optional) Base name for output files (default: `<data_file>_fit` for single file, `scaling_analysis/scaling_law_comparison` for multiple files)

**Example:**
```bash
# Fit scaling law to a single dataset
python scaling_analysis/fit_scaling_law.py --data scaling_analysis/scaling_data_scaling-experiment-v1.csv

# Compare multiple datasets
python scaling_analysis/fit_scaling_law.py --data \
    scaling_analysis/scaling_data_experiment-v1.csv \
    scaling_analysis/scaling_data_experiment-v2.csv \
    scaling_analysis/scaling_data_experiment-v3.csv

# Specify custom output base name
python scaling_analysis/fit_scaling_law.py --data my_data.csv --output my_analysis
```

**Output:**
- `<output>.png`: Plot showing data points and fitted curves (all datasets on same plot for comparison)
- `<output>_params.txt`: Human-readable fit parameters for all datasets
- `<output>_params.csv`: Machine-readable fit parameters for all datasets

## Complete Example Workflow

### Single Dataset Analysis

```bash
# 1. Download and prepare data for experiments tagged "chinchilla-scaling"
python scaling_analysis/prepare_scaling_data.py --tag chinchilla-scaling

# 2. Fit scaling laws and generate plots
python scaling_analysis/fit_scaling_law.py --data scaling_analysis/scaling_data_chinchilla-scaling.csv
```

This will create:
- `scaling_analysis/scaling_data_chinchilla-scaling.csv` - Prepared data
- `scaling_analysis/scaling_data_chinchilla-scaling_fit.png` - Plot
- `scaling_analysis/scaling_data_chinchilla-scaling_fit_params.txt` - Fit parameters (text)
- `scaling_analysis/scaling_data_chinchilla-scaling_fit_params.csv` - Fit parameters (CSV)

### Multiple Dataset Comparison

```bash
# 1. Download and prepare data for multiple experiment tags
python scaling_analysis/prepare_scaling_data.py --tag experiment-v1
python scaling_analysis/prepare_scaling_data.py --tag experiment-v2
python scaling_analysis/prepare_scaling_data.py --tag experiment-v3

# 2. Compare all datasets in a single plot
python scaling_analysis/fit_scaling_law.py --data \
    scaling_analysis/scaling_data_experiment-v1.csv \
    scaling_analysis/scaling_data_experiment-v2.csv \
    scaling_analysis/scaling_data_experiment-v3.csv \
    --output scaling_analysis/experiment_comparison
```

This will create:
- `scaling_analysis/experiment_comparison.png` - Comparison plot with all datasets
- `scaling_analysis/experiment_comparison_params.txt` - Fit parameters for all datasets (text)
- `scaling_analysis/experiment_comparison_params.csv` - Fit parameters for all datasets (CSV)

## Data Format

### Prepared Data CSV
The CSV file created by `prepare_scaling_data.py` has the following columns:
- `run_id`: Neptune run identifier
- `pflops`: Total PFLOPs at the end of training
- `final_loss`: Final evaluation loss (last entry in the loss curve)

**Note:** The `fit_scaling_law.py` script supports commented lines in CSV files. Any line starting with `#` will be ignored. This is useful for temporarily excluding data points without deleting them:

```csv
run_id,pflops,final_loss
TRAIN-101,100.5,5.234
# TRAIN-102,150.2,4.987  <- This line will be ignored
TRAIN-103,200.8,4.756
```

### Fit Parameters CSV
The CSV file created by `fit_scaling_law.py` has the following columns:
- `dataset`: Dataset label (derived from filename)
- `file`: Original data file path
- `a`: Coefficient parameter
- `b`: Exponent parameter
- `c`: Shift parameter (irreducible loss floor)
- `c_fixed`: Boolean indicating if `c` was fixed (True for all datasets after the first when comparing multiple datasets)
- `n_points`: Number of data points
- `pflops_min`: Minimum PFLOPs in dataset
- `pflops_max`: Maximum PFLOPs in dataset
- `loss_min`: Minimum loss in dataset
- `loss_max`: Maximum loss in dataset

## Model

The script fits a **Shifted Power Law** model to the data:

**Shifted Power Law**: `loss = a * pflops^b + c`

The shifted power law accounts for an irreducible loss floor (the `c` parameter), which typically provides a better fit than a simple power law for language model scaling.

### Shared `c` Parameter for Multiple Datasets

When comparing multiple datasets (e.g., different models on the same task), the `c` parameter represents the irreducible loss floor of the dataset/task, not a property of the model. Therefore:

- **For single dataset**: All three parameters (`a`, `b`, `c`) are fitted
- **For multiple datasets**:
  - The first dataset is fitted with all three parameters (`a`, `b`, `c`)
  - All subsequent datasets use the **same `c` value** from the first dataset
  - Only `a` and `b` are fitted for subsequent datasets

This ensures that all models being compared converge to the same theoretical minimum loss, making the comparison more meaningful. The shared `c` value is clearly indicated in:
- The plot title
- The parameter output files (marked as "shared" or "fixed")
- The CSV output (`c_fixed` column)

## Requirements

Make sure you have the required dependencies installed:
- `neptune-client`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `python-dotenv`

These should already be in your `requirements.txt`.

## Neptune Configuration

The scripts expect your Neptune API token to be set as an environment variable or in `~/.neptune/.env`:

```bash
export NEPTUNE_API_TOKEN="your-token-here"
```

Or create `~/.neptune/.env`:
```
NEPTUNE_API_TOKEN=your-token-here
```

## Troubleshooting

**"No runs found to process!"**
- Check that the tag exists and is spelled correctly
- Verify that experiments with this tag exist in your Neptune project

**"Data file must contain 'pflops' and 'final_loss' columns"**
- Make sure you're using the CSV file created by `prepare_scaling_data.py`
- Check that the data preparation step completed successfully

**Fit fails or produces poor results**
- Check that you have enough data points (at least 3-4 recommended)
- Verify that the data covers a reasonable range of compute scales
- Look at the raw data to ensure quality

## Other Scripts in This Directory

- `scaling_analysis.py`: Original script for downloading loss vs. flops curves
- `scaling_law.py`: Example script showing manual scaling law fitting

