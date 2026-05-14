# Statistical Analysis Pipeline for Kernel Autotuning

## Overview

For publishable research, you need **multiple independent runs** of each algorithm with sufficient trials to compute convergence statistics. The original 15 trials in one run is insufficient for publication.

## What Changed

### New Features

1. **Multiple Independent Runs**: Execute both Bayesian and Random search multiple times with different random seeds
2. **Configurable Trial Counts**: Increase from 15 to 30-50+ trials per run
3. **Statistical Tests**: 
   - Welch's t-test (parametric)
   - Mann-Whitney U test (non-parametric)
   - Cohen's d effect size
   - Confidence intervals
4. **Convergence Analysis**: Track "best-so-far" progress for each run
5. **Reproducible Results**: Seed-based determinism for all runs

## Recommended Setup for Publication

```bash
# Run 10 independent experiments with 40 trials each
python3 run_statistical_experiments.py \
  --M 512 --N 512 --K 512 \
  --num-runs 10 \
  --trials 40 \
  --output convergence_analysis.json
```

This generates:
- `convergence_analysis.json` - Raw convergence data from all runs
- `statistical_results.json` - Significance tests and effect sizes
- Console output with full statistical report

## Files

### `run_statistical_experiments.py`
- Orchestrates multiple independent tuning runs
- Calls `autotune.py` with different seeds for each run
- Collects convergence data and best times
- Triggers statistical analysis

**Usage:**
```bash
python3 run_statistical_experiments.py \
  --M 512 --N 512 --K 512 \
  --num-runs 5 \
  --trials 30
```

**Parameters:**
- `--M, --N, --K`: Problem dimensions
- `--num-runs`: Number of independent experiments (default: 5, recommend: 5-10)
- `--trials`: Trials per Bayesian/Random run (default: 40, recommend: 30-50)
- `--output`: JSON file with convergence data

### `statistical_analysis.py`
- Computes descriptive statistics
- Runs parametric and non-parametric significance tests
- Computes effect sizes
- Prints publication-ready report

**Usage:**
```bash
python3 statistical_analysis.py \
  --bayesian-results bayesian_runs.json \
  --random-results random_runs.json \
  --output statistical_results.json
```

## Statistical Interpretation

The statistical report includes:

1. **Descriptive Statistics**: Mean, median, std dev, range, IQR
2. **Effect Size (Cohen's d)**:
   - |d| < 0.2: Negligible
   - |d| < 0.5: Small
   - |d| < 0.8: Medium
   - |d| ≥ 0.8: Large

3. **Welch's t-test**: For normally distributed data, unequal variances
   - p < 0.01: Highly significant ***
   - p < 0.05: Significant **
   - p ≥ 0.05: Not significant

4. **Mann-Whitney U Test**: Non-parametric alternative (more robust)
   - Use when data is not normally distributed

## Example Output

```
================================================================================
STATISTICAL COMPARISON: BAYESIAN vs RANDOM SEARCH
================================================================================

Sample Sizes:
  Bayesian: 10 runs
  Random:   10 runs

Descriptive Statistics (Best Time Found):

  BAYESIAN:
    Mean:   10.2345 ms
    Median: 10.1234 ms
    Std:     0.5678 ms
    Range:  [9.5000, 11.2345] ms
    IQR:    [9.8901, 10.3456] ms

  RANDOM:
    Mean:   11.5678 ms
    Median: 11.4321 ms
    Std:     1.2345 ms
    Range:  [10.1234, 13.8901] ms
    IQR:    [10.5678, 12.3456] ms

Effect Size (Cohen's d): 1.234
  → Large effect size

Welch's t-test:
  t-statistic: -2.8765
  p-value:     0.0125
  Mean diff:   -1.3333 ms (95% CI: ±0.7834)
  Result:      SIGNIFICANT (p < 0.05) **

Mann-Whitney U test (non-parametric):
  U-statistic: 15.0000
  p-value:     0.0089
  Result:      SIGNIFICANT (p < 0.01) ***

WINNER: Bayesian
================================================================================
```
