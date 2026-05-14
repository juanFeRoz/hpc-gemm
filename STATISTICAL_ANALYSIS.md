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

### Updated `test_and_compare.py`
- Now supports `--num-runs` and `--trials` parameters
- Example for paper:
```bash
python3 test_and_compare.py \
  --M 2048 --N 2048 --K 2048 \
  --num-runs 10 \
  --trials 40 \
  --benchmark-runs 10
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

## Recommendations for Your Paper

1. **Minimum Rigor**: 5 independent runs × 30 trials each
2. **Good Rigor**: 10 independent runs × 40 trials each
3. **Excellent Rigor**: 10-20 independent runs × 50 trials each

2. **Report Both Tests**: Include both Welch's t-test and Mann-Whitney U results

3. **Include Convergence Plots**: Use convergence data to show how algorithms progress

4. **Report Effect Size**: Not just p-values, but Cohen's d for magnitude

5. **Problem Sizes**: Test on multiple problem sizes (e.g., 512, 1024, 2048)

## Example Paper Section

> We compared Bayesian optimization and random search for autotuning stencil kernels across 10 independent runs with 40 trials per run. Bayesian optimization found configurations with mean time 10.23 ms (SD=0.57), compared to random search at 11.57 ms (SD=1.23). A Welch's t-test showed this difference was statistically significant (t=-2.88, p=.0125) with a large effect size (Cohen's d=1.23). The Mann-Whitney U test confirmed significance (U=15, p=.0089), supporting the superiority of Bayesian optimization for this tuning task.

## Notes

- Seeds are deterministic based on run index (seed = run_idx × 1000)
- Both Bayesian and Random search use the same seed for fair comparison
- Convergence analysis tracks cumulative best-so-far for each evaluation
- All results are automatically saved for reproducibility and follow-up analysis
