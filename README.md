[README (1).md](https://github.com/user-attachments/files/28606762/README.1.md)
# HMM Feature-Selection Experiments

Reproduce the HMM regime-detection study and its summary statistics.

## What's here

| Script | Role | Reads | Writes |
|---|---|---|---|
| `experiment60stocksstatsv2.py` | Main pipeline. Downloads price data, fits 2-state Gaussian HMMs, runs Exp 1 / Exp 2 stacking / order experiment + statistical tests. | yfinance (network) | `exp1_results.csv`, `exp2_stacking.csv`, `order_experiment.csv`, `summary.csv`, `bootstrap_ci.csv`, `permutation_tests.csv`, plus PNG plots for `AAPL` |
| `get_stats_2.py` | Per-feature / per-tier μ-difference and σ-ratio summary. | `./results/exp1_results.csv` | stdout (CSV) |
| `stacking_stats_by_tier.py` | Minority-occupancy tables (trend vs vol) by cap tier, with bootstrap CIs. | `exp2_stacking.csv` | `stacking_trend_by_tier.csv`, `stacking_vol_by_tier.csv`|

The two stats scripts depend on the CSVs produced by the main script, so run the main script first.

## Setup

Requires Python 3.9+ and network access (the main script downloads prices via yfinance).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas yfinance hmmlearn matplotlib
```

## Running

### 1. Main pipeline (generates all CSVs)

```bash
python experiment60stocksstatsv2.py
```

This fits HMMs across all 60 tickers in three cap tiers. Expect it to take a while (53–60 downloads plus 5-restart HMM fits over multiple timeframes, plus a 10,000-iteration permutation test at the end). CSVs are written to the script's own directory.

### 2. Feature / tier summary

```bash
python get_stats_2.py
```

### 3. Stacking minority-occupancy by tier

```bash
python stacking_stats_by_tier.py
```

Reads `exp2_stacking.csv` from the current directory. Run it from wherever that CSV was written (or copy the CSV next to the script).

## Notes / reproducibility

- HMM fits use 5 random restarts with fixed per-restart seeds; bootstrap (n=1000) and permutation (n=10000) tests use seed `42`, so results are deterministic given the same downloaded data.
- Plots are only produced for tickers in `PLOT_TICKERS` (default `['AAPL']`). Edit that list to plot others.
- Timeframes differ per experiment (Exp 1: 2010–2024; Exp 2 vol: 2005–2024; Exp 2 bb: 2018–2024) and are set at the top of the main script.
- Tickers with fewer than 200 aligned rows are skipped automatically.
