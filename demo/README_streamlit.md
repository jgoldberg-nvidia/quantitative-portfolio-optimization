# cuFOLIO Demo — Streamlit Applications

Interactive web applications for GPU-accelerated portfolio optimization using Mean-CVaR and Mean-Variance models.

## Applications

### 1. Portfolio Optimizer (`cvar_streamlit_app.py`)
Single-period portfolio optimization with interactive parameter tuning.

- **Mean-CVaR** or **Mean-Variance** optimization (selectable)
- Real-time portfolio allocation visualization
- Constraint settings: weight bounds, leverage, cardinality, cash reserves
- Side-by-side GPU (cuOpt) vs CPU solver comparison
- Optional backtesting with historical/KDE/Gaussian methods

### 2. Efficient Frontier (`efficient_frontier_streamlit_app.py`)
Multi-portfolio efficient frontier generation with progressive GPU vs CPU comparison.

- Side-by-side GPU vs CPU performance racing
- Progressive frontier construction with real-time plot updates
- Special portfolio identification (Min Variance, Max Sharpe, Max Return)
- Optional discretized portfolio overlay

### 3. Rebalancing Strategies (`rebalancing_streamlit_app.py`)
Dynamic portfolio rebalancing simulation with multiple trigger strategies.

- Rebalancing triggers: loss threshold, drift from target, peak-to-trough decline, buy & hold
- Progressive backtesting with real-time cumulative return plots
- GPU vs CPU solver comparison with KDE timing breakdown
- Advanced mode for technical parameters (windows, transaction costs, turnover)
- Masked dataset and solver names for conference presentations

## Quick Start

```bash
# From the repository root:
pip install -e ".[demo]"

# Portfolio optimizer
streamlit run demo/cvar_streamlit_app.py

# Efficient frontier
streamlit run demo/efficient_frontier_streamlit_app.py

# Rebalancing strategies
streamlit run demo/rebalancing_streamlit_app.py
```

For GPU acceleration, install with CUDA extras:

```bash
pip install -e ".[demo,cuda12]"   # CUDA 12
pip install -e ".[demo,cuda13]"   # CUDA 13
```

## Requirements

- Python 3.10+
- Streamlit 1.28+
- CVXPY with supported solver (CLARABEL, HiGHS, etc.)
- Optional: NVIDIA GPU + CUDA for cuOpt acceleration

## Troubleshooting

**Import Errors**: Run `pip install -e .` from the repository root to register the `cufolio` package.

**Dataset Not Found**: Ensure CSV files exist in `data/stock_data/`.

**GPU Solver Unavailable**: The apps fall back to CPU-only mode automatically. Install `cuopt` for GPU support.
