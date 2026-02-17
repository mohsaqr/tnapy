# TNA - Transition Network Analysis for Python

A Python package providing **exact numerical equivalence** to the [R TNA package](https://cran.r-project.org/package=tna) for analyzing sequential data as transition networks.

## Features

- **8 Model Types**: relative, frequency, co-occurrence, reverse, n-gram, gap, window, attention
- **9 Centrality Measures**: OutStrength, InStrength, ClosenessIn, ClosenessOut, Closeness, Betweenness, BetweennessRSP, Diffusion, Clustering
- **Statistical Inference**: Bootstrap resampling, permutation tests, confidence intervals
- **10+ Visualization Functions**: Network plots, heatmaps, centrality charts, sequence plots
- **R Package Equivalence**: Verified numerical equivalence with comprehensive test suite

## Installation

```bash
# Development installation
pip install -e .

# Or install dependencies directly
pip install numpy pandas networkx scipy matplotlib seaborn
```

## Quick Start

```python
import tna
import pandas as pd

# Load example data (2000 learning sessions with 9 self-regulated learning behaviors)
df = tna.load_group_regulation()

# Build a TNA model (relative transition probabilities)
model = tna.tna(df)
print(model)

# Compute centrality measures
cent = tna.centralities(model)
print(cent)

# Visualize the network
tna.plot_network(model, layout='circular', edge_threshold=0.05)

# Visualize centralities
tna.plot_centralities(cent, measures=['OutStrength', 'InStrength', 'Betweenness'])
```

## Model Building

### Basic Models

```python
# Relative transition probabilities (default)
model = tna.tna(df)

# Frequency model (raw counts)
fmodel = tna.ftna(df)

# Co-occurrence model (bidirectional)
cmodel = tna.ctna(df)

# Attention model (exponential decay weighting)
amodel = tna.atna(df, beta=0.1)
```

### Advanced Model Types

```python
# All model types via build_model()
model = tna.build_model(df, type_='relative')      # Row-normalized probabilities
model = tna.build_model(df, type_='frequency')     # Raw transition counts
model = tna.build_model(df, type_='co-occurrence') # Bidirectional co-occurrence
model = tna.build_model(df, type_='reverse')       # Reverse order transitions
model = tna.build_model(df, type_='n-gram', params={'n': 2})  # Higher-order n-grams
model = tna.build_model(df, type_='gap', params={'max_gap': 3, 'decay': 0.5})  # Gap-weighted
model = tna.build_model(df, type_='window', params={'size': 3})  # Sliding window
model = tna.build_model(df, type_='attention', params={'beta': 0.1})  # Attention-weighted
```

### Scaling Options

```python
# Apply scaling to weight matrix
model = tna.tna(df, scaling='minmax')  # Min-max normalization [0, 1]
model = tna.tna(df, scaling='max')     # Divide by maximum
model = tna.tna(df, scaling='rank')    # Rank-based scaling
model = tna.tna(df, scaling=['minmax', 'max'])  # Multiple scalings
```

## Centrality Measures

```python
# Compute all centrality measures
cent = tna.centralities(model)

# Compute specific measures
cent = tna.centralities(model, measures=['OutStrength', 'InStrength', 'Betweenness'])

# With normalization
cent = tna.centralities(model, normalize=True)

# Include self-loops
cent = tna.centralities(model, loops=True)
```

### Available Measures

| Measure | Description |
|---------|-------------|
| `OutStrength` | Sum of outgoing edge weights |
| `InStrength` | Sum of incoming edge weights |
| `ClosenessIn` | Incoming closeness centrality |
| `ClosenessOut` | Outgoing closeness centrality |
| `Closeness` | Overall closeness (treats graph as undirected) |
| `Betweenness` | Standard betweenness centrality |
| `BetweennessRSP` | Randomized Shortest Path betweenness |
| `Diffusion` | Diffusion centrality (Banerjee et al. 2014) |
| `Clustering` | Weighted clustering coefficient (Zhang & Horvath 2005) |

## Data Preparation

### From Long Format Data

```python
# Prepare raw event data
prepared = tna.prepare_data(
    data=events_df,
    actor='user_id',
    time='timestamp',
    action='event_type',
    time_threshold=900  # 15 minutes session timeout
)

# Build model from prepared data
model = tna.tna(prepared)

# Access statistics
print(prepared.statistics)  # n_sessions, n_actors, etc.
```

### From Wide Format Data

```python
# Direct from wide format (rows=sequences, cols=time steps)
df = pd.DataFrame({
    'step1': ['A', 'B', 'A'],
    'step2': ['B', 'C', 'C'],
    'step3': ['C', 'A', 'B']
})
model = tna.tna(df)
```

## Statistical Inference

### Bootstrap Analysis

```python
# Bootstrap confidence intervals for model parameters
boot = tna.bootstrap_tna(df, n_boot=1000, ci=0.95, seed=42)

# Get summary with CIs for all edges
summary = boot.summary()

# Find significant edges
sig_edges = boot.significant_edges(threshold=0)

# Bootstrap centrality measures
cent_ci = tna.bootstrap_centralities(
    df,
    measures=['OutStrength', 'InStrength', 'Betweenness'],
    n_boot=1000,
    ci=0.95
)
```

### Permutation Tests

```python
# Compare two groups
result = tna.permutation_test(
    group1_df, group2_df,
    n_perm=1000,
    statistic='weights',  # or 'density', 'centrality'
    alternative='two-sided',
    seed=42
)
print(f"P-value: {result.p_value}")
print(f"Significant: {result.is_significant(0.05)}")

# Edge-wise comparison with multiple testing correction
edges = tna.permutation_test_edges(
    group1_df, group2_df,
    n_perm=1000,
    correction='fdr'  # or 'bonferroni', 'none'
)
```

### Confidence Intervals

```python
# Percentile method
ci = tna.confidence_interval(boot_samples, ci=0.95, method='percentile')

# BCa method (bias-corrected and accelerated)
ci = tna.bca_ci(data, boot_samples, statistic_func=np.mean, ci=0.95)
```

## Visualization

### Network Plots

```python
# Basic network plot
tna.plot_network(model)

# Customized network
tna.plot_network(
    model,
    layout='circular',           # or 'spring', 'kamada_kawai'
    node_size='OutStrength',     # Size by centrality
    edge_threshold=0.05,         # Hide weak edges
    node_color='steelblue',
    edge_cmap='Blues'
)

# Network with bootstrap confidence intervals
tna.plot_network_ci(boot, edge_alpha='significance')
```

### Centrality Plots

```python
# Bar charts for centralities
tna.plot_centralities(
    cent,
    measures=['OutStrength', 'InStrength', 'Betweenness'],
    ncol=3
)
```

### Heatmap

```python
# Transition matrix heatmap
tna.plot_heatmap(model, cmap='Blues', annotate=True)
```

### Model Comparison

```python
# Side-by-side comparison of two models
tna.plot_comparison(
    model1, model2,
    plot_type='heatmap',
    labels=('Group 1', 'Group 2')
)
```

### Sequence Visualization

```python
# State distribution over time
tna.plot_sequences(df, plot_type='distribution')

# State frequencies
tna.plot_frequencies(df)

# Histogram of sequence lengths
tna.plot_histogram(df)
```

### Statistical Plots

```python
# Bootstrap distribution
tna.plot_bootstrap(boot, plot_type='weights')
tna.plot_bootstrap(boot, plot_type='centrality', measure='OutStrength')

# Permutation test null distribution
tna.plot_permutation(result)
```

## Example Datasets

```python
# Wide format: 2000 sessions x 20 time steps
df = tna.load_group_regulation()

# Long format: Actor, Time, Action columns
df_long = tna.load_group_regulation_long()
```

## API Reference

### Model Building

| Function | Description |
|----------|-------------|
| `tna(x)` | Build relative transition probability model |
| `ftna(x)` | Build frequency (raw counts) model |
| `ctna(x)` | Build co-occurrence model |
| `atna(x, beta)` | Build attention-weighted model |
| `build_model(x, type_)` | Build model with specified type |

### Data Preparation

| Function | Description |
|----------|-------------|
| `prepare_data(data, actor, time, action)` | Prepare long-format event data |
| `create_seqdata(x)` | Create sequence data from various formats |

### Centralities

| Function | Description |
|----------|-------------|
| `centralities(model, measures)` | Compute centrality measures |

### Statistical Inference

| Function | Description |
|----------|-------------|
| `bootstrap_tna(x, n_boot)` | Bootstrap analysis of TNA model |
| `bootstrap_centralities(x, measures, n_boot)` | Bootstrap centrality CIs |
| `permutation_test(x1, x2, n_perm)` | Permutation test for group comparison |
| `permutation_test_edges(x1, x2, n_perm)` | Edge-wise permutation tests |
| `confidence_interval(samples, ci)` | Calculate confidence interval |
| `bca_ci(data, samples, func, ci)` | BCa confidence interval |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_network(model)` | Plot transition network |
| `plot_centralities(cent)` | Plot centrality bar charts |
| `plot_heatmap(model)` | Plot transition matrix heatmap |
| `plot_comparison(m1, m2)` | Compare two models |
| `plot_sequences(df)` | Plot sequence patterns |
| `plot_frequencies(df)` | Plot state frequencies |
| `plot_histogram(df)` | Plot sequence length histogram |
| `plot_bootstrap(boot)` | Visualize bootstrap results |
| `plot_permutation(result)` | Visualize permutation test |
| `plot_network_ci(boot)` | Network with confidence intervals |

### Utilities

| Function | Description |
|----------|-------------|
| `row_normalize(matrix)` | Row-normalize a matrix |
| `minmax_scale(matrix)` | Min-max scaling to [0, 1] |
| `max_scale(matrix)` | Divide by maximum |
| `rank_scale(matrix)` | Rank-based scaling |

## R Package Equivalence

This package is designed to produce numerically equivalent results to the R TNA package. Key equivalences:

- **Transition matrices**: Identical computation of relative, frequency, and co-occurrence matrices
- **Centrality measures**: Exact ports of R implementations including custom measures (diffusion, weighted clustering)
- **Data format**: Compatible with R's wide-format sequence data

### Verification

```python
# Python
model_py = tna.tna(df)
cent_py = tna.centralities(model_py)

# Results match R within floating-point precision:
# - Max absolute difference < 1e-10 for transition matrices
# - Max absolute difference < 1e-6 for centrality measures
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{tna_python,
  title = {TNA: Transition Network Analysis for Python},
  author  = "Saqr, Mohammed and Tikka, Santtu and López-Pernas, Sonsoles",
  year = {2026},
  url = {https://github.com/mohsaqr/tnapy}
}
```

Also cite Transition Network Analysis as a method

```bibtex
@INPROCEEDINGS{Saqr2025-ku,
  title     = "Transition Network Analysis: A Novel Framework for Modeling,
               Visualizing, and Identifying the Temporal Patterns of Learners
               and Learning Processes",
  author    = "Saqr, Mohammed and López-Pernas, Sonsoles and Törmänen, Tiina and
               Kaliisa, Rogers and Misiejuk, Kamila and Tikka, Santtu",
  booktitle = "Proceedings of Learning Analytics \& Knowledge (LAK '25)",
  publisher = "ACM",
  address   = "New York, NY, USA",
  doi       = "10.1145/3706468.3706513",
  pages     = "351 - 361",
  year      =  2025
}

```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
