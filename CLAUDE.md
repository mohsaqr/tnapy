# TNA Python Package

## Project Overview
Python implementation of the R TNA (Transition Network Analysis) package for analyzing sequential data as transition networks. Aims for numerical equivalence with R TNA where possible.

## Project Structure
```
tna/                    # Main package
  __init__.py           # Public API exports
  model.py              # TNA class, tna(), ftna(), ctna(), atna(), build_model()
  prepare.py            # TNAData, prepare_data(), create_seqdata()
  centralities.py       # centralities(), betweenness_network(), AVAILABLE_MEASURES
  prune.py              # prune()
  cliques.py            # cliques(), CliqueResult
  communities.py        # communities(), CommunityResult
  group.py              # GroupTNA, group_tna(), group_ftna(), group_ctna(), group_atna()
  compare.py            # compare_sequences()
  bootstrap.py          # bootstrap_tna(), permutation_test(), BootstrapResult, PermutationResult
  plot.py               # plot_network(), plot_centralities(), plot_heatmap(), plot_comparison(),
                        #   plot_sequences(), plot_frequencies(), plot_histogram(), plot_communities()
  colors.py             # color_palette(), DEFAULT_COLORS, create_color_map()
  utils.py              # row_normalize(), minmax_scale(), max_scale(), rank_scale(), apply_scaling()
  data.py               # load_group_regulation(), load_group_regulation_long()
  data/                 # CSV data files
tests/                  # pytest tests
tutorial.ipynb          # Main tutorial notebook (Colab-compatible)
tmp/                    # Rendered HTML outputs (not tracked in git)
```

## Key Architecture Decisions
- **GroupTNA duck typing**: `_is_group_tna()` uses `hasattr(x, 'models')` to avoid circular imports. Every analysis function checks this at the top and dispatches per-group.
- **R equivalence**: Bootstrap/permutation algorithms replicate R's exact approach (resampling 3D transition arrays, not raw sequences). Centralities use inverted weights (`1/w`) matching R igraph conventions.
- **Plot backend**: matplotlib with FancyArrowPatch for directed edges, Arc for self-loops. No qgraph dependency.

## Development Commands
```bash
# Run tests
.venv/bin/python -m pytest tests/ -v

# Execute tutorial notebook
jupyter nbconvert --to notebook --execute --output tmp/tutorial_executed.ipynb tutorial.ipynb \
  --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=tnapy

# Render to HTML
jupyter nbconvert --to html tmp/tutorial_executed.ipynb --output tutorial_executed.html
```

## Important Notes
- Always use the `tnapy` Jupyter kernel (not system Python) — it has the venv with all dependencies.
- System Python at `/opt/homebrew/bin/python3` lacks numpy/pandas (PEP 668 managed).
- The `.venv/` directory contains the project virtualenv.
- R ground truth values are validated to machine epsilon (~1e-15).
- Tutorial notebook must be Colab-compatible (pip install from GitHub, `%matplotlib inline`).
