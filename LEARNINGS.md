# Project Learnings

### 2026-02-15
- [closeness centrality]: igraph::closeness() defaults to normalized=FALSE (returns 1/sum(distances)), not normalized=TRUE ((n-1)/sum(distances)). Python implementation was using n_reachable/total_dist which equals (n-1)/total_dist for fully connected graphs — 8x off for the 9-state group_regulation dataset.
- [closeness mode=all]: igraph mode="all" treats directed edges as bidirectional. For each edge pair (u,v), the effective undirected weight is max(w(u,v), w(v,u)). NetworkX's DiGraph.to_undirected() does NOT properly combine bidirectional edge weights — must manually build undirected graph with max weight per edge pair.
- [ctna implementation]: Python ctna counts only adjacent bidirectional co-occurrences (each adjacent pair counted in both directions). R ctna uses window-based co-occurrence counting, producing much larger counts. Raw co-occurrence matrices are intentionally different between R and Python.
- [R TNA centralities]: R TNA 1.2.0 calls igraph functions with weights=1/edge_weight (invert=TRUE default). The centrality_funs dispatch closeness with mode="in"/"out"/"all" and betweenness without normalized parameter.
- [test validation]: Original test_r_equivalence.py only checked structural properties (shapes, ranges, non-negative, finite). Never compared against actual R output values. Tests can pass while being numerically wrong.
- [R ground truth]: Full precision R values obtained from R TNA 1.2.0 / igraph 2.2.1. Weight matrix and centralities match Python within ~1e-15 (machine epsilon) after fixes.
- [R bootstrap algorithm]: R TNA bootstrap resamples **per-sequence 3D transition arrays** (not raw sequences). Core unit: `trans[n_sequences, n_states, n_states]` built via `compute_transitions()`, then `trans[sample(idx, n, replace=TRUE), , ]`. Original Python resampled raw rows — fundamentally different algorithm.
- [R bootstrap stability]: R stability method counts exceedances `(wb <= w*0.75) + (wb >= w*1.25)` per iteration. P-values = `(count+1)/(iter+1)` — can exceed 1.0 for zero-weight edges (both conditions fire). This is intentional R behavior.
- [R permutation test]: R permutation combines 3D transitions from both groups, shuffles sequence indices, splits into two groups, computes weight difference per edge. Effect size = `diff_true / sd(perm_diffs)` with `ddof=1`.
- [R as.vector column-major]: R's `as.vector()` outputs matrices in column-major order (column 1 first, then column 2...). When comparing 81-element vectors from R 9x9 matrices, must use `order='F'` (Fortran/column-major) when reshaping in Python.
- [R p.adjust]: R's `p.adjust()` Holm method sorts ascending, applies `max(1, n-i+1) * p[i]`, enforces monotonicity with cumulative max, then restores original order. BH/FDR sorts ascending, applies `n/rank * p`, enforces monotonicity with cumulative min from the right.

### 2026-02-16
- [matplotlib arrows]: FancyArrowPatch arrowstyle `->,head_length=X,head_width=Y` combined with `mutation_scale` controls arrowhead size. Original `/15` and `/25` ratios with `mutation_scale=15` produce small arrows. Ratios of `/10` and `/15` with `mutation_scale=20` were too large. Final sweet spot: `/15` and `/22` with `mutation_scale=15`.
- [edge label placement]: Positioning edge labels at 65% along the edge towards the destination (`x1 + 0.65 * (x2 - x1)`) makes labels visually associate with their target node, much clearer than the default midpoint (50%).
- [notebook DPI]: Default `figure.dpi = 100` produces low-resolution plots in HTML exports. `150` is a good balance for quality vs file size.
- [jupyter kernel]: The project has a dedicated `tnapy` kernel (at `~/Library/Jupyter/kernels/tnapy/`) using the `.venv` virtualenv. The system Python (`/opt/homebrew/bin/python3`) lacks numpy/pandas. Always use `--ExecutePreprocessor.kernel_name=tnapy` when running `nbconvert --execute`.
- [GroupTNA duck typing]: Used `_is_group_tna(x)` with `hasattr(x, 'models')` to avoid circular imports between `group.py` and other modules (`centralities.py`, `prune.py`, `communities.py`, `cliques.py`, `bootstrap.py`, `plot.py`).
- [GroupTNA dispatch pattern]: Each analysis function checks `_is_group_tna(model)` at the top and dispatches per-group, returning combined results (DataFrame with 'group' column for centralities, GroupTNA for prune, dict for communities/cliques/bootstrap).
- [plot GroupTNA]: Multi-panel plots use `fig_w = figsize[0] * n_groups` to scale figure width by group count, with `plt.subplots(1, n_groups)`. Must handle `n_groups == 1` case where `axes` is not a list.
