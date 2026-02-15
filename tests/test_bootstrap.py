"""Tests for statistical inference (bootstrap and permutation tests).

Tests cover bootstrap resampling matching R TNA's algorithm,
permutation tests, confidence intervals, and plotting.
"""

import numpy as np
import pandas as pd
import pytest

import tna


# =============================================================================
# R Ground Truth Values (R TNA 1.2.0, seed=42, iter=100)
# =============================================================================

# R weights_orig in column-major order (9x9 = 81 values)
R_WEIGHTS_ORIG_VEC = [
    0.000000000000000, 0.002949852507375, 0.004740085321536,
    0.016243654822335, 0.071374335611238, 0.002467395135707,
    0.011165387299372, 0.000974500568459, 0.234662576687117,
    0.273084479371316, 0.027138643067847, 0.014852267340812,
    0.036040609137056, 0.047582890407492, 0.325343672893902,
    0.055826936496860, 0.025174598018516, 0.033742331288344,
    0.477406679764244, 0.497935103244838, 0.082003476062569,
    0.134517766497462, 0.321184510250569, 0.320408882622489,
    0.159106769016050, 0.290401169400682, 0.466257668711656,
    0.021611001964637, 0.119174041297935, 0.187707378732817,
    0.023350253807107, 0.084282460136674, 0.034191046880508,
    0.057920446615492, 0.017216176709436, 0.044478527607362,
    0.058939096267191, 0.059587020648968, 0.188023384420920,
    0.273604060913706, 0.194887370286004, 0.101868170602749,
    0.375436147941382, 0.067890206269287, 0.062883435582822,
    0.119842829076621, 0.115634218289086, 0.072681308263549,
    0.172081218274112, 0.105796001012402, 0.076841734226295,
    0.090718771807397, 0.146824752314439, 0.070552147239264,
    0.033398821218075, 0.033038348082596, 0.046610838995102,
    0.086294416243655, 0.022272842318400, 0.036305956996828,
    0.018143754361479, 0.075523794055547, 0.012269938650307,
    0.015717092337917, 0.141002949852507, 0.395797124348238,
    0.239086294416244, 0.011642622120982, 0.099753260486429,
    0.215631542219121, 0.374208218288127, 0.075153374233129,
    0.000000000000000, 0.003539823008850, 0.007584136514457,
    0.018781725888325, 0.140976967856239, 0.002819880155093,
    0.016050244242847, 0.001786584375508, 0.000000000000000,
]


class TestBootstrapAlgorithm:
    """Tests verifying the bootstrap algorithm matches R TNA."""

    @pytest.fixture
    def model(self):
        """Build TNA model from group_regulation data."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    def test_bootstrap_weights_match_model(self, model):
        """Test that bootstrap weights_orig matches the original model weights."""
        boot = tna.bootstrap_tna(model, iter=10, seed=42)
        np.testing.assert_array_almost_equal(
            boot.weights_orig, model.weights, decimal=15
        )

    def test_bootstrap_weights_match_r(self, model):
        """Test that weights_orig matches R ground truth."""
        boot = tna.bootstrap_tna(model, iter=10, seed=42)
        r_weights = np.array(R_WEIGHTS_ORIG_VEC).reshape((9, 9), order='F')
        np.testing.assert_array_almost_equal(
            boot.weights_orig, r_weights, decimal=10
        )

    def test_bootstrap_pvalue_formula(self, model):
        """Test p-value formula: (count + 1) / (iter + 1)."""
        boot = tna.bootstrap_tna(model, iter=100, seed=42)
        # Min possible p-value is 1/(iter+1) = 1/101
        min_pval = 1.0 / 101
        # Check that all p-values are multiples of 1/(iter+1)
        # Actually, since stability counts 2 conditions, p = (count+1)/(iter+1)
        # where count can be 0 to 2*iter
        denominators = boot.p_values * 101
        # Each should be close to an integer
        assert np.all(np.abs(denominators - np.round(denominators)) < 1e-10)

    def test_bootstrap_stability_method(self, model):
        """Test stability method: counts outside 0.75-1.25 * weights."""
        boot = tna.bootstrap_tna(
            model, iter=50, seed=42, method='stability',
            consistency_range=(0.75, 1.25)
        )
        # Consistency range bounds
        np.testing.assert_array_almost_equal(
            boot.cr_lower, model.weights * 0.75, decimal=15
        )
        np.testing.assert_array_almost_equal(
            boot.cr_upper, model.weights * 1.25, decimal=15
        )

    def test_bootstrap_threshold_method(self, model):
        """Test threshold method."""
        boot = tna.bootstrap_tna(model, iter=50, seed=42, method='threshold')
        # P-values should be valid (count + 1) / (iter + 1)
        assert np.all(boot.p_values >= 1 / 51)
        assert np.all(boot.p_values <= 1.0)

    def test_bootstrap_significance_filtering(self, model):
        """Test that weights_sig = (p < level) * weights."""
        boot = tna.bootstrap_tna(model, iter=100, seed=42, level=0.05)
        expected_sig = (boot.p_values < 0.05) * boot.weights_orig
        np.testing.assert_array_almost_equal(
            boot.weights_sig, expected_sig, decimal=15
        )

    def test_bootstrap_ci_ordering(self, model):
        """Test ci_lower <= ci_upper."""
        boot = tna.bootstrap_tna(model, iter=100, seed=42)
        assert np.all(boot.ci_lower <= boot.ci_upper)

    def test_bootstrap_mean_within_ci(self, model):
        """Test that bootstrap mean is generally within CIs."""
        boot = tna.bootstrap_tna(model, iter=200, seed=42)
        # Mean should be between CI bounds for most edges
        within = (boot.weights_mean >= boot.ci_lower) & (boot.weights_mean <= boot.ci_upper)
        # At least 80% should be within
        assert np.mean(within) > 0.8

    def test_bootstrap_sd_positive(self, model):
        """Test bootstrap SD is non-negative."""
        boot = tna.bootstrap_tna(model, iter=100, seed=42)
        assert np.all(boot.weights_sd >= 0)

    def test_bootstrap_zero_diagonal_pvalues(self, model):
        """Test that zero-weight edges get max p-value (both conditions fire)."""
        boot = tna.bootstrap_tna(model, iter=100, seed=42)
        # For zero-weight diagonal entries (self-loops removed), both conditions fire
        # p = (2*iter + 1) / (iter + 1) ≈ 1.99
        diag_mask = (model.weights == 0)
        # At least the diagonal entries with 0 weight should have high p-values
        for i in range(9):
            if model.weights[i, i] == 0:
                expected = (2 * 100 + 1) / (100 + 1)
                assert abs(boot.p_values[i, i] - expected) < 1e-10

    def test_bootstrap_seed_reproducibility(self, model):
        """Test that same seed produces same results."""
        boot1 = tna.bootstrap_tna(model, iter=50, seed=123)
        boot2 = tna.bootstrap_tna(model, iter=50, seed=123)
        np.testing.assert_array_equal(boot1.p_values, boot2.p_values)
        np.testing.assert_array_equal(boot1.weights_mean, boot2.weights_mean)
        np.testing.assert_array_equal(boot1.ci_lower, boot2.ci_lower)
        np.testing.assert_array_equal(boot1.ci_upper, boot2.ci_upper)

    def test_bootstrap_different_seeds(self, model):
        """Test that different seeds produce different results."""
        boot1 = tna.bootstrap_tna(model, iter=50, seed=123)
        boot2 = tna.bootstrap_tna(model, iter=50, seed=456)
        assert not np.allclose(boot1.p_values, boot2.p_values)


class TestBootstrapResult:
    """Tests for BootstrapResult class."""

    @pytest.fixture
    def boot_result(self):
        """Bootstrap result for testing."""
        df = tna.load_group_regulation()
        model = tna.tna(df)
        return tna.bootstrap_tna(model, iter=50, seed=42)

    def test_result_attributes(self, boot_result):
        """Test BootstrapResult has correct attributes."""
        assert isinstance(boot_result.weights_orig, np.ndarray)
        assert isinstance(boot_result.weights_sig, np.ndarray)
        assert isinstance(boot_result.weights_mean, np.ndarray)
        assert isinstance(boot_result.weights_sd, np.ndarray)
        assert isinstance(boot_result.p_values, np.ndarray)
        assert isinstance(boot_result.cr_lower, np.ndarray)
        assert isinstance(boot_result.cr_upper, np.ndarray)
        assert isinstance(boot_result.ci_lower, np.ndarray)
        assert isinstance(boot_result.ci_upper, np.ndarray)
        assert isinstance(boot_result.boot_summary, pd.DataFrame)
        assert isinstance(boot_result.model, tna.TNA)
        assert isinstance(boot_result.labels, list)

    def test_result_shapes(self, boot_result):
        """Test all matrix outputs have correct shape."""
        n = len(boot_result.labels)
        for attr in ['weights_orig', 'weights_sig', 'weights_mean', 'weights_sd',
                      'p_values', 'cr_lower', 'cr_upper', 'ci_lower', 'ci_upper']:
            mat = getattr(boot_result, attr)
            assert mat.shape == (n, n), f"{attr} shape is {mat.shape}, expected ({n}, {n})"

    def test_summary_dataframe(self, boot_result):
        """Test summary() returns correct DataFrame."""
        summary = boot_result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'from' in summary.columns
        assert 'to' in summary.columns
        assert 'weight' in summary.columns
        assert 'p_value' in summary.columns
        assert 'sig' in summary.columns
        # Only non-zero weight edges
        assert all(summary['weight'] > 0)

    def test_significant_edges(self, boot_result):
        """Test significant_edges() method."""
        sig_edges = boot_result.significant_edges()
        assert isinstance(sig_edges, list)
        for edge in sig_edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3  # (from, to, weight)

    def test_backward_compat_properties(self, boot_result):
        """Test backward compatibility properties."""
        assert isinstance(boot_result.estimate, tna.TNA)
        assert isinstance(boot_result.n_boot, int)
        assert boot_result.n_boot == boot_result.iter
        assert isinstance(boot_result.ci_level, float)
        assert boot_result.ci_level == 1 - boot_result.level
        assert isinstance(boot_result.weights_ci, tuple)
        assert len(boot_result.weights_ci) == 2
        assert isinstance(boot_result.inits_ci, tuple)


class TestBootstrapTNAInput:
    """Tests for bootstrap_tna input handling."""

    def test_model_input(self):
        """Test bootstrap with TNA model input (R-style)."""
        df = tna.load_group_regulation()
        model = tna.tna(df)
        boot = tna.bootstrap_tna(model, iter=10, seed=42)
        assert isinstance(boot, tna.BootstrapResult)
        assert boot.weights_orig.shape == model.weights.shape

    def test_dataframe_input(self):
        """Test bootstrap with DataFrame input (backward compat)."""
        df = tna.load_group_regulation()
        boot = tna.bootstrap_tna(df, iter=10, seed=42)
        assert isinstance(boot, tna.BootstrapResult)

    def test_tnadata_input(self):
        """Test bootstrap with TNAData input."""
        long_df = tna.load_group_regulation_long()
        prepared = tna.prepare_data(
            long_df, actor='Actor', time='Time', action='Action'
        )
        boot = tna.bootstrap_tna(prepared, iter=10, seed=42)
        assert isinstance(boot, tna.BootstrapResult)


class TestBootstrapCentralities:
    """Tests for bootstrap_centralities function."""

    @pytest.fixture
    def model(self):
        """Build TNA model."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    def test_centralities_basic(self, model):
        """Test basic centrality bootstrap."""
        result = tna.bootstrap_centralities(
            model,
            measures=['OutStrength', 'InStrength'],
            iter=20,
            seed=42
        )
        assert isinstance(result, pd.DataFrame)
        assert 'measure' in result.columns
        assert 'state' in result.columns
        assert 'estimate' in result.columns
        assert 'ci_lower' in result.columns
        assert 'ci_upper' in result.columns
        assert 'se' in result.columns

    def test_centralities_all_measures(self, model):
        """Test with all measures."""
        result = tna.bootstrap_centralities(
            model, measures=None, iter=10, seed=42
        )
        measures_in_result = result['measure'].unique()
        assert len(measures_in_result) == len(tna.AVAILABLE_MEASURES)

    def test_centralities_ci_ordering(self, model):
        """Test CI lower <= upper."""
        result = tna.bootstrap_centralities(
            model, measures=['OutStrength'], iter=20, seed=42
        )
        assert all(result['ci_lower'] <= result['ci_upper'])

    def test_centralities_se_positive(self, model):
        """Test standard errors are non-negative."""
        result = tna.bootstrap_centralities(
            model, measures=['OutStrength'], iter=20, seed=42
        )
        assert all(result['se'] >= 0)


class TestPermutationAlgorithm:
    """Tests verifying the permutation test algorithm matches R TNA."""

    @pytest.fixture
    def two_models(self):
        """Create two models by splitting group_regulation data."""
        df = tna.load_group_regulation()
        n = len(df)
        half = n // 2
        model1 = tna.tna(df.iloc[:half])
        model2 = tna.tna(df.iloc[half:])
        return model1, model2

    def test_permutation_edge_stats_structure(self, two_models):
        """Test edge stats has correct structure."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=50, seed=42)

        stats = result.edges['stats']
        assert 'edge_name' in stats.columns
        assert 'diff_true' in stats.columns
        assert 'effect_size' in stats.columns
        assert 'p_value' in stats.columns
        assert len(stats) == 81  # 9x9

    def test_permutation_pvalue_formula(self, two_models):
        """Test p-value formula: (count + 1) / (iter + 1)."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=100, seed=42)

        p_vals = result.edges['stats']['p_value'].values
        # All p-values should be multiples of 1/(iter+1)
        denominators = p_vals * 101
        assert np.all(np.abs(denominators - np.round(denominators)) < 1e-10)

    def test_permutation_effect_size(self, two_models):
        """Test effect sizes: diff_true / sd(perm_diffs)."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=50, seed=42)

        stats = result.edges['stats']
        # Effect sizes should be finite for non-zero SD
        finite_mask = np.isfinite(stats['effect_size'].values)
        # Most should be finite
        assert np.mean(finite_mask) > 0.9

    def test_permutation_diffs_true_matches_stats(self, two_models):
        """Test that diffs_true matrix matches stats DataFrame."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=50, seed=42)

        # The diffs_true matrix should equal weights_x - weights_y
        expected_diffs = model1.weights - model2.weights
        np.testing.assert_array_almost_equal(
            result.edges['diffs_true'], expected_diffs, decimal=15
        )

    def test_permutation_diffs_sig(self, two_models):
        """Test that diffs_sig = diffs_true * (p < level)."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=50, seed=42, level=0.05)

        # Rebuild p-value matrix from stats
        a = len(model1.labels)
        p_matrix = np.zeros((a, a))
        for _, row in result.edges['stats'].iterrows():
            parts = row['edge_name'].split(' -> ')
            i = model1.labels.index(parts[0])
            j = model1.labels.index(parts[1])
            p_matrix[i, j] = row['p_value']

        expected_sig = result.edges['diffs_true'] * (p_matrix < 0.05)
        np.testing.assert_array_almost_equal(
            result.edges['diffs_sig'], expected_sig, decimal=15
        )

    def test_permutation_identical_groups(self):
        """Test with identical groups (most p-values should be high)."""
        df = tna.load_group_regulation()
        model = tna.tna(df)
        result = tna.permutation_test(model, model, iter=50, seed=42)

        # Differences should be zero
        np.testing.assert_array_almost_equal(
            result.edges['diffs_true'], np.zeros((9, 9)), decimal=15
        )

    def test_permutation_seed_reproducibility(self, two_models):
        """Test seed produces reproducible results."""
        model1, model2 = two_models
        result1 = tna.permutation_test(model1, model2, iter=50, seed=999)
        result2 = tna.permutation_test(model1, model2, iter=50, seed=999)

        pd.testing.assert_frame_equal(
            result1.edges['stats'], result2.edges['stats']
        )

    def test_permutation_with_centralities(self, two_models):
        """Test permutation with centrality measures."""
        model1, model2 = two_models
        result = tna.permutation_test(
            model1, model2, iter=20, seed=42,
            measures=['OutStrength', 'InStrength']
        )

        assert result.centralities is not None
        assert 'stats' in result.centralities
        assert 'diffs_true' in result.centralities
        assert 'diffs_sig' in result.centralities

        cent_stats = result.centralities['stats']
        assert 'state' in cent_stats.columns
        assert 'centrality' in cent_stats.columns
        assert 'diff_true' in cent_stats.columns
        assert 'effect_size' in cent_stats.columns
        assert 'p_value' in cent_stats.columns

    def test_permutation_adjust_none(self, two_models):
        """Test permutation with no p-value adjustment."""
        model1, model2 = two_models
        result = tna.permutation_test(model1, model2, iter=50, seed=42, adjust='none')
        p_vals = result.edges['stats']['p_value'].values
        assert np.all(p_vals >= 0)
        assert np.all(p_vals <= 1)


class TestPermutationResult:
    """Tests for PermutationResult class."""

    def test_backward_compat_properties(self):
        """Test backward compatibility properties."""
        df = tna.load_group_regulation()
        model1 = tna.tna(df.head(500))
        model2 = tna.tna(df.tail(500))
        result = tna.permutation_test(model1, model2, iter=20, seed=42)

        assert isinstance(result.observed, float)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1
        assert isinstance(result.is_significant(), bool)
        assert result.alternative == 'two-sided'


class TestPermutationTestEdges:
    """Tests for permutation_test_edges convenience wrapper."""

    @pytest.fixture
    def two_groups(self):
        """Create two groups for testing."""
        np.random.seed(42)
        group1 = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C'] * 30,
            'step2': ['B', 'C', 'B', 'A'] * 30,
        })
        group2 = pd.DataFrame({
            'step1': ['C', 'A', 'C', 'B'] * 30,
            'step2': ['A', 'B', 'A', 'C'] * 30,
        })
        return group1, group2

    def test_edges_basic(self, two_groups):
        """Test basic edge-wise permutation test."""
        group1, group2 = two_groups
        result = tna.permutation_test_edges(
            group1, group2, n_perm=20, seed=42
        )

        assert isinstance(result, pd.DataFrame)
        assert 'from' in result.columns
        assert 'to' in result.columns
        assert 'diff' in result.columns
        assert 'p_value' in result.columns
        assert 'p_adjusted' in result.columns
        assert 'significant' in result.columns

    def test_edges_p_value_range(self, two_groups):
        """Test p-values are in valid range."""
        group1, group2 = two_groups
        result = tna.permutation_test_edges(
            group1, group2, n_perm=20, seed=42
        )
        assert all(result['p_value'] >= 0)
        assert all(result['p_value'] <= 1)


class TestConfidenceInterval:
    """Tests for confidence interval functions."""

    def test_percentile_ci(self):
        """Test percentile confidence interval."""
        np.random.seed(42)
        values = np.random.normal(10, 2, 1000)
        lower, upper = tna.confidence_interval(values, ci=0.95, method='percentile')
        assert lower < 10 < upper
        assert lower < upper

    def test_basic_ci(self):
        """Test basic bootstrap confidence interval."""
        np.random.seed(42)
        values = np.random.normal(5, 1, 1000)
        lower, upper = tna.confidence_interval(values, ci=0.95, method='basic')
        assert lower < upper

    def test_ci_level_affects_width(self):
        """Test that higher CI level produces wider interval."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 500)
        ci_90 = tna.confidence_interval(values, ci=0.90, method='percentile')
        ci_95 = tna.confidence_interval(values, ci=0.95, method='percentile')
        ci_99 = tna.confidence_interval(values, ci=0.99, method='percentile')
        assert (ci_90[1] - ci_90[0]) < (ci_95[1] - ci_95[0]) < (ci_99[1] - ci_99[0])

    def test_invalid_ci_method(self):
        """Test error on invalid method."""
        values = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown CI method"):
            tna.confidence_interval(values, method='invalid')


class TestBCaCI:
    """Tests for BCa confidence interval."""

    def test_bca_basic(self):
        """Test basic BCa CI computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 50)
        bootstrap_values = np.array([
            np.mean(np.random.choice(data, len(data), replace=True))
            for _ in range(200)
        ])
        lower, upper = tna.bca_ci(
            data, bootstrap_values, statistic_func=np.mean, ci=0.95
        )
        assert lower < upper
        assert lower < np.mean(data) < upper


class TestPAdjust:
    """Tests for p-value adjustment functions."""

    def test_none_adjustment(self):
        """Test no adjustment returns original p-values."""
        from tna.bootstrap import _p_adjust
        p = np.array([0.01, 0.05, 0.1, 0.5])
        adjusted = _p_adjust(p, method='none')
        np.testing.assert_array_equal(adjusted, p)

    def test_bonferroni(self):
        """Test Bonferroni adjustment."""
        from tna.bootstrap import _p_adjust
        p = np.array([0.01, 0.05, 0.1, 0.5])
        adjusted = _p_adjust(p, method='bonferroni')
        expected = np.minimum(p * 4, 1.0)
        np.testing.assert_array_almost_equal(adjusted, expected)

    def test_fdr(self):
        """Test FDR adjustment."""
        from tna.bootstrap import _p_adjust
        p = np.array([0.01, 0.04, 0.1, 0.5])
        adjusted = _p_adjust(p, method='fdr')
        # BH: adjusted should be non-decreasing when sorted
        sorted_adj = np.sort(adjusted)
        assert np.all(sorted_adj[:-1] <= sorted_adj[1:])
        # Adjusted >= original
        assert np.all(adjusted >= p - 1e-15)


class TestPlotBootstrap:
    """Tests for bootstrap plotting functions."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create bootstrap result for plotting tests."""
        df = tna.load_group_regulation()
        model = tna.tna(df.head(200))
        return tna.bootstrap_tna(model, iter=10, seed=42)

    def test_plot_bootstrap_weights(self, bootstrap_result):
        """Test plotting bootstrap weights."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        fig = tna.plot_bootstrap(bootstrap_result, plot_type='weights')
        assert fig is not None
        plt.close(fig)

    def test_plot_bootstrap_centrality(self, bootstrap_result):
        """Test plotting bootstrap centrality."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        fig = tna.plot_bootstrap(
            bootstrap_result, plot_type='centrality', measure='OutStrength'
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_bootstrap_centrality_requires_measure(self, bootstrap_result):
        """Test that centrality plot requires measure."""
        pytest.importorskip("matplotlib")
        with pytest.raises(ValueError, match="measure must be specified"):
            tna.plot_bootstrap(bootstrap_result, plot_type='centrality')


class TestPlotPermutation:
    """Tests for permutation plotting functions."""

    @pytest.fixture
    def permutation_result(self):
        """Create permutation result for plotting tests."""
        df = tna.load_group_regulation()
        model1 = tna.tna(df.head(500))
        model2 = tna.tna(df.tail(500))
        return tna.permutation_test(model1, model2, iter=20, seed=42)

    def test_plot_permutation(self, permutation_result):
        """Test plotting permutation results."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        ax = tna.plot_permutation(permutation_result)
        assert ax is not None
        plt.close()

    def test_plot_permutation_method(self, permutation_result):
        """Test PermutationResult.plot() method."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        ax = permutation_result.plot()
        assert ax is not None
        plt.close()


class TestPlotNetworkCI:
    """Tests for network CI plotting."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create bootstrap result for plotting tests."""
        df = tna.load_group_regulation()
        model = tna.tna(df.head(200))
        return tna.bootstrap_tna(model, iter=10, seed=42)

    def test_plot_network_ci(self, bootstrap_result):
        """Test network CI plotting."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        ax = tna.plot_network_ci(bootstrap_result)
        assert ax is not None
        plt.close()

    def test_plot_network_ci_alpha_modes(self, bootstrap_result):
        """Test different edge_alpha modes."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        ax_sig = tna.plot_network_ci(bootstrap_result, edge_alpha='significance')
        plt.close()
        ax_ci = tna.plot_network_ci(bootstrap_result, edge_alpha='ci_width')
        plt.close()
        assert ax_sig is not None
        assert ax_ci is not None


class TestIntegration:
    """Integration tests for bootstrap module."""

    def test_full_bootstrap_workflow(self):
        """Test complete bootstrap analysis workflow."""
        df = tna.load_group_regulation()
        model = tna.tna(df)

        # Bootstrap
        boot = tna.bootstrap_tna(model, iter=30, seed=42)

        # Summary
        summary = boot.summary()
        assert len(summary) > 0

        # Significant edges
        sig_edges = boot.significant_edges()
        assert isinstance(sig_edges, list)

        # Bootstrap centralities
        cent_ci = tna.bootstrap_centralities(
            model, measures=['OutStrength', 'InStrength'], iter=20, seed=42
        )
        assert len(cent_ci) > 0

    def test_full_permutation_workflow(self):
        """Test complete group comparison workflow."""
        df = tna.load_group_regulation()
        model1 = tna.tna(df.head(500))
        model2 = tna.tna(df.tail(500))

        # Permutation test
        result = tna.permutation_test(model1, model2, iter=30, seed=42)
        assert isinstance(result, tna.PermutationResult)

        # Edge stats
        stats = result.edges['stats']
        assert len(stats) == 81

        # Significant edges
        sig = stats[stats['p_value'] < 0.05]
        assert isinstance(sig, pd.DataFrame)

    def test_permutation_with_data_input(self):
        """Test permutation test with raw data (backward compat)."""
        df = tna.load_group_regulation()
        df1 = df.head(500)
        df2 = df.tail(500)

        # Edge-wise test with raw data
        edges = tna.permutation_test_edges(
            df1, df2, n_perm=20, seed=42
        )
        assert isinstance(edges, pd.DataFrame)
        assert 'significant' in edges.columns


class TestTransitions3D:
    """Tests for 3D transition computation."""

    def test_3d_transitions_sum_matches_2d(self):
        """Test that summing 3D transitions gives same result as 2D computation."""
        from tna.transitions import compute_transitions_3d, compute_weights_from_3d
        from tna.prepare import create_seqdata

        df = tna.load_group_regulation()
        model = tna.tna(df)
        seq_data, labels, _ = create_seqdata(df)

        trans_3d = compute_transitions_3d(seq_data, model.labels, type_='relative')
        weights = compute_weights_from_3d(trans_3d, type_='relative')

        np.testing.assert_array_almost_equal(weights, model.weights, decimal=15)

    def test_3d_transitions_shape(self):
        """Test 3D transition array has correct shape."""
        from tna.transitions import compute_transitions_3d
        from tna.prepare import create_seqdata

        df = tna.load_group_regulation()
        seq_data, labels, _ = create_seqdata(df)

        trans_3d = compute_transitions_3d(seq_data, labels, type_='relative')
        assert trans_3d.shape == (len(df), len(labels), len(labels))

    def test_3d_transitions_nonnegative(self):
        """Test 3D transition counts are non-negative."""
        from tna.transitions import compute_transitions_3d
        from tna.prepare import create_seqdata

        df = tna.load_group_regulation()
        seq_data, labels, _ = create_seqdata(df)

        trans_3d = compute_transitions_3d(seq_data, labels, type_='relative')
        assert np.all(trans_3d >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
