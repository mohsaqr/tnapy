"""Tests for statistical inference (bootstrap and permutation tests).

Tests cover bootstrap resampling, confidence intervals, permutation tests,
and visualization functions for statistical significance analysis.
"""

import numpy as np
import pandas as pd
import pytest

import tna


class TestBootstrapResult:
    """Tests for BootstrapResult class."""

    @pytest.fixture
    def simple_sequences(self):
        """Simple test sequences."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'] * 20,
            'step2': ['B', 'C', 'B', 'A', 'A'] * 20,
            'step3': ['C', 'A', 'C', 'B', 'C'] * 20,
        })

    @pytest.fixture
    def bootstrap_result(self, simple_sequences):
        """Bootstrap result for testing."""
        return tna.bootstrap_tna(simple_sequences, n_boot=50, seed=42)

    def test_bootstrap_result_attributes(self, bootstrap_result):
        """Test BootstrapResult has correct attributes."""
        assert isinstance(bootstrap_result.estimate, tna.TNA)
        assert isinstance(bootstrap_result.replicates, list)
        assert len(bootstrap_result.replicates) == bootstrap_result.n_boot
        assert isinstance(bootstrap_result.weights_ci, tuple)
        assert len(bootstrap_result.weights_ci) == 2
        assert isinstance(bootstrap_result.inits_ci, tuple)
        assert len(bootstrap_result.inits_ci) == 2

    def test_bootstrap_ci_shape(self, bootstrap_result):
        """Test CI arrays have correct shape."""
        n_states = len(bootstrap_result.estimate.labels)
        assert bootstrap_result.weights_ci[0].shape == (n_states, n_states)
        assert bootstrap_result.weights_ci[1].shape == (n_states, n_states)
        assert bootstrap_result.inits_ci[0].shape == (n_states,)
        assert bootstrap_result.inits_ci[1].shape == (n_states,)

    def test_bootstrap_ci_ordering(self, bootstrap_result):
        """Test lower CI <= upper CI."""
        assert np.all(bootstrap_result.weights_ci[0] <= bootstrap_result.weights_ci[1])
        assert np.all(bootstrap_result.inits_ci[0] <= bootstrap_result.inits_ci[1])

    def test_bootstrap_summary(self, bootstrap_result):
        """Test summary() method."""
        summary = bootstrap_result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'from' in summary.columns
        assert 'to' in summary.columns
        assert 'estimate' in summary.columns
        assert 'ci_lower' in summary.columns
        assert 'ci_upper' in summary.columns
        assert 'se' in summary.columns

        n_states = len(bootstrap_result.estimate.labels)
        assert len(summary) == n_states * n_states

    def test_significant_edges(self, bootstrap_result):
        """Test significant_edges() method."""
        sig_edges = bootstrap_result.significant_edges(threshold=0)
        assert isinstance(sig_edges, list)
        for edge in sig_edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3  # (from, to, estimate)


class TestBootstrapTNA:
    """Tests for bootstrap_tna function."""

    @pytest.fixture
    def group_regulation(self):
        """Load group regulation data."""
        return tna.load_group_regulation()

    def test_bootstrap_basic(self, group_regulation):
        """Test basic bootstrap functionality."""
        boot = tna.bootstrap_tna(group_regulation, n_boot=20, seed=42)

        assert boot.n_boot == 20
        assert boot.ci_level == 0.95
        assert len(boot.replicates) == 20

    def test_bootstrap_seed_reproducibility(self, group_regulation):
        """Test that seed produces reproducible results."""
        boot1 = tna.bootstrap_tna(group_regulation, n_boot=10, seed=123)
        boot2 = tna.bootstrap_tna(group_regulation, n_boot=10, seed=123)

        np.testing.assert_array_almost_equal(
            boot1.weights_ci[0], boot2.weights_ci[0]
        )
        np.testing.assert_array_almost_equal(
            boot1.weights_ci[1], boot2.weights_ci[1]
        )

    def test_bootstrap_different_seeds(self, group_regulation):
        """Test that different seeds produce different results."""
        boot1 = tna.bootstrap_tna(group_regulation, n_boot=10, seed=123)
        boot2 = tna.bootstrap_tna(group_regulation, n_boot=10, seed=456)

        # Results should differ (with high probability)
        assert not np.allclose(boot1.weights_ci[0], boot2.weights_ci[0])

    def test_bootstrap_ci_level(self, group_regulation):
        """Test different confidence levels."""
        boot_95 = tna.bootstrap_tna(group_regulation, n_boot=50, ci=0.95, seed=42)
        boot_90 = tna.bootstrap_tna(group_regulation, n_boot=50, ci=0.90, seed=42)

        # 95% CI should be wider than 90% CI
        width_95 = boot_95.weights_ci[1] - boot_95.weights_ci[0]
        width_90 = boot_90.weights_ci[1] - boot_90.weights_ci[0]

        # On average, 95% CI should be wider
        assert np.mean(width_95) >= np.mean(width_90)

    def test_bootstrap_with_tnadata(self):
        """Test bootstrap with TNAData input."""
        long_df = tna.load_group_regulation_long()
        prepared = tna.prepare_data(
            long_df,
            actor='Actor',
            time='Time',
            action='Action'
        )
        boot = tna.bootstrap_tna(prepared, n_boot=10, seed=42)

        assert isinstance(boot, tna.BootstrapResult)

    def test_bootstrap_model_type(self, group_regulation):
        """Test bootstrap with different model types."""
        boot_rel = tna.bootstrap_tna(group_regulation, n_boot=10, type_='relative', seed=42)
        boot_freq = tna.bootstrap_tna(group_regulation, n_boot=10, type_='frequency', seed=42)

        assert boot_rel.estimate.type_ == 'relative'
        assert boot_freq.estimate.type_ == 'frequency'


class TestBootstrapCentralities:
    """Tests for bootstrap_centralities function."""

    @pytest.fixture
    def simple_sequences(self):
        """Simple test sequences."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'] * 30,
            'step2': ['B', 'C', 'B', 'A', 'A'] * 30,
            'step3': ['C', 'A', 'C', 'B', 'C'] * 30,
        })

    def test_centralities_basic(self, simple_sequences):
        """Test basic centrality bootstrap."""
        result = tna.bootstrap_centralities(
            simple_sequences,
            measures=['OutStrength', 'InStrength'],
            n_boot=20,
            seed=42
        )

        assert isinstance(result, pd.DataFrame)
        assert 'measure' in result.columns
        assert 'state' in result.columns
        assert 'estimate' in result.columns
        assert 'ci_lower' in result.columns
        assert 'ci_upper' in result.columns
        assert 'se' in result.columns

    def test_centralities_all_measures(self, simple_sequences):
        """Test with all measures."""
        result = tna.bootstrap_centralities(
            simple_sequences,
            measures=None,  # All measures
            n_boot=10,
            seed=42
        )

        measures_in_result = result['measure'].unique()
        assert len(measures_in_result) == len(tna.AVAILABLE_MEASURES)

    def test_centralities_ci_ordering(self, simple_sequences):
        """Test CI lower <= upper."""
        result = tna.bootstrap_centralities(
            simple_sequences,
            measures=['OutStrength'],
            n_boot=20,
            seed=42
        )

        assert all(result['ci_lower'] <= result['ci_upper'])

    def test_centralities_se_positive(self, simple_sequences):
        """Test standard errors are non-negative."""
        result = tna.bootstrap_centralities(
            simple_sequences,
            measures=['OutStrength'],
            n_boot=20,
            seed=42
        )

        assert all(result['se'] >= 0)


class TestPermutationResult:
    """Tests for PermutationResult class."""

    def test_is_significant(self):
        """Test is_significant method."""
        # Significant result
        result_sig = tna.PermutationResult(
            observed=2.0,
            null_distribution=np.random.randn(100),
            p_value=0.01,
            n_perm=100,
            alternative='two-sided'
        )
        assert result_sig.is_significant(alpha=0.05) is True
        assert result_sig.is_significant(alpha=0.001) is False

        # Non-significant result
        result_nonsig = tna.PermutationResult(
            observed=0.5,
            null_distribution=np.random.randn(100),
            p_value=0.30,
            n_perm=100,
            alternative='two-sided'
        )
        assert result_nonsig.is_significant(alpha=0.05) is False


class TestPermutationTest:
    """Tests for permutation_test function."""

    @pytest.fixture
    def two_groups(self):
        """Create two groups for testing."""
        np.random.seed(42)
        # Group 1: A->B more common
        group1 = pd.DataFrame({
            'step1': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2]),
            'step2': np.random.choice(['A', 'B', 'C'], 100, p=[0.2, 0.5, 0.3]),
            'step3': np.random.choice(['A', 'B', 'C'], 100, p=[0.3, 0.3, 0.4]),
        })
        # Group 2: C->A more common
        group2 = pd.DataFrame({
            'step1': np.random.choice(['A', 'B', 'C'], 100, p=[0.2, 0.3, 0.5]),
            'step2': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.2, 0.3]),
            'step3': np.random.choice(['A', 'B', 'C'], 100, p=[0.4, 0.3, 0.3]),
        })
        return group1, group2

    def test_permutation_basic(self, two_groups):
        """Test basic permutation test."""
        group1, group2 = two_groups
        result = tna.permutation_test(group1, group2, n_perm=50, seed=42)

        assert isinstance(result, tna.PermutationResult)
        assert result.n_perm == 50
        assert len(result.null_distribution) == 50
        assert 0 <= result.p_value <= 1
        assert isinstance(result.observed, (float, np.floating))

    def test_permutation_alternatives(self, two_groups):
        """Test different alternatives."""
        group1, group2 = two_groups

        result_two = tna.permutation_test(
            group1, group2, n_perm=30, alternative='two-sided', seed=42
        )
        result_greater = tna.permutation_test(
            group1, group2, n_perm=30, alternative='greater', seed=42
        )
        result_less = tna.permutation_test(
            group1, group2, n_perm=30, alternative='less', seed=42
        )

        assert result_two.alternative == 'two-sided'
        assert result_greater.alternative == 'greater'
        assert result_less.alternative == 'less'

    def test_permutation_statistics(self, two_groups):
        """Test different test statistics."""
        group1, group2 = two_groups

        result_weights = tna.permutation_test(
            group1, group2, n_perm=20, statistic='weights', seed=42
        )
        result_density = tna.permutation_test(
            group1, group2, n_perm=20, statistic='density', seed=42
        )
        result_cent = tna.permutation_test(
            group1, group2, n_perm=20, statistic='centrality',
            measure='OutStrength', seed=42
        )

        assert isinstance(result_weights.observed, (float, np.floating))
        assert isinstance(result_density.observed, (float, np.floating))
        assert isinstance(result_cent.observed, (float, np.floating))

    def test_permutation_centrality_requires_measure(self, two_groups):
        """Test that centrality statistic requires measure parameter."""
        group1, group2 = two_groups

        with pytest.raises(ValueError, match="measure must be specified"):
            tna.permutation_test(
                group1, group2, n_perm=10, statistic='centrality', seed=42
            )

    def test_permutation_seed_reproducibility(self, two_groups):
        """Test seed produces reproducible results."""
        group1, group2 = two_groups

        result1 = tna.permutation_test(group1, group2, n_perm=20, seed=999)
        result2 = tna.permutation_test(group1, group2, n_perm=20, seed=999)

        assert result1.observed == result2.observed
        np.testing.assert_array_equal(
            result1.null_distribution, result2.null_distribution
        )

    def test_permutation_identical_groups(self):
        """Test with identical groups (p-value should be ~1)."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'B'] * 25,
            'step2': ['B', 'A', 'B', 'A'] * 25,
        })

        # Split into two identical groups
        result = tna.permutation_test(df, df.copy(), n_perm=50, seed=42)

        # P-value should be high (no difference)
        assert result.p_value > 0.1


class TestPermutationTestEdges:
    """Tests for permutation_test_edges function."""

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

    def test_edges_corrections(self, two_groups):
        """Test different multiple testing corrections."""
        group1, group2 = two_groups

        result_bonf = tna.permutation_test_edges(
            group1, group2, n_perm=20, correction='bonferroni', seed=42
        )
        result_fdr = tna.permutation_test_edges(
            group1, group2, n_perm=20, correction='fdr', seed=42
        )
        result_none = tna.permutation_test_edges(
            group1, group2, n_perm=20, correction='none', seed=42
        )

        # Bonferroni should be more conservative than FDR
        assert all(result_bonf['p_adjusted'] >= result_fdr['p_adjusted'])

        # No correction should equal raw p-values
        np.testing.assert_array_almost_equal(
            result_none['p_value'].values,
            result_none['p_adjusted'].values
        )

    def test_edges_p_value_range(self, two_groups):
        """Test p-values are in valid range."""
        group1, group2 = two_groups
        result = tna.permutation_test_edges(
            group1, group2, n_perm=20, seed=42
        )

        assert all(result['p_value'] >= 0)
        assert all(result['p_value'] <= 1)
        assert all(result['p_adjusted'] >= 0)
        assert all(result['p_adjusted'] <= 1)


class TestConfidenceInterval:
    """Tests for confidence interval functions."""

    def test_percentile_ci(self):
        """Test percentile confidence interval."""
        np.random.seed(42)
        values = np.random.normal(10, 2, 1000)

        lower, upper = tna.confidence_interval(values, ci=0.95, method='percentile')

        assert lower < 10 < upper
        assert lower < upper
        # 95% CI should contain most of the distribution
        assert lower > values.min()
        assert upper < values.max()

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

        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_90 < width_95 < width_99

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
        bootstrap_values = np.array([np.mean(np.random.choice(data, len(data), replace=True))
                                    for _ in range(200)])

        lower, upper = tna.bca_ci(
            data, bootstrap_values,
            statistic_func=np.mean,
            ci=0.95
        )

        assert lower < upper
        assert lower < np.mean(data) < upper

    def test_bca_skewed_data(self):
        """Test BCa with skewed data."""
        np.random.seed(42)
        # Exponential distribution (skewed)
        data = np.random.exponential(2, 100)
        bootstrap_values = np.array([np.mean(np.random.choice(data, len(data), replace=True))
                                    for _ in range(300)])

        lower, upper = tna.bca_ci(
            data, bootstrap_values,
            statistic_func=np.mean,
            ci=0.95
        )

        assert lower < upper
        assert lower > 0  # Exponential data is positive


class TestPlotBootstrap:
    """Tests for bootstrap plotting functions."""

    @pytest.fixture
    def bootstrap_result(self):
        """Create bootstrap result for plotting tests."""
        df = tna.load_group_regulation()
        return tna.bootstrap_tna(df.head(200), n_boot=10, seed=42)

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
            bootstrap_result,
            plot_type='centrality',
            measure='OutStrength'
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
        df1 = df.head(100)
        df2 = df.tail(100)
        return tna.permutation_test(df1, df2, n_perm=20, seed=42)

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
        return tna.bootstrap_tna(df.head(200), n_boot=10, seed=42)

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

    def test_full_workflow(self):
        """Test complete bootstrap analysis workflow."""
        # Load data
        df = tna.load_group_regulation()

        # Bootstrap analysis
        boot = tna.bootstrap_tna(df.head(500), n_boot=30, ci=0.95, seed=42)

        # Get summary
        summary = boot.summary()
        assert len(summary) > 0

        # Get significant edges
        sig_edges = boot.significant_edges(threshold=0.05)
        assert isinstance(sig_edges, list)

        # Bootstrap centralities
        cent_ci = tna.bootstrap_centralities(
            df.head(500),
            measures=['OutStrength', 'InStrength'],
            n_boot=20,
            seed=42
        )
        assert len(cent_ci) > 0

    def test_group_comparison_workflow(self):
        """Test complete group comparison workflow."""
        df = tna.load_group_regulation()

        # Split into groups
        df1 = df.head(500)
        df2 = df.tail(500)

        # Permutation test
        result = tna.permutation_test(df1, df2, n_perm=30, seed=42)
        assert isinstance(result.p_value, float)

        # Edge-wise testing
        edges = tna.permutation_test_edges(
            df1, df2, n_perm=20, correction='fdr', seed=42
        )
        sig_edges = edges[edges['significant']]
        assert isinstance(sig_edges, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
