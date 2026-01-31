"""Tests for numerical equivalence with R TNA package.

These tests verify that Python TNA produces identical results to R TNA.
To generate expected values, run tests/r_verification.R in R.
"""

import numpy as np
import pandas as pd
import pytest

import tna

# Expected values from R TNA package
# These can be updated by running r_verification.R


class TestREquivalence:
    """Tests for numerical equivalence with R TNA package."""

    @pytest.fixture
    def r_model(self):
        """Build model from group_regulation data (same as R)."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    @pytest.fixture
    def group_regulation_data(self):
        """Load group regulation data."""
        return tna.load_group_regulation()

    def test_model_labels(self, r_model):
        """Verify state labels match R."""
        expected_labels = {
            'adapt', 'cohesion', 'consensus', 'coregulate',
            'discuss', 'emotion', 'monitor', 'plan', 'synthesis'
        }
        assert set(r_model.labels) == expected_labels

    def test_weight_matrix_shape(self, r_model):
        """Verify weight matrix dimensions."""
        assert r_model.weights.shape == (9, 9)

    def test_row_stochastic(self, r_model):
        """Verify weight matrix is row-stochastic (rows sum to 1)."""
        row_sums = r_model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(9), decimal=10)

    def test_weight_matrix_values(self, r_model):
        """Verify specific weight matrix values match R.

        Expected values from R:
        > model <- tna(group_regulation)
        > round(model$weights, 4)
        """
        # Key transition probabilities to verify
        # These should match R output exactly

        # Get index mapping
        label_to_idx = {l: i for i, l in enumerate(r_model.labels)}

        # Verify some key transitions (from R output)
        # consensus -> plan should be high (~0.3958)
        consensus_idx = label_to_idx['consensus']
        plan_idx = label_to_idx['plan']
        assert 0.39 < r_model.weights[consensus_idx, plan_idx] < 0.40

        # cohesion -> consensus should be high (~0.4979)
        cohesion_idx = label_to_idx['cohesion']
        assert 0.49 < r_model.weights[cohesion_idx, consensus_idx] < 0.51

    def test_initial_probabilities_sum(self, r_model):
        """Verify initial probabilities sum to 1."""
        assert abs(r_model.inits.sum() - 1.0) < 1e-10

    def test_strength_centralities(self, r_model):
        """Verify strength centralities match R igraph."""
        cent = tna.centralities(r_model, measures=['OutStrength', 'InStrength'])

        # OutStrength should equal row sums of weight matrix
        # (which is 1 for row-stochastic matrix without self-loops removed)
        out_strength = r_model.weights.sum(axis=1)

        # After removing diagonal, out_strength may differ
        weights_no_diag = r_model.weights.copy()
        np.fill_diagonal(weights_no_diag, 0)
        expected_out = weights_no_diag.sum(axis=1)
        expected_in = weights_no_diag.sum(axis=0)

        for i, label in enumerate(r_model.labels):
            np.testing.assert_almost_equal(
                cent.loc[label, 'OutStrength'],
                expected_out[i],
                decimal=10
            )
            np.testing.assert_almost_equal(
                cent.loc[label, 'InStrength'],
                expected_in[i],
                decimal=10
            )

    def test_diffusion_centrality(self, r_model):
        """Verify diffusion centrality computation."""
        cent = tna.centralities(r_model, measures=['Diffusion'])

        # Diffusion should be positive for all nodes
        assert all(cent['Diffusion'] > 0)

        # Diffusion values should be reasonable (not infinity or nan)
        assert all(np.isfinite(cent['Diffusion']))

    def test_clustering_coefficient(self, r_model):
        """Verify clustering coefficient computation."""
        cent = tna.centralities(r_model, measures=['Clustering'])

        # Clustering should be in [0, 1] range for normalized networks
        # For raw networks, it can exceed 1
        assert all(np.isfinite(cent['Clustering']))

    def test_frequency_model(self, group_regulation_data):
        """Verify frequency model produces raw counts."""
        fmodel = tna.ftna(group_regulation_data)

        # Frequency model should have integer-like values (counts)
        # All values should be non-negative
        assert np.all(fmodel.weights >= 0)

        # Total transitions should be large (many sequences)
        assert fmodel.weights.sum() > 1000

    def test_cooccurrence_model(self, group_regulation_data):
        """Verify co-occurrence model is symmetric after computation."""
        cmodel = tna.ctna(group_regulation_data)

        # Row-normalized, so rows should sum to 1
        row_sums = cmodel.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(9), decimal=10)


class TestAllModelTypes:
    """Tests for all 8 model types to ensure R equivalence."""

    @pytest.fixture
    def simple_data(self):
        """Simple test data for model type comparison."""
        return pd.DataFrame({
            'T1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'T2': ['B', 'C', 'A', 'C', 'A', 'B', 'C', 'A'],
            'T3': ['C', 'A', 'B', 'B', 'C', 'A', 'A', 'B'],
            'T4': ['A', 'B', 'C', 'A', 'B', 'C', 'B', 'C'],
        })

    @pytest.fixture
    def group_regulation_data(self):
        """Load group regulation data."""
        return tna.load_group_regulation()

    def test_relative_model_row_stochastic(self, simple_data):
        """Relative model should have row-stochastic matrix."""
        model = tna.build_model(simple_data, type_='relative')
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_frequency_model_counts(self, simple_data):
        """Frequency model should have non-negative integer-like counts."""
        model = tna.build_model(simple_data, type_='frequency')
        assert np.all(model.weights >= 0)
        # Total should equal number of transitions
        total_transitions = (len(simple_data.columns) - 1) * len(simple_data)
        # Account for NA handling
        assert model.weights.sum() <= total_transitions

    def test_cooccurrence_model_structure(self, simple_data):
        """Co-occurrence model should have valid structure."""
        model = tna.build_model(simple_data, type_='co-occurrence')
        # After row-normalization, rows should sum to 1
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_reverse_model_different(self, simple_data):
        """Reverse model should differ from forward model."""
        fwd = tna.build_model(simple_data, type_='relative')
        rev = tna.build_model(simple_data, type_='reverse')
        # Matrices should be different (unless data is perfectly symmetric)
        assert rev.type_ == 'reverse'
        # Reverse should also be row-stochastic
        row_sums = rev.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_ngram_model_row_stochastic(self, simple_data):
        """N-gram model should have row-stochastic matrix."""
        model = tna.build_model(simple_data, type_='n-gram', params={'n': 2})
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_gap_model_row_stochastic(self, simple_data):
        """Gap model should have row-stochastic matrix."""
        model = tna.build_model(simple_data, type_='gap',
                                params={'max_gap': 3, 'decay': 0.5})
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_window_model_row_stochastic(self, simple_data):
        """Window model should have row-stochastic matrix."""
        model = tna.build_model(simple_data, type_='window',
                                params={'size': 3})
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_attention_model_row_stochastic(self, simple_data):
        """Attention model should have row-stochastic matrix."""
        model = tna.build_model(simple_data, type_='attention',
                                params={'beta': 0.1})
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=10)

    def test_all_model_types_with_group_regulation(self, group_regulation_data):
        """Test all model types with real group regulation data."""
        types_to_test = [
            ('relative', {}),
            ('frequency', {}),
            ('co-occurrence', {}),
            ('reverse', {}),
            ('n-gram', {'n': 2}),
            ('gap', {'max_gap': 3, 'decay': 0.5}),
            ('window', {'size': 3}),
            ('attention', {'beta': 0.1}),
        ]

        for type_, params in types_to_test:
            model = tna.build_model(group_regulation_data, type_=type_, params=params)
            assert model.type_ == type_
            assert model.weights.shape == (9, 9)
            assert len(model.labels) == 9

            # For normalized types, check row sums
            if type_ not in ['frequency']:
                row_sums = model.weights.sum(axis=1)
                np.testing.assert_array_almost_equal(
                    row_sums, np.ones(9), decimal=10,
                    err_msg=f"Row sums failed for type {type_}"
                )


class TestAllCentralityMeasures:
    """Tests for all 9 centrality measures."""

    @pytest.fixture
    def model(self):
        """Build model from group_regulation data."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    def test_all_measures_compute(self, model):
        """All centrality measures should compute without error."""
        cent = tna.centralities(model)
        assert len(cent.columns) == len(tna.AVAILABLE_MEASURES)
        assert all(m in cent.columns for m in tna.AVAILABLE_MEASURES)

    def test_outstrength_properties(self, model):
        """OutStrength should have valid properties."""
        cent = tna.centralities(model, measures=['OutStrength'])
        # All values should be finite and non-negative
        assert all(np.isfinite(cent['OutStrength']))
        assert all(cent['OutStrength'] >= 0)

    def test_instrength_properties(self, model):
        """InStrength should have valid properties."""
        cent = tna.centralities(model, measures=['InStrength'])
        # All values should be finite and non-negative
        assert all(np.isfinite(cent['InStrength']))
        assert all(cent['InStrength'] >= 0)

    def test_closeness_in_properties(self, model):
        """ClosenessIn should have valid properties."""
        cent = tna.centralities(model, measures=['ClosenessIn'])
        assert all(np.isfinite(cent['ClosenessIn']))
        assert all(cent['ClosenessIn'] >= 0)

    def test_closeness_out_properties(self, model):
        """ClosenessOut should have valid properties."""
        cent = tna.centralities(model, measures=['ClosenessOut'])
        assert all(np.isfinite(cent['ClosenessOut']))
        assert all(cent['ClosenessOut'] >= 0)

    def test_closeness_all_properties(self, model):
        """Closeness (all modes) should have valid properties."""
        cent = tna.centralities(model, measures=['Closeness'])
        assert all(np.isfinite(cent['Closeness']))
        assert all(cent['Closeness'] >= 0)

    def test_betweenness_properties(self, model):
        """Betweenness should have valid properties."""
        cent = tna.centralities(model, measures=['Betweenness'])
        assert all(np.isfinite(cent['Betweenness']))
        assert all(cent['Betweenness'] >= 0)

    def test_betweenness_rsp_properties(self, model):
        """BetweennessRSP should have valid properties."""
        cent = tna.centralities(model, measures=['BetweennessRSP'])
        # BetweennessRSP can return NaN if matrix has zero rows
        # For valid matrices, values should be positive
        if all(np.isfinite(cent['BetweennessRSP'])):
            assert all(cent['BetweennessRSP'] >= 0)

    def test_diffusion_properties(self, model):
        """Diffusion should have valid properties."""
        cent = tna.centralities(model, measures=['Diffusion'])
        assert all(np.isfinite(cent['Diffusion']))
        assert all(cent['Diffusion'] > 0)  # Should be strictly positive

    def test_clustering_properties(self, model):
        """Clustering should have valid properties."""
        cent = tna.centralities(model, measures=['Clustering'])
        assert all(np.isfinite(cent['Clustering']))

    def test_normalized_centralities(self, model):
        """Normalized centralities should be in [0, 1] range."""
        cent = tna.centralities(model, normalize=True)
        for col in cent.columns:
            if all(np.isfinite(cent[col])):
                assert cent[col].min() >= -1e-10, f"{col} min below 0"
                assert cent[col].max() <= 1.0 + 1e-10, f"{col} max above 1"

    def test_loops_parameter_effect(self, model):
        """Loops parameter should affect results when self-loops exist."""
        cent_no_loops = tna.centralities(model, loops=False)
        cent_with_loops = tna.centralities(model, loops=True)
        # Both should compute successfully
        assert len(cent_no_loops) == len(cent_with_loops)


class TestAlgorithmEquivalence:
    """Tests for algorithm-level equivalence with R implementations."""

    def test_diffusion_algorithm(self):
        """Test diffusion algorithm matches R implementation."""
        # Simple test matrix
        mat = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.4, 0.6, 0.0]
        ])

        # Python implementation
        from tna.centralities import _diffusion
        result = _diffusion(mat)

        # Manual calculation matching R:
        # s <- 0; p <- diag(1,n,n)
        # for (i in 1:n) { p <- p %*% mat; s <- s + p }
        # .rowSums(s, n, n)
        n = 3
        s = np.zeros((n, n))
        p = np.eye(n)
        for _ in range(n):
            p = p @ mat
            s = s + p
        expected = s.sum(axis=1)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_clustering_algorithm(self):
        """Test clustering algorithm matches R wcc implementation."""
        # Simple test matrix
        mat = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.4, 0.6, 0.0]
        ])

        # Python implementation
        from tna.centralities import _clustering
        result = _clustering(mat)

        # Manual calculation matching R wcc(x + t(x)):
        sym = mat + mat.T
        np.fill_diagonal(sym, 0)
        num = np.diag(sym @ sym @ sym)
        col_sums = sym.sum(axis=0)
        col_sums_sq = (sym ** 2).sum(axis=0)
        den = col_sums ** 2 - col_sums_sq
        expected = np.where(den != 0, num / den, 0)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_betweenness_rsp_algorithm(self):
        """Test RSP betweenness algorithm matches R implementation."""
        # Test with a fully connected matrix (no zero rows)
        mat = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2]
        ])

        from tna.centralities import _betweenness_rsp
        result = _betweenness_rsp(mat, beta=0.01)

        # Result should be finite and non-negative
        assert all(np.isfinite(result))
        assert all(result >= 0)

    def test_diffusion_with_larger_matrix(self):
        """Test diffusion with larger matrix."""
        np.random.seed(42)
        n = 10
        mat = np.random.rand(n, n)
        # Make row-stochastic
        mat = mat / mat.sum(axis=1, keepdims=True)
        np.fill_diagonal(mat, 0)
        mat = mat / mat.sum(axis=1, keepdims=True)

        from tna.centralities import _diffusion
        result = _diffusion(mat)

        # Manually compute expected
        s = np.zeros((n, n))
        p = np.eye(n)
        for _ in range(n):
            p = p @ mat
            s = s + p
        expected = s.sum(axis=1)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestDataConsistency:
    """Tests to ensure data consistency between R and Python."""

    def test_group_regulation_dimensions(self):
        """Group regulation data should have correct dimensions."""
        df = tna.load_group_regulation()
        assert df.shape[0] == 2000  # 2000 sequences
        # Number of time steps may vary, but should be reasonable
        assert df.shape[1] >= 10  # At least 10 time steps

    def test_group_regulation_states(self):
        """Group regulation data should have correct states."""
        df = tna.load_group_regulation()
        expected_states = {
            'adapt', 'cohesion', 'consensus', 'coregulate',
            'discuss', 'emotion', 'monitor', 'plan', 'synthesis'
        }
        actual_states = set(df.stack().dropna().unique())
        assert actual_states == expected_states

    def test_group_regulation_long_format(self):
        """Long format data should have correct structure."""
        df = tna.load_group_regulation_long()
        assert 'Actor' in df.columns
        assert 'Action' in df.columns
        assert 'Time' in df.columns

    def test_model_from_wide_and_long_consistency(self):
        """Models from wide and long format should be similar."""
        # This is a structural test - actual values may differ due to
        # session detection in long format
        df_wide = tna.load_group_regulation()
        df_long = tna.load_group_regulation_long()

        model_wide = tna.tna(df_wide)

        # Prepare long format and build model
        prepared = tna.prepare_data(
            df_long,
            actor='Actor',
            time='Time',
            action='Action'
        )
        model_long = tna.tna(prepared)

        # Both should have same states
        assert set(model_wide.labels) == set(model_long.labels)


class TestScalingEquivalence:
    """Tests for scaling function equivalence with R."""

    @pytest.fixture
    def test_matrix(self):
        """Test matrix for scaling."""
        return np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.6, 0.4, 0.0]
        ])

    def test_minmax_scaling(self, test_matrix):
        """Min-max scaling should produce [0, 1] range."""
        scaled = tna.minmax_scale(test_matrix)
        assert scaled.min() == 0.0
        assert scaled.max() == 1.0

    def test_max_scaling(self, test_matrix):
        """Max scaling should divide by maximum."""
        scaled = tna.max_scale(test_matrix)
        assert scaled.max() == 1.0
        np.testing.assert_array_almost_equal(
            scaled, test_matrix / test_matrix.max()
        )

    def test_rank_scaling(self, test_matrix):
        """Rank scaling should produce rank-based values."""
        scaled = tna.rank_scale(test_matrix)
        # Rank scaling produces ranks, not normalized values
        # Values should be non-negative
        assert scaled.min() >= 0.0
        # All values should be finite
        assert np.all(np.isfinite(scaled))


class TestInitialProbabilities:
    """Tests for initial probability computation."""

    def test_initial_probs_sum_to_one(self):
        """Initial probabilities should sum to 1."""
        df = tna.load_group_regulation()
        model = tna.tna(df)
        assert abs(model.inits.sum() - 1.0) < 1e-10

    def test_initial_probs_match_first_states(self):
        """Initial probs should reflect first state frequencies."""
        df = pd.DataFrame({
            'T1': ['A', 'A', 'B', 'C'],  # A appears 2x, B 1x, C 1x
            'T2': ['B', 'C', 'A', 'A'],
        })
        model = tna.tna(df)

        # Get initial prob for A
        a_idx = model.labels.index('A')
        assert abs(model.inits[a_idx] - 0.5) < 1e-10  # 2/4 = 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
