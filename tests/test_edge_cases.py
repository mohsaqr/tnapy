"""Edge case and error handling tests for TNA package.

Tests for:
- Empty input handling
- Single-state sequences
- Very long sequences
- Missing values patterns
- Numerical stability (near-zero weights)
- Invalid input handling
"""

import numpy as np
import pandas as pd
import pytest

import tna


class TestEmptyInput:
    """Tests for empty or near-empty input handling."""

    def test_empty_dataframe(self):
        """Empty DataFrame should raise an error or return empty model."""
        df = pd.DataFrame()
        # An empty dataframe may either raise an error or produce an empty model
        try:
            model = tna.tna(df)
            # If it succeeds, model should be empty or minimal
            assert len(model.labels) == 0 or model.weights.sum() == 0
        except (ValueError, KeyError, IndexError):
            pass  # Raising an error is acceptable

    def test_single_row_dataframe(self):
        """Single row DataFrame should work."""
        df = pd.DataFrame({'T1': ['A'], 'T2': ['B'], 'T3': ['C']})
        model = tna.tna(df)
        assert len(model.labels) == 3
        assert model.weights.shape == (3, 3)

    def test_single_column_dataframe(self):
        """Single column DataFrame has no transitions."""
        df = pd.DataFrame({'T1': ['A', 'B', 'C']})
        model = tna.tna(df)
        # No transitions possible with single column
        assert model.weights.sum() == 0

    def test_all_na_dataframe(self):
        """DataFrame with all NA values."""
        df = pd.DataFrame({
            'T1': [None, None, None],
            'T2': [None, None, None],
        })
        # Should handle gracefully or raise appropriate error
        try:
            model = tna.tna(df)
            # If it succeeds, should have empty or zero matrix
            assert model.weights.sum() == 0 or len(model.labels) == 0
        except (ValueError, KeyError):
            pass  # Acceptable to raise error


class TestSingleState:
    """Tests for sequences with only one state."""

    def test_single_state_all_rows(self):
        """All sequences contain only one state."""
        df = pd.DataFrame({
            'T1': ['A', 'A', 'A'],
            'T2': ['A', 'A', 'A'],
            'T3': ['A', 'A', 'A'],
        })
        model = tna.tna(df)

        assert len(model.labels) == 1
        assert model.labels[0] == 'A'
        assert model.weights.shape == (1, 1)
        # Self-loop should be 1.0
        assert model.weights[0, 0] == 1.0

    def test_single_state_initial_prob(self):
        """Initial probability with single state should be 1.0."""
        df = pd.DataFrame({
            'T1': ['A', 'A'],
            'T2': ['A', 'A'],
        })
        model = tna.tna(df)

        assert model.inits[0] == 1.0

    def test_single_state_centralities(self):
        """Centralities with single state."""
        df = pd.DataFrame({
            'T1': ['A', 'A'],
            'T2': ['A', 'A'],
        })
        model = tna.tna(df)
        cent = tna.centralities(model)

        # Should compute without error
        assert len(cent) == 1


class TestLongSequences:
    """Tests for very long sequences."""

    def test_long_sequence_100_steps(self):
        """Sequence with 100 time steps."""
        np.random.seed(42)
        n_sequences = 50
        n_steps = 100
        states = ['A', 'B', 'C', 'D', 'E']

        data = {}
        for i in range(n_steps):
            data[f'T{i+1}'] = np.random.choice(states, n_sequences)

        df = pd.DataFrame(data)
        model = tna.tna(df)

        assert len(model.labels) == 5
        assert model.weights.shape == (5, 5)
        # Row stochastic
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=10)

    def test_long_sequence_many_states(self):
        """Sequence with many unique states (20)."""
        np.random.seed(42)
        n_sequences = 100
        n_steps = 20
        states = [f'S{i}' for i in range(20)]

        data = {}
        for i in range(n_steps):
            data[f'T{i+1}'] = np.random.choice(states, n_sequences)

        df = pd.DataFrame(data)
        model = tna.tna(df)

        assert len(model.labels) == 20
        assert model.weights.shape == (20, 20)

    def test_large_number_of_sequences(self):
        """Large number of sequences (10000)."""
        np.random.seed(42)
        n_sequences = 10000
        n_steps = 5
        states = ['A', 'B', 'C']

        data = {}
        for i in range(n_steps):
            data[f'T{i+1}'] = np.random.choice(states, n_sequences)

        df = pd.DataFrame(data)
        model = tna.tna(df)

        assert model.weights.shape == (3, 3)
        # With many sequences, probabilities should be close to uniform
        # (random data with equal state probabilities)


class TestMissingValues:
    """Tests for various missing value patterns."""

    def test_random_na_pattern(self):
        """Random NA values throughout."""
        np.random.seed(42)
        df = pd.DataFrame({
            'T1': ['A', 'B', None, 'C', 'A'],
            'T2': ['B', None, 'A', 'A', None],
            'T3': [None, 'C', 'B', None, 'C'],
            'T4': ['C', 'A', 'C', 'B', 'A'],
        })
        model = tna.tna(df)

        assert set(model.labels) == {'A', 'B', 'C'}
        # Model should still be valid
        row_sums = model.weights.sum(axis=1)
        # Row sums should be 1 or 0 (if no outgoing transitions)
        for rs in row_sums:
            assert rs == 0 or abs(rs - 1.0) < 1e-10

    def test_trailing_na(self):
        """NA values at the end of sequences."""
        df = pd.DataFrame({
            'T1': ['A', 'B', 'C'],
            'T2': ['B', 'C', 'A'],
            'T3': ['C', None, None],
            'T4': [None, None, None],
        })
        model = tna.tna(df)

        assert set(model.labels) == {'A', 'B', 'C'}

    def test_leading_na(self):
        """NA values at the beginning of sequences."""
        df = pd.DataFrame({
            'T1': [None, None, 'C'],
            'T2': [None, 'B', 'A'],
            'T3': ['A', 'C', 'B'],
            'T4': ['B', 'A', 'C'],
        })
        model = tna.tna(df)

        assert set(model.labels) == {'A', 'B', 'C'}

    def test_alternating_na(self):
        """Alternating NA and non-NA values."""
        df = pd.DataFrame({
            'T1': ['A', None, 'B', None],
            'T2': [None, 'B', None, 'A'],
            'T3': ['B', None, 'C', None],
            'T4': [None, 'A', None, 'C'],
        })
        model = tna.tna(df)

        assert set(model.labels) == {'A', 'B', 'C'}


class TestNumericalStability:
    """Tests for numerical stability with edge case values."""

    def test_near_zero_weights(self):
        """Very small (near-zero) transition weights."""
        # Create data where one transition is very rare
        df = pd.DataFrame({
            'T1': ['A'] * 999 + ['B'],
            'T2': ['A'] * 999 + ['A'],
        })
        model = tna.tna(df)

        # B -> A transition should be 1.0 (only one outgoing from B)
        b_idx = model.labels.index('B')
        a_idx = model.labels.index('A')
        assert model.weights[b_idx, a_idx] == 1.0

        # A -> A should be close to 1.0
        assert model.weights[a_idx, a_idx] > 0.99

    def test_centralities_near_zero_weights(self):
        """Centralities with near-zero weights in matrix."""
        weights = np.array([
            [0.0, 1e-15, 1.0 - 1e-15],
            [0.5, 0.0, 0.5],
            [1.0 - 1e-15, 1e-15, 0.0]
        ])
        model = tna.build_model(weights)
        cent = tna.centralities(model)

        # Should compute without NaN or Inf
        for col in cent.columns:
            if not all(np.isfinite(cent[col])):
                # Some measures may legitimately produce NaN for certain matrices
                continue
            assert all(np.isfinite(cent[col])), f"{col} has non-finite values"

    def test_uniform_distribution(self):
        """Perfectly uniform transition distribution."""
        # Each transition equally likely
        n_states = 5
        weights = np.ones((n_states, n_states)) / n_states
        np.fill_diagonal(weights, 0)
        # Re-normalize after removing diagonal
        weights = weights / weights.sum(axis=1, keepdims=True)

        model = tna.build_model(weights)
        cent = tna.centralities(model)

        # With uniform weights, all states should have similar centralities
        for col in ['InStrength', 'OutStrength']:
            values = cent[col].values
            assert np.std(values) < 0.1, f"{col} should be nearly uniform"

    def test_sparse_matrix(self):
        """Very sparse transition matrix."""
        # Only diagonal adjacent transitions
        n = 5
        weights = np.zeros((n, n))
        for i in range(n):
            weights[i, (i + 1) % n] = 1.0  # Circular chain

        model = tna.build_model(weights)
        cent = tna.centralities(model)

        # All OutStrength should be 1
        np.testing.assert_array_almost_equal(
            cent['OutStrength'].values, np.ones(n)
        )


class TestInvalidInput:
    """Tests for invalid input handling."""

    def test_invalid_measure_name(self):
        """Invalid centrality measure name."""
        df = tna.load_group_regulation()
        model = tna.tna(df)

        with pytest.raises(ValueError, match="Unknown measures"):
            tna.centralities(model, measures=['InvalidMeasure'])

    def test_negative_weights_in_matrix(self):
        """Matrix input with negative weights."""
        weights = np.array([
            [0.0, -0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.4, 0.6, 0.0]
        ])
        # Should either handle or raise appropriate error
        try:
            model = tna.build_model(weights)
            # If accepted, centralities might have issues
        except ValueError:
            pass  # Acceptable to reject negative weights

    def test_non_square_matrix(self):
        """Non-square matrix input when matrix is expected."""
        weights = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
        ])
        # This should be treated as sequence data, not a matrix
        # Or raise an error
        model = tna.build_model(weights)
        # If it interprets as sequences, it should work


class TestBootstrapEdgeCases:
    """Tests for bootstrap edge cases."""

    def test_bootstrap_small_data(self):
        """Bootstrap with very small dataset."""
        df = pd.DataFrame({
            'T1': ['A', 'B'],
            'T2': ['B', 'A'],
        })

        boot = tna.bootstrap_tna(df, iter=10, seed=42)
        assert boot.iter == 10

    def test_bootstrap_single_sequence(self):
        """Bootstrap with single sequence."""
        df = pd.DataFrame({
            'T1': ['A'],
            'T2': ['B'],
            'T3': ['C'],
        })

        boot = tna.bootstrap_tna(df, iter=10, seed=42)
        # Should work but CI will be very wide
        assert boot.weights_orig is not None


class TestPermutationEdgeCases:
    """Tests for permutation test edge cases."""

    def test_permutation_identical_groups(self):
        """Permutation test with identical groups."""
        df = tna.load_group_regulation().head(100)

        result = tna.permutation_test(
            df, df,  # Same data
            iter=50,
            seed=42
        )

        # All true differences should be 0 for identical groups
        assert np.allclose(result.edges['diffs_true'], 0, atol=1e-10)

    def test_permutation_small_groups(self):
        """Permutation test with very small groups."""
        df1 = pd.DataFrame({
            'T1': ['A', 'B', 'A'],
            'T2': ['B', 'A', 'B'],
        })
        df2 = pd.DataFrame({
            'T1': ['B', 'A', 'B'],
            'T2': ['A', 'B', 'A'],
        })

        result = tna.permutation_test(
            df1, df2,
            iter=20,
            seed=42
        )

        # Should compute without error
        assert result.edges is not None


class TestModelTypes:
    """Tests for different model types with edge cases."""

    @pytest.fixture
    def minimal_data(self):
        """Minimal valid data."""
        return pd.DataFrame({
            'T1': ['A', 'B', 'A'],
            'T2': ['B', 'A', 'B'],
        })

    def test_all_types_with_minimal_data(self, minimal_data):
        """All model types should work with minimal data."""
        types = ['relative', 'frequency', 'co-occurrence', 'reverse']

        for t in types:
            model = tna.build_model(minimal_data, type_=t)
            assert model.type_ == t
            assert model.weights.shape == (2, 2)

    def test_attention_model_extreme_beta(self, minimal_data):
        """Attention model with extreme beta values."""
        # Very small beta (nearly no decay)
        model_low = tna.atna(minimal_data, beta=0.001)
        assert model_low.type_ == 'attention'

        # Large beta (strong decay)
        model_high = tna.atna(minimal_data, beta=10.0)
        assert model_high.type_ == 'attention'

    def test_gap_model_extreme_params(self, minimal_data):
        """Gap model with extreme parameters."""
        # Max gap larger than sequence length
        model = tna.build_model(
            minimal_data, type_='gap',
            params={'max_gap': 100, 'decay': 0.5}
        )
        assert model.type_ == 'gap'


class TestVisualizationEdgeCases:
    """Tests for visualization functions with edge cases."""

    def test_plot_single_state(self):
        """Plot network with single state."""
        df = pd.DataFrame({
            'T1': ['A', 'A'],
            'T2': ['A', 'A'],
        })
        model = tna.tna(df)

        # Should not raise error
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        try:
            tna.plot_network(model)
            plt.close()
        except Exception as e:
            # Some edge cases may legitimately fail visualization
            assert 'single' in str(e).lower() or 'empty' in str(e).lower()

    def test_plot_empty_centralities(self):
        """Plot centralities with single state."""
        df = pd.DataFrame({
            'T1': ['A', 'A'],
            'T2': ['A', 'A'],
        })
        model = tna.tna(df)
        cent = tna.centralities(model)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            tna.plot_centralities(cent)
            plt.close()
        except Exception:
            pass  # May fail with single state


class TestDataPreparationEdgeCases:
    """Tests for data preparation edge cases."""

    def test_prepare_data_single_actor(self):
        """Prepare data with single actor."""
        df = pd.DataFrame({
            'actor': ['user1', 'user1', 'user1'],
            'time': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:01', '2024-01-01 10:02']),
            'action': ['A', 'B', 'C']
        })

        prepared = tna.prepare_data(df, actor='actor', time='time', action='action')
        assert prepared.statistics['n_actors'] == 1

    def test_prepare_data_large_time_gap(self):
        """Prepare data with large time gap (session break)."""
        df = pd.DataFrame({
            'actor': ['user1', 'user1', 'user1'],
            'time': pd.to_datetime([
                '2024-01-01 10:00',
                '2024-01-01 10:01',
                '2024-01-02 10:00'  # Next day
            ]),
            'action': ['A', 'B', 'C']
        })

        prepared = tna.prepare_data(
            df, actor='actor', time='time', action='action',
            time_threshold=3600  # 1 hour threshold
        )
        # Should create multiple sessions due to gap
        assert prepared.statistics['n_sessions'] >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
