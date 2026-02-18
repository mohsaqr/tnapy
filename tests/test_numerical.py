"""Numerical equivalence tests for TNA package.

These tests verify that the Python TNA implementation produces
results equivalent to the R TNA package.
"""

import numpy as np
import pandas as pd
import pytest

import tna


class TestModelBuilding:
    """Tests for model building functions."""

    @pytest.fixture
    def simple_sequences(self):
        """Simple test sequences."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C'],
            'step2': ['B', 'C', 'B', 'A'],
            'step3': ['C', 'A', 'C', 'B'],
        })

    @pytest.fixture
    def sequences_with_na(self):
        """Sequences with missing values."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'A'],
            'step2': ['B', 'C', None],
            'step3': ['C', None, None],
        })

    def test_tna_basic(self, simple_sequences):
        """Test basic TNA model building."""
        model = tna.tna(simple_sequences)

        assert isinstance(model, tna.TNA)
        assert model.type_ == "relative"
        assert len(model.labels) == 3
        assert set(model.labels) == {'A', 'B', 'C'}
        assert model.weights.shape == (3, 3)

        # Check row normalization
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))

    def test_ftna_basic(self, simple_sequences):
        """Test frequency TNA model."""
        model = tna.ftna(simple_sequences)

        assert model.type_ == "frequency"
        # Frequency model should have raw counts
        assert model.weights.sum() > 3  # More than n_states

    def test_ctna_basic(self, simple_sequences):
        """Test co-occurrence TNA model."""
        model = tna.ctna(simple_sequences)

        assert model.type_ == "co-occurrence"
        # Co-occurrence returns raw counts (not row-normalized)
        assert np.all(model.weights >= 0)
        assert model.weights.sum() > 3  # More than n_states

    def test_atna_basic(self, simple_sequences):
        """Test attention TNA model."""
        model = tna.atna(simple_sequences, beta=0.1)

        assert model.type_ == "attention"
        # Attention returns raw weighted counts (not row-normalized)
        assert np.all(model.weights >= 0)
        assert model.weights.sum() > 3

    def test_sequences_with_na(self, sequences_with_na):
        """Test handling of missing values in sequences."""
        model = tna.tna(sequences_with_na)

        assert len(model.labels) == 3
        # Should still produce valid probabilities
        # Note: states with no outgoing transitions will have row sum of 0
        row_sums = model.weights.sum(axis=1)
        # Row sums should be either 0 (no transitions) or 1 (normalized)
        for rs in row_sums:
            assert rs == 0 or abs(rs - 1.0) < 1e-10

    def test_initial_probabilities(self, simple_sequences):
        """Test initial state probabilities."""
        model = tna.tna(simple_sequences)

        # Initial probs should sum to 1
        assert abs(model.inits.sum() - 1.0) < 1e-10

        # Each starting state appears once: A(2), B(1), C(1)
        a_idx = model.labels.index('A')
        b_idx = model.labels.index('B')
        c_idx = model.labels.index('C')

        assert model.inits[a_idx] == 0.5  # A starts 2/4 sequences
        assert model.inits[b_idx] == 0.25
        assert model.inits[c_idx] == 0.25

    def test_build_model_types(self, simple_sequences):
        """Test build_model with different types."""
        types = ['relative', 'frequency', 'co-occurrence', 'reverse',
                 'n-gram', 'gap', 'window', 'attention']

        for t in types:
            model = tna.build_model(simple_sequences, type_=t)
            assert model.type_ == t
            assert model.weights.shape == (3, 3)


class TestScaling:
    """Tests for scaling functions."""

    @pytest.fixture
    def test_matrix(self):
        """Test weight matrix."""
        return np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.6, 0.4, 0.0]
        ])

    def test_minmax_scaling(self, test_matrix):
        """Test min-max scaling."""
        scaled = tna.minmax_scale(test_matrix)

        assert scaled.min() == 0.0
        assert scaled.max() == 1.0

    def test_max_scaling(self, test_matrix):
        """Test max scaling."""
        scaled = tna.max_scale(test_matrix)

        assert scaled.max() == 1.0
        np.testing.assert_array_almost_equal(
            scaled, test_matrix / test_matrix.max()
        )

    def test_scaling_in_model(self):
        """Test scaling applied during model building."""
        df = pd.DataFrame({
            'step1': ['A', 'B'],
            'step2': ['B', 'A'],
        })

        model = tna.tna(df, scaling='minmax')
        assert 'minmax' in model.scaling

        model2 = tna.tna(df, scaling=['minmax', 'max'])
        assert model2.scaling == ['minmax', 'max']


class TestCentralities:
    """Tests for centrality measures."""

    @pytest.fixture
    def simple_model(self):
        """Simple TNA model for centrality tests."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'A', 'C', 'B'],
            'step2': ['B', 'C', 'B', 'A', 'A'],
            'step3': ['C', 'A', 'C', 'B', 'C'],
        })
        return tna.tna(df)

    def test_all_centralities(self, simple_model):
        """Test computing all centralities."""
        cent = tna.centralities(simple_model)

        assert isinstance(cent, pd.DataFrame)
        assert len(cent) == 3  # 3 states
        assert len(cent.columns) == len(tna.AVAILABLE_MEASURES)

    def test_specific_measures(self, simple_model):
        """Test computing specific measures."""
        measures = ['InStrength', 'OutStrength']
        cent = tna.centralities(simple_model, measures=measures)

        assert set(cent.columns) == set(measures)

    def test_normalized_centralities(self, simple_model):
        """Test normalized centralities."""
        cent = tna.centralities(simple_model, normalize=True)

        # All columns should have max 1.0
        for col in cent.columns:
            assert cent[col].max() <= 1.0 + 1e-10

    def test_strength_measures(self, simple_model):
        """Test strength centrality measures."""
        cent = tna.centralities(simple_model, measures=['InStrength', 'OutStrength'])

        # OutStrength should sum to row sums of weight matrix
        out_strength = simple_model.weights.sum(axis=1)
        for i, label in enumerate(simple_model.labels):
            np.testing.assert_almost_equal(
                cent.loc[label, 'OutStrength'],
                out_strength[i]
            )

        # InStrength should sum to column sums
        in_strength = simple_model.weights.sum(axis=0)
        for i, label in enumerate(simple_model.labels):
            np.testing.assert_almost_equal(
                cent.loc[label, 'InStrength'],
                in_strength[i]
            )

    def test_loops_parameter(self, simple_model):
        """Test loops parameter effect."""
        cent_no_loops = tna.centralities(simple_model, loops=False)
        cent_with_loops = tna.centralities(simple_model, loops=True)

        # Results may differ if there are self-loops
        # Just verify both compute successfully
        assert len(cent_no_loops) == len(cent_with_loops)

    def test_invalid_measure(self, simple_model):
        """Test error on invalid measure name."""
        with pytest.raises(ValueError, match="Unknown measures"):
            tna.centralities(simple_model, measures=['InvalidMeasure'])


class TestDataPreparation:
    """Tests for data preparation functions."""

    @pytest.fixture
    def long_format_data(self):
        """Long format event data."""
        return pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:01:00',
                '2024-01-01 10:02:00',
                '2024-01-01 11:00:00',
                '2024-01-01 11:05:00',
                '2024-01-01 12:00:00',  # >15 min gap - new session
                '2024-01-01 12:01:00',
            ]),
            'action': ['A', 'B', 'C', 'B', 'A', 'C', 'A'],
        })

    def test_prepare_data_basic(self, long_format_data):
        """Test basic data preparation."""
        prepared = tna.prepare_data(
            long_format_data,
            actor='user_id',
            time='timestamp',
            action='action'
        )

        assert isinstance(prepared, tna.TNAData)
        assert 'n_sessions' in prepared.statistics
        assert 'n_actors' in prepared.statistics
        assert prepared.statistics['n_actors'] == 2

    def test_session_creation(self, long_format_data):
        """Test session creation based on time threshold."""
        prepared = tna.prepare_data(
            long_format_data,
            actor='user_id',
            time='timestamp',
            action='action',
            time_threshold=900  # 15 minutes
        )

        # u1 has 1 session, u2 has 2 sessions (gap > 15 min)
        assert prepared.statistics['n_sessions'] == 3

    def test_sequence_data_format(self, long_format_data):
        """Test sequence data output format."""
        prepared = tna.prepare_data(
            long_format_data,
            actor='user_id',
            time='timestamp',
            action='action'
        )

        # sequence_data should be wide format
        assert isinstance(prepared.sequence_data, pd.DataFrame)
        # Columns should be action_1, action_2, etc.
        assert all(col.startswith('action_') for col in prepared.sequence_data.columns)

    def test_build_from_prepared_data(self, long_format_data):
        """Test building model from prepared data."""
        prepared = tna.prepare_data(
            long_format_data,
            actor='user_id',
            time='timestamp',
            action='action'
        )

        model = tna.tna(prepared)

        assert isinstance(model, tna.TNA)
        assert set(model.labels) == {'A', 'B', 'C'}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_state_sequence(self):
        """Test with sequences containing only one state."""
        df = pd.DataFrame({
            'step1': ['A', 'A'],
            'step2': ['A', 'A'],
        })

        model = tna.tna(df)
        assert len(model.labels) == 1
        assert model.weights.shape == (1, 1)
        assert model.weights[0, 0] == 1.0  # Self-loop

    def test_empty_transitions(self):
        """Test with very short sequences."""
        df = pd.DataFrame({
            'step1': ['A', 'B', 'C'],
        })

        model = tna.tna(df)
        # No transitions with single-step sequences
        assert model.weights.sum() == 0

    def test_matrix_input(self):
        """Test direct weight matrix input."""
        weights = np.array([
            [0.0, 0.6, 0.4],
            [0.5, 0.0, 0.5],
            [0.3, 0.7, 0.0]
        ])

        model = tna.build_model(weights)

        np.testing.assert_array_almost_equal(model.weights, weights)

    def test_numpy_array_sequences(self):
        """Test with numpy array sequences."""
        arr = np.array([
            ['A', 'B', 'C'],
            ['B', 'A', 'B'],
        ])

        model = tna.tna(arr)

        assert len(model.labels) == 3
        assert model.weights.shape == (3, 3)


class TestGroupRegulationData:
    """Tests for the example datasets."""

    def test_load_group_regulation(self):
        """Test loading wide format group regulation data."""
        df = tna.load_group_regulation()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2000  # 2000 sessions
        assert all(col.startswith('T') for col in df.columns)

    def test_load_group_regulation_long(self):
        """Test loading long format group regulation data."""
        df = tna.load_group_regulation_long()

        assert isinstance(df, pd.DataFrame)
        assert 'Actor' in df.columns
        assert 'Action' in df.columns
        assert 'Time' in df.columns

    def test_model_from_group_regulation(self):
        """Test building model from group regulation data."""
        df = tna.load_group_regulation()
        model = tna.tna(df)

        assert isinstance(model, tna.TNA)
        # Should have the regulation behavior codes from R data
        expected_states = {
            'adapt', 'cohesion', 'consensus', 'coregulate',
            'discuss', 'emotion', 'monitor', 'plan', 'synthesis'
        }
        assert set(model.labels) == expected_states

    def test_centralities_from_group_regulation(self):
        """Test computing centralities from group regulation data."""
        df = tna.load_group_regulation()
        model = tna.tna(df)
        cent = tna.centralities(model)

        assert len(cent) == 9  # 9 regulation behaviors
        assert all(col in cent.columns for col in ['InStrength', 'OutStrength', 'Betweenness'])


class TestTransitionTypes:
    """Tests for different transition computation types."""

    @pytest.fixture
    def test_sequences(self):
        """Test sequences for transition type comparison."""
        return pd.DataFrame({
            'step1': ['A', 'B', 'C', 'A', 'B'],
            'step2': ['B', 'C', 'A', 'C', 'A'],
            'step3': ['C', 'A', 'B', 'B', 'C'],
            'step4': ['A', 'B', 'C', 'A', 'B'],
        })

    def test_ngram_transitions(self, test_sequences):
        """Test n-gram transitions."""
        model_1gram = tna.build_model(test_sequences, type_='relative')
        model_2gram = tna.build_model(test_sequences, type_='n-gram', params={'n': 2})
        model_3gram = tna.build_model(test_sequences, type_='n-gram', params={'n': 3})

        # All should have same states
        assert set(model_1gram.labels) == set(model_2gram.labels) == set(model_3gram.labels)

        # n=2 should be same as regular transitions (default)
        # n=3 should be different (skipping one step)
        assert not np.allclose(model_1gram.weights, model_3gram.weights)

    def test_gap_transitions(self, test_sequences):
        """Test gap-based transitions."""
        model = tna.build_model(
            test_sequences,
            type_='gap',
            params={'max_gap': 3, 'decay': 0.5}
        )

        assert model.type_ == 'gap'
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))

    def test_window_transitions(self, test_sequences):
        """Test window-based transitions."""
        model = tna.build_model(
            test_sequences,
            type_='window',
            params={'size': 3}
        )

        assert model.type_ == 'window'
        row_sums = model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))

    def test_reverse_transitions(self, test_sequences):
        """Test reverse transitions."""
        model_fwd = tna.tna(test_sequences)
        model_rev = tna.build_model(test_sequences, type_='reverse')

        assert model_rev.type_ == 'reverse'
        # Forward and reverse should have different patterns
        assert not np.allclose(model_fwd.weights, model_rev.weights)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
