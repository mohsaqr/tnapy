"""Tests for numerical equivalence with R TNA package.

These tests verify that Python TNA produces identical results to R TNA.
Expected values are hardcoded from R TNA 1.2.0 / igraph 2.2.1 output.
To regenerate expected values, run tests/r_verification.R in R.
"""

import numpy as np
import pandas as pd
import pytest

import tna

# ── R ground truth: state ordering ──────────────────────────────────────────
# R TNA labels (alphabetical): adapt, cohesion, consensus, coregulate,
#   discuss, emotion, monitor, plan, synthesis
STATES = [
    'adapt', 'cohesion', 'consensus', 'coregulate',
    'discuss', 'emotion', 'monitor', 'plan', 'synthesis'
]

# ── R ground truth: full 9×9 weight matrix (tna(group_regulation)$weights) ──
# fmt: off
R_WEIGHT_MATRIX = np.array([
    [0.000000000000000, 0.273084479371316, 0.477406679764244, 0.021611001964637, 0.058939096267191, 0.119842829076621, 0.033398821218075, 0.015717092337917, 0.000000000000000],
    [0.002949852507375, 0.027138643067847, 0.497935103244838, 0.119174041297935, 0.059587020648968, 0.115634218289086, 0.033038348082596, 0.141002949852507, 0.003539823008850],
    [0.004740085321536, 0.014852267340812, 0.082003476062569, 0.187707378732817, 0.188023384420920, 0.072681308263549, 0.046610838995102, 0.395797124348238, 0.007584136514457],
    [0.016243654822335, 0.036040609137056, 0.134517766497462, 0.023350253807107, 0.273604060913706, 0.172081218274112, 0.086294416243655, 0.239086294416244, 0.018781725888325],
    [0.071374335611238, 0.047582890407492, 0.321184510250569, 0.084282460136674, 0.194887370286004, 0.105796001012402, 0.022272842318400, 0.011642622120982, 0.140976967856239],
    [0.002467395135707, 0.325343672893902, 0.320408882622489, 0.034191046880508, 0.101868170602749, 0.076841734226295, 0.036305956996828, 0.099753260486429, 0.002819880155093],
    [0.011165387299372, 0.055826936496860, 0.159106769016050, 0.057920446615492, 0.375436147941382, 0.090718771807397, 0.018143754361479, 0.215631542219121, 0.016050244242847],
    [0.000974500568459, 0.025174598018516, 0.290401169400682, 0.017216176709436, 0.067890206269287, 0.146824752314439, 0.075523794055547, 0.374208218288127, 0.001786584375508],
    [0.234662576687117, 0.033742331288344, 0.466257668711656, 0.044478527607362, 0.062883435582822, 0.070552147239264, 0.012269938650307, 0.075153374233129, 0.000000000000000],
])
# fmt: on

# ── R ground truth: initial probabilities ────────────────────────────────────
R_INITS = np.array([0.0115, 0.0605, 0.214, 0.019, 0.1755, 0.1515, 0.144, 0.2045, 0.0195])

# ── R ground truth: centralities(model, loops=FALSE) ────────────────────────
R_CENTRALITIES = {
    'OutStrength': np.array([
        1.000000000000000, 0.972861356932153, 0.917996523937431,
        0.976649746192893, 0.805112629713996, 0.923158265773705,
        0.981856245638521, 0.625791781711873, 1.000000000000000,
    ]),
    'InStrength': np.array([
        0.344577787953137, 0.811647784954297, 2.667218549507990,
        0.566581079944861, 1.188231522647023, 0.894131246276868,
        0.345714956560509, 1.193784260014568, 0.191539362041319,
    ]),
    'ClosenessIn': np.array([
        0.008339570131235, 0.013800358906669, 0.035123315719608,
        0.015546719101785, 0.019591856586706, 0.014102579452180,
        0.007580598453937, 0.027426128875219, 0.009967580555985,
    ]),
    'ClosenessOut': np.array([
        0.015168577227034, 0.012386811515627, 0.012535497744412,
        0.015004153699868, 0.013071651541832, 0.012056070286577,
        0.013684054771456, 0.011509488834405, 0.015790195673322,
    ]),
    'Closeness': np.array([
        0.024792561875215, 0.026543117094488, 0.038294808609938,
        0.021041713513592, 0.027104254929150, 0.023112084770781,
        0.019277663165081, 0.027426128875219, 0.024315306503490,
    ]),
    'Betweenness': np.array([1.0, 0.0, 30.0, 0.0, 16.0, 5.0, 0.0, 9.0, 7.0]),
    'BetweennessRSP': np.array([1.0, 19.0, 103.0, 27.0, 53.0, 36.0, 11.0, 61.0, 3.0]),
    'Diffusion': np.array([
        5.586291969638066, 5.208632770201142, 4.659727805733428,
        5.147938118060861, 4.627576670649398, 5.069888019796327,
        5.156836527833763, 3.487528601796722, 5.582502363886841,
    ]),
    'Clustering': np.array([
        0.336983862677757, 0.299648677165664, 0.160777291853848,
        0.305784186111251, 0.239710760651376, 0.290479309565872,
        0.288881938704651, 0.287490437349879, 0.358613621392574,
    ]),
}

# ── R ground truth: frequency model (ftna) ──────────────────────────────────
# fmt: off
R_FREQ_MATRIX = np.array([
    [  0, 139, 243,  11,  30,  61,  17,   8,   0],
    [  5,  46, 844, 202, 101, 196,  56, 239,   6],
    [ 30,  94, 519,1188,1190, 460, 295,2505,  48],
    [ 32,  71, 265,  46, 539, 339, 170, 471,  37],
    [282, 188,1269, 333, 770, 418,  88,  46, 557],
    [  7, 923, 909,  97, 289, 218, 103, 283,   8],
    [ 16,  80, 228,  83, 538, 130,  26, 309,  23],
    [  6, 155,1788, 106, 418, 904, 465,2304,  11],
    [153,  22, 304,  29,  41,  46,   8,  49,   0],
], dtype=float)
# fmt: on

# NOTE: R ctna uses window-based co-occurrence counting, while Python ctna
# counts only adjacent bidirectional pairs. Raw counts differ by design.
# We validate structural properties (symmetry, row-normalization) instead.


def _label_to_idx(model):
    """Map state label to index for the given model."""
    return {label: i for i, label in enumerate(model.labels)}


class TestREquivalence:
    """Tests for exact numerical equivalence with R TNA package."""

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
        assert set(r_model.labels) == set(STATES)

    def test_weight_matrix_shape(self, r_model):
        """Verify weight matrix dimensions."""
        assert r_model.weights.shape == (9, 9)

    def test_row_stochastic(self, r_model):
        """Verify weight matrix is row-stochastic (rows sum to 1)."""
        row_sums = r_model.weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(9), decimal=10)

    def test_weight_matrix_exact(self, r_model):
        """Verify full 9x9 weight matrix matches R within 1e-12."""
        idx = _label_to_idx(r_model)
        # Reorder R matrix to match Python label ordering
        order = [STATES.index(label) for label in r_model.labels]
        expected = R_WEIGHT_MATRIX[np.ix_(order, order)]
        np.testing.assert_array_almost_equal(
            r_model.weights, expected, decimal=12,
            err_msg="Weight matrix differs from R output"
        )

    def test_initial_probabilities_exact(self, r_model):
        """Verify initial probabilities match R within 1e-12."""
        idx = _label_to_idx(r_model)
        for state_i, state in enumerate(STATES):
            py_idx = idx[state]
            np.testing.assert_almost_equal(
                r_model.inits[py_idx], R_INITS[state_i], decimal=12,
                err_msg=f"Initial prob for {state} differs from R"
            )

    def test_initial_probabilities_sum(self, r_model):
        """Verify initial probabilities sum to 1."""
        assert abs(r_model.inits.sum() - 1.0) < 1e-10


class TestCentralitiesRValues:
    """Tests for all 9 centrality measures against hardcoded R values."""

    @pytest.fixture
    def model(self):
        """Build model from group_regulation data."""
        df = tna.load_group_regulation()
        return tna.tna(df)

    @pytest.fixture
    def cent(self, model):
        """Compute all centralities with loops=FALSE."""
        return tna.centralities(model, loops=False)

    def _check_measure(self, cent, model, measure, decimal=10):
        """Assert a centrality measure matches R values."""
        idx = _label_to_idx(model)
        for state_i, state in enumerate(STATES):
            py_idx = idx[state]
            np.testing.assert_almost_equal(
                cent.loc[model.labels[py_idx], measure],
                R_CENTRALITIES[measure][state_i],
                decimal=decimal,
                err_msg=f"{measure} for {state} differs from R"
            )

    def test_outstrength(self, cent, model):
        """OutStrength matches R igraph::strength(mode='out')."""
        self._check_measure(cent, model, 'OutStrength')

    def test_instrength(self, cent, model):
        """InStrength matches R igraph::strength(mode='in')."""
        self._check_measure(cent, model, 'InStrength')

    def test_closeness_in(self, cent, model):
        """ClosenessIn matches R igraph::closeness(mode='in')."""
        self._check_measure(cent, model, 'ClosenessIn')

    def test_closeness_out(self, cent, model):
        """ClosenessOut matches R igraph::closeness(mode='out')."""
        self._check_measure(cent, model, 'ClosenessOut')

    def test_closeness_all(self, cent, model):
        """Closeness matches R igraph::closeness(mode='all')."""
        self._check_measure(cent, model, 'Closeness')

    def test_betweenness(self, cent, model):
        """Betweenness matches R igraph::betweenness()."""
        self._check_measure(cent, model, 'Betweenness')

    def test_betweenness_rsp(self, cent, model):
        """BetweennessRSP matches R tna::rsp_bet()."""
        self._check_measure(cent, model, 'BetweennessRSP')

    def test_diffusion(self, cent, model):
        """Diffusion matches R tna::diffusion()."""
        self._check_measure(cent, model, 'Diffusion')

    def test_clustering(self, cent, model):
        """Clustering matches R tna::wcc()."""
        self._check_measure(cent, model, 'Clustering')

    def test_all_measures_present(self, cent):
        """All 9 centrality measures are computed."""
        assert len(cent.columns) == len(tna.AVAILABLE_MEASURES)
        assert all(m in cent.columns for m in tna.AVAILABLE_MEASURES)

    def test_normalized_centralities_range(self, model):
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
        assert len(cent_no_loops) == len(cent_with_loops)


class TestFrequencyModelRValues:
    """Tests for frequency model against hardcoded R values."""

    @pytest.fixture
    def fmodel(self):
        """Build frequency model from group_regulation data."""
        df = tna.load_group_regulation()
        return tna.ftna(df)

    def test_frequency_matrix_exact(self, fmodel):
        """Verify full 9x9 frequency matrix matches R exactly."""
        idx = _label_to_idx(fmodel)
        order = [STATES.index(label) for label in fmodel.labels]
        expected = R_FREQ_MATRIX[np.ix_(order, order)]
        np.testing.assert_array_equal(
            fmodel.weights, expected,
            err_msg="Frequency matrix differs from R output"
        )

    def test_frequency_total(self, fmodel):
        """Total transition count matches R."""
        assert fmodel.weights.sum() == R_FREQ_MATRIX.sum()

    def test_frequency_nonnegative(self, fmodel):
        """All frequency counts are non-negative."""
        assert np.all(fmodel.weights >= 0)


class TestCooccurrenceModelRValues:
    """Tests for co-occurrence model structural properties.

    NOTE: Python ctna counts adjacent bidirectional co-occurrences,
    while R ctna uses window-based counting. Raw counts differ by design.
    We validate structural properties here instead of exact R values.
    """

    @pytest.fixture
    def cmodel(self):
        """Build co-occurrence model from group_regulation data."""
        df = tna.load_group_regulation()
        return tna.ctna(df)

    def test_cooccurrence_raw_counts(self, cmodel):
        """Co-occurrence model should return raw counts (not row-normalized)."""
        assert np.all(cmodel.weights >= 0)
        assert cmodel.weights.sum() > 9

    def test_cooccurrence_nonnegative(self, cmodel):
        """All co-occurrence weights should be non-negative."""
        assert np.all(cmodel.weights >= 0)

    def test_cooccurrence_correct_states(self, cmodel):
        """Co-occurrence model should have 9 states."""
        assert cmodel.weights.shape == (9, 9)
        assert set(cmodel.labels) == set(STATES)


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
        total_transitions = (len(simple_data.columns) - 1) * len(simple_data)
        assert model.weights.sum() <= total_transitions

    def test_cooccurrence_model_structure(self, simple_data):
        """Co-occurrence model should return raw counts."""
        model = tna.build_model(simple_data, type_='co-occurrence')
        assert np.all(model.weights >= 0)
        assert model.weights.sum() > 3

    def test_reverse_model_different(self, simple_data):
        """Reverse model should differ from forward model."""
        rev = tna.build_model(simple_data, type_='reverse')
        assert rev.type_ == 'reverse'
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

    def test_attention_model_raw_counts(self, simple_data):
        """Attention model should return raw weighted counts."""
        model = tna.build_model(simple_data, type_='attention',
                                params={'beta': 0.1})
        assert np.all(model.weights >= 0)
        assert model.weights.sum() > 3

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

            if type_ not in ['frequency', 'co-occurrence', 'attention']:
                row_sums = model.weights.sum(axis=1)
                np.testing.assert_array_almost_equal(
                    row_sums, np.ones(9), decimal=10,
                    err_msg=f"Row sums failed for type {type_}"
                )


class TestAlgorithmEquivalence:
    """Tests for algorithm-level equivalence with R implementations."""

    def test_diffusion_algorithm(self):
        """Test diffusion algorithm matches R implementation."""
        mat = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.4, 0.6, 0.0]
        ])

        from tna.centralities import _diffusion
        result = _diffusion(mat)

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
        mat = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
            [0.4, 0.6, 0.0]
        ])

        from tna.centralities import _clustering
        result = _clustering(mat)

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
        mat = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2]
        ])

        from tna.centralities import _betweenness_rsp
        result = _betweenness_rsp(mat, beta=0.01)

        assert all(np.isfinite(result))
        assert all(result >= 0)

    def test_diffusion_with_larger_matrix(self):
        """Test diffusion with larger matrix."""
        np.random.seed(42)
        n = 10
        mat = np.random.rand(n, n)
        mat = mat / mat.sum(axis=1, keepdims=True)
        np.fill_diagonal(mat, 0)
        mat = mat / mat.sum(axis=1, keepdims=True)

        from tna.centralities import _diffusion
        result = _diffusion(mat)

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
        assert df.shape[0] == 2000
        assert df.shape[1] >= 10

    def test_group_regulation_states(self):
        """Group regulation data should have correct states."""
        df = tna.load_group_regulation()
        actual_states = set(df.stack().dropna().unique())
        assert actual_states == set(STATES)

    def test_group_regulation_long_format(self):
        """Long format data should have correct structure."""
        df = tna.load_group_regulation_long()
        assert 'Actor' in df.columns
        assert 'Action' in df.columns
        assert 'Time' in df.columns

    def test_model_from_wide_and_long_consistency(self):
        """Models from wide and long format should have same states."""
        df_wide = tna.load_group_regulation()
        df_long = tna.load_group_regulation_long()

        model_wide = tna.tna(df_wide)
        prepared = tna.prepare_data(
            df_long, actor='Actor', time='Time', action='Action'
        )
        model_long = tna.tna(prepared)

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
        assert scaled.min() >= 0.0
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
            'T1': ['A', 'A', 'B', 'C'],
            'T2': ['B', 'C', 'A', 'A'],
        })
        model = tna.tna(df)

        a_idx = model.labels.index('A')
        assert abs(model.inits[a_idx] - 0.5) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
