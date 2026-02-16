"""Tests for cluster_sequences() and import_onehot()."""

import numpy as np
import pandas as pd
import pytest

from tna.cluster import (
    ClusterResult,
    _hamming_distance,
    _lcs_distance,
    _levenshtein_distance,
    _osa_distance,
    cluster_sequences,
)
from tna.prepare import TNAData, import_onehot


# ======================================================================
# Distance function tests
# ======================================================================


class TestHammingDistance:
    def test_identical(self):
        assert _hamming_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_all_different(self):
        assert _hamming_distance(["A", "B", "C"], ["D", "E", "F"]) == 3.0

    def test_partial_mismatch(self):
        assert _hamming_distance(["A", "B", "C"], ["A", "X", "C"]) == 1.0

    def test_different_lengths(self):
        # Shorter sequence padded with sentinel → extra position counts as mismatch
        assert _hamming_distance(["A", "B"], ["A", "B", "C"]) == 1.0

    def test_weighted(self):
        # Position 0 match, position 1 mismatch → exp(-1*1) ≈ 0.3679
        d = _hamming_distance(["A", "B"], ["A", "X"], weighted=True, lambda_=1.0)
        assert abs(d - np.exp(-1.0)) < 1e-10

    def test_weighted_first_position(self):
        # Position 0 mismatch → exp(-1*0) = 1.0
        d = _hamming_distance(["X"], ["Y"], weighted=True, lambda_=1.0)
        assert abs(d - 1.0) < 1e-10


class TestLevenshteinDistance:
    def test_identical(self):
        assert _levenshtein_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_single_insert(self):
        assert _levenshtein_distance(["A", "B"], ["A", "X", "B"]) == 1.0

    def test_single_delete(self):
        assert _levenshtein_distance(["A", "X", "B"], ["A", "B"]) == 1.0

    def test_single_substitute(self):
        assert _levenshtein_distance(["A", "B", "C"], ["A", "X", "C"]) == 1.0

    def test_empty_vs_nonempty(self):
        assert _levenshtein_distance([], ["A", "B", "C"]) == 3.0
        assert _levenshtein_distance(["A", "B"], []) == 2.0

    def test_classic_example(self):
        # "kitten" → "sitting" = 3 edits
        a = list("kitten")
        b = list("sitting")
        assert _levenshtein_distance(a, b) == 3.0


class TestOSADistance:
    def test_identical(self):
        assert _osa_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_transposition(self):
        # Adjacent swap: AB → BA = 1 (transposition)
        assert _osa_distance(["A", "B"], ["B", "A"]) == 1.0

    def test_substitution(self):
        assert _osa_distance(["A", "B", "C"], ["A", "X", "C"]) == 1.0

    def test_osa_vs_levenshtein(self):
        # OSA should be ≤ Levenshtein (transposition = 1 op instead of 2)
        a = ["C", "A"]
        b = ["A", "C"]
        assert _osa_distance(a, b) <= _levenshtein_distance(a, b)

    def test_empty(self):
        assert _osa_distance([], ["A", "B"]) == 2.0
        assert _osa_distance(["A"], []) == 1.0


class TestLCSDistance:
    def test_identical(self):
        assert _lcs_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_no_common(self):
        # LCS=0, dist = max(3,3) - 0 = 3 (R TNA formula)
        assert _lcs_distance(["A", "B", "C"], ["D", "E", "F"]) == 3.0

    def test_partial(self):
        # "ABCD" vs "ACBD": LCS = "ABD" (len 3), dist = max(4,4) - 3 = 1
        a = ["A", "B", "C", "D"]
        b = ["A", "C", "B", "D"]
        assert _lcs_distance(a, b) == 1.0

    def test_one_empty(self):
        assert _lcs_distance([], ["A", "B"]) == 2.0
        assert _lcs_distance(["A"], []) == 1.0

    def test_subsequence(self):
        # "AC" is subsequence of "ABC" → LCS=2, dist = max(2,3) - 2 = 1
        assert _lcs_distance(["A", "C"], ["A", "B", "C"]) == 1.0


# ======================================================================
# Clustering tests
# ======================================================================


@pytest.fixture
def simple_sequences():
    """Two clear clusters: AAA-like and BBB-like."""
    return pd.DataFrame({
        "s1": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "s2": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "s3": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })


class TestClusterSequencesPAM:
    def test_basic(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        assert isinstance(result, ClusterResult)
        assert result.k == 2
        assert len(result.assignments) == 8
        assert set(result.assignments) == {1, 2}

    def test_cluster_separation(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        # First 4 should be in one cluster, last 4 in another
        assert len(set(result.assignments[:4])) == 1
        assert len(set(result.assignments[4:])) == 1
        assert result.assignments[0] != result.assignments[4]

    def test_silhouette_range(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        assert -1.0 <= result.silhouette <= 1.0

    def test_sizes_sum(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        assert result.sizes.sum() == 8

    def test_distance_matrix_shape(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        assert result.distance.shape == (8, 8)

    def test_distance_matrix_symmetric(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        np.testing.assert_array_equal(result.distance, result.distance.T)

    def test_distance_matrix_zero_diagonal(self, simple_sequences):
        result = cluster_sequences(simple_sequences, k=2, method="pam")
        np.testing.assert_array_equal(np.diag(result.distance), 0)


class TestClusterSequencesHierarchical:
    @pytest.mark.parametrize("method", [
        "ward.D", "ward.D2", "complete", "average", "single",
        "mcquitty", "median", "centroid",
    ])
    def test_all_methods(self, simple_sequences, method):
        result = cluster_sequences(
            simple_sequences, k=2, method=method
        )
        assert result.k == 2
        assert len(result.assignments) == 8
        assert result.method == method

    def test_complete_separation(self, simple_sequences):
        result = cluster_sequences(
            simple_sequences, k=2, method="complete"
        )
        assert len(set(result.assignments[:4])) == 1
        assert len(set(result.assignments[4:])) == 1


class TestClusterSequencesDissimilarity:
    @pytest.mark.parametrize("dissimilarity", ["hamming", "lv", "osa", "lcs"])
    def test_all_metrics(self, simple_sequences, dissimilarity):
        result = cluster_sequences(
            simple_sequences, k=2, dissimilarity=dissimilarity
        )
        assert result.dissimilarity == dissimilarity
        assert result.k == 2

    def test_weighted_hamming(self, simple_sequences):
        result = cluster_sequences(
            simple_sequences, k=2, dissimilarity="hamming",
            weighted=True, lambda_=0.5,
        )
        assert result.k == 2


class TestClusterSequencesInput:
    def test_tnadata_input(self):
        """Accept TNAData objects."""
        seq_df = pd.DataFrame({
            "a1": ["X", "X", "Y", "Y"],
            "a2": ["X", "X", "Y", "Y"],
        })
        tna_data = TNAData(
            long_data=pd.DataFrame(),
            sequence_data=seq_df,
            meta_data=pd.DataFrame(),
        )
        result = cluster_sequences(tna_data, k=2)
        assert result.k == 2
        assert len(result.assignments) == 4

    def test_invalid_dissimilarity(self, simple_sequences):
        with pytest.raises(ValueError, match="Unknown dissimilarity"):
            cluster_sequences(simple_sequences, k=2, dissimilarity="invalid")

    def test_invalid_method(self, simple_sequences):
        with pytest.raises(ValueError, match="Unknown method"):
            cluster_sequences(simple_sequences, k=2, method="invalid")

    def test_k_too_small(self, simple_sequences):
        with pytest.raises(ValueError, match="k must be >= 2"):
            cluster_sequences(simple_sequences, k=1)

    def test_k_too_large(self, simple_sequences):
        with pytest.raises(ValueError, match="exceeds"):
            cluster_sequences(simple_sequences, k=100)

    def test_na_handling(self):
        df = pd.DataFrame({
            "s1": ["A", "A", "*", "*"],
            "s2": ["A", "A", "%", "%"],
        })
        result = cluster_sequences(df, k=2)
        assert result.k == 2


# ======================================================================
# import_onehot tests
# ======================================================================


class TestImportOnehot:
    def test_basic_r_format(self):
        """R format: 1 row per group, W{w}_T{t} columns."""
        df = pd.DataFrame({
            "A": [1, 0, 0, 1, 0],
            "B": [0, 1, 0, 0, 1],
            "C": [0, 0, 1, 0, 0],
        })
        result = import_onehot(df, cols=["A", "B", "C"])
        # No actor → all rows in 1 group → 1 output row
        # 5 windows of size 1, 3 cols each → 15 columns
        assert result.shape == (1, 15)
        assert result["W0_T1"].iloc[0] == "A"
        assert pd.isna(result["W0_T2"].iloc[0])
        assert pd.isna(result["W1_T1"].iloc[0])
        assert result["W1_T2"].iloc[0] == "B"
        assert result["W2_T3"].iloc[0] == "C"
        assert result["W3_T1"].iloc[0] == "A"
        assert result["W4_T2"].iloc[0] == "B"

    def test_multiple_active(self):
        """Multiple one-hot columns active in same row → multiple slots."""
        df = pd.DataFrame({
            "A": [1, 0],
            "B": [1, 0],
            "C": [0, 1],
        })
        result = import_onehot(df, cols=["A", "B", "C"])
        # 1 group, 2 windows of size 1, 3 cols each → 6 columns
        assert result.shape == (1, 6)
        assert result["W0_T1"].iloc[0] == "A"
        assert result["W0_T2"].iloc[0] == "B"
        assert pd.isna(result["W0_T3"].iloc[0])
        assert pd.isna(result["W1_T1"].iloc[0])
        assert pd.isna(result["W1_T2"].iloc[0])
        assert result["W1_T3"].iloc[0] == "C"

    def test_tumbling_window(self):
        """Tumbling: non-overlapping windows expand rows × cols."""
        df = pd.DataFrame({
            "A": [1, 0, 1],
            "B": [0, 1, 0],
        })
        result = import_onehot(df, cols=["A", "B"], window_size=2)
        # 1 group, 2 tumbling windows: W0(2rows×2cols=4), W1(1row×2cols=2) → 6 cols
        assert result.shape == (1, 6)
        # W0: row0(A=A,B=NA), row1(A=NA,B=B)
        assert result["W0_T1"].iloc[0] == "A"
        assert pd.isna(result["W0_T2"].iloc[0])
        assert pd.isna(result["W0_T3"].iloc[0])
        assert result["W0_T4"].iloc[0] == "B"
        # W1: row2(A=A,B=NA)
        assert result["W1_T1"].iloc[0] == "A"
        assert pd.isna(result["W1_T2"].iloc[0])

    def test_sliding_window(self):
        """Sliding: overlapping windows, collapsed to first non-NA per col."""
        df = pd.DataFrame({
            "A": [1, 0, 1],
            "B": [0, 1, 0],
        })
        result = import_onehot(
            df, cols=["A", "B"], window_size=2, window_type="sliding"
        )
        # 2 sliding windows (3-2+1=2), each collapsed to 2 slots → 4 cols
        assert result.shape == (1, 4)
        # R: W0_T1=A, W0_T2=B, W1_T1=A, W1_T2=B
        assert result["W0_T1"].iloc[0] == "A"
        assert result["W0_T2"].iloc[0] == "B"
        assert result["W1_T1"].iloc[0] == "A"
        assert result["W1_T2"].iloc[0] == "B"

    def test_aggregate(self):
        """Aggregate collapses tumbling windows to first non-NA per col."""
        df = pd.DataFrame({
            "A": [1, 1, 0],
            "B": [0, 0, 1],
        })
        # Without aggregate: 1 window of 3 rows × 2 cols = 6 slots
        result_no_agg = import_onehot(
            df, cols=["A", "B"], window_size=3, aggregate=False
        )
        assert result_no_agg.shape == (1, 6)
        # R: W0_T1=A, W0_T2=NA, W0_T3=A, W0_T4=NA, W0_T5=NA, W0_T6=B
        assert result_no_agg["W0_T1"].iloc[0] == "A"
        assert result_no_agg["W0_T6"].iloc[0] == "B"

        # With aggregate: first non-NA per col → 2 slots
        result_agg = import_onehot(
            df, cols=["A", "B"], window_size=3, aggregate=True
        )
        assert result_agg.shape == (1, 2)
        # R: W0_T1=A, W0_T2=B
        assert result_agg["W0_T1"].iloc[0] == "A"
        assert result_agg["W0_T2"].iloc[0] == "B"

    def test_actor_grouping(self):
        """Each actor becomes one row."""
        df = pd.DataFrame({
            "user": ["u1", "u1", "u2", "u2"],
            "A": [1, 0, 0, 1],
            "B": [0, 1, 1, 0],
        })
        result = import_onehot(df, cols=["A", "B"], actor="user", window_size=2)
        # 2 actors, each with 1 tumbling window of 2 rows × 2 cols = 4 slots
        assert result.shape == (2, 4)
        # u1: row0(A=A,B=NA), row1(A=NA,B=B)
        assert result["W0_T1"].iloc[0] == "A"
        assert result["W0_T4"].iloc[0] == "B"
        # u2: row0(A=NA,B=B), row1(A=A,B=NA)
        assert pd.isna(result["W0_T1"].iloc[1])
        assert result["W0_T2"].iloc[1] == "B"
        assert result["W0_T3"].iloc[1] == "A"

    def test_session_grouping(self):
        """Each session becomes one row."""
        df = pd.DataFrame({
            "sess": ["s1", "s1", "s2", "s2"],
            "X": [1, 0, 1, 0],
            "Y": [0, 1, 0, 1],
        })
        result = import_onehot(
            df, cols=["X", "Y"], session="sess", window_size=2
        )
        # 2 sessions, each 1 window of 2 rows × 2 cols = 4 slots
        assert result.shape == (2, 4)

    def test_sliding_with_actor(self):
        """Sliding windows within actors, padded columns across groups."""
        df = pd.DataFrame({
            "user": ["u1", "u1", "u1", "u2", "u2"],
            "A": [1, 0, 1, 0, 1],
            "B": [0, 1, 0, 1, 0],
        })
        result = import_onehot(
            df, cols=["A", "B"], actor="user",
            window_size=2, window_type="sliding",
        )
        # u1: 3 rows → 2 sliding windows, each 2 slots → 4 cols
        # u2: 2 rows → 1 sliding window, 2 slots → 2 cols (padded to 4)
        assert result.shape == (2, 4)
        # u1: W0=[A,B], W1=[A,B]
        assert result["W0_T1"].iloc[0] == "A"
        assert result["W0_T2"].iloc[0] == "B"
        # u2: W0=[A,B], W1 is NaN (padded)
        assert result["W0_T1"].iloc[1] == "A"
        assert result["W0_T2"].iloc[1] == "B"
        assert pd.isna(result["W1_T1"].iloc[1])

    def test_missing_columns(self):
        df = pd.DataFrame({"A": [1, 0]})
        with pytest.raises(ValueError, match="Columns not found"):
            import_onehot(df, cols=["A", "Z"])

    def test_invalid_values(self):
        df = pd.DataFrame({"A": [1, 2], "B": [0, 1]})
        with pytest.raises(ValueError, match="only 0 and 1"):
            import_onehot(df, cols=["A", "B"])

    def test_invalid_window_type(self):
        df = pd.DataFrame({"A": [1, 0], "B": [0, 1]})
        with pytest.raises(ValueError, match="Unknown window_type"):
            import_onehot(df, cols=["A", "B"], window_type="bad")

    def test_compatible_with_tna(self):
        """Output should be usable by tna.tna()."""
        from tna import tna as tna_func

        df = pd.DataFrame({
            "user": [f"u{i}" for i in range(10) for _ in range(5)],
            "A": [1, 0, 1, 0, 1] * 10,
            "B": [0, 1, 0, 1, 0] * 10,
        })
        seq = import_onehot(df, cols=["A", "B"], actor="user")
        # 10 actors → 10 rows, each a sequence
        model = tna_func(seq)
        assert model.weights.shape[0] == model.weights.shape[1]
