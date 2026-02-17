"""Tests for compare_sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tna


# Reference files for regression tests
TEST_DATA_DIR = Path(__file__).parent / "data"
REF_CS_NOTEST = TEST_DATA_DIR / "r_cs_notest.csv"
REF_CS_CUSTOM = TEST_DATA_DIR / "r_cs_custom.csv"


@pytest.fixture
def group_model():
    """Build a GroupTNA model from the group regulation dataset."""
    prep = tna.prepare_data(
        tna.load_group_regulation_long(),
        actor="Actor",
        action="Action",
        time="Time",
    )
    return tna.group_tna(prep, group="Achiever")


class TestCompareSequencesNoTest:
    """Tests for compare_sequences without permutation test."""

    def test_shape_and_columns(self, group_model):
        res = tna.compare_sequences(group_model)
        assert res.shape == (918, 5)
        assert list(res.columns) == [
            "pattern",
            "freq_High",
            "freq_Low",
            "prop_High",
            "prop_Low",
        ]

    def test_frequencies_match_reference(self, group_model):
        """Verify frequencies match reference data exactly."""
        res = tna.compare_sequences(group_model)
        ref = pd.read_csv(REF_CS_NOTEST)
        merged = pd.merge(res, ref, on="pattern", suffixes=("_py", "_ref"))
        assert len(merged) == 918
        assert (merged["freq_High_py"] == merged["freq_High_ref"]).all()
        assert (merged["freq_Low_py"] == merged["freq_Low_ref"]).all()

    def test_proportions_match_reference(self, group_model):
        """Verify proportions match reference data to machine epsilon."""
        res = tna.compare_sequences(group_model)
        ref = pd.read_csv(REF_CS_NOTEST)
        merged = pd.merge(res, ref, on="pattern", suffixes=("_py", "_ref"))
        np.testing.assert_allclose(
            merged["prop_High_py"].values,
            merged["prop_High_ref"].values,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            merged["prop_Low_py"].values,
            merged["prop_Low_ref"].values,
            atol=1e-12,
        )

    def test_order_matches_reference(self, group_model):
        """Output order matches reference: by length then alphabetical."""
        res = tna.compare_sequences(group_model)
        ref = pd.read_csv(REF_CS_NOTEST)
        assert (res["pattern"].values == ref["pattern"].values).all()

    def test_unigram_proportions_sum_to_one(self, group_model):
        """Unigram proportions should sum to 1 per group."""
        res = tna.compare_sequences(group_model)
        unigrams = res[~res["pattern"].str.contains("->")]
        np.testing.assert_allclose(unigrams["prop_High"].sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(unigrams["prop_Low"].sum(), 1.0, atol=1e-12)

    def test_custom_sub_and_min_freq(self, group_model):
        """Match reference with sub=1:3 and min_freq=10."""
        res = tna.compare_sequences(group_model, sub=range(1, 4), min_freq=10)
        ref = pd.read_csv(REF_CS_CUSTOM)
        assert res.shape == ref.shape
        merged = pd.merge(res, ref, on="pattern", suffixes=("_py", "_ref"))
        assert len(merged) == len(res)
        assert (merged["freq_High_py"] == merged["freq_High_ref"]).all()
        assert (merged["freq_Low_py"] == merged["freq_Low_ref"]).all()


class TestCompareSequencesWithTest:
    """Tests for compare_sequences with permutation test."""

    def test_test_output_columns(self, group_model):
        res = tna.compare_sequences(
            group_model, test=True, iter_=10, seed=42
        )
        assert "effect_size" in res.columns
        assert "p_value" in res.columns
        assert res.shape[1] == 7

    def test_test_output_shape(self, group_model):
        res = tna.compare_sequences(
            group_model, test=True, iter_=10, seed=42
        )
        assert res.shape[0] == 918

    def test_sorted_by_pvalue(self, group_model):
        res = tna.compare_sequences(
            group_model, test=True, iter_=10, seed=42
        )
        assert (res["p_value"].diff().dropna() >= -1e-15).all()

    def test_pvalues_in_range(self, group_model):
        res = tna.compare_sequences(
            group_model, test=True, iter_=10, seed=42
        )
        assert (res["p_value"] >= 0).all()
        assert (res["p_value"] <= 1).all()

    def test_bonferroni_unigram_pvalue(self, group_model):
        """With 9 unigrams and all raw p = 1/(iter+1), Bonferroni = 9/(iter+1)."""
        res = tna.compare_sequences(
            group_model, test=True, iter_=100, seed=42
        )
        unigram_p = res[~res["pattern"].str.contains("->")]["p_value"]
        # 9 unigrams, raw p = 1/101, bonferroni = 9/101
        expected = 9 / 101
        # At least some unigrams should have this p-value
        assert any(abs(p - expected) < 1e-10 for p in unigram_p)

    def test_frequencies_unchanged_by_test(self, group_model):
        """Frequencies should be identical whether test is run or not."""
        res_notest = tna.compare_sequences(group_model)
        res_test = tna.compare_sequences(
            group_model, test=True, iter_=10, seed=42
        )
        merged = pd.merge(
            res_notest, res_test, on="pattern", suffixes=("_no", "_t")
        )
        assert (merged["freq_High_no"] == merged["freq_High_t"]).all()
        assert (merged["freq_Low_no"] == merged["freq_Low_t"]).all()


class TestCompareSequencesEdgeCases:
    """Edge case tests."""

    def test_min_freq_too_high(self, group_model):
        """Very high min_freq should return empty DataFrame."""
        res = tna.compare_sequences(group_model, min_freq=100000)
        assert len(res) == 0
        assert "pattern" in res.columns

    def test_sub_single_length(self, group_model):
        """Only unigrams."""
        res = tna.compare_sequences(group_model, sub=[1])
        assert all("->" not in p for p in res["pattern"])
        assert len(res) == 9  # 9 unique actions

    def test_sub_bigrams_only(self, group_model):
        """Only bigrams."""
        res = tna.compare_sequences(group_model, sub=[2])
        assert all("->" in p for p in res["pattern"])
        assert all(p.count("->") == 1 for p in res["pattern"])

    def test_invalid_input(self):
        """Should raise TypeError for non-GroupTNA input."""
        with pytest.raises(TypeError):
            tna.compare_sequences(pd.DataFrame({"a": [1, 2, 3]}))

    def test_single_group_error(self):
        """Should raise ValueError for single group."""
        # Build a model with one group
        prep = tna.prepare_data(
            tna.load_group_regulation_long(),
            actor="Actor",
            action="Action",
            time="Time",
        )
        from tna.group import GroupTNA
        gm = tna.group_tna(prep, group="Achiever")
        single = GroupTNA(models={"High": gm["High"]})
        with pytest.raises(ValueError, match="at least 2 groups"):
            tna.compare_sequences(single)
