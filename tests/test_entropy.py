"""
Tests for eregion.core.entropy 
Reference: S. J. Watts, L. Crow, "The Shannon Entropy of a Histogram",
https://arxiv.org/pdf/2210.02848
"""
import numpy as np
import pytest

from eregion.core.entropy import (
    shannon_entropy_histogram,
    histogram_efficiency,
    differential_entropy_knn,
    optimal_bin_width,
    entropy_optimal_histogram,
    _as_float_array,
)


def test_as_float_array_preserves_float64_without_copy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = _as_float_array(arr)
    assert out is arr


def test_as_float_array_preserves_float32_without_copy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _as_float_array(arr)
    assert out is arr
    assert out.dtype == np.float32


def test_differential_entropy_knn_matches_known_normal_value_float32():
    # float32 input should still give a correct estimate
    rng = np.random.default_rng(10)
    sample = rng.normal(0, 1, 100_000).astype(np.float32)
    assert differential_entropy_knn(sample, dither=True, rng=rng) == pytest.approx(2.047, abs=0.05)

def test_shannon_entropy_fair_coin_is_one_bit():
    # From the paper example
    assert shannon_entropy_histogram([50, 50]) == pytest.approx(1.0)


def test_shannon_entropy_unfair_coin_matches_paper_example():
    # From the paper example
    counts = [25, 75]
    assert shannon_entropy_histogram(counts) == pytest.approx(0.811, abs=1e-3)


def test_shannon_entropy_deterministic_histogram_is_zero():
    assert shannon_entropy_histogram([10, 0, 0]) == pytest.approx(0.0)


def test_shannon_entropy_ignores_empty_bins():
    # Empty bins should not raise (0 * log(0) := 0) or change the result
    assert shannon_entropy_histogram([50, 50, 0, 0]) == pytest.approx(1.0)


def test_shannon_entropy_uniform_over_n_bins_is_log2_n():
    counts = [10] * 8
    assert shannon_entropy_histogram(counts) == pytest.approx(3.0)  # log2(8) = 3


def test_shannon_entropy_nats_base():
    counts = [50, 50]
    h_bits = shannon_entropy_histogram(counts, base=2)
    h_nats = shannon_entropy_histogram(counts, base=np.e)
    assert h_nats == pytest.approx(h_bits * np.log(2))


def test_shannon_entropy_raises_for_empty_histogram():
    with pytest.raises(ValueError):
        shannon_entropy_histogram([0, 0, 0])


def test_histogram_efficiency_is_one_for_uniform_histogram():
    counts = [10] * 8
    h = shannon_entropy_histogram(counts)
    assert histogram_efficiency(h, n_bins=8) == pytest.approx(1.0)


def test_histogram_efficiency_raises_for_nonpositive_bins():
    with pytest.raises(ValueError):
        histogram_efficiency(1.0, n_bins=0)


def test_differential_entropy_knn_matches_known_normal_value():
    # From the paper table 1.2: standard normal (sigma=1) has h = 2.047 bits
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 100_000)
    assert differential_entropy_knn(sample) == pytest.approx(2.047, abs=0.05)


def test_differential_entropy_knn_matches_known_uniform_value():
    # Continuous uniform(0,1) has differential entropy h = 0 bits
    rng = np.random.default_rng(1)
    sample = rng.uniform(0, 1, 100_000)
    assert differential_entropy_knn(sample) == pytest.approx(0.0, abs=0.05)


def test_differential_entropy_knn_matches_known_exponential_value():
    # Exponential(mean=1) has differential entropy h = log2(e) = 1.4427 bits
    rng = np.random.default_rng(2)
    sample = rng.exponential(1.0, 100_000)
    assert differential_entropy_knn(sample) == pytest.approx(np.log2(np.e), abs=0.05)


def test_optimal_bin_width_raises_for_invalid_M():
    rng = np.random.default_rng(4)
    sample = rng.normal(0, 1, 1000)
    with pytest.raises(ValueError):
        optimal_bin_width(sample, M=1.0)
    with pytest.raises(ValueError):
        optimal_bin_width(sample, M=3.5)


def test_optimal_bin_width_is_positive_and_reasonable():
    rng = np.random.default_rng(5)
    sample = rng.normal(0, 1, 10_000)
    delta = optimal_bin_width(sample, M=2.5)
    assert delta > 0
    # bin width should be a small fraction of the data's spread
    assert delta < (sample.max() - sample.min())


def test_entropy_optimal_histogram_returns_expected_keys():
    rng = np.random.default_rng(6)
    sample = rng.normal(0, 1, 5000)
    result = entropy_optimal_histogram(sample, M=2.5)
    assert set(result) == {"counts", "bin_edges", "bin_width", "entropy_bits", "efficiency", "M_bin"}


def test_entropy_optimal_histogram_counts_sum_to_n():
    rng = np.random.default_rng(7)
    sample = rng.normal(0, 1, 5000)
    result = entropy_optimal_histogram(sample, M=2.5)
    assert result["counts"].sum() == sample.size


def test_entropy_optimal_histogram_edges_cover_full_data_range():
    rng = np.random.default_rng(8)
    sample = rng.exponential(1.0, 5000)
    result = entropy_optimal_histogram(sample, M=2.5)
    assert result["bin_edges"][0] <= sample.min()
    assert result["bin_edges"][-1] >= sample.max()


def test_entropy_optimal_histogram_M_bin_roughly_matches_input_M():
    # Per the paper, M_X/M_B estimated from a well-binned histogram should
    # roughly track the input M for large N
    rng = np.random.default_rng(9)
    sample = rng.normal(0, 1, 20_000)
    result = entropy_optimal_histogram(sample, M=2.5)
    assert 1.5 < result["M_bin"] < 4.0
