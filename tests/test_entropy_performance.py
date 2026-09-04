"""
Performance/regression tests for eregion.core.entropy. 
    pytest tests/test_entropy_performance.py -m slow -v
"""
import time
import warnings

import numpy as np
import pytest

from eregion.core.entropy import (
    differential_entropy_knn,
    optimal_bin_width,
    entropy_optimal_histogram,
)

IMAGE_SHAPE = (8192, 8192)  # ~67.1 million pixels
KNN_BUDGET_S = 60
FULL_PIPELINE_BUDGET_S = 90


def _simulated_detector_image(rng):
    return np.round(rng.normal(1000.0, 5.0, size=IMAGE_SHAPE))


@pytest.mark.slow
def test_differential_entropy_knn_performance_on_8k_image():
    rng = np.random.default_rng(0)
    image = _simulated_detector_image(rng)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        h = differential_entropy_knn(image, dither=True, rng=rng)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(h)
    assert elapsed < KNN_BUDGET_S, f"differential_entropy_knn took {elapsed:.1f}s on an 8k x 8k image (budget {KNN_BUDGET_S}s)"


@pytest.mark.slow
def test_entropy_optimal_histogram_performance_on_8k_image():
    rng = np.random.default_rng(1)
    image = _simulated_detector_image(rng)
    n = image.size

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = entropy_optimal_histogram(image, M=2.5, dither=True, rng=rng)
    elapsed = time.perf_counter() - t0

    # correctness sanity
    assert result["counts"].sum() == n
    assert result["bin_width"] > 0
    assert result["entropy_bits"] > 0

    assert elapsed < FULL_PIPELINE_BUDGET_S, (
        f"entropy_optimal_histogram took {elapsed:.1f}s on an 8k x 8k image "
        f"(budget {FULL_PIPELINE_BUDGET_S}s)"
    )


@pytest.mark.slow
def test_optimal_bin_width_performance_on_8k_image():
    rng = np.random.default_rng(2)
    image = _simulated_detector_image(rng)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        delta = optimal_bin_width(image, M=2.5, dither=True, rng=rng)
    elapsed = time.perf_counter() - t0

    assert delta > 0
    assert elapsed < KNN_BUDGET_S, f"optimal_bin_width took {elapsed:.1f}s on an 8k x 8k image (budget {KNN_BUDGET_S}s)"
