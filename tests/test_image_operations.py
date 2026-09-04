import time

import numpy as np
import pytest

from eregion.core.image_operations import do_digital_binning


def test_do_digital_binning_sums_correctly_default_axis():
    data = np.arange(12, dtype=float).reshape(4, 3)

    binned = do_digital_binning(data, binsizes=[1, 3])

    expected_rows = np.array(
        [
            data[0:1].sum(axis=0),
            data[1:4].sum(axis=0),
        ]
    )
    assert np.array_equal(binned[:2], expected_rows)
    # rows beyond len(binsizes) are left as the zero-initialized fill value
    assert np.array_equal(binned[2:], np.zeros((2, 3)))


def test_do_digital_binning_explicit_axis_zero_matches_default():
    data = np.arange(20, dtype=float).reshape(5, 4)

    default = do_digital_binning(data, binsizes=[2, 3])
    explicit = do_digital_binning(data, binsizes=[2, 3], binaxis=0)

    assert np.array_equal(default, explicit)


def test_do_digital_binning_single_bin_sums_entire_axis_into_first_row():
    data = np.arange(12, dtype=float).reshape(4, 3)

    binned = do_digital_binning(data, binsizes=[4])

    assert np.array_equal(binned[0], data.sum(axis=0))
    assert np.array_equal(binned[1:], np.zeros((3, 3)))


def test_do_digital_binning_all_ones_reproduces_original_rows():
    data = np.arange(12, dtype=float).reshape(4, 3)

    binned = do_digital_binning(data, binsizes=[1, 1, 1, 1])

    assert np.array_equal(binned, data)


def test_do_digital_binning_does_not_mutate_input():
    data = np.arange(12, dtype=float).reshape(4, 3)
    original = data.copy()

    do_digital_binning(data, binsizes=[1, 3])

    assert np.array_equal(data, original)


def test_do_digital_binning_preserves_integer_dtype():
    data = np.arange(12, dtype=np.int32).reshape(4, 3)

    binned = do_digital_binning(data, binsizes=[2, 2])

    assert binned.dtype == data.dtype
    assert np.array_equal(binned[0], data[0:2].sum(axis=0))
    assert np.array_equal(binned[1], data[2:4].sum(axis=0))


def test_do_digital_binning_raises_when_binsizes_do_not_sum_to_axis_length():
    data = np.zeros((4, 3))

    with pytest.raises(AssertionError):
        do_digital_binning(data, binsizes=[1, 2])


def test_do_digital_binning_binaxis_one_sums_along_columns():
    data = np.arange(12, dtype=float).reshape(4, 3)

    binned = do_digital_binning(data, binsizes=[1, 2], binaxis=1)

    expected_cols = np.array(
        [
            data[:, 0:1].sum(axis=1),
            data[:, 1:3].sum(axis=1),
        ]
    ).T
    assert np.array_equal(binned[:, :2], expected_cols)
    assert np.array_equal(binned[:, 2:], np.zeros((4, 1)))


def test_do_digital_binning_binaxis_one_matches_transposed_binaxis_zero():
    data = np.arange(20, dtype=float).reshape(5, 4)
    binsizes = [1, 3]

    binned_axis1 = do_digital_binning(data, binsizes=binsizes, binaxis=1)
    binned_axis0_of_transpose = do_digital_binning(data.T, binsizes=binsizes, binaxis=0)

    assert np.array_equal(binned_axis1, binned_axis0_of_transpose.T)


def test_do_digital_binning_performance_on_large_image():
    rng = np.random.default_rng(seed=0)
    nrows, ncols = 4096, 4096
    data = rng.random((nrows, ncols))

    # emulate incremental binning (Kaye et al.) with many small, growing bins
    binsizes = []
    total = 0
    increment = 1
    while total < nrows:
        next_bin = min(increment, nrows - total)
        binsizes.append(next_bin)
        total += next_bin
        increment += 1

    start = time.perf_counter()
    binned = do_digital_binning(data, binsizes=binsizes)
    elapsed = time.perf_counter() - start

    assert binned.shape == data.shape
    assert elapsed < 2.0, f"do_digital_binning took too long: {elapsed:.3f}s for {len(binsizes)} bins"
