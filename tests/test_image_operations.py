import time

import numpy as np
import pytest

from eregion.core.image_operations import do_digital_binning, find_vignetted_region, find_cosmic_rays


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


def test_do_digital_binning_raises_when_binsizes_do_not_sum_to_axis_length():
    data = np.zeros((4, 3))

    with pytest.raises(ValueError):
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


def _vignetted_flat(shape=(200, 200), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    flat = 1000.0 + rng.normal(0, 10, shape)
    # dark corner: illumination falls off with distance inside a 60 px radius around (0, 0)
    r = np.hypot(yy, xx)
    flat[r < 60] *= 0.3 + 0.7 * (r[r < 60] / 60) ** 2
    # hot/cold pixels which must not create mask islands
    flat[100, 100] = 0.0
    flat[150, 50] = 1e5
    return flat


def test_find_vignetted_region_masks_corner_not_center():
    mask = find_vignetted_region(_vignetted_flat())
    assert mask.dtype == bool and mask.shape == (200, 200)
    assert mask[:20, :20].all()
    assert not mask[80:, 80:].any()
    assert not mask[100, 100]


def test_find_vignetted_region_edge_connected_drops_interior_blob():
    flat = _vignetted_flat()
    yy, xx = np.indices(flat.shape)
    blob = np.hypot(yy - 120, xx - 120) < 15
    flat[blob] *= 0.2
    assert not find_vignetted_region(flat, edge_connected=True)[blob].any()
    assert find_vignetted_region(flat, edge_connected=False)[120, 120]


def test_find_vignetted_region_ignores_nonfinite():
    flat = _vignetted_flat()
    flat[150:160, 150:160] = np.nan
    assert not find_vignetted_region(flat)[150:160, 150:160].any()


@pytest.mark.parametrize(
    "image, kwargs",
    [
        (np.ones(10), {}),
        (np.ones((10, 10)), {"threshold": 1.5}),
        (np.ones((10, 10)), {"reference_percentile": 0}),
        (np.ones((10, 10)), {"smooth_size": 0}),
        (np.ones((10, 10)), {"morph_radius": -1}),
        (np.ones((10, 10)), {"min_area": -1}),
        (np.full((10, 10), np.nan), {}),
        (np.zeros((10, 10)), {}),
    ],
)
def test_find_vignetted_region_bad_input_raises(image, kwargs):
    with pytest.raises(ValueError):
        find_vignetted_region(image, **kwargs)


CR_ARGS = {"effective_gain": 1.0, "readnoise": 5.0}
CR_PIXELS = [(20, 30), (50, 70), (80, 15)]


def _flat_with_cosmics(shape=(100, 100), seed=0):
    rng = np.random.default_rng(seed)
    image = rng.normal(1000.0, np.sqrt(1000.0), shape)
    for y, x in CR_PIXELS:
        image[y, x] += 5000.0
    image[60, 40:44] += 3000.0  # short track
    return image


def test_find_cosmic_rays_flags_injected_cosmics():
    mask = find_cosmic_rays(_flat_with_cosmics(), **CR_ARGS)
    assert mask.dtype == bool
    for y, x in CR_PIXELS:
        assert mask[y, x]
    assert mask[60, 40:44].all()
    assert mask.sum() < 50


def test_find_cosmic_rays_ignores_input_mask_and_nans():
    image = _flat_with_cosmics()
    image[5, 5] = np.nan
    input_mask = np.zeros(image.shape, dtype=bool)
    input_mask[CR_PIXELS[0]] = True
    mask = find_cosmic_rays(image, mask=input_mask, **CR_ARGS)
    assert not mask[CR_PIXELS[0]]
    assert not mask[5, 5]
    assert mask[CR_PIXELS[1]]


def test_find_cosmic_rays_raises_on_bad_input():
    image = _flat_with_cosmics()
    with pytest.raises(ValueError, match="2D"):
        find_cosmic_rays(image[None], **CR_ARGS)
    with pytest.raises(ValueError, match="effective_gain and readnoise"):
        find_cosmic_rays(image, effective_gain=1.0)
    with pytest.raises(ValueError, match="mask shape"):
        find_cosmic_rays(image, mask=np.zeros((10, 10), dtype=bool), **CR_ARGS)
