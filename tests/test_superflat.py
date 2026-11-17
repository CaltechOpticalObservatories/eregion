import numpy as np
import pytest
import xarray as xr

from eregion.datamodels import DetImage, ImageBundle, Output
from eregion.tasks.superflat import SuperCombine


def make_images(n=40, shape=(8, 6), seed=0, with_std=True):
    rng = np.random.default_rng(seed)
    stack = rng.normal(1000.0, 20.0, size=(n, *shape))
    images = []
    for i, arr in enumerate(stack):
        da = xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])})
        ds = da.to_dataset(name="data")
        if with_std:
            ds["std"] = xr.full_like(da, 2.0)
        full = (slice(0, shape[0]), slice(0, shape[1]))
        output = Output(id="0", ext_id=0, ext_slice=full, data_slice=full)
        images.append(DetImage(data=ds, output_objects={"0": output}, meta={"name": f"img{i}"}))
    return stack, images


def test_welford_and_mean_methods_agree_and_match_numpy():
    stack, images_a = make_images()
    _, images_b = make_images()

    welford = SuperCombine(method="welford").run(ImageBundle(images_a)).data[0]
    mean = SuperCombine(method="mean").run(ImageBundle(images_b)).data[0]

    for res in (welford, mean):
        np.testing.assert_allclose(res.get_data("data").values, stack.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(res.get_data("std").values, stack.std(axis=0, ddof=1), rtol=1e-6)
        np.testing.assert_array_equal(res.get_data("count").values, len(stack))
        # variance of the mean propagated from the input std planes: n * 2**2 / n**2
        np.testing.assert_allclose(res.get_data("variance").values, 4.0 / len(stack))
        assert res.meta["num_images"] == len(stack)
    assert "median" in welford.get_data("all").data_vars
    assert "median" not in mean.get_data("all").data_vars


def test_streaming_batches_give_stable_results_and_leave_inputs_untouched():
    stack, images = make_images(n=10)
    task = SuperCombine(method="mean")

    task.run(ImageBundle(images[:5]))
    first = task.run(ImageBundle(images[5:])).data[0].get_data("all").copy(deep=True)
    again = task.final_image_result.data[0].get_data("all")

    xr.testing.assert_identical(first, again)
    np.testing.assert_allclose(first["variance"].values, 4.0 / len(stack))
    assert "num_images" not in images[0].meta
    assert all(img._data is None for img in images)


def test_masked_pixels_are_excluded():
    stack, images = make_images(n=5, with_std=False)
    out = images[0].outputs["0"]
    mask = xr.zeros_like(out.data, dtype=bool)
    mask[0, 0] = True
    out.set_mask("bad", mask)

    res = SuperCombine(method="welford").run(ImageBundle(images)).data[0]

    assert res.get_data("count").values[0, 0] == 4
    np.testing.assert_allclose(res.get_data("data").values[0, 0], stack[1:, 0, 0].mean())
    assert res.outputs["0"].masks is None and images[0].outputs["0"].parent is images[0]


def test_shape_mismatch_raises():
    _, images = make_images(n=2)
    _, other = make_images(n=1, shape=(4, 4))
    with pytest.raises(ValueError, match="shape"):
        SuperCombine(method="welford").run(ImageBundle(images + other))
