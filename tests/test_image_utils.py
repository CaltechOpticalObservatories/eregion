import numpy as np
import pytest
import xarray as xr

from eregion.utils.image_utils import ensure_dataarray, ensure_numpy, set_slice_in_data, slice_data


def make_dataset(dims=("y", "x"), shape=(4, 5), fill=None):
    """
    Build a DetImage-like xr.Dataset with 'data' and 'std' data_vars and integer coords on every dim.
    :param dims: dimension names, in the order they are laid out in the arrays.
    :param shape: size of each dim in `dims`.
    :param fill: constant value for 'data'; if None, 'data' holds np.arange values.
    :return: xr.Dataset with data_vars ('data', 'std').
    """
    data = np.full(shape, fill, dtype=float) if fill is not None else np.arange(np.prod(shape), dtype=float).reshape(shape)
    return xr.Dataset({"data": (dims, data), "std": (dims, np.ones(shape))},
                      coords={d: np.arange(s) for d, s in zip(dims, shape)})


def test_ensure_dataarray_from_2d_ndarray():
    arr = np.arange(6).reshape(2, 3)
    da = ensure_dataarray(arr)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x")
    assert da.shape == (2, 3)
    assert np.array_equal(da.values, arr)


def test_ensure_dataarray_renames_unknown_2d_dims():
    da = xr.DataArray(np.arange(6).reshape(2, 3), dims=("row", "col"))
    out = ensure_dataarray(da)

    assert out.dims == ("y", "x")
    assert np.array_equal(out.values, da.values)


def test_ensure_dataarray_reorders_3d_dims():
    da = xr.DataArray(np.arange(24).reshape(2, 3, 4), dims=("t", "y", "x"))
    out = ensure_dataarray(da)

    assert out.dims == ("y", "x", "t")
    assert out.shape == (3, 4, 2)


def test_ensure_numpy_from_dataarray():
    arr = np.arange(4).reshape(2, 2)
    da = xr.DataArray(arr, dims=("y", "x"))

    out = ensure_numpy(da)

    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr)


def test_slice_data_tuple_uses_python_exclusive_stop_semantics():
    da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("y", "x"), coords={"y": np.arange(5), "x": np.arange(5)})
    out = slice_data(da, (slice(1, 4), slice(0, 3)))

    assert out.shape == (3, 3)
    assert np.array_equal(out.values, da.values[1:4, 0:3])


def test_slice_data_dict_with_negative_step():
    da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("y", "x"), coords={"y": np.arange(5), "x": np.arange(5)})
    out = slice_data(da, {"y": slice(4, 1, -1), "x": slice(4, 2, -1)})

    assert out.shape == (3, 2)
    assert np.array_equal(out.values, da.values[4:1:-1, 4:2:-1])


####################################### xr.Dataset support (DetImage.data) #############################################
def test_ensure_dataarray_keeps_2d_dataset_and_all_data_vars():
    ds = make_dataset()
    out = ensure_dataarray(ds)

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"data", "std"}
    assert out.sizes == {"y": 4, "x": 5}
    for var in out.data_vars:
        assert out[var].dims == ("y", "x")
        assert np.array_equal(out[var].values, ds[var].values)


def test_ensure_dataarray_reorders_transposed_2d_dataset():
    ds = make_dataset(dims=("x", "y"), shape=(5, 4))
    out = ensure_dataarray(ds)

    assert isinstance(out, xr.Dataset)
    for var in out.data_vars:
        assert out[var].dims == ("y", "x")
        assert np.array_equal(out[var].values, ds[var].values.T)


def test_ensure_dataarray_reorders_3d_dataset():
    ds = make_dataset(dims=("t", "y", "x"), shape=(2, 3, 4))
    out = ensure_dataarray(ds)

    assert isinstance(out, xr.Dataset)
    for var in out.data_vars:
        assert out[var].dims == ("y", "x", "t")
    assert np.array_equal(out["data"].values, np.transpose(ds["data"].values, (1, 2, 0)))


def test_ensure_dataarray_rejects_unsupported_dataset_ndim():
    ds = make_dataset(dims=("y",), shape=(4,))

    with pytest.raises(TypeError, match="data must be an xarray.DataArray, xarray.Dataset, or numpy.ndarray"):
        ensure_dataarray(ds)


def test_ensure_dataarray_renames_unknown_2d_dataset_dims():
    ds = make_dataset(dims=("row", "col"), shape=(4, 5))
    out = ensure_dataarray(ds)

    assert out["data"].dims == ("y", "x")


def test_ensure_numpy_rejects_dataset():
    with pytest.raises(TypeError, match="data must be an xarray.DataArray, or numpy.ndarray"):
        ensure_numpy(make_dataset())


def test_slice_data_dataset_slices_every_data_var():
    ds = make_dataset(shape=(5, 5))
    out = slice_data(ds, (slice(1, 4), slice(0, 3)))

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"data", "std"}
    assert out.sizes == {"y": 3, "x": 3}
    for var in out.data_vars:
        assert np.array_equal(out[var].values, ds[var].values[1:4, 0:3])
    # .sel keeps the parent's coordinate labels rather than reindexing from 0
    assert np.array_equal(out["y"].values, np.arange(1, 4))
    assert np.array_equal(out["x"].values, np.arange(0, 3))


def test_slice_data_dataset_dict_with_negative_step():
    ds = make_dataset(shape=(5, 5))
    out = slice_data(ds, {"y": slice(4, 1, -1), "x": slice(4, 2, -1)})

    assert out.sizes == {"y": 3, "x": 2}
    for var in out.data_vars:
        assert np.array_equal(out[var].values, ds[var].values[4:1:-1, 4:2:-1])


def test_slice_data_3d_dataset_slices_time_axis():
    ds = make_dataset(dims=("y", "x", "t"), shape=(2, 3, 4))
    out = slice_data(ds, (slice(0, 2), slice(0, 2), slice(1, 3)))

    assert out.sizes == {"y": 2, "x": 2, "t": 2}
    assert np.array_equal(out["data"].values, ds["data"].values[0:2, 0:2, 1:3])


def test_slice_data_dataset_rejects_non_slice_entries():
    ds = make_dataset()

    with pytest.raises(TypeError, match=r"slicer must be a dict of \{dim: slice\}"):
        slice_data(ds, (slice(0, 2), 1))


def test_set_slice_in_data_dataset_full_extent_updates_in_place():
    target = make_dataset()
    out = set_slice_in_data(make_dataset(fill=3.0), target, (slice(0, 4), slice(0, 5)))

    assert out is target
    assert np.all(target["data"].values == 3.0)
    assert np.all(target["std"].values == 1.0)


def test_set_slice_in_data_dataset_without_slicer_uses_slicedata_coords():
    target = make_dataset()
    out = set_slice_in_data(make_dataset(fill=7.0), target, None)

    assert np.all(out["data"].values == 7.0)


def test_set_slice_in_data_dataarray_subregion():
    target = make_dataset()["data"].copy()
    slicedata = make_dataset(fill=9.0)["data"].isel(y=slice(1, 3), x=slice(0, 2))
    out = set_slice_in_data(slicedata, target, (slice(1, 3), slice(0, 2)))

    assert np.all(out.values[1:3, 0:2] == 9.0)
    # Everything outside the slice is untouched
    assert out.values[0, 0] == 0.0
    assert out.values[3, 4] == 19.0


def test_set_slice_in_data_rejects_mismatched_data_vars():
    target = make_dataset()
    slicedata = make_dataset(fill=1.0).drop_vars("std")

    with pytest.raises(ValueError, match="must have the same data_vars"):
        set_slice_in_data(slicedata, target, (slice(0, 4), slice(0, 5)))


def test_set_slice_in_data_dataset_subregion():
    target = make_dataset()
    slicedata = make_dataset(fill=9.0).isel(y=slice(1, 3), x=slice(0, 2))
    out = set_slice_in_data(slicedata, target, (slice(1, 3), slice(0, 2)))

    assert np.all(out["data"].values[1:3, 0:2] == 9.0)
    assert out["data"].values[0, 0] == 0.0


def test_set_slice_in_data_rejects_mixed_types():
    with pytest.raises(TypeError, match="must be both xarray.DataArray or xarray.Dataset"):
        set_slice_in_data(make_dataset(fill=1.0), make_dataset()["data"].copy(), (slice(0, 4), slice(0, 5)))

