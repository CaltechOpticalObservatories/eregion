import numpy as np
import pytest
import xarray as xr

from eregion.utils.image_utils import ensure_dataarray, ensure_numpy, slice_data


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


def test_slice_data_rejects_non_slice_entries():
    da = xr.DataArray(np.arange(9).reshape(3, 3), dims=("y", "x"), coords={"y": np.arange(3), "x": np.arange(3)})

    with pytest.raises(ValueError, match="must be slices"):
        slice_data(da, (slice(0, 2), 1))
