import numpy as np
import xarray as xr
from typing import Hashable, TypeVar
from copy import deepcopy

# type for xr.DataArray | xr.Dataset that can be used in type hints for functions that accept either type
XRDATA = TypeVar("XRDATA", bound=xr.DataArray | xr.Dataset)

def has_dim_order(data: XRDATA, target: tuple[str, ...]) -> bool:
    """
    Check that data already carries its dims in the target order. A Dataset keeps dim ordering on each data_var
    (Dataset.dims is an unordered name->size mapping), so every data_var is checked.
    :param data: xr.DataArray or xr.Dataset
    :param target: expected dim names in order, e.g. ('y','x')
    :return: True if the dims are already ordered as target.
    """
    if isinstance(data, xr.Dataset):
        return all(data[var].dims == target for var in data.data_vars)
    return data.dims == target

def ensure_dataarray(data: XRDATA | np.ndarray) -> XRDATA:
    """
    Coerce 2D data to xr.DataArray with dims ('y','x') and integer coords.
    :param data: xr.Dataset, xr.DataArray, or np.ndarray
    :raises TypeError: if data is not xarray.DataArray, xarray.Dataset, or numpy.ndarray
    :return: xarray.DataArray or xarray.Dataset with dims ('y','x') or ('y','x','t')
    """
    ndim = data.ndim if isinstance(data, np.ndarray) else len(data.sizes)

    match (data, ndim):
        case (xr.DataArray() | xr.Dataset(), 2) | (xr.DataArray() | xr.Dataset(), 3):
            target = ("y", "x") if ndim == 2 else ("y", "x", "t")
            if not has_dim_order(data, target):
                try:
                    data = data.transpose(..., *target)
                except ValueError:  # dims are not named 'y'/'x'(/'t'); rename them positionally first
                    data = data.rename(dict(zip(list(data.sizes), target))).transpose(*target)
            return data
        case (np.ndarray(), 2):
            y_size, x_size = data.shape
            return xr.DataArray(data, dims=("y", "x"), coords={"y": np.arange(y_size), "x": np.arange(x_size),})
        case (np.ndarray(), 3):
            y_size, x_size, t_size = data.shape
            return xr.DataArray(data, dims=("y", "x", "t"),
                                coords={"y": np.arange(y_size), "x": np.arange(x_size), "t": np.arange(t_size)})
        case _:
            raise TypeError("data must be an xarray.DataArray, xarray.Dataset, or numpy.ndarray")

def ensure_numpy(data: xr.DataArray | np.ndarray) -> np.ndarray:
    """
    Coerce xarray.DataArray to numpy.ndarray
    :type data: Union[xr.DataArray, np.ndarray]
    :rtype: np.ndarray
    :raises TypeError: if data is not xarray.DataArray or numpy.ndarray
    :return: numpy.ndarray
    """
    match data:
        case np.ndarray():
            return data
        case xr.DataArray():
            return data.values
        case _:
            raise TypeError("data must be an xarray.DataArray, or numpy.ndarray")

def slice_data(data: XRDATA, slicer: tuple[slice, ...] | dict[Hashable, slice]) -> XRDATA:
    """
    Slice a 2D or 3D DataArray using ('y','x','t) slices. The slicing is done with .sel method, NOT .isel, so slices
    should be specific labels (coordinate values) and not index positions. The slicer must conform to numpy slicing
    where the stop index is exclusive, function will automatically decrease the stop index by 1 to make it inclusive
    for the .sel method.
    """
    if isinstance(slicer, tuple):
        dim = ["y", "x", "t"]
        slicer = {dim[i]:s for i, s in enumerate(slicer)}
    slcr = decrease_slicer_stop_index(deepcopy(slicer))
    return data.sel(**slcr)

def set_slice_in_data(slicedata: XRDATA,
                      targetdata: XRDATA,
                      slicer: tuple[slice, ...] | dict[Hashable, slice] | None = None) -> XRDATA:
    """
    Set a slice of a DataArray/Set with another DataArray/Set. The .loc method is used for setting slice,
    so slices should be specific labels (coordinate values) and not index positions. The slicer must conform to numpy slicing
    where the stop index is exclusive, function will automatically decrease the stop index by 1 to make it inclusive
    for the .loc method.
    """
    if isinstance(slicer, tuple):
        dim = ["y", "x", "t"]
        slicer = {dim[i]:s for i, s in enumerate(slicer)}
    slcr = decrease_slicer_stop_index(deepcopy(slicer)) if slicer is not None else slicedata.coords

    if set(targetdata.dims) != set(slicedata.dims):
        raise ValueError("targetdata and slicedata must have the same dims.")

    match (targetdata, slicedata):
        case (xr.DataArray(), xr.DataArray()):
            targetdata.loc[slcr] = ensure_numpy(slicedata)
        case (xr.Dataset(), xr.Dataset()):
            if set(targetdata.data_vars) != set(slicedata.data_vars):
                raise ValueError("targetdata and slicedata must have the same data_vars.")
            for var in targetdata.data_vars:
                targetdata[var].loc[slcr] = ensure_numpy(slicedata[var])
        case _:
            raise TypeError("targetdata and slicedata must be both xarray.DataArray or xarray.Dataset")
    return targetdata

def decrease_slicer_stop_index(slicer: dict[Hashable, slice]) -> dict[Hashable, slice]:
    if isinstance(slicer, dict) and all(isinstance(s, slice) for s in slicer.values()):
        for k, sl in slicer.items():
            step = sl.step if sl.step is not None else 1
            slicer[k] = slice(sl.start, sl.stop - step, step) if sl.stop is not None else sl
    else:
        raise TypeError("slicer must be a dict of {dim: slice}.")
    return slicer


