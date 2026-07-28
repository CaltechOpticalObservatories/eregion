import numpy as np
import xarray as xr
from typing import Literal
from copy import deepcopy

def ensure_dataarray(data: xr.DataArray | np.ndarray) -> xr.DataArray:
    """
    Coerce 2D data to xr.DataArray with dims ('y','x') and integer coords.
    :type data: Union[xr.DataArray, np.ndarray]
    :rtype: xr.DataArray
    :raises TypeError: if data is not xarray.DataArray or numpy.ndarray
    :return: xarray.DataArray with dims ('y','x') or ('y','x','t')
    """
    match (data, data.ndim):
        case (xr.DataArray(), 2):
            if data.dims != ("y", "x"): # Ensure dims ordering is ('y','x'); rename if unnamed
                try:
                    data = data.transpose(..., "y", "x")
                except Exception:
                    data = data.rename({data.dims[0]: "y", data.dims[1]: "x"}).transpose("y", "x")
            return data

        case (xr.DataArray(), 3):
            if data.dims != ("y", "x", "t"):
                try:
                    data = data.transpose(..., "y", "x", "t")
                except Exception:
                    data = data.rename({data.dims[0]: "y", data.dims[1]: "x",
                                        data.dims[2]: "t"}).transpose("y", "x", "t")
            return data

        case (np.ndarray(), 2):
            y_size, x_size = data.shape
            return xr.DataArray(
                data,
                dims=("y", "x"),
                coords={
                    "y": np.arange(y_size),
                    "x": np.arange(x_size),
                },
            )

        case (np.ndarray(), 3):
            y_size, x_size, t_size = data.shape
            return xr.DataArray(
                data,
                dims=("y", "x", "t"),
                coords={
                    "y": np.arange(y_size),
                    "x": np.arange(x_size),
                    "t": np.arange(t_size),
                },
            )

        case _:
            raise TypeError("data must be an xarray.DataArray, or numpy.ndarray")

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

def slice_data(data: xr.DataArray | xr.Dataset, slicer: tuple[slice, ...] | dict[Literal, slice]) -> xr.DataArray | xr.Dataset:
    """
    Slice a 2D or 3D DataArray using ('y','x','t) positional slices.
    """
    slcr = decrease_slicer_stop_index(deepcopy(slicer))
    return data.sel(**slcr)

def decrease_slicer_stop_index(slicer: tuple[slice, ...] | dict[Literal, slice]) -> dict[Literal, slice]:
    if isinstance(slicer, tuple):
        dim = ["y", "x", "t"]
        slicer = {dim[i]:s for i, s in enumerate(slicer)}

    if isinstance(slicer, dict) and all(isinstance(s, slice) for s in slicer.values()):
        for k, sl in slicer.items():
            step = sl.step if sl.step is not None else 1
            slicer[k] = slice(sl.start, sl.stop - step, step) if sl.start is not None and sl.stop is not None else sl
    else:
        raise TypeError("slicer must be a tuple of slices or a dict of {dim: slice}.")

    return slicer


