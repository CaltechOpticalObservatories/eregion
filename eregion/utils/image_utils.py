import numpy as np
import xarray as xr

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

def slice_data(data: xr.DataArray, slicer: tuple[slice, ...] | dict[str, slice]) -> xr.DataArray:
    """
    Slice a 2D or 3D DataArray using ('y','x','t) positional slices.
    """
    match slicer:
        case tuple():
            if not all(isinstance(s, slice) for s in slicer):
                raise ValueError("All elements of slicer tuple must be slices.")
            match (data.ndim, data.dims):
                case (2, ("y", "x")):
                    slicer = {"y": slicer[0], "x": slicer[1]}
                case (3, ("y", "x", "t")):
                    slicer = {"y": slicer[0], "x": slicer[1], "t": slicer[2]}
                case _:
                    raise ValueError("DataArray must be 2D with dims ('y','x') or 3D with dims ('y','x','t').")
        case dict():
            if not all(isinstance(s, slice) for s in slicer.values()):
                raise ValueError("All values of slicer dict must be slices.")
        case _:
            raise ValueError("slicer must be a tuple of slices or a dict of {dim: slice}.")

    # hack to not include last element of slices but still use .sel() which includes the stop index
    for k, sl in slicer.items():
        if sl.step > 0:
            slicer[k] = slice(sl.start, sl.stop - 1, sl.step)
        else:
            slicer[k] = slice(sl.start, sl.stop + 1, sl.step)

    return data.sel(**slicer)



