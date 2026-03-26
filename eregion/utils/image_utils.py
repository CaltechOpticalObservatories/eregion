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

def slice_data(data: xr.DataArray, slicer: tuple[slice, ...]) -> xr.DataArray:
    """
    Slice a 2D or 3D DataArray using ('y','x','t) positional slices.
    """
    if not isinstance(slicer, tuple) or not all(isinstance(s, slice) for s in slicer):
        raise ValueError("slicer must be a tuple of slice objects.")

    match (data.ndim, data.dims):
        case (2, ("y", "x")):
            return data.isel(y=slicer[0], x=slicer[1])
        case (3, ("y", "x", "t")):
            return data.isel(y=slicer[0], x=slicer[1], t=slicer[2])
        case _:
            raise ValueError("DataArray must be 2D with dims ('y','x') or 3D with dims ('y','x','t').")

