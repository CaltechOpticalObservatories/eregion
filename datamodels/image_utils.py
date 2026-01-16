import numpy as np
import xarray as xr
from typing import Union, Tuple, List, Any
from numpy.typing import NDArray
from astropy.io import fits

from prefect import task

def ensure_dataarray(data: Union[xr.DataArray, NDArray]) -> xr.DataArray:
    """
    Coerce 2D data to xr.DataArray with dims ('y','x') and integer coords.
    :type data: Union[xr.DataArray, NDArray]
    :rtype: xr.DataArray
    :raises TypeError: if data is not xarray.DataArray or numpy.ndarray
    :return: xarray.DataArray with dims ('y','x') or ('y','x','t')
    """

    if isinstance(data, xr.DataArray):
        # Ensure dims ordering is ('y','x'); rename if unnamed
        if data.ndim == 2 and data.dims != ("y", "x"):
            try:
                data = data.transpose(..., "y", "x")
            except Exception:
                data = data.rename({data.dims[0]: "y", data.dims[1]: "x"}).transpose("y", "x")
        elif data.ndim == 3 and data.dims != ("y", "x", "t"):
            try:
                data = data.transpose(..., "y", "x", "t")
            except Exception:
                data = data.rename({data.dims[0]: "y", data.dims[1]: "x",
                                    data.dims[2]: "t"}).transpose("y", "x", "t")
        return data

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            y_size, x_size = data.shape
            return xr.DataArray(
                data,
                dims=("y", "x"),
                coords={
                    "y": np.arange(y_size),
                    "x": np.arange(x_size),
                },
            )
        elif data.ndim == 3:
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

    raise TypeError("data must be an xarray.DataArray, or numpy.ndarray")

def slice_data(data: xr.DataArray, slicer: tuple[slice, ...]) -> xr.DataArray:
    """
    Slice a 2D or 3D DataArray using ('y','x','t) positional slices.
    """
    if not isinstance(slicer, tuple) or not all(isinstance(s, slice) for s in slicer):
        raise ValueError("slicer must be a tuple of slice objects.")
    if data.ndim == 2:
        if data.dims != ("y", "x"):
            raise ValueError("DataArray must be 2D with dims ('y','x').")
        return data.isel(y=slicer[0], x=slicer[1])
    elif data.ndim == 3:
        if data.dims != ("y", "x", "t"):
            raise ValueError("DataArray must be 3D with dims ('y','x','t').")
        return data.isel(y=slicer[0], x=slicer[1], t=slicer[2])
    else:
        raise ValueError("DataArray must be 2D or 3D.")

@task
def load_image_fits(filename: str) -> Tuple[List[Any], List[fits.Header]]:
    """
    Load FITS file and return the data as a list with hdu extensions as the first axis.
    :param filename: str
        Path to the FITS file.
    :return: Tuple[List[Any], List[fits.Header]]
        A tuple containing a list of data arrays (image/table/...) for each HDU and a list of corresponding FITS headers.
    """
    input_data_array, input_headers = [], []
    with fits.open(filename) as hdulist:
        for hdu in hdulist:
            input_data_array.append(hdu.data)
            input_headers.append(hdu.header)
    return input_data_array, input_headers
