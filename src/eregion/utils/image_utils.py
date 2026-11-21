import numpy as np
import xarray as xr
from typing import Hashable, TypeVar

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
            # dims without coords make .sel/.loc fall back to (stop-exclusive) positions, so add integer coords
            missing = {d: np.arange(data.sizes[d]) for d in target if d not in data.coords}
            return data.assign_coords(missing) if missing else data
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

def to_slicer(region: tuple[slice, ...] | dict[Hashable, slice],
              dims: tuple[str, ...] = ("y", "x", "t")) -> dict[Hashable, slice]:
    """
    Return a new {dim: slice} dict from a dict, or from a tuple of slices matched to dims in order. Slice steps,
    including negative ones, are kept.
    :param region: tuple of slices or dict of {dim: slice}.
    :param dims: dimension names a tuple is matched to.
    :return: dict of {dim: slice}
    :raises TypeError: if region is not a tuple or dict of slices.
    """
    if isinstance(region, (tuple, list)):
        if len(region) > len(dims):
            raise ValueError(f"region has {len(region)} slices but only dims {dims} are known")
        slicer = dict(zip(dims, region))
    elif isinstance(region, dict):
        slicer = dict(region)
    else:
        raise TypeError(f"region must be a tuple or dict of slices, got {type(region).__name__}")
    if not all(isinstance(s, slice) for s in slicer.values()):
        raise TypeError("slicer must be a dict of {dim: slice}.")
    return slicer

def to_label_slicer(region: tuple[slice, ...] | dict[Hashable, slice]) -> dict[Hashable, slice]:
    """
    Turn a numpy-style slicer (stop exclusive) into one for label based selection with .sel/.loc, where the stop is
    inclusive, by moving each stop back one step. Returns a new dict.
    :param region: tuple of slices or dict of {dim: slice}, in pixel (coordinate label) values.
    :return: dict of {dim: slice} with inclusive stops.
    """
    out = {}
    for dim, sl in to_slicer(region).items():
        step = sl.step if sl.step is not None else 1
        out[dim] = slice(sl.start, sl.stop - step, step) if sl.stop is not None else sl
    return out

def slice_data(data: XRDATA, slicer: tuple[slice, ...] | dict[Hashable, slice]) -> XRDATA:
    """
    Slice a 2D or 3D DataArray/Dataset using ('y','x','t') slices. The slicing is done with .sel, NOT .isel, so slices
    are coordinate labels, not index positions. The slicer uses numpy semantics (stop exclusive); see
    to_label_slicer. A negative step returns the region in reversed order.
    """
    return data.sel(**to_label_slicer(slicer))

def set_slice_in_data(slicedata: XRDATA,
                      targetdata: XRDATA,
                      slicer: tuple[slice, ...] | dict[Hashable, slice] | None = None) -> XRDATA:
    """
    Set a slice of a DataArray/Dataset with another DataArray/Dataset, in place, with .loc. Slices are coordinate
    labels with numpy semantics (stop exclusive); see to_label_slicer. Without a slicer, slicedata's coords are used.
    """
    slcr = to_label_slicer(slicer) if slicer is not None else slicedata.coords

    if set(targetdata.dims) != set(slicedata.dims):
        raise ValueError("targetdata and slicedata must have the same dims.")

    match (targetdata, slicedata):
        case (xr.DataArray(), xr.DataArray()):
            targetdata.loc[slcr] = slicedata.to_numpy()
        case (xr.Dataset(), xr.Dataset()):
            if set(targetdata.data_vars) != set(slicedata.data_vars):
                raise ValueError("targetdata and slicedata must have the same data_vars.")
            for var in targetdata.data_vars:
                targetdata[var].loc[slcr] = slicedata[var].to_numpy()
        case _:
            raise TypeError("targetdata and slicedata must be both xarray.DataArray or xarray.Dataset")
    return targetdata
