from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Optional, Any, Literal, Callable
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pathlib
import json
import logging
from copy import deepcopy

from eregion.utils import (
    ensure_dataarray,
    slice_data,
    to_slicer,
    to_label_slicer,
    XRDATA
)
from eregion.core.image_stats import STATFUNCS, as_masked_array, check_mask_aware, do_statistics, reduce_array
from eregion.datamodels.meta import DetImageMeta
from eregion.datamodels import output as output_module
from eregion.datamodels.output import Output
from eregion.datamodels.operators import ArithmeticOperators
from eregion.utils import load_class

logger = logging.getLogger(__name__)

########################################## VAR NAMES AND KINDS ######################################################
MASK_SUFFIX = "_mask"
LABEL_SUFFIX = "_label"
RESERVED_NAMES = frozenset({"combined", "all", "mask", "label"})
NETCDF_FORMAT = 2

MaskSelection = str | Sequence[str] | None


def var_kind(name: str) -> Literal['data', 'mask', 'label']:
    """
    Return the kind of a DetImage var from its name: '<prefix>_mask' is a bool mask, '<prefix>_label' an integer
    label image (0 is background), anything else a data var.
    :param name: str, var name.
    :return: 'data', 'mask' or 'label'
    :raises TypeError: if name is not a str.
    :raises ValueError: if name is empty, reserved, or a bare suffix.
    """
    if not isinstance(name, str):
        raise TypeError(f"var names must be str, got {type(name).__name__}")
    if not name or name in RESERVED_NAMES:
        raise ValueError(f"'{name}' can't be a var name; reserved names are {sorted(RESERVED_NAMES)}")
    for suffix, kind in ((MASK_SUFFIX, 'mask'), (LABEL_SUFFIX, 'label')):
        if name.endswith(suffix):
            if len(name) == len(suffix):
                raise ValueError(f"{kind} vars must be named '<prefix>{suffix}', got '{name}'")
            return kind
    return 'data'


def _check_dtype(name: str, kind: str, dtype: np.dtype, target_dtype: Optional[np.dtype] = None) -> None:
    """ Masks must be bool, labels integer, and float values can't go into an integer data var. """
    if kind == 'mask' and dtype != np.bool_:
        raise TypeError(f"mask var '{name}' must be bool, got {dtype}")
    if kind == 'label' and not np.issubdtype(dtype, np.integer):
        raise TypeError(f"label var '{name}' must be integer, got {dtype}")
    if (kind == 'data' and target_dtype is not None and np.issubdtype(target_dtype, np.integer)
            and not np.issubdtype(dtype, np.integer)):
        raise TypeError(f"data var '{name}' is {target_dtype}; writing {dtype} values would truncate them")


def _select_mask_names(masks: MaskSelection, available: list[str], owner: str) -> list[str]:
    """ Resolve a masks selection: 'combined' (all masks), one name, a list of names, or None (no masks). """
    if masks is None:
        return []
    if isinstance(masks, str):
        if masks == 'combined':
            return list(available)
        masks = [masks]
    names = list(masks)
    for name in names:
        if var_kind(name) != 'mask':
            raise ValueError(f"'{name}' is not a mask name; mask vars are named '<prefix>{MASK_SUFFIX}'")
        if name not in available:
            raise KeyError(f"{owner} has no mask '{name}'; its masks are {available} (or 'combined')")
    return names


def combine_masks(ds: xr.Dataset, names: list[str], coords: dict[str, Any], dims: tuple[str, ...] = ('y', 'x')
                  ) -> xr.DataArray:
    """
    Return the OR of the named bool mask vars of ds as a new DataArray, all False if names is empty.
    :param ds: xr.Dataset holding the masks.
    :param names: list of mask var names.
    :param coords: coords of the returned array (used when names is empty).
    :param dims: dims of the returned array.
    :return: bool xr.DataArray
    """
    if names:
        values = np.logical_or.reduce([ds[name].values for name in names])
        return xr.DataArray(values, dims=ds[names[0]].dims, coords=ds[names[0]].coords)
    shape = tuple(len(coords[dim]) for dim in dims)
    return xr.DataArray(np.zeros(shape, dtype=bool), dims=dims, coords=coords)


############################################## DETIMAGE CLASS #######################################################
class DetImage(ArithmeticOperators):
    """
    Detector image holding 2D (y, x) arrays and the Output regions that make it up.

    All arrays live in one xr.Dataset (`dataset`), on the same y/x coordinates. The kind of each var comes from its
    name:

    - '<prefix>_mask': bool mask, True where a pixel is masked (e.g. 'sigma_clip_mask', 'cosmic_ray_mask').
    - '<prefix>_label': integer label image, 0 is background (e.g. 'cosmic_ray_label').
    - anything else: a data var. 'data' is the main image; 'std' and 'count' are common for combined images.

    'combined', 'all', 'mask' and 'label' are reserved and can't be var names. In the `masks` parameter of the methods
    below, 'combined' means the OR of all masks.

    Reading and writing:

    - get_var(name, region) returns any var, whole or a region. The result may share memory with the image, so write
      through set_var.
    - get_mask(masks, region) returns the OR of the selected masks as a bool array (all False if there are none).
    - get_masked(name, masks, region) returns a numpy masked array.
    - set_var(name, values, region, combine) writes a whole var or a region of it; combine=True ORs a mask.
    - set_vars(values) writes several whole vars, remove_vars(*names) drops vars.
    - set_loader(loader) / load() / unload() handle data read from disk on demand. Masks and labels are kept in memory
      when the data vars are unloaded.

    A region is a dict or tuple of slices in pixel coordinates, numpy style (stop exclusive); a negative step returns
    and expects values in that reversed order. Output methods take the same arguments, with region names
    ('image', 'serial_overscan', ...) as well.

    Parameters:
        data: optional 2D array (np.ndarray or xr.DataArray) stored as 'data', or an xr.Dataset of vars.
        output_objects: Prebuilt Output regions.
        meta: Dict or DetImageMeta; validated if provided.
        loader: optional Callable(filename) -> (data arrays, headers) used to load the data vars on demand.
        kwargs: Backward-compatible meta fields (merged if meta is a dict).

    Arithmetic: `image + 2`, `image - other_image`, `2 * image`, `1 / image` return a new DetImage, applying the
    operation output by output (see unary_op/binary_op). add/subtract/multiply/divide do the same and take
    inplace=True and the region/propagate_std/combine_masks options. The result keeps the labels (and 'count') of
    the left operand.
    """

    ndim: int = 2

    def __init__(
            self,
            data: Optional[XRDATA | np.ndarray] = None,
            output_objects: Optional[dict[str, Output]] = None,
            meta: Optional[DetImageMeta | dict[str, Any]] = None,
            *,
            loader: Optional[Callable] = None,
            **kwargs: Any,
    ):
        self.outputs: dict[str, Output] = {}
        self.meta: DetImageMeta | dict[str, Any] = {}
        self._ds: xr.Dataset = xr.Dataset()
        self._loader: Callable | None = None

        # Normalize meta to DetImageMeta if possible; allow empty until placement on focal plane.
        if meta is None and kwargs:
            meta = kwargs
        if isinstance(meta, DetImageMeta):
            self.meta: DetImageMeta | dict[str, Any] = meta
        elif isinstance(meta, dict) and meta:
            # Validate only if sufficient keys are present; otherwise store as-is.
            if {"name", "properties"}.issubset(meta.keys()):
                self.meta = DetImageMeta.model_validate(meta)
            else:
                self.meta = dict(meta)
        # Merge any additional kwargs into meta
        self.meta.update(kwargs)
        self.id: str | int = self.meta.get("name", "unknown")
        self.image_type: dict[str, Any] = self.meta.get(
            "image_type", {"type": "unknown"}
        )

        match data:
            case None:
                pass
            case xr.Dataset():
                self.set_vars(data)
            case np.ndarray() | xr.DataArray():
                self.set_var('data', data)
            case func if callable(func):
                raise TypeError("Pass a data loader with DetImage(loader=...) or set_loader(), not as data.")
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}. Must be xr.DataArray, xr.Dataset or "
                                f"np.ndarray.")
        if loader is not None:
            self.set_loader(loader)

        # Outputs
        if output_objects is None and self.is_loaded:
            logger.info("Creating default Output for DetImage (covers full array).")
            h, w = self.shape
            self.outputs["0"] = Output(
                id="0",
                input_array_axis=0,
                input_slice=(slice(0, h), slice(0, w)),
                output_slice=(slice(0, h), slice(0, w)),
                parent=self,
            )
        else:
            self.outputs.update(output_objects) if output_objects else {}
            for out_id, out in self.outputs.items():
                out.parent = self

    def add_output(self, output: Output, overwrite: bool = True):
        output.parent = self
        if output.id in self.outputs:
            logger.debug(
                f"Output with id {output.id} already exists, overwrite is set to {overwrite}."
            )
            if overwrite:
                self.outputs[output.id] = output
        else:
            self.outputs[output.id] = output

    ########################################## Loading ##########################################
    @property
    def is_loaded(self) -> bool:
        """ True if the data vars are in memory. """
        return 'data' in self._ds.data_vars

    @property
    def has_loader(self) -> bool:
        """ True if the data vars can be (re)loaded from disk with load(). """
        return self._loader is not None

    def set_loader(self, loader: Callable):
        """
        Set the function that loads the data vars on demand. It is called as loader(meta['filename']) and must
        return (data arrays, headers) indexable by each Output's input_array_axis.
        :param loader: Callable
        """
        if not callable(loader):
            raise TypeError(f"loader must be callable, got {type(loader).__name__}")
        self._loader = loader

    def load(self, force: bool = False):
        """
        Load the data vars with the loader, keeping existing masks and labels. Does nothing if already loaded,
        unless force is True.
        :param force: bool, reload even if the data vars are in memory.
        :raises ValueError: if there is no loader or no filename in meta.
        """
        if self.is_loaded and not force:
            return
        if self._loader is None:
            raise ValueError(f"DetImage {self.id} has no loader, cannot load data.")
        filename = self.meta.get("filename")
        if not filename:
            raise ValueError(f"DetImage {self.id} has no 'filename' in meta to load data from.")
        idata, iheaders = self._loader(filename)
        temp = np.zeros(self.shape)
        for out_id, output in self.outputs.items():
            hdr, seldata = output._select_header_and_data(iheaders, idata)
            temp[*output.output_slice] = seldata
            output.header = hdr
        loaded = ensure_dataarray(temp).to_dataset(name='data')
        self._ds = xr.merge([self._ds.drop_vars(self._names('data')), loaded], join='exact', compat='override',
                            combine_attrs='drop')

    def unload(self, discard: bool = False):
        """
        Drop the data vars to free memory; masks and labels stay. With a loader they are loaded again on the next
        access; without one they are gone for good, which has to be asked for with discard=True.
        :param discard: bool, allow unloading an image that has no loader (its data vars are lost).
        :raises ValueError: if there is no loader and discard is False.
        """
        if self._loader is None and not discard:
            raise ValueError(f"DetImage {self.id} has no loader; unloading would lose its data "
                             f"(pass discard=True to drop it anyway).")
        self._ds = self._ds.drop_vars(self._names('data'))

    def _ensure_loaded(self):
        if not self.is_loaded and self._loader is not None:
            self.load()

    ########################################## Introspection ##########################################
    def _names(self, kind: str) -> list[str]:
        return [name for name in self._ds.data_vars if var_kind(name) == kind]

    @property
    def dataset(self) -> xr.Dataset:
        """ The Dataset holding every var (data, masks, labels), loaded if needed. Write through set_var. """
        self._ensure_loaded()
        return self._ds

    @property
    def data(self) -> xr.DataArray:
        """ The main image array, 'data'. """
        return self.get_var('data')

    @property
    def data_names(self) -> list[str]:
        """ Names of the data vars (loads the data if needed). """
        self._ensure_loaded()
        return self._names('data')

    @property
    def mask_names(self) -> list[str]:
        """ Names of the mask vars. """
        return self._names('mask')

    @property
    def label_names(self) -> list[str]:
        """ Names of the label vars. """
        return self._names('label')

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the detector image: from meta['properties'], then meta['shape'], then the stored vars,
        then the outputs' output_slices. Raises ValueError if none of these are available.
        """
        if 'properties' in self.meta:
            if self.meta['properties'] and 'y_size' in self.meta['properties']:
                return self.meta['properties']['y_size'], self.meta['properties']['x_size']
        if 'shape' in self.meta:
            return self.meta['shape']
        if 'y' in self._ds.dims and 'x' in self._ds.dims:
            return self._ds.sizes['y'], self._ds.sizes['x']
        if (opshape := self._shape_from_outputs) is not None:
            return opshape
        raise ValueError("Cannot determine shape of DetImage from metadata or outputs.")

    @property
    def _shape_from_outputs(self) -> Optional[tuple[int, ...]]:
        if len(self.outputs) > 0:
            imsize = [0] * self.ndim
            for _, output in self.outputs.items():
                imsize[0] = max(imsize[0], output.output_slice[0].stop)
                imsize[1] = max(imsize[1], output.output_slice[1].stop)
            return tuple(imsize)
        return None

    def validate_size_from_outputs(self) -> Optional[bool]:
        """validate whether the size set in the image metadata matches with the size
           set in the images' Output objects

        returns
        -------

        bool
           If both Output objects and metadata properties are present, returns whether
           these agree on the total size of the image. If either of these are not present,
           comparison is not possible, return None

        """
        if not ("properties" in self.meta and "y_size" in self.meta["properties"]):
            return None
        propsize = tuple([self.meta["properties"][_] for _ in ["y_size", "x_size"]])

        if (opshape := self._shape_from_outputs) is not None:
            logger.debug("both size attributes present, comparing")
            return all(a == b for a, b in zip(propsize, opshape))
        return None

    def _full_coords(self) -> dict[str, Any]:
        """ The y/x coords of the stored vars, or integer coords from shape if nothing is stored yet. """
        if 'y' in self._ds.coords and 'x' in self._ds.coords:
            return {'y': self._ds.coords['y'].values, 'x': self._ds.coords['x'].values}
        h, w = self.shape
        return {'y': np.arange(h), 'x': np.arange(w)}

    ########################################## Reading ##########################################
    def get_var(self, name: str = 'data', region: Optional[tuple | dict] = None) -> xr.DataArray:
        """
        Return a data, mask or label var, whole or a region of it. Data vars are loaded if needed. The result may
        share memory with the image; write through set_var.
        :param name: str, var name.
        :param region: dict or tuple of pixel slices (stop exclusive), or None for the whole image.
        :return: xr.DataArray
        :raises KeyError: if the image has no var of that name.
        """
        if var_kind(name) == 'data':
            self._ensure_loaded()
        if name not in self._ds.data_vars:
            raise KeyError(f"DetImage {self.id} has no var '{name}'; its vars are {list(self._ds.data_vars)}")
        da = self._ds[name]
        return slice_data(da, region) if region is not None else da

    def get_mask(self, masks: MaskSelection = 'combined', region: Optional[tuple | dict] = None) -> xr.DataArray:
        """
        Return the OR of the selected masks as a new bool array; all False if no mask is selected or none exist.
        :param masks: 'combined' (all masks), a mask name, a list of mask names, or None (no masks).
        :param region: dict or tuple of pixel slices, or None for the whole image.
        :return: bool xr.DataArray
        :raises KeyError: for a mask name the image does not have.
        """
        names = _select_mask_names(masks, self.mask_names, f"DetImage {self.id}")
        mask = combine_masks(self._ds, names, self._full_coords())
        return slice_data(mask, region) if region is not None else mask

    def get_masked(self, name: str = 'data', masks: MaskSelection = 'combined',
                   region: Optional[tuple | dict] = None, mask_invalid: bool = False) -> np.ma.MaskedArray:
        """
        Return a data var as a numpy masked array, for numpy.ma calculations that skip masked pixels.
        :param name: str, data var name.
        :param masks: masks to apply, as in get_mask.
        :param region: dict or tuple of pixel slices, or None for the whole image.
        :param mask_invalid: bool, if True also mask NaN and inf pixels.
        :return: np.ma.MaskedArray
        """
        if var_kind(name) != 'data':
            raise ValueError(f"get_masked takes a data var, got the {var_kind(name)} var '{name}'")
        data = self.get_var(name, region)
        return as_masked_array(data.values, self.get_mask(masks, region).values, mask_invalid=mask_invalid)

    def _get_filled(self, name: str, masks: MaskSelection, fill_value=np.nan) -> xr.DataArray:
        """ A data var with masked pixels replaced by fill_value (for plotting). """
        data = self.get_var(name)
        if masks is None:
            return data
        return data.where(~self.get_mask(masks).values, other=fill_value)

    ########################################## Writing ##########################################
    def set_var(self, name: str, values: np.ndarray | xr.DataArray, region: Optional[tuple | dict] = None,
                combine: bool = False):
        """
        Write a whole var (region=None) or a region of it. The var kind comes from the name (see the class
        docstring) and the values are checked against it: masks must be bool, labels integer.
        Writing a region of a var that does not exist yet creates it first, filled with NaN (float data), False
        (mask) or 0 (label).
        :param name: str, var name.
        :param values: np.ndarray or xr.DataArray. A whole var must match the image shape; a region must match the
            region's shape, and an xr.DataArray must have the region's coords in the same order.
        :param region: dict or tuple of pixel slices (stop exclusive, steps kept), or None for the whole var.
        :param combine: bool, masks only: OR the values with the existing mask instead of replacing it.
        """
        kind = var_kind(name)
        if combine and kind != 'mask':
            raise ValueError(f"combine=True ORs masks; '{name}' is a {kind} var")
        if kind == 'data':
            self._ensure_loaded()
        if region is None:
            self._set_whole(name, kind, values, combine)
        else:
            self._set_region(name, kind, values, to_slicer(region, dims=('y', 'x')), combine)

    def _set_whole(self, name: str, kind: str, values, combine: bool):
        target = self._ds[name] if name in self._ds.data_vars else None
        if isinstance(values, xr.DataArray):
            arr = ensure_dataarray(values)
            if 'y' in self._ds.coords:
                coords = self._full_coords()
                if not (np.array_equal(arr['y'].values, coords['y']) and np.array_equal(arr['x'].values, coords['x'])):
                    raise ValueError(f"'{name}' has different y/x coords than the image; align it first")
        elif 'y' in self._ds.coords:
            values = np.asarray(values)
            coords = self._full_coords()
            expected = (len(coords['y']), len(coords['x']))
            if values.shape != expected:
                raise ValueError(f"'{name}' has shape {values.shape}, the image is {expected}")
            arr = xr.DataArray(values, dims=('y', 'x'), coords=coords)
        else:
            # first var of the image: it sets the coords
            arr = ensure_dataarray(np.asarray(values))
        # a whole var is replaced, so its dtype may change (only region writes can truncate)
        _check_dtype(name, kind, arr.dtype)
        if combine and target is not None:
            arr = arr | target
        self._ds[name] = arr

    def _set_region(self, name: str, kind: str, values, slicer: dict, combine: bool):
        if name not in self._ds.data_vars:
            coords = self._full_coords()
            shape = (len(coords['y']), len(coords['x']))
            if kind == 'mask':
                blank = np.zeros(shape, dtype=bool)
            elif kind == 'label':
                blank = np.zeros(shape, dtype=np.int32)
            elif np.issubdtype(np.asarray(values).dtype, np.integer):
                raise ValueError(f"integer data var '{name}' does not exist; set the whole var first")
            else:
                blank = np.full(shape, np.nan)
            self._ds[name] = xr.DataArray(blank, dims=('y', 'x'), coords=coords)
        target = self._ds[name]
        current = slice_data(target, slicer)
        if isinstance(values, xr.DataArray):
            values = ensure_dataarray(values)
            if not (np.array_equal(values['y'].values, current['y'].values)
                    and np.array_equal(values['x'].values, current['x'].values)):
                raise ValueError(f"values for '{name}' do not have the region's coords (in the region's order)")
            values = values.values
        values = np.asarray(values)
        if values.shape != current.shape:
            raise ValueError(f"values for '{name}' have shape {values.shape}, the region is {current.shape}")
        _check_dtype(name, kind, values.dtype, target.dtype)
        if combine:
            values = values | current.values
        target.loc[to_label_slicer(slicer)] = values

    def set_vars(self, values: Mapping[str, np.ndarray | xr.DataArray] | xr.Dataset):
        """
        Write several whole vars, see set_var.
        :param values: mapping of var name to values, or an xr.Dataset.
        """
        items = values.data_vars.items() if isinstance(values, xr.Dataset) else values.items()
        for name, value in items:
            self.set_var(name, value)

    def remove_vars(self, *names: str):
        """
        Drop vars of any kind except 'data' (use unload() for that).
        :raises KeyError: if a name is not a var of the image.
        """
        for name in names:
            var_kind(name)
            if name == 'data':
                raise ValueError("'data' can't be removed; use unload() to free it")
            if name not in self._ds.data_vars:
                raise KeyError(f"DetImage {self.id} has no var '{name}'")
        self._ds = self._ds.drop_vars(list(names))

    def ensure_float(self):
        """ Cast integer 'data' and 'std' arrays to float64, so that arithmetic written back into them is not truncated. """
        self._ensure_loaded()
        for var in ('data', 'std'):
            if var in self._ds.data_vars and not np.issubdtype(self._ds[var].dtype, np.floating):
                self._ds[var] = self._ds[var].astype(np.float64)

    def copy(self) -> DetImage:
        """
        Return an independent copy of this image: every var (data, masks, labels), meta and outputs (with their
        headers) are copied, and the copied outputs point to the new image. An unloaded image stays unloaded and shares
        the loader. Cheaper than copy.deepcopy, which also walks every Output's parent link.
        :return: DetImage
        """
        # meta and image_type are copied together so that an image_type shared with meta stays shared
        meta, image_type = deepcopy((self.meta, self.image_type))
        outputs = {
            out_id: out.model_copy(update={
                'parent': None,
                'header': deepcopy(out.header),
                **deepcopy(out.__pydantic_extra__ or {}),  # model_copy only copies extra fields shallowly
            })
            for out_id, out in self.outputs.items()
        }
        new = type(self)(output_objects=outputs, meta=meta)
        new.id = self.id
        new.image_type = image_type
        new._ds = self._ds.copy(deep=True)
        new._loader = self._loader
        return new

    ######################################## Arithmetic ########################################
    def unary_op(self, op: str, value: float | np.ndarray, inplace: bool = False, **kwargs) -> DetImage:
        """
        Apply `data <op> value` to every output (see Output.unary_op). Pixels outside every output are not changed.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param value: float, or np.ndarray broadcastable to the operated region of every output.
        :param inplace: bool, if True modify this image, else a copy.
        :param kwargs: passed to Output.unary_op (region, propagate_std).
        :return: the modified image (self if inplace).
        """
        target = self if inplace else self.copy()
        for output in target.outputs.values():
            output.unary_op(op, value, **kwargs)
        return target

    def binary_op(self, op: str, other: DetImage, inplace: bool = False, **kwargs) -> DetImage:
        """
        Apply `self <op> other` output by output, matching outputs by id (see Output.binary_op). Pixels outside every
        output are not changed.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param other: DetImage of the same class, with the same output ids.
        :param inplace: bool, if True modify this image, else a copy.
        :param kwargs: passed to Output.binary_op (region, combine_masks, propagate_std).
        :return: the modified image (self if inplace).
        """
        if type(other) is not type(self):
            raise TypeError(f"Cannot {op} a {type(other).__name__} and a {type(self).__name__}")
        if set(other.outputs) != set(self.outputs):
            raise ValueError(f"Images {self.id} and {other.id} have different outputs: "
                             f"{sorted(self.outputs)} vs {sorted(other.outputs)}")
        target = self if inplace else self.copy()
        for out_id, output in target.outputs.items():
            output.binary_op(op, other.outputs[out_id], **kwargs)
        return target

    def add(self, other: DetImage | float | np.ndarray, inplace: bool = False, **kwargs) -> DetImage:
        """ Add a number, an array or another DetImage. See unary_op() and binary_op(). """
        return self._dispatch('add', other, inplace, **kwargs)

    def subtract(self, other: DetImage | float | np.ndarray, inplace: bool = False, **kwargs) -> DetImage:
        """ Subtract a number, an array or another DetImage. See unary_op() and binary_op(). """
        return self._dispatch('subtract', other, inplace, **kwargs)

    def multiply(self, other: DetImage | float | np.ndarray, inplace: bool = False, **kwargs) -> DetImage:
        """ Multiply by a number, an array or another DetImage. See unary_op() and binary_op(). """
        return self._dispatch('multiply', other, inplace, **kwargs)

    def divide(self, other: DetImage | float | np.ndarray, inplace: bool = False, **kwargs) -> DetImage:
        """ Divide by a number, an array or another DetImage. See unary_op() and binary_op(). """
        return self._dispatch('divide', other, inplace, **kwargs)

    def _dispatch(self, op: str, other: DetImage | float | np.ndarray, inplace: bool, **kwargs) -> DetImage:
        if isinstance(other, DetImage):
            return self.binary_op(op, other, inplace=inplace, **kwargs)
        return self.unary_op(op, other, inplace=inplace, **kwargs)

    def _operate(self, op: str, other: DetImage | float | np.ndarray, reflected: bool = False) -> DetImage:
        """ Backs the arithmetic operators: returns a new DetImage. """
        if isinstance(other, DetImage):
            if type(other) is not type(self) or reflected:
                return NotImplemented
            return self.binary_op(op, other)
        if not self._is_operand(other):
            return NotImplemented
        return self.unary_op(op, other, reflected=reflected)

    ######################################## Reductions ########################################
    def reduce(self, stat: str | Callable = 'median', per_output: bool = True, region=None, axis: Optional[int] = None,
               name: str = 'data', masks: MaskSelection = 'combined', mask_invalid: bool = False) -> dict[str, Any] | Any:
        """
        Reduce the image to one statistic, per output or over the whole image.
        :param stat: str or Callable, as in Output.reduce.
        :param per_output: bool, if True return {output id: value} computed by Output.reduce, else a single value
            computed over the whole image.
        :param region: per output: a region name or slicer (see Output); whole image: a pixel slicer or None.
        :param axis: int or None, axis to collapse (0 = 'y', 1 = 'x'); None reduces to a single value.
        :param name: str, data var to reduce.
        :param masks: masks whose pixels are ignored, as in get_mask; None ignores no pixels.
        :param mask_invalid: bool, if True also ignore NaN and inf pixels.
        :return: dict of output id to value, or a single value.
        """
        if per_output:
            return {out_id: output.reduce(stat, region=region, axis=axis, name=name, masks=masks,
                                          mask_invalid=mask_invalid)
                    for out_id, output in self.outputs.items()}
        if masks is not None or mask_invalid:
            check_mask_aware(stat)
        return reduce_array(self.get_masked(name, masks, region, mask_invalid), stat, axis=axis)

    def stats(self, which: dict[str, Callable] = STATFUNCS, per_output: bool = True, region=None,
              axis: Optional[int] = None, name: str = 'data', masks: MaskSelection = 'combined',
              mask_invalid: bool = False, prepend_kw: str = '') -> dict[str, Any]:
        """
        Compute several statistics with core.image_stats.do_statistics, per output or over the whole image.
        :param which: dict of statistic name to function, default STATFUNCS.
        :param per_output: bool, if True return {output id: stats dict}, else one stats dict for the whole image.
        :param prepend_kw: str, prefix for the statistic names.
        Other parameters are as in reduce().
        :return: dict
        """
        if per_output:
            return {out_id: output.stats(which, region=region, axis=axis, name=name, masks=masks,
                                         mask_invalid=mask_invalid, prepend_kw=prepend_kw)
                    for out_id, output in self.outputs.items()}
        if masks is not None or mask_invalid:
            for stat in which.values():
                check_mask_aware(stat)
        masked = self.get_masked(name, masks, region, mask_invalid)
        return do_statistics(masked, which=which, axis=axis, prepend_kw=prepend_kw)

    def show(self, ax=None, save=None, name: str = 'data', masks: MaskSelection = 'combined', **imshow_kwargs):
        """
        Plot a data var of the image, with masked pixels left blank.
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :param save: str, optional path to save the plot. If None, the plot is not saved.
        :param name: str, data var to plot. Default is 'data'.
        :param masks: masks to blank out, as in get_mask; None plots every pixel.
        :param imshow_kwargs: dict, additional keyword arguments to pass to the imshow function.
        :return: Matplotlib Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        self._get_filled(name, masks).plot.imshow(ax=ax, **imshow_kwargs)
        if save is not None:
            ax.figure.savefig(save)
        return ax

    ########################################## Files ##########################################
    def to_netcdf(self, filepath: str | pathlib.Path, **kwargs):
        """
        Save the image to a netcdf file: every var (data, masks, labels) as a netcdf variable, and meta, image_type
        and outputs as attributes.
        :param filepath: str | pathlib.Path; '.nc' is appended if missing.
        :param kwargs: passed to xr.Dataset.to_netcdf.
        """
        ds_to_save = self.dataset.copy()  # shallow copy, so the attrs below do not end up on the image
        out_dict = {}
        outclass = Output.__name__
        for out_id, output in self.outputs.items():
            out_dict[out_id] = output.to_json(exclude={"parent"})
            outclass = output.__class__.__name__
        ds_to_save.attrs = {
            "eregion_format": NETCDF_FORMAT,
            "meta": self.meta.to_json(),
            "image_type": json.dumps(self.image_type),
            "outputs": json.dumps(out_dict),
            "output_class": outclass,
        }

        filepath = pathlib.Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".nc":
            # append rather than with_suffix(), which would cut names like 'flat_0.5s' at the last dot
            filepath = filepath.with_name(filepath.name + ".nc")
        ds_to_save.to_netcdf(str(filepath), mode="w", **kwargs)

    @classmethod
    def from_netcdf(cls, filepath: str | pathlib.Path):
        """
        Load a DetImage saved with to_netcdf(). Files written before eregion_format 2 (masks and labels listed in
        the 'mask_keys'/'label_keys' attributes) are converted: masks are cast to bool and renamed to '<name>_mask',
        labels renamed to '<name>_label' ('cosmic_ray_labels' becomes 'cosmic_ray_label').
        :param filepath: str, path to the netcdf file.
        :return: DetImage object.
        """
        loaded_ds = xr.load_dataset(str(filepath))
        attrs = dict(loaded_ds.attrs)
        loaded_ds.attrs = {}
        if attrs.get("eregion_format") != NETCDF_FORMAT:
            loaded_ds = cls._convert_legacy(loaded_ds, attrs, filepath)

        outputs = {}
        outclass = load_class(attrs["output_class"], default_module=output_module)
        for out_id, output_dict in json.loads(attrs["outputs"]).items():
            outputs[out_id] = outclass.from_json(output_dict)

        detimg = cls(output_objects=outputs, meta=DetImageMeta.from_json(attrs["meta"]))
        detimg.set_vars(ensure_dataarray(loaded_ds))
        detimg.image_type = json.loads(attrs["image_type"])
        return detimg

    @staticmethod
    def _convert_legacy(ds: xr.Dataset, attrs: dict, filepath) -> xr.Dataset:
        """ Rename and cast the masks/labels of a file written before eregion_format 2. """
        mask_keys = json.loads(attrs.get("mask_keys", "[]"))
        label_keys = json.loads(attrs.get("label_keys", "[]"))
        listed = set(mask_keys) | set(label_keys)
        unlisted = [name for name in ds.data_vars if name not in listed and var_kind(name) != 'data']
        if unlisted:
            raise ValueError(f"{filepath}: data vars {unlisted} end in '{MASK_SUFFIX}' or '{LABEL_SUFFIX}'; "
                             f"they can't be told apart from masks and labels")
        renames = {}
        for key in mask_keys:
            if not key.endswith(MASK_SUFFIX):
                renames[key] = key + MASK_SUFFIX
        for key in label_keys:
            if key.endswith(LABEL_SUFFIX + "s"):
                renames[key] = key[:-1]
            elif not key.endswith(LABEL_SUFFIX):
                renames[key] = key + LABEL_SUFFIX
        ds = ds.rename(renames)
        for key in mask_keys:
            name = renames.get(key, key)
            # masks were saved as float with NaN outside the outputs that set them
            ds[name] = xr.where(ds[name].notnull(), ds[name], 0).astype(bool)
        if renames:
            logger.warning(f"{filepath} was saved by an older eregion; renamed {renames}")
        return ds


# Output.parent is annotated with DetImage, which output.py can only import for type checking (image.py imports
# output.py), so the Output models are completed here, once DetImage exists.
for _output_class in (output_module.Output, output_module.CCDOutput,
                      output_module.CMOSOutput, output_module.IRDetectorOutput):
    _output_class.model_rebuild(_types_namespace={"DetImage": DetImage})
