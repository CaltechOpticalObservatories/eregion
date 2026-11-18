from __future__ import annotations
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
    set_slice_in_data,
    XRDATA
)
from eregion.core.image_stats import STATFUNCS, as_masked_array, check_mask_aware, do_statistics, reduce_array
from eregion.datamodels.meta import DetImageMeta
from eregion.datamodels import output as output_module
from eregion.datamodels.output import Output
from eregion.datamodels.operators import ArithmeticOperators
from eregion.utils import load_class

logger = logging.getLogger(__name__)

############################################## DETIMAGE CLASS #######################################################
class DetImage(ArithmeticOperators):
    """
    Base detector image holding 2D (only spatial) pixel data and outputs. The data arrays are stored in .data
    as an xarray.Dataset. Since xarray.Dataset requires a data variable name to store arrays in, the default name used
    for the main image array is 'data'. Additional arrays commonly stored are 'std' for standard deviation and
    'count' for number of pixels used when storing combined multiple 2D arrays.

    Parameters:
        data: optional, 2D image array (np.ndarray or xr.DataArray or xr.Dataset) or Callable
        output_objects: Prebuilt Output regions.
        meta: Dict or DetImageMeta; validated if provided.
        kwargs: Backward-compatible meta fields (merged if meta is a dict).

    Attributes:
        ndim: int - Number of dimensions (fixed at 2 for spatial).
        outputs: dict[str, Output] - Mapping of output IDs to Output objects.
        meta: DetImageMeta or dict - Metadata for the detector image.
        id: str | int - Identifier for the corresponding detector taken from 'name' key in detector config.
        image_type: dict[str, Any] - Mapping of any image identifying keys to their values like 'type', 'exptime', 'mode', etc.
        masks: xr.Dataset or None - Full image boolean masks built from the outputs' masks.
        labels: xr.Dataset or None - Full image integer label images (e.g. 'cosmic_ray_labels'), set with set_labels().
        _data: Internal attribute for storing image data when loaded.
        _dataloader: Internal variable for storing data loader Callable for on-demand loading.

    Arithmetic: `image + 2`, `image - other_image`, `2 * image`, `1 / image` return a new DetImage, applying the
    operation output by output (see unary_op/binary_op). add/subtract/multiply/divide do the same and take
    inplace=True and the region/propagate_std/combine_masks options.
    """

    ndim: int = 2

    def __init__(
            self,
            data: Optional[XRDATA | np.ndarray | Callable] = None,
            output_objects: Optional[dict[str, Output]] = None,
            meta: Optional[DetImageMeta | dict[str, Any]] = None,
            **kwargs: Any,
    ):

        self.outputs: dict[str, Output] = {}
        self.meta: DetImageMeta | dict[str, Any] = {}
        self._data: xr.Dataset | None = None
        self._dataloader: Callable | None = None
        self.masks: xr.Dataset | None = None
        self.labels: xr.Dataset | None = None

        if data is not None:
            self.set_data(data)

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

        # Outputs
        if output_objects is None and self._data is not None:
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

    def set_data(self, data: XRDATA | np.ndarray | Callable):
        """
        If data is a callable, store it as _dataloader for on-demand loading. If data is an array, store it as _data.
        """
        match data:
            case func if callable(func):
                logger.debug("Callable provided for on-demand data loading.")
                self._dataloader = data
            case np.ndarray() | xr.DataArray():
                logger.debug("np.ndarray provided, converting to xr.Dataset storing directly in _data.")
                self._data = ensure_dataarray(data).to_dataset(name='data')
            case xr.Dataset():
                data = ensure_dataarray(data)
                if 'data' not in data.data_vars:
                    raise ValueError("xr.Dataset provided must contain a data_var named 'data'.")
                else:
                    self._data = data
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}. "
                                f"Must be xr.DataArray, xr.Dataset, np.ndarray, or Callable.")

    def _load_from_disk(self):
        """
        Load data from disk into _data attribute. self.data returns xarray _data when called.
        """
        if self._dataloader is not None:
            assert self.meta["filename"] is not None
            assert len(self.meta["filename"]) > 0
            idata, iheaders = self._dataloader(self.meta["filename"])
            temp = np.zeros(self.shape)
            for out_id, output in self.outputs.items():
                hdr, seldata = output._select_header_and_data(iheaders, idata)
                temp[*output.output_slice] = seldata
                output.header = hdr
            temp = ensure_dataarray(temp)
            self._data = temp.to_dataset(name='data')
            del idata, iheaders
        else:
            raise ValueError(
                "No dataloader function provided for this DetImage, cannot load data."
            )

    def unload(self):
        """
        Unload data from _data attribute to free up memory.
        """
        del self._data
        self._data = None

    def get_data(self, data_var: Optional[str | Literal['all']] = 'data') -> XRDATA:
        """
        Return the data array or dataset from _data attribute. If _data is None, attempt to load from disk.
        :param data_var: str or 'all', name of the data variable in _data to return. If 'all', returns the entire dataset.
        """
        if self._data is None:
            self._load_from_disk()
        data_var = data_var if data_var is not None else 'data'
        if data_var == 'all':
            return self._data
        else:
            if data_var not in self._data.data_vars:
                raise ValueError(f"Data variable '{data_var}' not found in DetImage data.")
            return self._data[data_var]

    @property
    def data(self):
        """ Return the main image array 'data' from the xr.Dataset stored in _data. If _data is None, attempt to load from disk. """
        return self.get_data('data')

    def set_data_slice(self,
                       slicedata: XRDATA,
                       slicer: dict[str, slice],
                       data_var: Optional[str | Literal['all']] = 'data'):
        """
        Set a slice (specified by slicer) in the xr.Dataset in _data attribute to provided slicedata.
        This is useful for updating specific regions of the detector image.
        :param slicedata: Data to set in the specified slice
        :param slicer: Dictionary of slices specifying the region in _data to update. Keys should match the dimensions of _data.
        :param data_var: Name of the data variable in _data to update. If 'all', slicedata must be an xr.Dataset with matching data_vars.
        :return:
        """
        if self._data is None:
            raise ValueError("No data loaded in DetImage to set a slice.")
        match (data_var, slicedata):
            case ('all', xr.Dataset()):
                self._data = set_slice_in_data(slicedata, self._data, slicer)
            case ('all', _):
                raise TypeError("When data_var is 'all', slicedata must be an xarray.Dataset.")
            case (_, xr.DataArray()):
                self._data[data_var] = set_slice_in_data(slicedata, self._data[data_var], slicer)
            case (_, xr.Dataset()):
                if data_var not in slicedata.data_vars:
                    raise ValueError(f"Data variable '{data_var}' not found in provided slicedata xr.Dataset.")
                self._data[data_var] = set_slice_in_data(slicedata[data_var], self._data[data_var], slicer)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the detector image. First check meta['properties'], then meta['shape'], then _data.shape,
        then outputs' output_slices. If none of these are available, raise ValueError.
        """
        if 'properties' in self.meta:
            if self.meta['properties'] and 'y_size' in self.meta['properties']:
                return self.meta['properties']['y_size'], self.meta['properties']['x_size']
        if 'shape' in self.meta:
            return self.meta['shape']
        if "properties" in self.meta:
            if self.meta["properties"] and "y_size" in self.meta["properties"]:
                return (
                    self.meta["properties"]["y_size"],
                    self.meta["properties"]["x_size"],
                )
        if "shape" in self.meta:
            return self.meta["shape"]
        if self._data is not None:
            return self._data['data'].shape
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

    def set_labels(self, label_key: str, labels: xr.DataArray) -> None:
        """
        Store a full image integer label array (0 is background) in self.labels under label_key, replacing any
        existing labels with the same key.
        :param label_key: str, name of the label data_var in self.labels.
        :param labels: xr.DataArray of integers with the same dims and shape as self.data.
        :return: None
        """
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"labels must have an integer dtype, got {labels.dtype}.")
        data = self.data
        if labels.dims != data.dims or labels.shape != data.shape:
            raise ValueError(f"labels dims/shape {labels.dims}{labels.shape} do not match image data "
                             f"{data.dims}{data.shape}.")
        if self.labels is None:
            self.labels = labels.to_dataset(name=label_key)
        else:
            self.labels[label_key] = labels

    def _build_full_mask(self) -> bool:
        """
        Build a full mask from the individual output masks.
        :return: True if a full mask was built, False otherwise.
        """
        self.masks = xr.Dataset(coords=self.data.coords)
        for out_id, output in self.outputs.items():
            if output.masks is not None:
                # merge output mask dataset with masks, output mask coords are a subset of img masks
                # Using fill_value=np.nan so the merge doesn't raise conflict error (compat="override" picks the first dataset's value so can't use that)
                self.masks = self.masks.merge(
                    output.masks, join="outer", fill_value=np.nan, compat="no_conflicts")
        # After masks are merged, replace NaNs with zeros (since this is an explicit masks array it should have 1/0 for mask/not-mask and Nan does not imply masked.
        for dvar in self.masks.data_vars:
            da = self.masks[dvar]
            self.masks[dvar] = da.where(da.notnull(), other=False)

        if len(self.masks.data_vars) == 0:
            self.masks = None
            return False
        return True

    def get_mask(self, mask_key: str | Literal['all', 'combined'] = 'combined') -> XRDATA | None:
        """
        Return the mask array or dataset. Build masks from outputs if self.masks is empty.
        :param mask_key: str or 'all' or 'combined'. Returns dataset if 'all'. Returns DataArray if key is specified. Returns OR combined mask from all keys if 'combined'.
        :return: xr.DataArray or xr.Dataset
        """
        built = self._build_full_mask() if self.masks is None else True
        if built:
            match mask_key:
                case 'all':
                    return self.masks
                case 'combined':
                    return self.combined_mask
                case k if k in self.masks:
                    return self.masks[k]
                case k:
                    logger.error(f"{k} is not present in the masks dataset. Available keys are {list(self.masks.data_vars)}")
                    return None
        else:
            logger.warning(f"No masks found in any of the outputs to build the full mask from.")
            return None

    @property
    def combined_mask(self):
        built = self._build_full_mask() if self.masks is None else True
        if built:
            combmask = np.any(self.masks.to_array().to_numpy(), axis=0)
            return xr.DataArray(combmask, dims=self.masks.dims, coords=self.masks.coords)
        else:
            logger.error("No masks to combine.")
            return None

    def get_masked_data(self, data_var: str = 'data',  mask_key: str = 'combined', fill_value = np.nan) -> xr.DataArray:
        mask = self.get_mask(mask_key)
        datarr: xr.DataArray = self.get_data(data_var)
        if mask is not None:
            return datarr.where(~mask.to_numpy(), other=fill_value)
        else:
            return datarr

    def has_data_var(self, data_var: str) -> bool:
        """ Return True if the image data (loaded on demand) has a data variable named data_var. """
        return data_var in self.get_data('all').data_vars

    def ensure_float(self):
        """ Cast integer 'data' and 'std' arrays to float64, so that arithmetic written back into them is not truncated. """
        for var in ('data', 'std'):
            if self.has_data_var(var) and not np.issubdtype(self._data[var].dtype, np.floating):
                self._data[var] = self._data[var].astype(np.float64)

    def copy(self) -> DetImage:
        """
        Return an independent copy of this image: data, meta, outputs (with their headers and masks) are copied, and
        the copied outputs point to the new image. An image that is not loaded yet stays unloaded and shares the
        data loader. Cheaper than copy.deepcopy, which also walks every Output's parent link.
        :return: DetImage
        """
        # meta and image_type are copied together so that an image_type shared with meta stays shared
        meta, image_type = deepcopy((self.meta, self.image_type))
        outputs = {
            out_id: out.model_copy(update={
                'parent': None,
                'header': deepcopy(out.header),
                'masks': out.masks.copy(deep=True) if out.masks is not None else None,
                **deepcopy(out.__pydantic_extra__ or {}),  # model_copy only copies extra fields shallowly
            })
            for out_id, out in self.outputs.items()
        }
        new = type(self)(output_objects=outputs, meta=meta)
        new.id = self.id
        new.image_type = image_type
        new._data = self._data.copy(deep=True) if self._data is not None else None
        new._dataloader = self._dataloader
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
    def reduce(self, stat: str | Callable = 'median', per_output: bool = True, **kwargs) -> dict[str, Any] | Any:
        """
        Reduce the image to one statistic, per output or over the whole image.
        :param stat: str or Callable, as in Output.reduce.
        :param per_output: bool, if True return {output id: value} computed by Output.reduce, else a single value
            computed over the whole image (only region='full' applies then).
        :param kwargs: region, axis, data_var, mask_key, use_mask, mask_invalid; see Output.reduce.
        :return: dict of output id to value, or a single value.
        """
        if per_output:
            return {out_id: output.reduce(stat, **kwargs) for out_id, output in self.outputs.items()}
        if kwargs.get('use_mask', True) or kwargs.get('mask_invalid', False):
            check_mask_aware(stat)
        return reduce_array(self._masked_image(**kwargs), stat, axis=kwargs.get('axis'))

    def stats(self, which: dict[str, Callable] = STATFUNCS, per_output: bool = True, prepend_kw: str = '',
              **kwargs) -> dict[str, Any]:
        """
        Compute several statistics with core.image_stats.do_statistics, per output or over the whole image.
        :param which: dict of statistic name to function, default STATFUNCS.
        :param per_output: bool, if True return {output id: stats dict}, else one stats dict for the whole image.
        :param prepend_kw: str, prefix for the statistic names.
        :param kwargs: region, axis, data_var, mask_key, use_mask, mask_invalid; see Output.reduce.
        :return: dict
        """
        if per_output:
            return {out_id: output.stats(which, prepend_kw=prepend_kw, **kwargs)
                    for out_id, output in self.outputs.items()}
        if kwargs.get('use_mask', True) or kwargs.get('mask_invalid', False):
            for stat in which.values():
                check_mask_aware(stat)
        return do_statistics(self._masked_image(**kwargs), which=which, axis=kwargs.get('axis'), prepend_kw=prepend_kw)

    def get_masked_array(self, data_var: str = 'data', mask_key: str = 'combined',
                         mask_invalid: bool = False) -> np.ma.MaskedArray:
        """
        Return the whole image as a numpy masked array, for numpy.ma based calculations that skip masked pixels.
        Unlike get_masked_data, masked pixels are not replaced by a fill value.
        :param data_var: str, data variable to return.
        :param mask_key: str, mask to apply ('combined' ORs all masks; nothing is masked if the image has no masks).
        :param mask_invalid: bool, if True also mask NaN and inf pixels.
        :return: np.ma.MaskedArray
        :raises KeyError: if mask_key is not 'combined' and not one of the image's masks.
        """
        has_masks = self.masks is not None or self._build_full_mask()
        if mask_key == 'combined':
            mask = self.combined_mask if has_masks else None
        elif has_masks and mask_key in self.masks:
            mask = self.masks[mask_key]
        else:
            available = list(self.masks.data_vars) if has_masks else []
            raise KeyError(f"DetImage {self.id} has no mask '{mask_key}'; its masks are {available} (or 'combined')")
        return as_masked_array(self.get_data(data_var).values, mask.values if mask is not None else None,
                               mask_invalid=mask_invalid)

    def _masked_image(self, region: str = 'full', axis: Optional[int] = None, data_var: str = 'data',
                      mask_key: str = 'combined', use_mask: bool = True,
                      mask_invalid: bool = False) -> np.ma.MaskedArray:
        if region != 'full':
            raise ValueError(f"Only region='full' applies to the whole image, got '{region}'; use per_output=True")
        if not use_mask:
            return as_masked_array(self.get_data(data_var).values, mask_invalid=mask_invalid)
        return self.get_masked_array(data_var, mask_key=mask_key, mask_invalid=mask_invalid)

    def show(
            self,
            ax=None,
            save=None,
            with_mask=True,
            mask_key='combined',
            data_var='data',
            **imshow_kwargs
    ):
        """
        Plot the detector image data with optional mask overlay.
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :param save: str, optional path to save the plot. If None, the plot is not saved.
        :param with_mask: bool, if True, overlay the mask specified by mask_key on the image if available.
        :param mask_key: str, key in self.masks to use for masking the image. Default is 'sigma_clip_mask'.
        :param data_var: str, name of the data variable in the DetImage.data to plot. Default is 'data'.
        :param imshow_kwargs: dict, additional keyword arguments to pass to the imshow function.
        :return: Matplotlib Axes object containing the plot.
        """
        if self.data is None:
            raise ValueError("DetImage has no data to show.")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

        if with_mask:
            data = self.get_masked_data(data_var, mask_key, fill_value=np.nan)
        else:
            data = self.get_data(data_var)
        im = data.plot.imshow(ax=ax, **imshow_kwargs)

        if save is not None:
            ax.figure.savefig(save)
        return ax

    def to_netcdf(self, filepath: str | pathlib.Path, **kwargs):
        """
        Save the object to disk in netcdf format. Use xr.Dataset to hold everything. The .data and .masks are in data_vars,
        the .meta and .outputs in attrs.
        :param filepath: str | pathlib.Path
        """
        data = self.get_data('all')
        if isinstance(data, xr.DataArray):
            ds_to_save = data.to_dataset(name='data')
        else:
            # shallow copy so adding masks/labels below does not change self._data
            ds_to_save = data.copy()
        # add masks if they exist
        if self.masks is not None:
            ds_to_save.update(self.masks)
            ds_to_save.attrs['mask_keys'] = json.dumps(list(self.masks.data_vars.keys()))
        else:
            ds_to_save.attrs['mask_keys'] = json.dumps([])
        # add labels if they exist
        if self.labels is not None:
            ds_to_save.update(self.labels)
            ds_to_save.attrs['label_keys'] = json.dumps(list(self.labels.data_vars.keys()))
        else:
            ds_to_save.attrs['label_keys'] = json.dumps([])
        # convert meta to dict to store in attrs
        ds_to_save.attrs["meta"] = self.meta.to_json()
        ds_to_save.attrs["image_type"] = json.dumps(self.image_type)
        # convert outputs to dict of dicts
        out_dict = {}
        for out_id, output in self.outputs.items():
            op = output.to_json(exclude={"masks", "parent"})
            out_dict[out_id] = op
            outclass = output.__class__.__name__
        ds_to_save.attrs["outputs"] = json.dumps(out_dict)
        ds_to_save.attrs["output_class"] = outclass

        filepath = pathlib.Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".nc":
            # append rather than with_suffix(), which would cut names like 'flat_0.5s' at the last dot
            filepath = filepath.with_name(filepath.name + ".nc")
        ds_to_save.to_netcdf(str(filepath), mode="w", **kwargs)

    @classmethod
    def from_netcdf(cls, filepath: str | pathlib.Path):
        """
        Load a DetImage object from a netcdf file saved with to_netcdf().
        :param filepath: str, path to the netcdf file.
        :return: DetImage object.
        """
        loaded_ds = xr.load_dataset(str(filepath))
        mask_keys = json.loads(loaded_ds.attrs['mask_keys'])
        masks = loaded_ds[mask_keys] if len(mask_keys) > 0 else None
        # files saved before labels were added have no 'label_keys' attr
        label_keys = json.loads(loaded_ds.attrs.get('label_keys', '[]'))
        labels = loaded_ds[label_keys] if len(label_keys) > 0 else None
        data = loaded_ds.drop_vars(mask_keys + label_keys)
        meta = DetImageMeta.from_json(loaded_ds.attrs['meta'])

        outputs = {}
        outclass = load_class(loaded_ds.attrs["output_class"], default_module=output_module)
        outputs_attr = json.loads(loaded_ds.attrs["outputs"])
        for out_id, output_dict in outputs_attr.items():
            output = outclass.from_json(output_dict)
            # extract subdataset for output masks
            output.masks = slice_data(masks, output.output_slice) if masks is not None else None
            outputs[out_id] = output

        detimg = cls(data=data, output_objects=outputs, meta=meta)
        detimg.masks = masks
        detimg.labels = labels
        detimg.image_type = json.loads(loaded_ds.attrs["image_type"])
        return detimg


# Output.parent is annotated with DetImage, which output.py can only import for type checking (image.py imports
# output.py), so the Output models are completed here, once DetImage exists.
for _output_class in (output_module.Output, output_module.CCDOutput,
                      output_module.CMOSOutput, output_module.IRDetectorOutput):
    _output_class.model_rebuild(_types_namespace={"DetImage": DetImage})
