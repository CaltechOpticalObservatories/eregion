from __future__ import annotations
from typing import Optional, Any, Literal, Callable, Generic, TypeVar, Self
from pydantic import Field, ConfigDict, field_serializer
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import glob2
import json
from copy import deepcopy

from .mappable import Mappable
from eregion.utils import (ensure_dataarray, slice_data, ensure_numpy, configure_logger,
                           decrease_slicer_stop_index, set_slice_in_data, XRDATA)

logger = configure_logger(__name__)


############################################### META CLASSES FOR DETIMAGE #############################################
class DetectorProperties(Mappable):
    """
    Physical and sampling properties for a detector tile.
    """
    pixel_size: float = Field(gt=0, description="Pixel size in mm.")
    x_size: int = Field(gt=0, description="Tile width in pixels.")
    y_size: int = Field(gt=0, description="Tile height in pixels.")

    model_config = ConfigDict(extra="allow")


class FocalPlanePosition(Mappable):
    """
    Center position of the detector on the focal plane (same units as pixel_size, typically mm).
    """
    x_cen: float = Field(..., description="Center X in length units.")
    y_cen: float = Field(..., description="Center Y in length units.")

    model_config = ConfigDict(extra="allow")


class DetImageMeta(Mappable):
    """
    Metadata for a detector image, validated for focal-plane assembly.
    """
    name: str = Field(default=None)
    filename: Optional[str] = Field(default=None)
    properties: DetectorProperties
    focal_plane_position: Optional[FocalPlanePosition]
    image_type: dict[str, Any] = Field(default_factory=dict,
                                       description="Dictionary of identifying keys for this image, e.g. type, exptime, mode, etc.")

    model_config = ConfigDict(extra="allow")


############################################### OUTPUT BASE CLASS #####################################################
class Output(Mappable):
    """
    One amplifier/output region within a detector image.

    """
    id: str = Field(..., alias="id")
    input_array_axis: int = Field(..., alias="ext_id",
                                  description="Axis index in the input array, "
                                              "or extension ID in FITS file which contains the data for this output.")
    input_slice: tuple[slice, ...] = Field(..., alias="ext_slice",
                                           description="List of Slice objects defining the portion of the data array at input_array_axis "
                                                       "in input array or FITS that corresponds to this Detector Output.")
    output_slice: tuple[slice, ...] = Field(..., alias="data_slice",
                                            description="List of Slice objects defining the portion of the full detector data array "
                                                        "that this Detector Output maps to.")
    header: Optional[fits.Header | dict] = Field(default_factory=fits.Header,
                                                 description="FITS header in dict form for this output if available.")
    parent: Optional["DetImage"] = Field(default=None, description="Parent detector image.", exclude=True)
    masks: Optional[xr.Dataset] = Field(default=None,
                                        description="Optional xr.Dataset containing masks for this output.",
                                        exclude=True)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

    @field_serializer("header", when_used="json")
    def serialize_header(self, value: fits.Header | dict | None) -> dict | None:
        return dict(value) if isinstance(value, fits.Header) else value

    def get_output_data(self, data_var: Optional[str | Literal['all']] = 'data') -> XRDATA:
        """
        Return the array corresponding to this output from the parent DetImage's .data attribute using the output_slice.
        :param data_var: str or 'all'
            Name of the data variable in the DetImage.data (xr.Dataset) to slice. If 'all', slices the entire dataset.
        :return: xr.DataArray or xr.Dataset corresponding to this output.
        """
        if self.parent is None or getattr(self.parent, "data", None) is None:
            raise ValueError("Attach this Output to a DetImage with valid data.")
        return slice_data(self.parent.get_data(data_var), self.output_slice)

    @property
    def data(self):
        """
        Return the array corresponding to this output from the parent DetImage's .data attribute using the output_slice.
        DetImage.data is expected to be a xr.Dataset with a data_var named 'data'.
        """
        return slice_data(self.parent.data, self.output_slice)

    @property
    def image_region(self) -> dict[str, slice]:
        """Return the slice defining image region (i.e. only the active light capturing pixels) for this output."""
        return {axis: slc for axis, slc in zip(['y', 'x'], self.output_slice)}

    def get_image_region(self,
                         return_masks: bool = False,
                         data_var: Optional[str | Literal['all']] = 'data') -> tuple[XRDATA, Optional[XRDATA]]:
        """
        Return the image region data and optionally the corresponding masks for this output from the parent DetImage.
        :param return_masks: bool, if True, return the corresponding masks for this output if available.
        :param data_var: str or 'all'
            Name of the data variable in the DetImage.data (xr.Dataset) to slice. If 'all', slices the entire dataset.
        :return: tuple of (image region data, image region masks) where masks is None if return_masks is False
        or no masks are available.
        """
        imslc = self.image_region
        imdata = slice_data(self.parent.get_data(data_var), imslc)
        immask = slice_data(self.masks, imslc) if (return_masks and self.masks is not None) else None
        return imdata, immask

    def set_data_in_parent(self,
                           new_data: XRDATA,
                           slicer: Optional[dict[str, slice]] = None,
                           data_var: Optional[str | Literal['all']] = 'data'):
        """
        Set the data for this output in the parent DetImage's .data attribute using the provided new_data and slicer.
        :param new_data: xr.DataArray or xr.Dataset containing the new data to set for this output.
        :param slicer: dict
            Mapping of dimension names to slice objects defining the region of the parent DetImage's data to update.
            If None, uses the output_slice of this output.
        :param data_var: str or 'all'
            Name of the data variable in the DetImage.data (xr.Dataset) to update. If 'all', updates the entire dataset.
        """
        if self.parent is None or getattr(self.parent, "data", None) is None:
            raise ValueError("Attach this Output to a DetImage with valid data.")
        new_data = ensure_dataarray(new_data)
        # Assign new data to the appropriate slice in the parent DetImage
        slicer = {new_data.dims[i]: slc for i, slc in enumerate(self.output_slice)} if slicer is None else slicer
        self.parent.set_data_slice(new_data, slicer, data_var=data_var)

    def show(self, ax=None, save=None, with_mask=True, mask_key='sigma_clip_mask', data_var='data', **imshow_kwargs):
        """
        Plot this output's image.
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :param save: Optional[str]
            Path to save the plot.
        :param with_mask: bool, if True, overlay the mask specified by mask_key on the image if available.
        :param mask_key: str, key in self.masks to use for masking the image. Default is 'sigma_clip_mask'.
        :param data_var: str, name of the data variable in the DetImage.data to plot. Default is 'data'.
        :param imshow_kwargs: dict, additional keyword arguments to pass to xarray.DataArray.plot.imshow().
        :return: Matplotlib Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

        data = slice_data(self.parent.get_data(data_var), self.output_slice)

        if with_mask and self.masks is not None and mask_key in self.masks:
            arr = np.ma.masked_array(data=data.values, mask=self.masks[mask_key].values)
            arr = arr.filled(0)
            temp = xr.DataArray(data=arr, coords=data.coords, dims=data.dims)
        else:
            temp = data

        im = temp.plot.imshow(ax=ax, **imshow_kwargs)

        if save is not None:
            ax.figure.savefig(save)
        return ax


class CCDOutput(Output):
    """
    CCD-specific output region with prescan/overscan info.
    """
    serial_prescan: slice = Field(slice(None),
                                  description="Slice object defining the serial prescan region for this output.")
    serial_overscan: slice = Field(slice(None),
                                   description="Slice object defining the serial overscan region for this output.")
    parallel_prescan: slice = Field(slice(None),
                                    description="Slice object defining the parallel prescan region for this output.")
    parallel_overscan: slice = Field(slice(None),
                                     description="Slice object defining the parallel overscan region for this output.")
    parallel_axis: Literal['x', 'y'] = Field('y',
                                             description="Name of the parallel readout axis for this output ('x' or 'y').")
    readout_pixel: tuple[int, int] = Field((0, 0),
                                           description="Tuple defining the (y, x) pixel coordinates of the readout amplifier "
                                                       "for this output in the full detector array.")

    @property
    def serial_axis(self) -> Literal['x', 'y']:
        return 'x' if self.parallel_axis == 'y' else 'y'

    @property
    def parallel_axint(self):
        return 0 if self.parallel_axis == 'y' else 1

    @property
    def serial_axint(self):
        return 0 if self.serial_axis == 'y' else 1

    @property
    def image_region(self) -> dict[str, slice]:
        parallel_step = -1 if self.parallel_prescan.stop > self.parallel_overscan.start else 1
        im_slc_parallel = slice(self.parallel_prescan.stop, self.parallel_overscan.start, parallel_step)
        serial_step = -1 if self.serial_prescan.stop > self.serial_overscan.start else 1
        im_slc_serial = slice(self.serial_prescan.stop, self.serial_overscan.start, serial_step)
        return {self.parallel_axis: im_slc_parallel, self.serial_axis: im_slc_serial}

    def get_scan(self,
                 axis: Literal['serial', 'parallel'],
                 kind: Literal['prescan', 'overscan'],
                 corner: bool = False,
                 data_var: Optional[str | Literal['all']] = 'data'
                 ) -> XRDATA:
        """
        Slice the data array to get the scan region (prescan or overscan) along the specified axis (serial or parallel).
        :param axis: serial or parallel
        :param kind: prescan or overscan
        :param corner: True to include the corner region (intersection of prescan and overscan) in the returned slice,
        False to exclude it.
        :param data_var: Name of the data variable in the DetImage to slice. If 'all', slices the entire dataset.
        :return: sliced xr.DataArray corresponding to the requested scan region.
        """
        slc = getattr(self, f"{axis}_{kind}")
        slicer = {dim: self.output_slice[i] for i, dim in enumerate(['y', 'x'])}
        slicer[getattr(self, f"{axis}_axis")] = slc
        if not corner:
            # Exclude the corner region by adjusting the slice to avoid overlap with the other axis scans
            other_axis = "serial" if axis == "parallel" else "parallel"
            other_prescan = getattr(self, f"{other_axis}_prescan")
            other_overscan = getattr(self, f"{other_axis}_overscan")
            step = -1 if other_prescan.stop > other_overscan.start else 1
            slicer[getattr(self, f"{other_axis}_axis")] = slice(other_prescan.stop, other_overscan.start, step)
        return slice_data(self.parent.get_data(data_var), slicer)

    def get_prescan(self, axis: Literal['serial', 'parallel'], corner: bool = False,
                    data_var: Optional[str | Literal['all']] = 'data') -> XRDATA:
        """
        Slice the data array to get the prescan region along the specified axis (serial or parallel).
        :param axis: serial or parallel
        :param corner: True to include the corner region (intersection of prescan and overscan) in the returned slice,
        False to exclude it.
        :param data_var: Name of the data variable in the DetImage to slice. If 'all', slices the entire dataset.
        :return: sliced xr.DataArray corresponding to the requested prescan region.
        """
        return self.get_scan(axis=axis, kind='prescan', corner=corner, data_var=data_var)

    def get_overscan(self, axis: Literal['serial', 'parallel'], corner: bool = False,
                     data_var: Optional[str | Literal['all']] = 'data') -> XRDATA:
        """
        Slice the data array to get the overscan region along the specified axis (serial or parallel).
        :param axis: serial or parallel
        :param corner: True to include the corner region (intersection of prescan and overscan) in the returned slice,
        False to exclude it.
        :param data_var: Name of the data variable in the DetImage to slice. If 'all', slices the entire dataset.
        :return: sliced xr.DataArray corresponding to the requested overscan region.
        """
        return self.get_scan(axis=axis, kind='overscan', corner=corner, data_var=data_var)

    def show(self, ax=None, shade_regions=False, save=None, with_mask=True, mask_key='sigma_clip_mask', data_var='data',
             **imshow_kwargs):
        ax = super().show(ax=ax, save=None, with_mask=with_mask, mask_key=mask_key, data_var=data_var, **imshow_kwargs)

        if shade_regions:
            ## Shade the prescan and overscan regions
            def _bounds(s: slice, n: int) -> tuple[int, int]:
                s0 = s.start if s.start is not None else (0 if s.step > 0 else n)
                s1 = (s.stop - 1 if s.step > 0 else s.stop + 1) if s.stop is not None else (n if s.step > 0 else 0)
                return s0, s1

            spandict = {0: ax.axvspan, 1: ax.axhspan}
            regions = [
                (self.serial_prescan, self.parallel_axis, "gold", "S Prescan"),
                (self.serial_overscan, self.parallel_axis, "red", "S Overscan"),
                (self.parallel_prescan, self.serial_axis, "cyan", "P Prescan"),
                (self.parallel_overscan, self.serial_axis, "blue", "P Overscan"),
            ]
            shape = self.data.shape
            for s, axis, color, label in regions:
                axis_idx = 0 if axis == 'y' else 1
                a, b = _bounds(s, shape[axis_idx])
                spandict[axis_idx](a, b, color=color, alpha=0.3, label=label)
            ax.scatter(self.readout_pixel[1],
                       self.readout_pixel[0],
                       marker='x', color='red', s=100)
            ax.legend(loc=(0.55, 1.05), fontsize=7)

        if save is not None:
            ax.figure.savefig(save)
        return ax


class CMOSOutput(Output):
    # CMOS specific output region can be added here if needed
    pass


class IRDetectorOutput(Output):
    # IR Detector specific output region can be added here if needed
    pass


############################################## DETIMAGE CLASS #######################################################
class DetImage:
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
        _data: Internal attribute for storing image data when loaded.
        _dataloader: Internal variable for storing data loader Callable for on-demand loading.

    """

    def __init__(
            self,
            data: Optional[XRDATA | np.ndarray | Callable] = None,
            output_objects: Optional[dict[str, Output]] = None,
            meta: Optional[DetImageMeta | dict[str, Any]] = None,
            **kwargs: Any,
    ):

        self.ndim: int = 2
        self.outputs: dict[str, Output] = {}
        self.meta: DetImageMeta | dict[str, Any] = {}
        self._data: Optional[xr.Dataset] = None
        self._dataloader: Optional[Callable] = None
        self.masks: Optional[xr.Dataset] = None

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
        self.id: str | int = self.meta.get('name', 'unknown')
        self.image_type: dict[str, Any] = self.meta.get('image_type', {'type': 'unknown'})

        # Outputs
        if output_objects is None and self._data is not None:
            logger.info("Creating default Output for DetImage (covers full array).")
            h, w = self.shape
            self.outputs['0'] = Output(id='0',
                                       input_array_axis=0,
                                       input_slice=(slice(0, h), slice(0, w)),
                                       output_slice=(slice(0, h), slice(0, w)),
                                       parent=self)
        else:
            self.outputs.update(output_objects) if output_objects else {}
            for out_id, out in self.outputs.items():
                out.parent = self

    def add_output(self, output: Output, overwrite: bool = True):
        output.parent = self
        if output.id in self.outputs:
            logger.debug(f"Output with id {output.id} already exists, overwrite is set to {overwrite}.")
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
            idata, iheaders = self._dataloader(self.meta['filename'])
            temp = np.zeros(self.shape)
            for out_id, output in self.outputs.items():
                output.header = iheaders[output.input_array_axis]
                temp[*output.output_slice] = idata[output.input_array_axis][*output.input_slice]
            temp = ensure_dataarray(temp)
            self._data = temp.to_dataset(name='data')
            del idata, iheaders
        else:
            raise ValueError("No dataloader function provided for this DetImage, cannot load data.")

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
                data = set_slice_in_data(slicedata, self._data[data_var], slicer)
                self._data[data_var] = data
            case (_, xr.Dataset()):
                if data_var not in slicedata.data_vars:
                    raise ValueError(f"Data variable '{data_var}' not found in provided slicedata xr.Dataset.")
                data = set_slice_in_data(slicedata[data_var], self._data[data_var], slicer)
                self._data[data_var] = data

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
        if self._data is not None:
            return tuple(self._data.sizes[dim] for dim in ['y', 'x'])
        if len(self.outputs) > 0:
            imsize = [0] * self.ndim
            for _, output in self.outputs.items():
                imsize[0] = max(imsize[0], output.output_slice[0].stop)
                imsize[1] = max(imsize[1], output.output_slice[1].stop)
            return tuple(imsize)
        raise ValueError("Cannot determine shape of DetImage from metadata or outputs.")

    def build_full_mask(self) -> bool:
        """
        Build a full mask from the individual output masks.
        :return: True if a full mask was built, False otherwise.
        """
        if self.masks is not None:
            return True
        self.masks = xr.Dataset(coords=self.data.coords)
        for out_id, output in self.outputs.items():
            if output.masks is not None:
                # combine output mask dataset with maskset, output mask coords are a subset of maskset
                self.masks = self.masks.merge(output.masks, join='outer', fill_value=np.nan, compat='no_conflicts')
        if len(self.masks.data_vars) == 0:
            self.masks = None
            return False
        return True

    def show(self, ax=None, save=None, with_mask=True, mask_key='sigma_clip_mask', data_var='data', **imshow_kwargs):
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
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

        data = self.get_data(data_var)
        if with_mask:
            if self.build_full_mask() and mask_key in self.masks.data_vars:
                arr = np.ma.masked_array(data=data.values, mask=self.masks[mask_key].values)
                arr = arr.filled(0)
                temp = xr.DataArray(data=arr, coords=data.coords, dims=data.dims)
            else:
                logger.warning(f"Mask key '{mask_key}' not found in DetImage masks. Showing unmasked data.")
                temp = data
        else:
            temp = data
        im = temp.plot.imshow(ax=ax, **imshow_kwargs)

        if save is not None:
            ax.figure.savefig(save)
        return ax

    def to_netcdf(self, filepath, **kwargs):
        """
        Save the object to disk in netcdf format. Use xr.Dataset to hold everything. The .data and .masks are in data_vars,
        the .meta and .outputs in attrs.
        :param filepath: str
        """
        data = self.get_data('all')
        if isinstance(data, xr.DataArray):
            ds_to_save = data.to_dataset(name='data')
        else:
            ds_to_save = data
        # add masks if they exist
        if self.masks is not None:
            ds_to_save.update(self.masks)
            ds_to_save.attrs['mask_keys'] = json.dumps(list(self.masks.data_vars.keys()))
        else:
            ds_to_save.attrs['mask_keys'] = json.dumps([])
        # convert meta to dict to store in attrs
        ds_to_save.attrs['meta'] = self.meta.to_json()
        ds_to_save.attrs['image_type'] = json.dumps(self.image_type)
        # convert outputs to dict of dicts
        out_dict = {}
        for out_id, output in self.outputs.items():
            op = output.to_json(exclude={'masks', 'parent'})
            out_dict[out_id] = op
            outclass = output.__class__.__name__
        ds_to_save.attrs['outputs'] = json.dumps(out_dict)
        ds_to_save.attrs['output_class'] = outclass

        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        if not filepath.endswith('.nc'):
            filepath = filepath + '.nc'
        ds_to_save.to_netcdf(filepath, mode='w', **kwargs)

    @classmethod
    def from_netcdf(cls, filepath):
        """
        Load a DetImage object from a netcdf file saved with to_netcdf().
        :param filepath: str, path to the netcdf file.
        :return: DetImage object.
        """
        loaded_ds = xr.load_dataset(filepath)
        mask_keys = json.loads(loaded_ds.attrs['mask_keys'])
        masks = loaded_ds[mask_keys] if len(mask_keys) > 0 else None
        data = loaded_ds.drop_vars(mask_keys)
        meta = DetImageMeta.from_json(loaded_ds.attrs['meta'])

        outputs = {}
        outclass = globals()[loaded_ds.attrs['output_class']]
        outputs_attr = json.loads(loaded_ds.attrs['outputs'])
        for out_id, output_dict in outputs_attr.items():
            output = outclass.from_json(output_dict)
            # extract subdataset for output masks
            output.masks = slice_data(masks, output.output_slice) if masks is not None else None
            outputs[out_id] = output

        detimg = cls(data=data, output_objects=outputs, meta=meta)
        detimg.masks = masks
        detimg.image_type = json.loads(loaded_ds.attrs['image_type'])
        return detimg


TImage = TypeVar("TImage")


class ImageBundle(Generic[TImage]):
    """
    Class to hold a list of images. Contains methods to tabulate metadata of the images for easy filtering.

    Attributes
    ----------
    images: list[TImage]
        List of input images.
    list: pd.DataFrame
        DataFrame containing image identifying metadata from image_class.image_type.

    Methods
    -------
    __call__(pd_query: str = '')
        Returns a new ImageBundle with images filtered based on the provided pandas query string. Calls filter() internally.
    filter(pd_query: str = '')
        Returns a list of images filtered based on the provided pandas query string.
    save(filepath: str, **kwargs)
        Saves all images in the ImageBundle to the specified folder path using their to_netcdf() method.
    load(filepath: str)
        Loads all netcdf files from the specified folder path and creates an ImageBundle from them using the from_netcdf() method of the image_class.
    __iter__()
        Returns an iterator over the list of images in the ImageBundle.
    __getitem__(i)
        Returns the image at index i in the ImageBundle.
    __setitem__(key, value)
        Sets the image at index key to value in the ImageBundle. Validates that value is of the correct image_class type.
    __add__(other: Self)
        Combines two ImageBundle instances of the same type and returns a new ImageBundle containing images from both.
    append(image)
        Appends a new image to the ImageBundle after validating its type.
    extend(other: Self)
        Extends the ImageBundle with images from another ImageBundle of the same type.

    """
    image_class: type[TImage] = DetImage

    def __init__(self, images: TImage | list[TImage] | None = None):
        """
        :param images: image_class | list[image_class] | None
            image_class could be DetImage, FocalPlaneImage, etc.
        :return: ImageBundle instance.
        """
        images = images if isinstance(images, list) else [images] if images is not None else []
        self.images: list[TImage] = [self._validate_image(image) for image in images]
        self.list: pd.DataFrame | None = None
        self._tabulate()

    def _validate_image(self, image: TImage) -> TImage:
        if not isinstance(image, self.image_class):
            raise TypeError(f"Expected {self.image_class.__name__}, got {type(image).__name__}")
        return image

    def _tabulate(self):
        """
        Loops through list of images and creates a dataframe from their image_type dict.

        Default columns are `det_id`, `filename`, `object`, containing image_class.id, image_class.meta['filename']
        and the image_class object itself. Rest of the columns/values are keys/values in the image_class.image_type.
        """
        tab = []
        for i, image in enumerate(self.images):
            imtype = deepcopy(image.image_type)
            imtype['det_id'] = image.id
            imtype['filename'] = image.meta.get('filename', None)
            imtype['object'] = image
            tab.append(imtype)
        self.list = pd.DataFrame(tab)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Create an ImageBundle from a pandas DataFrame. The DataFrame must contain a column named 'object' with image_class instances.
        :param df: pd.DataFrame
            DataFrame containing image_class instances in a column named 'object'.
        :return: ImageBundle instance.
        """
        if 'object' not in df.columns:
            raise ValueError("DataFrame must contain a column named 'object' with image_class instances.")
        images = df['object'].to_list()
        return cls(images=images)

    def __call__(self, pd_query: str = ''):
        """
        Returns a new ImageBundle with images filtered based on the provided pandas query string.
        :param pd_query: str
            Query string to filter the dataframe. Should be a valid pandas query string.
        :return: ImageBundle
            A new ImageBundle with the filtered images.
        """
        return type(self)(self.filter(pd_query))

    def filter(self, pd_query: str = '') -> list[TImage]:
        """
        Filters images based on the provided pandas query string. Returns a list of filtered images.
        :param pd_query: str
            Query string to filter the dataframe. Should be a valid pandas query string.
        :return: List of filtered DetImages.
        """
        df = self.list.query(pd_query) if pd_query != '' else self.list
        return df['object'].to_list()

    def groupby(self, by, sort=False, dropna=False, **kwargs):
        """
        Wrapper for pandas groupby on self.list dataframe. Returns a pandas groupby object.
        :param by: pandas groupby **by** parameter.
        :param sort: pandas groupby **sort** parameter. Default is False for faster operation.
        :param dropna: pandas groupby **dropna** parameter. Default is False to keep NaN groups.
        :param kwargs: pandas groupby additional keyword arguments.
        :return: pandas groupby object.
        """
        missing_keys = set(by).difference(self.list.columns)
        if missing_keys:
            missing = ', '.join(missing_keys)
            raise KeyError(f"Groupby keys not found in DataFrame columns: {missing}")
        return self.list.groupby(by, sort=sort, dropna=dropna, **kwargs)

    def save(self, filepath: str, **kwargs):
        """
        Call to_netcdf() for each image object, and save them in one folder path
        :param filepath: str
            Path to folder where images will be saved. The images are saved with image_{i}.nc filenames where
            "i" is their index in the ImageBundle list.
        :param **kwargs: Additional keyword arguments to pass to the to_netcdf() method of the image_class.
        """
        # check if image_class has to_netcdf method
        if not hasattr(self.image_class, 'to_netcdf'):
            raise AttributeError(f"{self.image_class.__name__} does not have a to_netcdf() method.")

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, image in enumerate(self.images):
            image.to_netcdf(os.path.join(filepath, f'image_{i}.nc'), **kwargs)

    @classmethod
    def load(cls, filepath: str):
        """
        Load all the netcdf files from {filepath} folder and create an ImageBundle from them.
        :param filepath: str
            Path to folder where images are saved. Looks for filenames of format image_{i}.nc
        :return: ImageBundle instance
        """
        # check if image_class has from_netcdf method
        if not hasattr(cls.image_class, 'from_netcdf'):
            raise AttributeError(f"{cls.image_class.__name__} does not have a from_netcdf() method.")

        files = sorted(glob2.glob(os.path.join(filepath, '*.nc')))
        images = []
        for file in files:
            im = cls.image_class.from_netcdf(file)
            images.append(im)
        return cls(images=images)

    def __repr__(self):
        return f"ImageBundle with {len(self)} images, {repr(self.list)}"

    def _repr_html_(self):
        return self.list._repr_html_()

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, i):
        return self.images[i]

    def __setitem__(self, key, value):
        self.images[key] = self._validate_image(value)
        self._tabulate()

    def __add__(self, other: Self):
        if type(self) is not type(other):
            raise TypeError(f"Cannot combine {self.__class__.__name__} with {other.__class__.__name__}.")
        return type(self)(self.images + other.images)

    def append(self, image):
        self.images.append(self._validate_image(image))
        self._tabulate()

    def extend(self, other: Self):
        if type(self) is not type(other):
            raise TypeError(f"Cannot extend {self.__class__.__name__} with {other.__class__.__name__}.")
        self.images.extend(other.images)
        self._tabulate()
