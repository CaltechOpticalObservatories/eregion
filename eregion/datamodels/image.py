from __future__ import annotations
from typing import Optional, Any, Literal, Callable
from pydantic import Field, ConfigDict, model_serializer, model_validator, SerializationInfo
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
from utils import ensure_dataarray, slice_data, ensure_numpy, configure_logger, decrease_slicer_stop_index

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

    model_config = ConfigDict(extra="allow")

    def update(self, other: dict[str, Any]):
        for key, value in other.items():
            if not hasattr(self, key):
                setattr(self, key, value)

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
                                        description="Optional xr.Dataset containing masks for this output.", exclude=True)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

    @model_serializer(when_used='json', mode='plain')
    def serializer(self, info: SerializationInfo):
        """
        Custom serializer for slice objects to convert them to a list of [start, stop, step] for JSON serialization.

        Also convert fits.Header to dict
        """
        def _to_dict(value):
            if isinstance(value, slice):
                return {"start": value.start, "stop": value.stop, "step": value.step}
            if isinstance(value, dict):
                return {k: _to_dict(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_dict(v) for v in value]
            if isinstance(value, tuple):
                return tuple([_to_dict(v) for v in value])
            if isinstance(value, fits.Header):
                return dict(value)
            return value

        data = self.model_dump(mode="python", include=info.include, exclude=info.exclude, by_alias=info.by_alias,
                               exclude_unset=info.exclude_unset, exclude_defaults=info.exclude_defaults,
                               exclude_none=info.exclude_none, round_trip=info.round_trip)
        return _to_dict(data)

    @model_validator(mode='before')
    @classmethod
    def parse(cls, kwargs):
        """
        Custom validator to convert dict representations of slices back to slice objects when loading from JSON.
        Check for the keys start, stop and step to identify if something is a slice. If so, convert to slice object.
        """
        for key, value in kwargs.items():
            if isinstance(value, tuple) or isinstance(value, list):
                for i,item in enumerate(value):
                    if isinstance(item, dict) and {"start", "stop", "step"}.issubset(item):
                        value[i] = slice(item["start"], item["stop"], item["step"])
                kwargs[key] = value
            elif isinstance(value, dict) and {"start", "stop", "step"}.issubset(value):
                kwargs[key] = slice(value["start"], value["stop"], value["step"])
        return kwargs

    @property
    def data(self):
        # Full data from parent DetImage corresponding to this output, including any prescan/overscan.
        if self.parent is None or getattr(self.parent, "data", None) is None:
            raise ValueError("Attach this Output to a DetImage with valid data.")
        return slice_data(self.parent.data, self.output_slice)

    def set_data_in_parent(self, new_data: xr.DataArray | np.ndarray):
        if self.parent is None or getattr(self.parent, "data", None) is None:
            raise ValueError("Attach this Output to a DetImage with valid data.")
        # Convert new_data to numpy array if it's an xarray DataArray, to ensure compatibility with parent data array.
        new_data_np = ensure_numpy(new_data)
        # Assign new data to the appropriate slice in the parent DetImage
        self.parent.set_data_slice(new_data_np, self.output_slice)

    def show(self, ax=None, save=None, **imshow_kwargs):
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        im = self.data.plot.imshow(ax=ax, **imshow_kwargs)
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
    def image_region(self) -> dict[Literal['x', 'y'], slice]:
        parallel_step = -1 if self.parallel_prescan.stop > self.parallel_overscan.start else 1
        im_slc_parallel = slice(self.parallel_prescan.stop, self.parallel_overscan.start, parallel_step)
        serial_step = -1 if self.serial_prescan.stop > self.serial_overscan.start else 1
        im_slc_serial = slice(self.serial_prescan.stop, self.serial_overscan.start, serial_step)
        return {self.parallel_axis: im_slc_parallel, self.serial_axis: im_slc_serial}

    def get_prescan(self, kind: Literal['serial', 'parallel']) -> xr.DataArray:
        slc = self.serial_prescan if kind == "serial" else self.parallel_prescan
        axis = self.serial_axis if kind == "serial" else self.parallel_axis
        return slice_data(self.data, {axis: slc})

    def get_overscan(self, kind: Literal['serial', 'parallel']) -> xr.DataArray:
        slc = self.serial_overscan if kind == "serial" else self.parallel_overscan
        axis = self.serial_axis if kind == "serial" else self.parallel_axis
        return slice_data(self.data, {axis: slc})

    def get_image_region(self):
        return slice_data(self.data, self.image_region)

    def show(self, ax=None, shade_regions=False, save=None, **imshow_kwargs):
        ax = super().show(ax=ax, save=None, **imshow_kwargs)

        if shade_regions:
            ## Shade the prescan and overscan regions
            def _bounds(s: slice, n: int) -> tuple[int, int]:
                s0 = s.start if s.start is not None else (0 if s.step > 0 else n)
                s1 = (s.stop-1 if s.step>0 else s.stop+1) if s.stop is not None else (n if s.step>0 else 0)
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
                axis_idx = 0 if axis=='y' else 1
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
    Base detector image holding 2D (only spatial) pixel data and outputs.

    Args:
        data: 2D image array (np.ndarray or xr.DataArray) or Callable
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
        data: Optional[xr.DataArray | np.ndarray | Callable] = None,
        output_objects: Optional[dict[str, Output]] = None,
        meta: Optional[DetImageMeta | dict[str, Any]] = None,
        **kwargs: Any,
    ):

        self.ndim: int = 2
        self.outputs: dict[str, Output] = {}
        self.meta: DetImageMeta | dict[str, Any] = {}
        self._data: xr.DataArray | None = None
        self._dataloader: Callable | None = None

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

        self.masks: xr.Dataset = None

    def output_by_id(self, output_id: str) -> Output:
        try:
            return self.outputs[output_id]
        except:
            raise ValueError(f"Output with id {output_id} not found.")

    def add_output(self, output: Output, overwrite: bool = True):
        output.parent = self
        if output.id in self.outputs:
            logger.debug(f"Output with id {output.id} already exists, overwrite is set to {overwrite}.")
            if overwrite:
                self.outputs[output.id] = output
        else:
            self.outputs[output.id] = output

    def set_data(self, data: xr.DataArray | np.ndarray | Callable):
        if isinstance(data, Callable):
            self._dataloader = data
        elif isinstance(data, xr.DataArray) or isinstance(data, np.ndarray):
            self._data = ensure_dataarray(data)
        else:
            raise TypeError(
                "If provided, data must be an xr.DataArray or np.ndarray or Callable for loading on demand.")

    def set_data_slice(self, slicedata, slicer):
        temp = self._data.copy(deep=True)
        slicer = decrease_slicer_stop_index(slicer)
        temp.loc[slicer] = slicedata
        self._data = temp
        del temp

    def _load_from_disk(self):
        """
        Load data from disk into _data attribute. self.data returns xarray _data when called.
        """
        if self._dataloader is not None:
            idata, iheaders = self._dataloader(self.meta['filename'])
            self._data = np.zeros(self.shape)
            for out_id, output in self.outputs.items():
                output.fits_header = iheaders[output.input_array_axis]
                self._data[*output.output_slice] = idata[output.input_array_axis][*output.input_slice]
            self._data = ensure_dataarray(self._data)
            del idata, iheaders
        else:
            raise ValueError("No dataloader function provided for this DetImage, cannot load data.")

    def unload(self):
        """
        Unload data from _data attribute to free up memory.
        """
        del self._data
        self._data = None

    @property
    def data(self) -> xr.DataArray:
        if self._data is None:
            self._load_from_disk()
        return self._data

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def shape(self) -> tuple[int, ...]:
        if 'properties' in self.meta:
            if self.meta['properties'] and 'y_size' in self.meta['properties']:
                return self.meta['properties']['y_size'], self.meta['properties']['x_size']
        if 'shape' in self.meta:
            return self.meta['shape']
        if self._data is not None:
            return self._data.shape
        if len(self.outputs) > 0:
            imsize = [0] * self.ndim
            for _, output in self.outputs.items():
                imsize[0] = max(imsize[0], output.output_slice[0].stop)
                imsize[1] = max(imsize[1], output.output_slice[1].stop)
            return tuple(imsize)
        raise ValueError("Cannot determine shape of DetImage from metadata or outputs.")

    def build_full_mask(self):
        self.masks = xr.Dataset(coords=self.data.coords)
        for out_id, output in self.outputs.items():
            if output.masks is not None:
                # combine output mask dataset with maskset, output mask coords are a subset of maskset
                self.masks = self.masks.merge(output.masks, join='outer', fill_value=np.nan, compat='no_conflicts')
            else:
                self.masks = None
                return False
        return True

    def show(self, ax=None, save=None, with_mask=True, mask_key='sigma_clip_mask', **imshow_kwargs):
        if self.data is None:
            raise ValueError("DetImage has no data to show.")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

        if with_mask:
            mask_built = False
            if self.masks is None:
                mask_built = self.build_full_mask()
            if mask_built and mask_key in self.masks.data_vars:
                mdata = np.ma.MaskedArray(self.data.values, mask=self.masks[mask_key].values)
                im = ax.imshow(mdata, **imshow_kwargs)
                plt.colorbar(im, ax=ax)
            else:
                logger.warning(f"Mask key '{mask_key}' not found in DetImage masks. Showing unmasked data.")
                im = self.data.plot.imshow(ax=ax, **imshow_kwargs)
        else:
            im = self.data.plot.imshow(ax=ax, **imshow_kwargs)

        if save is not None:
            ax.figure.savefig(save)
        return ax

    def to_netcdf(self, filepath, **kwargs):
        """
        Save the object to disk in netcdf format. Use xr.Dataset to hold everything. The .data and .masks are in data_vars,
        the .meta and .outputs in attrs.
        :param filepath: str
        """
        ds_to_save = self.data.to_dataset(name='data')
        # add masks if they exist
        ds_to_save = ds_to_save.update(self.masks) if self.masks is not None else ds_to_save
        # convert meta to dict to store in attrs
        ds_to_save.attrs['meta'] = self.meta.model_dump_json()
        ds_to_save.attrs['image_type'] = json.dumps(self.image_type)
        # convert outputs to dict of dicts
        out_dict = {}
        for out_id, output in self.outputs.items():
            op = output.model_dump_json(exclude={'masks', 'parent'})
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
        loaded_ds = xr.load_dataset(filepath)
        data = loaded_ds['data']
        masks = loaded_ds.drop_vars('data')
        meta = DetImageMeta.model_validate_json(loaded_ds.attrs['meta'])

        outputs = {}
        outclass = globals()[loaded_ds.attrs['output_class']]
        outputs_attr = json.loads(loaded_ds.attrs['outputs'])
        for out_id, output_dict in outputs_attr.items():
            output = outclass.model_validate_json(output_dict)
            # extract subdataset for output masks
            output.masks = slice_data(masks, output.output_slice)
            outputs[out_id] = output

        detimg = cls(data=data, output_objects=outputs, meta=meta)
        detimg.masks = masks
        detimg.image_type = json.loads(loaded_ds.attrs['image_type'])
        return detimg


class ImageBundle:
    """
    Class to hold a list of images. Contains methods to tabulate metadata of the images for easy filtering.
    """
    image_class = DetImage

    def __init__(self, images: image_class | list[image_class] | None = None):
        """
        :param images: List of input images.
        :type images: image_class | list[image_class] | None
        :return: ImageBundle instance.
        :rtype: ImageBundle
        Attributes
        ----------
        images: list[image_class]
            List of input images.
        list: pd.DataFrame
            DataFrame containing image identifying metadata from image_class.image_type.
        """
        images = images if isinstance(images, list) else [images] if images is not None else []
        assert all([isinstance(image, self.image_class) for image in images])
        self.images = images
        self.tabulate()

    def tabulate(self) -> pd.DataFrame:
        """
        Loops through list of images and creates a dataframe from their image_type dict.

        Default columns are `det_id`, `filename`, `object`, containing DetImage.id, DetImage.meta['filename']
        and the DetImage object itself. Rest of the columns/values are keys/values in the DetImage.image_type.
        """
        tab = []
        for i, image in enumerate(self.images):
            imtype = deepcopy(image.image_type)
            imtype['det_id'] = image.id
            imtype['filename'] = image.meta['filename']
            imtype['object'] = image
            tab.append(imtype)
        self.list = pd.DataFrame(tab)

    def filter(self, pd_query: str='') -> list[image_class]:
        """
        Filters images based on column values supplied as criteria.
        :param pd_query: str
            Query string to filter the dataframe. Should be a valid pandas query string.
        :return: List of filtered DetImages.
        """
        df = self.list.query(pd_query) if pd_query != '' else self.list
        return df['object'].to_list()

    def append(self, image):
        if isinstance(image, self.image_class):
            self.images.append(image)
            self.tabulate()
        else:
            raise TypeError(f"Not a {self.image_class.__name__}: {type(image)}")

    def save(self, filepath: str, **kwargs):
        """
        Call to_netcdf() for each detimage, and save them in one folder path
        """
        # check if image_class has to_netcdf method
        if not hasattr(self.image_class, 'to_netcdf'):
            raise AttributeError(f"{self.image_class.__name__} does not have a to_netcdf() method.")

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i,image in enumerate(self.images):
            image.to_netcdf(os.path.join(filepath, f'image_{i}.nc'), **kwargs)

    @classmethod
    def load(cls, filepath: str):
        """
        Load all netcdf files in a folder and create an ImageBundle from them.
        :param filepath:
        :return:
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

    def __call__(self, pd_query: str='') -> "ImageBundle":
        return ImageBundle(self.filter(pd_query))

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
        self.images[key] = value
        self.tabulate()

    def __add__(self, other: ImageBundle):
        self.images.extend(other.images)
        self.tabulate()
