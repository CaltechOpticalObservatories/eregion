from __future__ import annotations

from typing import Optional, Any, Literal, Callable
from pydantic import Field, ConfigDict
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits

from datamodels.mappable import Mappable
from utils import ensure_dataarray, slice_data, ensure_numpy, configure_logger
from core.image_operations import flip_and_rotate

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
                                                 description="FITS header or dict for this output if available.")
    parent: Optional["DetImage"] = Field(default=None, description="Parent detector image.")
    masks: Optional[dict[str, xr.DataArray]] = Field({},
                                                description="Optional dictionary of masks for this output, e.g., {'bad_pixels': mask_array}.")

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True,
                              populate_by_name=True)

    @property
    def data(self) -> xr.DataArray:
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
        im_slc_parallel = slice(self.parallel_prescan.stop, self.parallel_overscan.start)
        im_slc_serial = slice(self.serial_prescan.stop, self.serial_overscan.start)
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
        data: 2D image array or Callable
        output_objects: Prebuilt Output regions.
        meta: Dict or DetImageMeta; validated if provided.
        kwargs: Backward-compatible meta fields (merged if meta is a dict).

    Attributes:
        ndim: int - Number of dimensions (fixed at 2 for spatial).
        outputs: dict[str, Output] - Mapping of output IDs to Output objects.
        meta: DetImageMeta or dict - Metadata for the detector image.
        id: str | int - Identifier for the corresponding detector taken from 'name' key in detector config.
        image_type: dict[str, Any] - Mapping of any image identifying keys to their values like 'type', 'exptime', 'mode', etc.
        focal_plane: FocalPlaneImage - Pointer to the focal plane image object this DetImage is placed on, if any.
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
        self._data = None
        self._dataloader = None

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
            for outid, out in zip(self.outputs):
                out.parent = self

        self.focal_plane: Optional["FocalPlaneImage"] = None

    def output_by_id(self, output_id: str) -> Output:
        try:
            return self.outputs[output_id]
        except:
            raise ValueError(f"Output with id {output_id} not found.")

    def add_output(self, output: Output, overwrite: bool = True):
        output.parent = self
        if output.id in self.outputs:
            logger.warning(f"Output with id {output.id} already exists, overwrite is set to {overwrite}.")
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

    def set_data_slice(self, slicedata, slice):
        self._data[slice] = slicedata

    def load_from_disk(self):
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
            self.load_from_disk()
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

    def show(self, ax=None, save=None, **imshow_kwargs):
        if self.data is None:
            raise ValueError("DetImage has no data to show.")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        im = self.data.plot.imshow(ax=ax, **imshow_kwargs)
        if save is not None:
            ax.figure.savefig(save)
        return ax


class ImageBundle:
    """
    Class to hold a list of images. Contains methods to tabulate metadata of the images for easy filtering.
    """

    def __init__(self, images: list[DetImage] | None = None):
        """
        :param images: List of input images.

        Attributes
        ----------
        images: list[DetImage]
            List of input images.
        list: pd.DataFrame
            DataFrame containing image identifying metadata from DetImage.image_type.
        """
        self.images: list = images if images is not None else []
        self.list: pd.DataFrame = self.tabulate()

    def tabulate(self) -> pd.DataFrame:
        """
        Loops through list of images and creates a dataframe from their image_type dict.

        Default columns are `det_id`, `filename`, `object`, containing DetImage.id, DetImage.meta['filename']
        and the DetImage object itself. Rest of the columns/values are keys/values in the DetImage.image_type.
        """
        tab = []
        for i, image in enumerate(self.images):
            imtype = image.image_type
            imtype['det_id'] = image.id
            imtype['filename'] = image.meta['filename']
            imtype['object'] = image
            tab.append(imtype)
        return pd.DataFrame(tab)

    def filter(self, **criteria) -> list[DetImage]:
        """
        Filters images based on column values supplied as criteria.
        :param criteria: column_name=value entries to filter on.
        :return: List of filtered DetImages.
        """
        sel = ' & '.join([f'({k} == {repr(v)})' for k, v in criteria.items()])
        df = self.list.query(sel) if sel != '' else self.list
        return df['object'].to_list()

    def __call__(self, **criteria) -> "ImageBundle":
        return ImageBundle(self.filter(**criteria))

    def __repr__(self):
        print(f"ImageBundle with {len(self.images)} images:")
        return self.list.__repr__()

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)


class FocalPlaneImage:
    """
    Composite focal-plane image made by placing multiple DetImage tiles.
    :param num_detectors: int
        Number of detector tiles expected.
    :param dim: tuple[int, int]
        Dimensions of the focal plane image (height, width) in mm.
    :param det_images: Optional[List[DetImage]]
        List of DetImage objects to place on the focal plane.
    """

    def __init__(
        self,
        num_detectors: int,
        dim: Optional[tuple[int, int]] = None,
        det_images: Optional[list[DetImage]] = None,
        **kwargs,
    ):
        self.meta: dict = {}
        if kwargs:
            self.meta.update(kwargs)

        self.num_detectors = int(num_detectors)
        self.dim_mm = tuple(dim) if dim is not None else None
        self.pixel_size = None
        self.data: xr.DataArray | None = None   # To hold det image data in one array
        self.table: pd.DataFrame | None = None  # To keep track of det_image data position within focal-plane data array

        self.det_images: dict[str, DetImage] = {}
        if det_images is not None:
            self.update(det_images)

    def update(self, det_images: list[DetImage]) -> None:
        self.add_det_images(det_images)
        self.construct_focal_plane_image()

    def add_det_images(self, det_images):
        det_images = det_images if len(det_images) > 0 else [det_images]
        for det_image in det_images:
            if len(self.det_images) == self.num_detectors:
                logger.error(
                    f"Number of DetImages added have reached number of detectors present in this focal-plane.")
                break
            self.validate_det_image(det_image)
            det_image.focal_plane = self
            self.det_images[det_image.id] = det_image

    def validate_det_image(self, det_image: DetImage) -> None:
        """
        Accepts either validated DetImageMeta or legacy dict with required keys.
        """
        if not (isinstance(det_image.meta, DetImageMeta) or isinstance(det_image.meta, dict)):
            raise ValueError("DetImage.meta must be DetImageMeta or dict.")
        required = {"properties", "focal_plane_position"}
        if not required.issubset(det_image.meta.keys()):
            raise ValueError("DetImage.meta missing required keys for focal-plane placement.")
        # attempt to coerce for consistent downstream access
        det_image.meta = DetImageMeta.model_validate(det_image.meta)

        # check pixel size is same for all det images
        pixsize = det_image.meta.properties.pixel_size
        if self.pixel_size is None:
            self.pixel_size = pixsize
        elif pixsize != self.pixel_size:
            raise ValueError("All DetImage objects must have the same pixel_size for focal-plane assembly.")

    def construct_focal_plane_image(self):
        if len(self.det_images) == 0:
            raise ValueError("No DetImage objects to assemble.")

        frames = []
        for det_id, det_image in self.det_images.items():
            props = det_image.meta.properties  # type: ignore[union-attr]
            pos = det_image.meta.focal_plane_position  # type: ignore[union-attr]
            xhalf, yhalf = props.x_size / 2, props.y_size / 2
            frames.append(
                {
                    "det_id": det_id,  # type: ignore[arg-type]
                    "x_min": int(pos.x_cen / self.pixel_size - xhalf),
                    "x_max": int(pos.x_cen / self.pixel_size - xhalf) + props.x_size,
                    "y_min": int(pos.y_cen / self.pixel_size - yhalf),
                    "y_max": int(pos.y_cen / self.pixel_size - yhalf) + props.y_size,
                    "angle": pos.angle if hasattr(pos, "angle") else None,
                    "flip_x": pos.flip_x if hasattr(pos, "flip_x") else None,
                    "flip_y": pos.flip_y if hasattr(pos, "flip_y") else None,
                }
            )
        frames_df = pd.DataFrame(frames)

        # Verify that there are no overlapping detectors, i.e. area covered inside corners should not overlap
        for i in range(len(frames_df)):
            for j in range(i + 1, len(frames_df)):
                a = frames_df.iloc[i]
                b = frames_df.iloc[j]
                separated = (a["x_max"] <= b["x_min"]) or (a["x_min"] >= b["x_max"]) or \
                            (a["y_max"] <= b["y_min"]) or (a["y_min"] >= b["y_max"])
                if not separated:
                    raise ValueError(f"Detectors {a['det_id']} and {b['det_id']} overlap in focal plane.")

        # Verify the size of the focal plane image
        calc_dim = np.array([frames_df["y_max"].max() - frames_df["y_min"].min(),
                            frames_df["x_max"].max() - frames_df["x_min"].min()])
        if self.dim_mm is not None:
            dim_pix = (np.array(self.dim_mm) / self.pixel_size).astype(int)
            if any(calc_dim > dim_pix):
                logger.warning("Provided dim %s < computed dim %s.", dim_pix, calc_dim)
        else:
            dim_pix = calc_dim.astype(int)

        # Initialize DataArray
        self.data = xr.DataArray(
            np.zeros(dim_pix, dtype=float),
            dims=("y", "x"),
            coords={"y": np.arange(frames_df["y_min"].min(), frames_df["y_max"].max(), 1),
                    "x": np.arange(frames_df["x_min"].min(), frames_df["x_max"].max(), 1)}
        )

        # Place tiles
        for i in range(len(frames_df)):
            row = frames_df.iloc[i]
            di = self.det_images[row["det_id"]]
            if di.data is None:
                raise ValueError(f"DetImage at index {i} has no data.")
            else:
                imdata = flip_and_rotate(di.data.values, angle=row['angle'], flip_x=row['flip_x'],
                                         flip_y=row['flip_y'])

            slc = {'y':slice(row['y_min'], row['y_max']-1), 'x':slice(row['x_min'], row['x_max']-1)}
            self.data.loc[slc] = imdata

        self.table = frames_df

    def show(self, ax=None, save=None, show_det_id=False, **imshow_kwargs):
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(8, 8), tight_layout=True)
        im = self.data.plot.imshow(ax=ax, **imshow_kwargs)
        # Draw detector boundaries
        if hasattr(self, "table"):
            for _, row in self.table.iterrows():
                rect = plt.Rectangle(
                    (row["x_min"], row["y_min"]),
                    row["x_max"] - row["x_min"],
                    row["y_max"] - row["y_min"],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                if show_det_id:
                    ax.text(row["x_min"] + 150, row["y_min"] + 150, str(row["det_id"]), color="white", fontsize=8)
        if save is not None:
            ax.figure.savefig(save)
        return ax
