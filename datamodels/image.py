from __future__ import annotations

from typing import Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

from datamodels.image_utils import ensure_dataarray, slice_data

logger = logging.getLogger(__name__)


class DetectorProperties(BaseModel):
    """
    Physical and sampling properties for a detector tile.
    """
    pixel_size: float = Field(gt=0, description="Pixel size in mm.")
    x_size: int = Field(gt=0, description="Tile width in pixels.")
    y_size: int = Field(gt=0, description="Tile height in pixels.")

    model_config = ConfigDict(extra="allow")

class FocalPlanePosition(BaseModel):
    """
    Center position of the detector on the focal plane (same units as pixel_size, typically mm).
    """
    x_cen: float = Field(..., description="Center X in length units.")
    y_cen: float = Field(..., description="Center Y in length units.")

    model_config = ConfigDict(extra="allow")

class DetImageMeta(BaseModel):
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

class Output(BaseModel):
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
    fits_header: Optional[fits.Header] = Field(default=None,
                                                 description="FITS header for this output if available.")
    parent: Optional["DetImage"] = Field(default=None, description="Parent detector image.")

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
        # Ensure new_data is xr.DataArray
        new_data_da = ensure_dataarray(new_data)
        # Assign new data to the appropriate slice in the parent DetImage
        self.parent.data[self.output_slice] = new_data_da

    def show(self, ax=None, save=None, **imshow_kwargs):
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        image = self.data
        im = ax.imshow(image, **imshow_kwargs)
        ax.figure.colorbar(im, ax=ax)
        if save is not None:
            ax.figure.savefig(save)
        return ax


class CCDOutput(Output):
    """
    CCD-specific output region with prescan/overscan info.
    """
    serial_prescan: Optional[slice] = Field(None,
                                          description="Slice object defining the serial prescan region for this output.")
    serial_overscan: Optional[slice] = Field(None,
                                           description="Slice object defining the serial overscan region for this output.")
    parallel_prescan: Optional[slice] = Field(None,
                                           description="Slice object defining the parallel prescan region for this output.")
    parallel_overscan: Optional[slice] = Field(None,
                                            description="Slice object defining the parallel overscan region for this output.")
    parallel_axis: Optional[Literal['x', 'y']] = Field(None,
                                description="Name of the parallel readout axis for this output ('x' or 'y').")
    readout_pixel: Optional[tuple[int, int]] = Field(None,
                                           description="Tuple defining the (y, x) pixel coordinates of the readout amplifier "
                                                       "for this output in the full detector array.")

    @property
    def serial_axis(self) -> str:
        return 'x' if self.parallel_axis == 'y' else 'y'

    def get_prescan(self, kind: str) -> xr.DataArray:
        slc = self.serial_prescan if kind == "serial" else self.parallel_prescan
        axis = self.serial_axis if kind == "serial" else self.parallel_axis
        return self.data.isel(**{axis: slc})

    def get_overscan(self, kind: str) -> xr.DataArray:
        slc = self.serial_overscan if kind == "serial" else self.parallel_overscan
        axis = self.serial_axis if kind == "serial" else self.parallel_axis
        return self.data.isel(**{axis: slc})

    def show(self, ax=None, shade_regions=False, save=None, **imshow_kwargs):
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        image = self.data
        im = ax.imshow(image.values, **imshow_kwargs)

        if shade_regions:
            ## Shade the prescan and overscan regions
            spandict = {0: ax.axvspan, 1: ax.axhspan}

            def _bounds(s: slice, n: int) -> tuple[int, int]:
                s0 = 0 if s.start is None else (n + s.start if s.start < 0 else s.start)
                s1 = n if s.stop is None else (n + s.stop if s.stop < 0 else s.stop)
                return s0, s1

            regions = [
                (self.serial_prescan, self.parallel_axis, "gold", "Serial Prescan"),
                (self.serial_overscan, self.parallel_axis, "red", "Serial Overscan"),
                (self.parallel_prescan, self.serial_axis, "cyan", "Parallel Prescan"),
                (self.parallel_overscan, self.serial_axis, "blue", "Parallel Overscan"),
            ]
            shape = image.shape
            for s, axis, color, label in regions:
                axis_idx = 0 if axis=='y' else 1
                a, b = _bounds(s, shape[axis_idx])
                spandict[axis_idx](a, b, color=color, alpha=0.3, label=label)
            ax.legend(loc=(0.01, 1.01), fontsize=8)

        ax.figure.colorbar(im, ax=ax)
        if save is not None:
            ax.figure.savefig(save)
        return ax

class CMOSOutput(Output):
    # CMOS specific output region can be added here if needed
    pass

class IRDetectorOutput(Output):
    # IR Detector specific output region can be added here if needed
    pass

# Base class for detector image.
class DetImage:
    """
    Base detector image holding 2D (only spatial) pixel data and outputs.

    Args:
        data: 2D image array.
        output_objects: Prebuilt Output regions.
        image_type: Optional label like "bias", "flat".
        meta: Dict or DetImageMeta; validated if provided.
        kwargs: Backward-compatible meta fields (merged if meta is a dict).
    """
    def __init__(
        self,
        data: Optional[xr.DataArray | np.ndarray] = None,
        output_objects: Optional[dict[str, Output]] = None,
        image_type: Optional[str] = None,
        meta: Optional[DetImageMeta | dict[str, Any]] = None,
        **kwargs: Any,
    ):

        self.ndim = 2
        self.data: Optional[xr.DataArray] = ensure_dataarray(data) if data is not None else None
        self.image_type = image_type
        self.outputs = {}

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
        else:
            self.meta = {}
        # Merge any additional kwargs into meta
        self.meta.update(kwargs)


        # Outputs
        if output_objects is None and self.data is not None:
            logger.info("Creating default Output for DetImage (covers full array).")
            h, w = self.data.shape
            self.outputs['0'] = Output(id='0',
                                       input_array_axis=1,
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

    def add_output(self, output: Output):
        output.parent = self
        self.outputs[output.id] = output

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def show(self, ax=None, save=None, **imshow_kwargs):
        if self.data is None:
            raise ValueError("DetImage has no data to show.")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        im = ax.imshow(self.data.values, **imshow_kwargs)
        ax.figure.colorbar(im, ax=ax)
        if save is not None:
            ax.figure.savefig(save)
        return ax

class FocalPlaneImage:
    """
    Composite focal-plane image made by placing multiple DetImage tiles.
    :param num_detectors: int
        Number of detector tiles expected.
    :param dim: tuple[int, int]
        Dimensions of the focal plane image (height, width) in pixels.
    :param fp_center: Optional[Tuple[float, float]]
        Pixels coordinates (y, x) corresponding to the focal plane center with respect to the image array origin (top-left).
        Default is dim/2 i.e. (ydim/2, xdim/2).
    :param det_images: Optional[List[DetImage]]
        List of DetImage objects to place on the focal plane.
    """

    def __init__(
        self,
        num_detectors: int,
        dim: tuple[int, int],
        fp_center: Optional[tuple[float, float]] = None,
        det_images: Optional[list[DetImage]] = None,
        **kwargs,
    ):
        self.meta: dict = {}
        if kwargs:
            self.meta.update(kwargs)
        self.num_detectors = int(num_detectors)
        self.dim = tuple(dim)
        self.fp_cen_pix = fp_center if fp_center is not None else (dim[0] / 2, dim[1] / 2)
        self.det_images: list[DetImage] = []
        if det_images:
            for di in det_images:
                self.add_DetImage(di)

        if self.det_images:
            self.construct_focal_plane_image()

    def validate_det_image(self, det_image: DetImage) -> None:
        """
        Accepts either validated DetImageMeta or legacy dict with required keys.
        """
        if isinstance(det_image.meta, DetImageMeta):
            return
        if not isinstance(det_image.meta, dict):
            raise ValueError("DetImage.meta must be DetImageMeta or dict.")
        required = {"properties", "focal_plane_position"}
        if not required.issubset(det_image.meta.keys()):
            raise ValueError("DetImage.meta missing required keys for focal-plane placement.")
        # attempt to coerce for consistent downstream access
        det_image.meta = DetImageMeta.model_validate(det_image.meta)

    def construct_focal_plane_image(self):
        if not self.det_images:
            raise ValueError("No DetImage objects to assemble.")
        # Cast all to validated meta
        for di in self.det_images:
            self.validate_det_image(di)

        pixsize = self.det_images[0].meta.properties.pixel_size # type: ignore[union-attr]

        frames = []
        for det_image in self.det_images:
            props = det_image.meta.properties  # type: ignore[union-attr]
            pos = det_image.meta.focal_plane_position  # type: ignore[union-attr]
            xhalf, yhalf = props.x_size / 2, props.y_size / 2
            frames.append(
                {
                    "det_id": getattr(det_image.meta, "name", None) or "unknown",  # type: ignore[arg-type]
                    "x_min": pos.x_cen / pixsize - xhalf,
                    "x_max": pos.x_cen / pixsize + xhalf,
                    "y_min": pos.y_cen / pixsize - yhalf,
                    "y_max": pos.y_cen / pixsize + yhalf,
                }
            )

        frames_df = pd.DataFrame(frames, index=range(len(self.det_images)))

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
        fp_ymin = int(frames_df["y_min"].min())
        fp_ymax = int(frames_df["y_max"].max())
        fp_xmin = int(frames_df["x_min"].min())
        fp_xmax = int(frames_df["x_max"].max())
        calc_dim = (fp_ymax - fp_ymin, fp_xmax - fp_xmin)
        if calc_dim != self.dim:
            logger.warning("Provided dim %s != computed dim %s.", self.dim, calc_dim)

        # Calculate the positions to place each det_image in the focal plane array
        # Flip y-axis and shift origin (specified by self.fp_cen_pix) to top-left corner
        frames_df["y_min_fp"] = (self.fp_cen_pix[0] - frames_df["y_max"]).astype(int)
        frames_df["y_max_fp"] = (self.fp_cen_pix[0] - frames_df["y_min"]).astype(int)
        frames_df["x_min_fp"] = (frames_df["x_min"] + self.fp_cen_pix[1]).astype(int)
        frames_df["x_max_fp"] = (frames_df["x_max"] + self.fp_cen_pix[1]).astype(int)

        # Initialize DataArray
        self.data: xr.DataArray = xr.DataArray(
            np.zeros(self.dim, dtype=float),
            dims=("y", "x"),
            coords={"y": np.arange(self.dim[0]), "x": np.arange(self.dim[1])},
        )

        # Place tiles
        for i, det_image in enumerate(self.det_images):
            if det_image.data is None:
                raise ValueError(f"DetImage at index {i} has no data.")
            yslc = slice(int(frames_df.loc[i, "y_min_fp"]), int(frames_df.loc[i, "y_max_fp"]))
            xslc = slice(int(frames_df.loc[i, "x_min_fp"]), int(frames_df.loc[i, "x_max_fp"]))
            self.data[yslc, xslc] = det_image.data.values

        self.frames_df = frames_df

    def add_DetImage(self, det_image: DetImage):
        if len(self.det_images) >= self.num_detectors:
            raise ValueError(f"Number of det_images ({len(self.det_images)}) reached limit ({self.num_detectors}).")
        self.validate_det_image(det_image)
        det_image.focal_plane = self
        self.det_images.append(det_image)

    def show(self, ax=None, save=None, **imshow_kwargs):
        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(8, 8), tight_layout=True)
        im = ax.imshow(self.data.values, **imshow_kwargs)
        # Draw detector boundaries
        if hasattr(self, "frames_df"):
            for _, row in self.frames_df.iterrows():
                rect = plt.Rectangle(
                    (row["x_min_fp"], row["y_min_fp"]),
                    row["x_max_fp"] - row["x_min_fp"],
                    row["y_max_fp"] - row["y_min_fp"],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(row["x_min_fp"] + 150, row["y_min_fp"] + 150, str(row["det_id"]), color="white", fontsize=8)
        ax.figure.colorbar(im, ax=ax)
        if save is not None:
            ax.figure.savefig(save)
        return ax

