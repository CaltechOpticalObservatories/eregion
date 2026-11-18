from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Literal, TypeVar, Mapping, Iterable
from pydantic import Field, ConfigDict, field_serializer
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

from .mappable import Mappable
from eregion.utils import (
    ensure_dataarray,
    slice_data,
    XRDATA
)
if TYPE_CHECKING:
    from eregion.datamodels.image import DetImage

logger = logging.getLogger(__name__)
TData = TypeVar("TData")

############################################### OUTPUT BASE CLASS #####################################################
class Output(Mappable):
    """
    One amplifier/output region within a detector image.
    """

    id: str = Field(..., alias="id")
    input_array_axis: int | str = Field(
        ...,
        alias="ext_id",
        description="Axis index in the input array, "
        "or extension ID in FITS file which contains the data for this output."
        "If a string, find the FITS HDU which has that stringas its EXTNAME",
    )
    input_slice: tuple[slice, ...] = Field(
        ...,
        alias="ext_slice",
        description="List of Slice objects defining the portion of the data array at input_array_axis "
        "in input array or FITS that corresponds to this Detector Output.",
    )
    output_slice: tuple[slice, ...] = Field(
        ...,
        alias="data_slice",
        description="List of Slice objects defining the portion of the full detector data array "
        "that this Detector Output maps to.",
    )
    header: Optional[fits.Header | dict] = Field(
        default_factory=fits.Header,
        description="FITS header in dict form for this output if available.",
    )
    parent: Optional["DetImage"] = Field(
        default=None, description="Parent detector image.", exclude=True
    )
    masks: Optional[xr.Dataset] = Field(
        default=None,
        description="Optional xr.Dataset containing masks for this output. "
        "Write masks via set_mask() so the parent DetImage's full mask is rebuilt.",
        exclude=True,
    )

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

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
        return slice_data(self.parent.get_data(data_var), self.output_slice)

    def get_output_mask(self, mask_key: str | Literal['combined', 'all'] = 'combined') -> XRDATA | None:
        if self.masks:
            match mask_key:
                case 'combined':
                    return self.combined_mask
                case k if k in self.masks:
                    return self.masks[k]
                case 'all':
                    return self.masks
                case _:
                    logger.error(f"Unknown mask key {mask_key}")
                    return None
        else:
            return None

    def set_mask(self, mask_key: str, mask: xr.DataArray, combine: bool = True):
        """
        Set a mask for this output and clear the parent's full mask so it is rebuilt on next access.
        :param mask_key: str, name of the mask data_var in self.masks.
        :param mask: xr.DataArray, boolean mask with the same coords as this output.
        :param combine: bool, if True OR with an existing mask under mask_key, else replace it.
        """
        if self.masks is None:
            self.masks = mask.to_dataset(name=mask_key)
        elif combine and mask_key in self.masks:
            self.masks[mask_key] = self.masks[mask_key] | mask
        else:
            self.masks[mask_key] = mask
        if self.parent is not None:
            self.parent.masks = None

    def get_masked_output_data(self, data_var: str = 'data', mask_key: str = 'combined', fill_value=np.nan):
        mask = self.get_output_mask(mask_key)
        datarr = self.get_output_data(data_var)
        if mask:
            return datarr.where(~mask.to_numpy(), other=fill_value)
        else:
            return datarr

    @property
    def data(self):
        """
        Return the array corresponding to this output from the parent DetImage's .data attribute using the output_slice.
        DetImage.data is expected to be a xr.Dataset with a data_var named 'data'.
        """
        return slice_data(self.parent.data, self.output_slice)

    @property
    def combined_mask(self):
        if self.masks:
            combmask = np.any(self.masks.to_array().to_numpy(), axis=0)
            return xr.DataArray(combmask, dims=self.masks.dims, coords=self.masks.coords)
        else:
            logger.error("No masks to combine.")
            return None

    @property
    def image_region(self) -> dict[str, slice]:
        """ Return the slice defining image region (i.e. only the active light capturing pixels) for this output. """
        return {axis: slc for axis, slc in zip(["y", "x"], self.output_slice)}

    def get_image_region(self,
                         return_masks: bool = False,
                         mask_key: str = 'combined',
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
        immask = None
        if return_masks:
            mask = self.get_output_mask(mask_key)
            immask = slice_data(mask, imslc) if mask is not None else None

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
        new_data = ensure_dataarray(new_data)
        # Assign new data to the appropriate slice in the parent DetImage
        slicer = {new_data.dims[i]: slc for i, slc in enumerate(self.output_slice)} if slicer is None else slicer
        self.parent.set_data_slice(new_data, slicer, data_var=data_var)

    def _select_header_and_data(
        self, iheaders: Iterable[Mapping], idata: TData
    ) -> tuple[Mapping, TData]:
        """select which data represents this output in a collection of input headers and input data (big array).
        This is intended for internal use only and eventual destruction in the righteous fire of refactoring because it
        too tightly couples Outputs to FITS files. For now, however, it solves a problm in a slightly cleaner way than
        duplicating this logic in several other places

        parameters
        ---------
        :param iheaders
            collection of header information, probably loaded from a FITS file (or HDF5 file in my wildest of happy dreams)
        :param idata: TData
            array type which can be indexed into using a numerical index, and then a slice. We don't care much beyond that

        returns
        -------
        tuple[Mapping, TData]
            the tuple of the selected header corresponding to this output, and the sliced data corresponding to this output
        """

        ext_idx: int = int()

        match self.input_array_axis:
            case int():
                ext_idx = self.input_array_axis
            case str():
                # HIDEOUS: this is FITS specific, and not only that,
                # but NASA Registry of FITS conventions (1977 I believe)  specific.
                # You have absolutely no idea how much I want to ___redacted___ myself right now
                try:
                    ext_idx: int = next(
                        (
                            i
                            for i, v in enumerate(iheaders)
                            if v.get("EXTNAME", "") == self.input_array_axis
                        )
                    )
                except StopIteration as err:
                    raise ValueError(
                        f"HDU with EXTNAME of {self.input_array_axis} not found in provided data"
                    ) from err
            case _:
                raise TypeError("couldn't interpret input_array_axis or ext_id")

        header = iheaders[ext_idx]
        data = idata[ext_idx][*self.input_slice]

        return header, data

    def show(self, ax=None, save=None, with_mask=True, mask_key='sigma_clip_mask', data_var='data',
             **imshow_kwargs):
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

        if with_mask:
            temp = self.parent.get_masked_data(data_var, mask_key, fill_value=0)
        else:
            temp = self.parent.get_data(data_var)
        temp = slice_data(temp, self.output_slice)

        im = temp.plot.imshow(ax=ax, **imshow_kwargs)

        if save is not None:
            ax.figure.savefig(save)
        return ax

class CCDOutput(Output):
    """
    CCD-specific output region with prescan/overscan info.
    """

    serial_prescan: slice = Field(
        slice(None),
        description="Slice object defining the serial prescan region for this output.",
    )
    serial_overscan: slice = Field(
        slice(None),
        description="Slice object defining the serial overscan region for this output.",
    )
    parallel_prescan: slice = Field(
        slice(None),
        description="Slice object defining the parallel prescan region for this output.",
    )
    parallel_overscan: slice = Field(
        slice(None),
        description="Slice object defining the parallel overscan region for this output.",
    )
    parallel_axis: Literal["x", "y"] = Field(
        "y",
        description="Name of the parallel readout axis for this output ('x' or 'y').",
    )
    readout_pixel: tuple[int, int] = Field(
        (0, 0),
        description="Tuple defining the (y, x) pixel coordinates of the readout amplifier "
        "for this output in the full detector array.",
    )

    @property
    def serial_axis(self) -> Literal["x", "y"]:
        return "x" if self.parallel_axis == "y" else "y"

    @property
    def parallel_axint(self):
        return 0 if self.parallel_axis == "y" else 1

    @property
    def serial_axint(self):
        return 0 if self.serial_axis == "y" else 1

    @property
    def image_region(self) -> dict[str, slice]:
        parallel_step = (
            -1 if self.parallel_prescan.stop > self.parallel_overscan.start else 1
        )
        im_slc_parallel = slice(
            self.parallel_prescan.stop, self.parallel_overscan.start, parallel_step
        )
        serial_step = -1 if self.serial_prescan.stop > self.serial_overscan.start else 1
        im_slc_serial = slice(
            self.serial_prescan.stop, self.serial_overscan.start, serial_step
        )
        return {self.parallel_axis: im_slc_parallel, self.serial_axis: im_slc_serial}

    def get_scan(
        self,
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
            slicer[getattr(self, f"{other_axis}_axis")] = slice(
                other_prescan.stop, other_overscan.start, step
            )
        return slice_data(self.parent.get_data(data_var), slicer)

    def get_prescan(
        self, axis: Literal['serial', 'parallel'], corner: bool = False,
        data_var: Optional[str | Literal['all']] = 'data'
    ) -> XRDATA:
        """
        Slice the data array to get the prescan region along the specified axis (serial or parallel).
        :param axis: serial or parallel
        :param corner: True to include the corner region (intersection of prescan and overscan) in the returned slice,
        False to exclude it.
        :param data_var: Name of the data variable in the DetImage to slice. If 'all', slices the entire dataset.
        :return: sliced xr.DataArray corresponding to the requested prescan region.
        """
        return self.get_scan(axis=axis, kind='prescan', corner=corner, data_var=data_var)

    def get_overscan(
        self, axis: Literal['serial', 'parallel'], corner: bool = False,
        data_var: Optional[str | Literal['all']] = 'data'
    ) -> XRDATA:
        """
        Slice the data array to get the overscan region along the specified axis (serial or parallel).
        :param axis: serial or parallel
        :param corner: True to include the corner region (intersection of prescan and overscan) in the returned slice,
        False to exclude it.
        :param data_var: Name of the data variable in the DetImage to slice. If 'all', slices the entire dataset.
        :return: sliced xr.DataArray corresponding to the requested overscan region.
        """
        return self.get_scan(axis=axis, kind='overscan', corner=corner, data_var=data_var)

    def show(self, ax=None, save=None, with_mask=True, mask_key='sigma_clip_mask', data_var='data',
             shade_regions=False, **imshow_kwargs):
        ax = super().show(ax=ax, save=None, with_mask=with_mask, mask_key=mask_key, data_var=data_var, **imshow_kwargs)

        if shade_regions:
            ## Shade the prescan and overscan regions
            def _bounds(s: slice, n: int) -> tuple[int, int]:
                s0 = s.start if s.start is not None else (0 if s.step > 0 else n)
                s1 = (
                    (s.stop - 1 if s.step > 0 else s.stop + 1)
                    if s.stop is not None
                    else (n if s.step > 0 else 0)
                )
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
                axis_idx = 0 if axis == "y" else 1
                a, b = _bounds(s, shape[axis_idx])
                spandict[axis_idx](a, b, color=color, alpha=0.3, label=label)
            ax.scatter(
                self.readout_pixel[1],
                self.readout_pixel[0],
                marker="x",
                color="red",
                s=100,
            )
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
