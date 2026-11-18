from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, Literal, TypeVar, Mapping, Iterable
from pydantic import Field, ConfigDict, field_serializer
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

from .mappable import Mappable
from .operators import ArithmeticOperators
from eregion.utils import (
    ensure_dataarray,
    slice_data,
    XRDATA
)
from eregion.core.image_operations import arithmetic, propagate_std as propagate_std_through
from eregion.core.image_stats import STATFUNCS, as_masked_array, check_mask_aware, do_statistics, reduce_array
if TYPE_CHECKING:
    from eregion.datamodels.image import DetImage

logger = logging.getLogger(__name__)
TData = TypeVar("TData")

############################################### OUTPUT BASE CLASS #####################################################
class Output(ArithmeticOperators, Mappable):
    """
    One amplifier/output region within a detector image.

    Arithmetic: add/subtract/multiply/divide (and unary_op/binary_op) change the output's region of the parent
    DetImage in place. The operators (`out + 2`, `3 * out`, `out - other_out`) instead return a new xr.DataArray of
    the full output and leave the parent untouched.
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

    def get_output_data(self, data_var: Optional[str | Literal['all']] = 'data') -> XRDATA:
        """
        Return the array corresponding to this output from the parent DetImage's .data attribute using the output_slice.
        :param data_var: str or 'all'
            Name of the data variable in the DetImage.data (xr.Dataset) to slice. If 'all', slices the entire dataset.
        :return: xr.DataArray or xr.Dataset corresponding to this output.
        """
        return slice_data(self.parent.get_data(data_var), self.output_slice)

    def get_masked_output_data(self, data_var: str = 'data', mask_key: str = 'combined', fill_value=np.nan):
        mask = self.get_output_mask(mask_key)
        datarr = self.get_output_data(data_var)
        if mask is not None:
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

    ############################################ Regions ############################################
    @property
    def region_names(self) -> list[str]:
        """ Names of the regions accepted by get_region() and the arithmetic / reduce methods. """
        return ['full', 'image']

    def _region_slicer(self, region: str) -> dict[str, slice]:
        """
        Return the {dim: slice} slicer of a named region of this output, in parent DetImage coordinates.
        :param region: str, one of self.region_names. 'full' is the whole output_slice, 'image' the image_region.
        :return: dict of dimension name to slice.
        """
        match region:
            case 'full':
                return {dim: slc for dim, slc in zip(['y', 'x'], self.output_slice)}
            case 'image':
                return dict(self.image_region)
            case _:
                raise ValueError(f"Unknown region '{region}' for {type(self).__name__}, "
                                 f"expected one of {self.region_names}")

    def get_region(self,
                   region: str = 'full',
                   data_var: Optional[str | Literal['all']] = 'data',
                   mask_key: str = 'combined',
                   with_mask: bool = False) -> tuple[XRDATA, Optional[XRDATA]]:
        """
        Return the data of a named region of this output and, optionally, its mask.
        :param region: str, one of self.region_names.
        :param data_var: str or 'all', data variable of the parent DetImage to slice.
        :param mask_key: str, mask to return when with_mask is True ('combined' ORs all masks of this output).
        :param with_mask: bool, if True also return the mask of the region.
        :return: tuple of (region data, region mask). The mask is None if not requested, or if mask_key is
            'combined' and the output has no masks.
        :raises KeyError: if mask_key is not 'combined' and not one of this output's masks.
        """
        slicer = self._region_slicer(region)
        data = slice_data(self.parent.get_data(data_var), slicer)
        mask = self._lookup_mask(mask_key) if with_mask else None
        return data, (slice_data(mask, slicer) if mask is not None else None)

    def _lookup_mask(self, mask_key: str) -> Optional[xr.DataArray]:
        """ Return one mask of this output, the OR of all with 'combined' (None if it has no masks). """
        if mask_key == 'combined':
            return self.combined_mask if self.masks else None
        if not self.masks or mask_key not in self.masks:
            available = list(self.masks.data_vars) if self.masks else []
            raise KeyError(f"Output {self.id} has no mask '{mask_key}'; its masks are {available} (or 'combined')")
        return self.masks[mask_key]

    ######################################## Arithmetic ########################################
    def unary_op(self, op: str, value: float | np.ndarray, region: str = 'full', propagate_std: bool = True,
                 reflected: bool = False):
        """
        Apply `data <op> value` to a region of this output, in place in the parent DetImage.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param value: float or np.ndarray broadcastable to the region (e.g. a per-row median kept with keepdims).
        :param region: str, region of this output to operate on, one of self.region_names.
        :param propagate_std: bool, if True and the parent has a 'std' data variable, propagate it (value is exact).
        :param reflected: bool, if True apply `value <op> data` instead (e.g. 1 - data, 2 / data).
        :return: self
        """
        self.parent.ensure_float()
        slicer = self._region_slicer(region)
        data = slice_data(self.parent.data, slicer).values
        new_data = arithmetic(data, value, op, reflected=reflected)
        if propagate_std and self.parent.has_data_var('std'):
            std = slice_data(self.parent.get_data('std'), slicer).values
            if reflected:
                new_std = propagate_std_through(op, np.broadcast_to(value, data.shape), data, std_b=std)
            else:
                new_std = propagate_std_through(op, data, value, std_a=std)
            self.set_data_in_parent(new_std, slicer, data_var='std')
        self.set_data_in_parent(new_data, slicer, data_var='data')
        return self

    def binary_op(self, op: str, other: Output, region: str = 'full', combine_masks: bool = True,
                  propagate_std: bool = True):
        """
        Apply `self <op> other` pixel by pixel to a region of this output, in place in the parent DetImage. The two
        regions are matched by position, so they only need the same shape.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param other: Output of the same class as this output.
        :param region: str, region of both outputs to operate on, one of self.region_names.
        :param combine_masks: bool, if True OR each of other's masks into this output's mask of the same key.
        :param propagate_std: bool, if True and this output's parent has a 'std' data variable, propagate it
            (other's 'std' is used if its parent has one, else taken as zero). The two outputs are treated as
            independent, so e.g. subtracting an output from itself gives sqrt(2) * std, not 0.
        :return: self
        """
        if type(other) is not type(self):
            raise TypeError(f"Cannot {op} a {type(other).__name__} and a {type(self).__name__}; "
                            f"both outputs must be of the same class")
        self.parent.ensure_float()
        slicer = self._region_slicer(region)
        data = slice_data(self.parent.data, slicer).values
        other_data = slice_data(other.parent.data, other._region_slicer(region)).values
        if data.shape != other_data.shape:
            raise ValueError(f"Region '{region}' of output {self.id} has shape {data.shape}, "
                             f"but of output {other.id} has shape {other_data.shape}")
        new_data = arithmetic(data, other_data, op)
        if propagate_std and self.parent.has_data_var('std'):
            std = slice_data(self.parent.get_data('std'), slicer).values
            other_std = (slice_data(other.parent.get_data('std'), other._region_slicer(region)).values
                         if other.parent.has_data_var('std') else None)
            self.set_data_in_parent(propagate_std_through(op, data, other_data, std, other_std), slicer, data_var='std')
        self.set_data_in_parent(new_data, slicer, data_var='data')
        if combine_masks:
            self._or_masks_from(other)
        return self

    def _or_masks_from(self, other: Output):
        """ OR every mask of other into the mask of the same key of this output, matching pixels by position. """
        if other.masks is None:
            return
        full = slice_data(self.parent.data, self.output_slice)
        for key in other.masks.data_vars:
            other_mask = other.masks[key].values.astype(bool)
            if other_mask.shape != full.shape:
                raise ValueError(f"Mask '{key}' of output {other.id} has shape {other_mask.shape}, "
                                 f"but output {self.id} has shape {full.shape}")
            self.set_mask(key, full.copy(data=other_mask), combine=True)

    def add(self, other: Output | float | np.ndarray, **kwargs):
        """ Add a number, an array or another output to this output in place. See unary_op() and binary_op(). """
        return self._dispatch('add', other, **kwargs)

    def subtract(self, other: Output | float | np.ndarray, **kwargs):
        """ Subtract a number, an array or another output from this output in place. See unary_op() and binary_op(). """
        return self._dispatch('subtract', other, **kwargs)

    def multiply(self, other: Output | float | np.ndarray, **kwargs):
        """ Multiply this output by a number, an array or another output in place. See unary_op() and binary_op(). """
        return self._dispatch('multiply', other, **kwargs)

    def divide(self, other: Output | float | np.ndarray, **kwargs):
        """ Divide this output by a number, an array or another output in place. See unary_op() and binary_op(). """
        return self._dispatch('divide', other, **kwargs)

    def _dispatch(self, op: str, other: Output | float | np.ndarray, **kwargs):
        if isinstance(other, Output):
            return self.binary_op(op, other, **kwargs)
        return self.unary_op(op, other, **kwargs)

    def _operate(self, op: str, other: Output | float | np.ndarray, reflected: bool = False) -> xr.DataArray:
        """ Return `self <op> other` over the full output as a new DataArray, leaving the parent untouched. """
        data = slice_data(self.parent.data, self.output_slice)
        if isinstance(other, Output):
            if type(other) is not type(self) or reflected:
                return NotImplemented
            other = slice_data(other.parent.data, other.output_slice).values
            if other.shape != data.shape:
                raise ValueError(f"Output shapes {data.shape} and {other.shape} do not match")
        elif not self._is_operand(other):
            return NotImplemented
        return data.copy(data=arithmetic(data.values, other, op, reflected=reflected))

    ######################################## Reductions ########################################
    def _masked_region(self, region: str, data_var: str, mask_key: str, use_mask: bool,
                       mask_invalid: bool) -> np.ma.MaskedArray:
        data, mask = self.get_region(region, data_var=data_var, mask_key=mask_key, with_mask=use_mask)
        return as_masked_array(data.values, mask.values if mask is not None else None, mask_invalid=mask_invalid)

    def reduce(self, stat: str | Callable = 'median', region: str = 'full', axis: Optional[int] = None,
               data_var: str = 'data', mask_key: str = 'combined', use_mask: bool = True,
               mask_invalid: bool = False):
        """
        Reduce a region of this output to one statistic.
        :param stat: str or Callable, name in core.image_stats.STATFUNCS (e.g. 'median', 'mean', 'std', 'mad') or a
            function called as stat(masked_array, axis=axis) that respects the mask (e.g. np.ma.max). Plain numpy
            and scipy functions ignore masks, so they raise TypeError unless use_mask and mask_invalid are False.
        :param region: str, region of this output, one of self.region_names.
        :param axis: int or None, axis to collapse (0 = 'y', 1 = 'x'); None reduces to a single value.
        :param data_var: str, data variable of the parent DetImage to use.
        :param mask_key: str, mask whose pixels are ignored when use_mask is True.
        :param use_mask: bool, if True ignore masked pixels.
        :param mask_invalid: bool, if True also ignore NaN and inf pixels.
        :return: scalar or np.ndarray; NaN where every pixel is masked.
        :raises KeyError: if use_mask is True and mask_key is not 'combined' or one of this output's masks.
        """
        if use_mask or mask_invalid:
            check_mask_aware(stat)
        return reduce_array(self._masked_region(region, data_var, mask_key, use_mask, mask_invalid), stat, axis=axis)

    def stats(self, which: dict[str, Callable] = STATFUNCS, region: str = 'full', axis: Optional[int] = None,
              data_var: str = 'data', mask_key: str = 'combined', use_mask: bool = True,
              mask_invalid: bool = False, prepend_kw: str = '') -> dict[str, Any]:
        """
        Compute several statistics of a region of this output with core.image_stats.do_statistics.
        :param which: dict of statistic name to function, default STATFUNCS.
        :param prepend_kw: str, prefix for the keys of the returned dict.
        Other parameters are as in reduce().
        :return: dict of statistic name to value.
        """
        if use_mask or mask_invalid:
            for stat in which.values():
                check_mask_aware(stat)
        masked = self._masked_region(region, data_var, mask_key, use_mask, mask_invalid)
        return do_statistics(masked, which=which, axis=axis, prepend_kw=prepend_kw)

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
        return slice_data(self.parent.get_data(data_var), self._scan_slicer(axis, kind, corner))

    def _scan_slicer(self, axis: Literal['serial', 'parallel'], kind: Literal['prescan', 'overscan'],
                     corner: bool = False) -> dict[str, slice]:
        """ Return the {dim: slice} slicer of a scan region; see get_scan(). """
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
        return slicer

    @property
    def region_names(self) -> list[str]:
        return super().region_names + [f"{axis}_{kind}" for axis in ('serial', 'parallel')
                                       for kind in ('prescan', 'overscan')]

    def _region_slicer(self, region: str) -> dict[str, slice]:
        """ Adds the scan regions ('serial_overscan', ...) to the base regions; scans exclude the corners. """
        axis, _, kind = region.partition('_')
        if axis in ('serial', 'parallel') and kind in ('prescan', 'overscan'):
            return self._scan_slicer(axis, kind, corner=False)
        return super()._region_slicer(region)

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