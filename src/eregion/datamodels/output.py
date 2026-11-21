from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, Literal, TypeVar, Mapping, Iterable
from pydantic import Field, ConfigDict, field_serializer, model_validator
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

from .mappable import Mappable
from .operators import ArithmeticOperators
from eregion.utils import to_slicer
from eregion.core.image_operations import arithmetic, propagate_std as propagate_std_through
from eregion.core.image_stats import STATFUNCS, check_mask_aware, do_statistics, reduce_array
if TYPE_CHECKING:
    from eregion.datamodels.image import DetImage

logger = logging.getLogger(__name__)
TData = TypeVar("TData")

Region = str | dict | tuple | None
MaskSelection = str | list[str] | tuple[str, ...] | None

############################################### OUTPUT BASE CLASS #####################################################
class Output(ArithmeticOperators, Mappable):
    """
    One amplifier/output region within a detector image. An Output holds no arrays: it reads and writes the vars of
    its parent DetImage, restricted to its region.

    Reading and writing use the same methods and arguments as DetImage (get_var, get_mask, get_masked, set_var),
    where region is:

    - None: the whole output (output_slice),
    - a region name from region_names: 'full', 'image' and, for CCD outputs, 'serial_prescan', 'serial_overscan',
      'parallel_prescan', 'parallel_overscan',
    - a dict or tuple of slices in the parent image's pixel coordinates.

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

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_masks(cls, values: Any) -> Any:
        # masks are vars of the parent DetImage now; an extra 'masks' field would be silently ignored
        if isinstance(values, dict) and "masks" in values:
            raise TypeError("Outputs hold no masks; write them with output.set_var('<prefix>_mask', values)")
        return values

    @field_serializer("header", when_used="json")
    def serialize_header(self, value: fits.Header | dict | None) -> dict | None:
        return dict(value) if isinstance(value, fits.Header) else value

    ############################################ Regions ############################################
    @property
    def region_names(self) -> list[str]:
        """ Names of the regions accepted as region= by the methods of this output. """
        return ['full', 'image']

    def region_slicer(self, name: str) -> dict[str, slice]:
        """
        Return the {dim: slice} slicer of a named region of this output, in parent pixel coordinates.
        :param name: str, one of self.region_names. 'full' is the whole output_slice; for a plain Output 'image' is
            the same, CCD outputs exclude the pre/overscans.
        :return: dict of dimension name to slice.
        """
        if name in ('full', 'image'):
            return {dim: slc for dim, slc in zip(['y', 'x'], self.output_slice)}
        raise ValueError(f"Unknown region '{name}' for {type(self).__name__}, expected one of {self.region_names}")

    def _resolve(self, region: Region) -> dict[str, slice]:
        """ Turn a region (None, a region name, or a slicer) into a slicer in parent pixel coordinates. """
        if region is None:
            return self.region_slicer('full')
        if isinstance(region, str):
            return self.region_slicer(region)
        return to_slicer(region, dims=('y', 'x'))

    def _parent_or_raise(self) -> "DetImage":
        if self.parent is None:
            raise ValueError(f"Output {self.id} has no parent DetImage")
        return self.parent

    ######################################## Reading and writing ########################################
    @property
    def data(self) -> xr.DataArray:
        """ The parent's 'data' over this output. """
        return self.get_var('data')

    @property
    def mask_names(self) -> list[str]:
        """ Names of the parent's mask vars. """
        return self._parent_or_raise().mask_names

    @property
    def label_names(self) -> list[str]:
        """ Names of the parent's label vars. """
        return self._parent_or_raise().label_names

    def get_var(self, name: str = 'data', region: Region = None) -> xr.DataArray:
        """ A var of the parent over a region of this output; see DetImage.get_var. """
        return self._parent_or_raise().get_var(name, self._resolve(region))

    def get_mask(self, masks: MaskSelection = 'combined', region: Region = None) -> xr.DataArray:
        """ The OR of the selected masks over a region of this output; see DetImage.get_mask. """
        return self._parent_or_raise().get_mask(masks, self._resolve(region))

    def get_masked(self, name: str = 'data', masks: MaskSelection = 'combined', region: Region = None,
                   mask_invalid: bool = False) -> np.ma.MaskedArray:
        """ A data var over a region of this output as a numpy masked array; see DetImage.get_masked. """
        return self._parent_or_raise().get_masked(name, masks, self._resolve(region), mask_invalid)

    def set_var(self, name: str, values: np.ndarray | xr.DataArray, region: Region = None, combine: bool = False):
        """ Write a var of the parent over a region of this output; see DetImage.set_var. """
        self._parent_or_raise().set_var(name, values, self._resolve(region), combine)

    ######################################## Arithmetic ########################################
    def unary_op(self, op: str, value: float | np.ndarray, region: Region = None, propagate_std: bool = True,
                 reflected: bool = False):
        """
        Apply `data <op> value` to a region of this output, in place in the parent DetImage.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param value: float or np.ndarray broadcastable to the region (e.g. a per-row median kept with keepdims).
        :param region: region of this output to operate on (see the class docstring).
        :param propagate_std: bool, if True and the parent has a 'std' data var, propagate it (value is exact).
        :param reflected: bool, if True apply `value <op> data` instead (e.g. 1 - data, 2 / data).
        :return: self
        """
        parent = self._parent_or_raise()
        parent.ensure_float()
        slicer = self._resolve(region)
        data = parent.get_var('data', slicer).values
        new_data = arithmetic(data, value, op, reflected=reflected)
        if propagate_std and 'std' in parent.data_names:
            std = parent.get_var('std', slicer).values
            if reflected:
                new_std = propagate_std_through(op, np.broadcast_to(value, data.shape), data, std_b=std)
            else:
                new_std = propagate_std_through(op, data, value, std_a=std)
            parent.set_var('std', new_std, slicer)
        parent.set_var('data', new_data, slicer)
        return self

    def binary_op(self, op: str, other: Output, region: Region = None, combine_masks: bool = True,
                  propagate_std: bool = True):
        """
        Apply `self <op> other` pixel by pixel to a region of this output, in place in the parent DetImage. The two
        regions are matched by position, so they only need the same shape.
        :param op: str, one of 'add', 'subtract', 'multiply', 'divide'.
        :param other: Output of the same class as this output.
        :param region: region of both outputs to operate on (a region name, or a slicer used for both).
        :param combine_masks: bool, if True OR each of other's masks into this output's mask of the same name.
        :param propagate_std: bool, if True and this output's parent has a 'std' data var, propagate it
            (other's 'std' is used if its parent has one, else taken as zero). The two outputs are treated as
            independent, so e.g. subtracting an output from itself gives sqrt(2) * std, not 0.
        :return: self
        """
        if type(other) is not type(self):
            raise TypeError(f"Cannot {op} a {type(other).__name__} and a {type(self).__name__}; "
                            f"both outputs must be of the same class")
        parent = self._parent_or_raise()
        parent.ensure_float()
        slicer = self._resolve(region)
        data = parent.get_var('data', slicer).values
        other_data = other.get_var('data', region).values
        if data.shape != other_data.shape:
            raise ValueError(f"Region '{region}' of output {self.id} has shape {data.shape}, "
                             f"but of output {other.id} has shape {other_data.shape}")
        new_data = arithmetic(data, other_data, op)
        if propagate_std and 'std' in parent.data_names:
            std = parent.get_var('std', slicer).values
            other_std = other.get_var('std', region).values if 'std' in other.parent.data_names else None
            parent.set_var('std', propagate_std_through(op, data, other_data, std, other_std), slicer)
        parent.set_var('data', new_data, slicer)
        if combine_masks:
            self._or_masks_from(other)
        return self

    def _or_masks_from(self, other: Output):
        """ OR every mask of other into the mask of the same name of this output, matching pixels by position. """
        for name in other.mask_names:
            self.set_var(name, other.get_mask(name).values, combine=True)

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
        data = self.data
        if isinstance(other, Output):
            if type(other) is not type(self) or reflected:
                return NotImplemented
            other = other.data.values
            if other.shape != data.shape:
                raise ValueError(f"Output shapes {data.shape} and {other.shape} do not match")
        elif not self._is_operand(other):
            return NotImplemented
        return data.copy(data=arithmetic(data.values, other, op, reflected=reflected))

    ######################################## Reductions ########################################
    def reduce(self, stat: str | Callable = 'median', region: Region = None, axis: Optional[int] = None,
               name: str = 'data', masks: MaskSelection = 'combined', mask_invalid: bool = False):
        """
        Reduce a region of this output to one statistic.
        :param stat: str or Callable, name in core.image_stats.STATFUNCS (e.g. 'median', 'mean', 'std', 'mad') or a
            function called as stat(masked_array, axis=axis) that respects the mask (e.g. np.ma.max). Plain numpy
            and scipy functions ignore masks, so they raise TypeError unless masks is None and mask_invalid False.
        :param region: region of this output (see the class docstring).
        :param axis: int or None, axis to collapse (0 = 'y', 1 = 'x'); None reduces to a single value.
        :param name: str, data var to reduce.
        :param masks: masks whose pixels are ignored ('combined', a name, a list of names), or None for none.
        :param mask_invalid: bool, if True also ignore NaN and inf pixels.
        :return: scalar or np.ndarray; NaN where every pixel is masked.
        :raises KeyError: for a mask name the parent does not have.
        """
        if masks is not None or mask_invalid:
            check_mask_aware(stat)
        return reduce_array(self.get_masked(name, masks, region, mask_invalid), stat, axis=axis)

    def stats(self, which: dict[str, Callable] = STATFUNCS, region: Region = None, axis: Optional[int] = None,
              name: str = 'data', masks: MaskSelection = 'combined', mask_invalid: bool = False,
              prepend_kw: str = '') -> dict[str, Any]:
        """
        Compute several statistics of a region of this output with core.image_stats.do_statistics.
        :param which: dict of statistic name to function, default STATFUNCS.
        :param prepend_kw: str, prefix for the keys of the returned dict.
        Other parameters are as in reduce().
        :return: dict of statistic name to value.
        """
        if masks is not None or mask_invalid:
            for stat in which.values():
                check_mask_aware(stat)
        return do_statistics(self.get_masked(name, masks, region, mask_invalid), which=which, axis=axis,
                             prepend_kw=prepend_kw)

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

    def show(self, ax=None, save=None, name: str = 'data', masks: MaskSelection = 'combined', **imshow_kwargs):
        """
        Plot a data var over this output, with masked pixels left blank.
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :param save: Optional[str], path to save the plot.
        :param name: str, data var to plot. Default is 'data'.
        :param masks: masks to blank out, as in get_mask; None plots every pixel.
        :param imshow_kwargs: dict, additional keyword arguments to pass to xarray.DataArray.plot.imshow().
        :return: Matplotlib Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
        data = self.get_var(name)
        if masks is not None:
            data = data.where(~self.get_mask(masks).values)
        data.plot.imshow(ax=ax, **imshow_kwargs)
        if save is not None:
            ax.figure.savefig(save)
        return ax


def _between(prescan: slice, overscan: slice) -> slice:
    """ The slice from the end of a prescan to the start of an overscan, reversed if the readout runs backwards. """
    step = -1 if prescan.stop > overscan.start else 1
    return slice(prescan.stop, overscan.start, step)


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
    def region_names(self) -> list[str]:
        return super().region_names + [f"{axis}_{kind}" for axis in ('serial', 'parallel')
                                       for kind in ('prescan', 'overscan')]

    def region_slicer(self, name: str, corner: bool = False) -> dict[str, slice]:
        """
        Return the {dim: slice} slicer of a named region, in parent pixel coordinates. 'image' is the light
        sensitive area between the pre- and overscans; the scan regions ('serial_overscan', ...) span the image area
        along the other axis, or the whole output with corner=True. Slices run in readout order (step -1 when the
        prescan comes after the overscan in pixel coordinates).
        :param name: str, one of self.region_names.
        :param corner: bool, scan regions only: include the corners where the two axes' scans overlap.
        :return: dict of dimension name to slice.
        """
        axes = {'serial': self.serial_axis, 'parallel': self.parallel_axis}

        def between(axis: str) -> slice:
            return _between(getattr(self, f"{axis}_prescan"), getattr(self, f"{axis}_overscan"))

        axis, _, kind = name.partition('_')
        is_scan = axis in axes and kind in ('prescan', 'overscan')
        if corner and not is_scan:
            raise ValueError("corner=True applies to the scan regions only")
        if name == 'image':
            return {axes[axis]: between(axis) for axis in axes}
        if is_scan:
            slicer = super().region_slicer('full')
            slicer[axes[axis]] = getattr(self, name)
            if not corner:
                other = 'serial' if axis == 'parallel' else 'parallel'
                slicer[axes[other]] = between(other)
            return slicer
        return super().region_slicer(name)

    def show(self, ax=None, save=None, name: str = 'data', masks: MaskSelection = 'combined',
             shade_regions: bool = False, **imshow_kwargs):
        ax = super().show(ax=ax, save=None, name=name, masks=masks, **imshow_kwargs)

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
