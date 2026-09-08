from copy import deepcopy
from abc import abstractmethod
from typing import Optional, Any, Literal
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from itertools import batched
from functools import wraps

from eregion.utils import ensure_numpy, slice_data, decrease_slicer_stop_index
from eregion.datamodels import DetImage, Output, CCDOutput, ImageBundle
from eregion.core.image_operations import sigma_clip_image
from eregion.core.image_stats import do_statistics, STATFUNCS
from eregion.tasks import LazyTask, ImageResult

########### BasePreprocessingTask ###########
class BasePreprocessingTask(LazyTask):
    """
    Base class for image preprocessing tasks to reduce repetitive code. Most preproc tasks do some processing
    to each image independently and can benefit from parallelization for multiple images.

    Each subclass just needs to implement _process_single_image() method to process a single DetImage.
    The base lazy_run will handle batching and parallel execution.

    The results are wrapped in ImageResult subclass of TaskResult.
    """
    task_result = ImageResult

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def _process_single_image(self, img: DetImage) -> DetImage:
        return img

    def lazy_run(
            self,
            images: ImageBundle | list[DetImage],
            **kwargs
    ):
        """
        :param images: an iterable of DetImage or ImageBundle to process
        Keyword arguments

        batch_size: number of images to process per batch
        """
        image_list = images.images if isinstance(images, ImageBundle) else images
        batch_size = kwargs.get("batch_size", len(image_list))
        for batch in batched(image_list, batch_size):
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)
            yield self.task_result(data=results)

    @wraps(lazy_run)
    def run(self, *args, **kwargs) -> ImageResult:
        return super().run(*args, **kwargs)


########### Bias Subtraction Task ###########
class BiasSubtraction(BasePreprocessingTask):
    task_result = ImageResult

    def __init__(self, name: Optional[str] = "subtract_bias", **kwargs):
        """
        Subtract master bias frames from input images.
        :param name: Optional[str]
        :keyword only_image_area: bool, If True, only subtract bias from the image area, ignoring overscan/prescan regions. Default is True.
        :keyword quick: bool, If True, perform full image subtraction including overscan/prescan regions. Default is False.
        """
        super().__init__(name=name, **kwargs)
        self.master_bias = None

    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage by subtracting the corresponding master bias.
        :param img: DetImage
            The science image to be bias-subtracted.
        :return: DetImage
            The bias-subtracted science image.
        """
        # Load the appropriate master bias for this image based on metadata (e.g., detector name)
        # TODO: this should be improved to match the correct master bias using groupby keys.
        master_bias = self.master_bias.filter(f'det_id == "{img.id}"')
        if not master_bias:
            raise ValueError(f"No matching master bias found for DetImage with name '{img.id}'.")
        else:
            master_bias = master_bias[0]
            self.logger.info(f"Using master bias with det_id='{master_bias.id}' and type='{master_bias.image_type}'.")

        if self.meta.get('quick', False):
            self.logger.warning("kwarg 'quick' is True, performing full image subtraction including overscan/prescan regions.")
            sub_img = self._subtract(img.data, master_bias.data)
            img.set_data(sub_img)
        else:
            # overwrite img
            for output in img.outputs.values():
                mb_output = master_bias.outputs[output.id]
                if self.meta.get("only_image_area", True):
                    if 'image_region' in dir(output):
                        imslc = output.image_region
                        sub_output = self._subtract(slice_data(output.data, imslc),
                                                   slice_data(mb_output.data, imslc))
                        img.set_data_slice(sub_output, imslc)
                    else:
                        raise ValueError(f"No image_region attribute found for Output '{output.id}' of DetImage {img.id}.")
                else:
                    self.logger.warning("kwarg 'only_image_area' is False, performing full output subtraction including overscan/prescan regions.")
                    sub_output = self._subtract(output.data, mb_output.data)
                    output.set_data_in_parent(sub_output)

        img.image_type.update({'bias_subtracted': True})
        return img

    def lazy_run(self,
                 images,
                 master_bias: Optional[ImageBundle | list[DetImage] | DetImage] = None,
                 **kwargs):
        """
        Calls this class' _process_single_image() per input image which applies bias subtraction.
        :param images: Iterable of DetImage objects or ImageBundle to be processed.
        :param master_bias: ImageBundle | list[DetImage] The master bias frames to subtract from science images.
        :keyword batch_size: int, Number of images to process per batch
        """
        if master_bias is None:
            raise ValueError("master_bias must be provided for bias subtraction.")
        self.master_bias = master_bias if isinstance(master_bias, ImageBundle) else ImageBundle(master_bias)

        return super().lazy_run(images, **kwargs)

    @wraps(lazy_run)
    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

    @staticmethod
    def _subtract(image: np.ndarray | xr.DataArray, master_bias: np.ndarray | xr.DataArray) -> np.ndarray:
        """
        Convenience non-Prefect path for raw arrays.
        """
        if master_bias.shape != image.shape:
            raise ValueError(
                f"Master bias array shape {master_bias.shape} does not match image shape {image.shape}.")
        return ensure_numpy(image) - ensure_numpy(master_bias)

    def __call__(self, image, master_bias):
        return self._subtract(image, master_bias)


########### Scan Subtraction Task ###########
class ScanSubtraction(BasePreprocessingTask):
    task_result = ImageResult

    def __init__(self,
                 which_scan: str,
                 name: Optional[str] = None,
                 **kwargs
    ):
        """
        Subtract scan region from input image. Example, serial overscan subtraction.
        Basic statistics are calculated on the scan region and stored in the output header, before the subtraction
        value is calculated and subtracted from the output data. The subtraction is done on the full array, for example,
        in case of serial overscan the full slice along the parallel axis (prescan, image area, overscan) is used
        and subtracted from the full array (including serial overscan itself).

        :param which_scan: str, required
            One of 'serial_prescan', 'serial_overscan', 'parallel_prescan', 'parallel_overscan'.
        :param name: Optional[str]
            Name identifying the task instance. Default is None.
        :keyword method: str
            method to use for subtraction. Default is 'median_by_axis'.
        :keyword trim_start: int
            number of indices to trim from the start of the scan region before calculating the subtraction value. Default is 0.
        :keyword trim_end: int
            number of indices to trim from the end of the scan region before calculating the subtraction value. Default is 0.
        """
        if 'method' not in kwargs:
            self.logger.warning("'method' not specified, defaulting to 'median_by_axis'.")
            kwargs['method'] = 'median_by_axis'
        super().__init__(name=name, **kwargs)
        self.which_scan = which_scan.lower()
        if self.which_scan not in ["serial_prescan", "serial_overscan", "parallel_prescan", "parallel_overscan"]:
            raise ValueError(f"Invalid which_scan value: {self.which_scan}")

    def _subtract_scan_per_output(self, output: Output, trim_start: int = 0, trim_end: int = 0):
        axis, kind = self.which_scan.split("_")
        getfunc = getattr(output, "get_scan")

        scan_data = getfunc(axis=axis, kind=kind, corner=True) # full scan data including overlapping corners region
        axis_str = getattr(output, axis+"_axis")
        trim_slc = slice(trim_start, scan_data.sizes[axis_str] - trim_end)
        trimmed_scan_data = scan_data.isel({axis_str: trim_slc})

        BASICSTATS = {'mean': STATFUNCS['mean'], 'median': STATFUNCS['median'], 'std': STATFUNCS['std'],
                      'mad': STATFUNCS['mad']}
        trimmed_scan_stats = do_statistics(trimmed_scan_data.values, which=BASICSTATS,
                                           prepend_kw=f"HIERARCH {self.which_scan}_")

        methodkwargs = {'axis': getattr(output, axis+"_axint")} if self.method_name == "median_by_axis" else {}
        subtract_value = self.method(trimmed_scan_data.values, **methodkwargs)
        subtracted_scan = output.data.values - subtract_value

        output.set_data_in_parent(subtracted_scan)
        outhdr = getattr(output, "header", {})
        outhdr.update(trimmed_scan_stats)
        output.header = outhdr

        return output

    def _process_single_image(self, img: DetImage) -> DetImage:
        trim_start = self.meta.get("trim_start", 0)
        trim_end = self.meta.get("trim_end", 0)
        for out_id, output in img.outputs.items():
            img.outputs[out_id] = self._subtract_scan_per_output(output, trim_start, trim_end)

        img.image_type.update({f"{self.which_scan}_subtracted": True})
        return img

    @property
    def methods(self):
        """
        Return a dictionary of available methods for pre/overscan subtraction and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'simple_median': np.median,
            'median_by_axis': 'core.image_stats.median_by_axis',
            'simple_mean': np.mean,
        }


########### Cosmic Ray/Bad Pixel Masking ###########
class SigmaClipMasking(BasePreprocessingTask):
    task_result = ImageResult

    def __init__(self,
                 name: Optional[str] = "sigma_clip_masking",
                 sigma_clip_args: Optional[dict[str, Any]] = None,
                 clip_axis: Optional[Literal["serial", "parallel"]] = None,
                 **kwargs
    ):
        """
        Create a bad pixel mask/cosmic ray mask by sigma clipping. Clipping is performed per output.
        'simple' method applies clipping to entire output. 'ccd' method applies clipping separately to serial overscan,
        parallel overscan, and image region. Masks are saved as attributes of the output for later use.

        Currently only supports sigma clipping
        :param name: Optional[str]
        :param sigma_clip_args: Optional[dict[str, Any]]
            Arguments to pass to astropy.stats.sigma_clip function.
        :param clip_axis: Optional[Literal["serial", "parallel"]]
            Axis along which to perform clipping. Example, for image where parallel readout is along y-axis and serial
            readout along x-axis, to do the clipping per row set clip_axis="serial".
        :keyword method: Optional[str]
            Method to use for sigma clipping. Default is 'ccd' which applies sigma clipping per CCD output.
        """
        if 'method' not in kwargs:
            kwargs['method'] = 'ccd'  # ccd default for now
        super().__init__(name=name, **kwargs)
        self.sigma_clip_args = {"sigma": 5.0, "axis": None, "masked": True, "copy": True, "grow": 10.0}
        self.sigma_clip_args.update(sigma_clip_args or {})
        if clip_axis is not None and clip_axis in ["serial", "parallel"]:
            self.clip_axis = clip_axis+"_axint"
        else:
            self.clip_axis = None


    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage by applying sigma clipping to create a mask.
        :param img: DetImage
            The image to be processed.
        :return: DetImage
            The processed image with updated mask.
        """
        for out_id, output in img.outputs.items():
            img.outputs[out_id] = self.method(output)
        img.image_type.update({"bad_pixel_masked": True})
        return img

    @property
    def methods(self):
        """
        Return a dictionary of available methods for sigma clipping and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'simple': self._sigma_clip_per_output,  # simple sigma clipping on the entire output
            'ccd': self._sigma_clip_per_ccdoutput,  # ccd specific, clips overscans and image region separately
        }

    def _sigma_clip_per_ccdoutput(self, output: CCDOutput) -> CCDOutput:
        """
        Apply sigma clipping to a single output to create a mask. Masks are saved as attributes of the output for later use.
        """
        if self.clip_axis is not None and self.sigma_clip_args["axis"] is None:
            self.sigma_clip_args["axis"] = getattr(output, self.clip_axis)

        sigma_clip_args_overscan = deepcopy(self.sigma_clip_args)
        sigma_clip_args_overscan.pop("grow")

        # clip serial overscan
        soc_slcs = decrease_slicer_stop_index({output.serial_axis: output.serial_overscan})
        serial_overscan_data = output.get_overscan("serial", corner=True).values
        serial_overscan_clipped = sigma_clip_image(serial_overscan_data, **sigma_clip_args_overscan)

        # clip parallel overscan
        poc_slcs = decrease_slicer_stop_index({output.parallel_axis: output.parallel_overscan})
        parallel_overscan_data = output.get_overscan("parallel", corner=True).values
        parallel_overscan_clipped = sigma_clip_image(parallel_overscan_data, **sigma_clip_args_overscan)

        # clip image data region
        im_slcs = decrease_slicer_stop_index(output.image_region)
        image_data, _ = output.get_image_region()
        image_data_clipped = sigma_clip_image(image_data.values, **self.sigma_clip_args)

        # combine masks
        combined_mask = xr.zeros_like(output.data).astype(bool)
        combined_mask.sel(**im_slcs).values |= image_data_clipped.mask
        combined_mask.sel(**soc_slcs).values |= serial_overscan_clipped.mask
        combined_mask.sel(**poc_slcs).values |= parallel_overscan_clipped.mask

        if output.masks is None:
            output.masks = combined_mask.to_dataset(name="sigma_clip_mask")
        else:
            if "sigma_clip_mask" in output.masks:
                output.masks["sigma_clip_mask"] = (output.masks["sigma_clip_mask"] | combined_mask)
            else:
                output.masks["sigma_clip_mask"] = combined_mask
        return output

    def _sigma_clip_per_output(self, output: Output) -> Output:
        """
        Apply sigma clipping to a single output to create a mask. Masks are saved as attributes of the output for later use.
        """
        clipped = sigma_clip_image(output.data.values, **self.sigma_clip_args)
        mask = clipped.mask

        if output.masks is None:
            output.masks = mask.to_dataset(name="sigma_clip_mask")
        else:
            if "sigma_clip_mask" in output.masks:
                output.masks["sigma_clip_mask"] = (output.masks["sigma_clip_mask"] | mask)
            else:
                output.masks["sigma_clip_mask"] = mask
        return output


