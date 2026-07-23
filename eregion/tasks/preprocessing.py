from copy import deepcopy
from abc import abstractmethod
from typing import Optional, Iterable, Iterator, Any
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from tasks import LazyTask
from datamodels import DetImage, Output
from utils import ensure_dataarray, ensure_numpy, load_class
from core.image_operations import subtract_from_image, sigma_clip_image

########### BasePreprocessingTask ###########
class BasePreprocessingTask(LazyTask):
    """
    Base class for preproc tasks in this file to avoid repetitive code for lazy_run as most apply some processing
    to each image independently and can benefit from parallelization at the image level. Each subclass just needs to
    implement _process_single_image which defines how to process a single DetImage, and lazy_run will handle batching
    and parallel execution.
    """
    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage and return the processed DetImage.
        :param img: DetImage
            The image to be processed.
        :return: DetImage
            The processed image.
        """
        pass

    def lazy_run(
            self,
            images: Iterable[DetImage],
            batch_size: int = 1
    ):
        """
        Lazy execution: yield processed batches as they complete.
        - images: an iterable/stream of DetImage
        - batch_size: number of images to process per batch
        """

        def process_batch():
            output = Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)
            outdict = {}
            for out in output:
                outdict[out.image_type] = outdict.get(out.image_type, []) + [out]
            yield outdict

        batch: list[DetImage] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                yield from process_batch()
                batch = []
        if batch:
            yield from process_batch()


########### Bias Subtraction Task ###########
class BiasSubtraction(BasePreprocessingTask):
    required_keys = []

    def __init__(self,
                 master_biases: list[DetImage],
                 name: Optional[str] = "subtract_bias",
                 **kwargs
    ):
        """
        Initialize the BiasSubtraction task.
        :param master_biases: list[DetImage]
                The master bias frames to subtract from science images. It can be a list of DetImage objects in case of mosaics
        :param name: Optional[str]
        """
        super().__init__(name=name, **kwargs)

        if not isinstance(master_biases, list) or not all(isinstance(img, DetImage) for img in master_biases):
            raise ValueError("master_biases must be a list of DetImage instances.")
        # Verify image_type of master_biases is 'master_bias'
        if not all(img.image_type.lower()=='master_bias' for img in master_biases):
            raise ValueError("All master_biases must have image_type 'master_bias'.")
        self.master_biases = master_biases

    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage by subtracting the corresponding master bias.
        :param img: DetImage
            The science image to be bias-subtracted.
        :return: DetImage
            The bias-subtracted science image.
        """
        # Load the appropriate master bias for this image based on metadata (e.g., detector name)
        master_bias = None
        for mb in self.master_biases:
            if 'name' not in mb.meta.keys() or 'name' not in img.meta.keys():
                raise ValueError(f"Either master bias or input DetImage is missing 'name' in metadata. Cannot match master bias to image.")
            if mb.meta.name == img.meta.name:
                master_bias = mb
                break
        if master_bias is None:
            raise ValueError(f"No matching master bias found for DetImage with name '{img.meta['name']}'.")

        # initialize a new DetImage for the bias-subtracted result as a copy of the input image
        bias_subtracted = deepcopy(img)
        # Update its data, meta
        bias_subtracted.data = ensure_dataarray(self.subtract(img.data, master_bias.data))
        bias_subtracted.meta["bias_subtracted"] = True
        return bias_subtracted

    def lazy_run(self, images: Iterable[DetImage], batch_size: int = 1) -> Iterator[dict[str, list[DetImage]]]:
        """
        Inherit the lazy_run from BasePreprocessingTask which will handle batching and parallel execution of _process_single_image.
        :param images: Iterable of DetImage objects to be processed.
        :param batch_size: Number of images to process in each batch. Default is 1 (process images one at a time).
        :return: An iterator that yields dictionaries with image_type as keys and lists of processed DetImage objects as values.
        """
        return super().lazy_run(images, batch_size)

    @staticmethod
    def subtract(image: np.ndarray | xr.DataArray, master_bias: np.ndarray | xr.DataArray) -> np.ndarray:
        """
        Convenience non-Prefect path for raw arrays.
        """
        if master_bias.shape != image.shape:
            raise ValueError(
                f"Master bias array shape {master_bias.shape} does not match image shape {image.shape}.")
        bias_subtracted, _ = subtract_from_image(ensure_numpy(image), ensure_numpy(master_bias))
        return bias_subtracted

    def __call__(self, image, master_bias):
        return self.subtract(image, master_bias)


########### Scan Subtraction Task ###########
class ScanSubtraction(BasePreprocessingTask):
    def __init__(self,
                 which_scan: str,
                 name: Optional[str] = None,
                 **kwargs
    ):
        """
        Initialize the ScanSubtraction task
        :param which_scan: str, required
                One of 'serial_prescan', 'serial_overscan', 'parallel_prescan', 'parallel_overscan'.
        :param name: Optional[str]
        :param kwargs:
            method: str, optional
                Method to use for subtraction. Default is 'simple_median'.
            skip_rows: int, optional
                Number of rows to skip from the start of the scan region. Default is 0.
            skip_cols: int, optional
                Number of columns to skip from the start of the scan region. Default is 0.
        """
        super().__init__(name=name, **kwargs)
        self.which_scan = which_scan.lower()
        if self.which_scan not in ["serial_prescan", "serial_overscan", "parallel_prescan", "parallel_overscan"]:
            raise ValueError(f"Invalid which_scan value: {self.which_scan}")

        if self.method is None:
            self.logger.warning("No method specified for scan subtraction. Defaulting to 'simple_median'.")
            self.set_method("simple_median")

    def _subtract_scan_per_output(self, output: Output, skip_rows: int = 0, skip_cols: int = 0):
        axis, scan = self.which_scan.split("_")
        getfunc = getattr(output, "get_" + scan)

        scan_data = getfunc(kind=axis) # xr.DataArray
        trimmed_scan_data = scan_data.isel({'y': slice(skip_rows, None, None), 'x': slice(skip_cols, None, None)})

        medaxis = getattr(output, axis+"_axis")
        axisint = 0 if medaxis == 'y' else 1

        subtracted_scan, subtract_value = subtract_from_image(
            image=output.data.values,
            subtract_object=trimmed_scan_data.values,
            method=load_class(self.method),
            axis=axisint
        )
        return subtracted_scan, subtract_value

    def _process_single_image(self, image: DetImage) -> DetImage:
        """
        Process a single DetImage by subtracting the specified scan from each output.
        :param image: DetImage
            The image to be processed.
        :return: DetImage
            The processed image with scans subtracted.
        """
        skip_rows = self.meta.get("skip_rows", 0)
        skip_cols = self.meta.get("skip_cols", 0)
        outputs = image.outputs.values()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._subtract_scan_per_output)(output, skip_rows, skip_cols) for output in outputs
        )

        new_image = deepcopy(image)
        for output, (subtracted_scan, subtract_value) in zip(new_image.outputs.values(), results):
            output.set_data_in_parent(subtracted_scan)
            setattr(output, f"{self.which_scan}_median", subtract_value)

        new_image.meta[f"{self.which_scan}_subtracted"] = True
        return new_image

    def lazy_run(self, images: Iterable[DetImage], batch_size: int = 1) -> Iterator[dict[str, list[DetImage]]]:
        """
        Inherit the lazy_run from BasePreprocessingTask which will handle batching and parallel execution of _process_single_image.
        :param images: Iterable of DetImage objects to be processed.
        :param batch_size: Number of images to process in each batch. Default is 1 (process images one at a time).
        :return: An iterator that yields dictionaries with image_type as keys and lists of processed DetImage objects as values.
        """
        return super().lazy_run(images, batch_size)

    @property
    def methods(self):
        """
        Return a dictionary of available methods for creating master bias and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'simple_median': 'core.image_operations.simple_median',
            'median_by_axis': 'core.image_operations.median_by_axis',
            'simple_mean': 'core.image_operations.simple_mean',
        }


########### Cosmic Ray/Bad Pixel Masking ###########
class SigmaClipMasking(BasePreprocessingTask):
    required_keys = []

    def __init__(self,
                 name: Optional[str] = "sigma_clip_masking",
                 sigma_clip_args: Optional[dict[str, Any]] = None,
                 **kwargs
    ):
        """
        Initialize the SigmaClipMasking task.
        :param name: Optional[str]
        :param sigma_clip_args: Optional[dict[str, Any]]
            Arguments to pass to astropy.stats.sigma_clip function.
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.sigma_clip_args = sigma_clip_args or {"sigma": 5.0, "axis": None, "masked": True, "copy": True, "grow": 10.0}

    def _sigma_clip_per_output(self, output: Output) -> Output:
        """
        Apply sigma clipping to a single output to create a mask. Masks are saved as attributes of the output for later use.
        :param output: Output
            The output to be processed.
        :return: Output
            The output with an added mask attribute for sigma clipping.
        """
        sigma_clip_args_overscan = deepcopy(self.sigma_clip_args)
        sigma_clip_args_overscan.pop("grow")

        # clip serial overscan
        serial_overscan_data = output.get_overscan(kind="serial").values
        serial_overscan_clipped = sigma_clip_image(serial_overscan_data, **sigma_clip_args_overscan)

        # clip parallel overscan
        parallel_overscan_data = output.get_overscan(kind="parallel").values
        parallel_overscan_clipped = sigma_clip_image(parallel_overscan_data, **sigma_clip_args_overscan)

        # clip image data region
        im_slc_parallel = slice(output.parallel_prescan.stop, output.parallel_overscan.start)
        im_slc_serial = slice(output.serial_prescan.stop, output.serial_overscan.start)
        im_slcs = {output.parallel_axis: im_slc_parallel, output.serial_axis: im_slc_serial}
        image_data = output.data.isel(**im_slcs).values
        image_data_clipped = sigma_clip_image(image_data, **self.sigma_clip_args)

        # combine masks
        combined_mask = xr.zeros_like(output.data).astype(bool)
        combined_mask.isel(**im_slcs).values |= image_data_clipped.mask
        combined_mask.isel(**{output.serial_axis: output.serial_overscan}).values |= serial_overscan_clipped.mask
        combined_mask.isel(**{output.parallel_axis: output.parallel_overscan}).values |= parallel_overscan_clipped.mask

        # Set mask as an attribute of output
        if not hasattr(output, 'masks'):
            output.masks = {}
        output.masks['sigma_clip_mask'] = combined_mask
        return output

    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage by applying sigma clipping to create a mask.
        :param img: DetImage
            The image to be processed.
        :return: DetImage
            The processed image with updated mask.
        """
        new_image = deepcopy(img)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._sigma_clip_per_output)(output) for output in new_image.outputs.values()
        )
        for new_output in results:
            new_image.add_output(new_output, overwrite=True)

        new_image.meta["bad_pixel_masked"] = True
        return new_image

    def lazy_run(self, images: Iterable[DetImage], batch_size: int = 1) -> Iterator[dict[str, list[DetImage]]]:
        """
        Inherit the lazy_run from BasePreprocessingTask which will handle batching and parallel execution of _process_single_image.
        :param images: Iterable of DetImage objects to be processed.
        :param batch_size: Number of images to process in each batch. Default is 1 (process images one at a time).
        :return: An iterator that yields dictionaries with image_type as keys and lists of processed DetImage objects as values.
        """
        return super().lazy_run(images, batch_size)





