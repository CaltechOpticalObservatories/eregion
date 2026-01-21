from copy import deepcopy

from typing import Optional, Iterable, Iterator, Any
import numpy as np
import xarray as xr

from tasks.task import LazyTask
from datamodels.image import DetImage, Output
from datamodels.image_utils import ensure_dataarray
from core.image_operations import subtract_from_image, simple_median, median_by_axis, simple_mean, sigma_clip_image

from joblib import Parallel, delayed

########### Bias Subtraction Task ###########
class BiasSubtraction(LazyTask):
    required_keys = []

    def __init__(self,
                 master_bias: DetImage | np.ndarray | xr.DataArray,
                 name: Optional[str] = "subtract_bias",
                 **kwargs
                 ):
        """
        Initialize the BiasSubtraction task. kwargs must include 'master_bias' which can be a DetImage or numpy array.
        :param master_bias: DetImage or np.ndarray or xr.DataArray
                The master bias frame to subtract from science images.
        :param name: Optional[str]
        """

        super().__init__(name=name, **kwargs)

        if isinstance(master_bias, DetImage):
            self.master_bias_data = master_bias.data
        elif isinstance(master_bias, np.ndarray) or isinstance(master_bias, xr.DataArray):
            self.master_bias_data = ensure_dataarray(master_bias)
        else:
            raise ValueError("master_bias must be either a DetImage, a numpy array, or an xarray DataArray.")
        self.meta.update({"master_bias": master_bias})


    def _process_single_image(self, img: DetImage) -> DetImage:
        """
        Process a single DetImage by subtracting the master bias.
        :param img: DetImage
            The science image to be bias-subtracted.
        :return: DetImage
            The bias-subtracted science image.
        """
        # check that dimensions match
        if img.data.shape != self.master_bias_data.shape:
            raise ValueError(f"Image shape {img.data.shape} does not match master bias shape {self.master_bias_data.shape}.")
        # initialize a new DetImage for the bias-subtracted result as a copy of the input image
        bias_subtracted = deepcopy(img)
        bias_subtracted.data = ensure_dataarray(img.data - self.master_bias_data)
        bias_subtracted.meta["bias_subtracted"] = True
        bias_subtracted.image_type = f'bias_sub_{img.image_type}'
        return bias_subtracted


    def lazy_run(
        self,
        images: Iterable[DetImage],
        batch_size: int = 1,
    ) -> Iterator[list[DetImage]]:
        """
        Lazy execution: yield processed batches as they complete.
        - images: an iterable/stream of DetImage
        - batch_size: number of images to process per batch
        """

        batch: list[DetImage] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)
                batch = []

        if batch:
            yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)


    def run(self, images: list[DetImage]) -> list[DetImage]:
        """
        Subtract the master bias from the given images.
        :param images: list of DetImage
            List of science images to be bias-subtracted.
        :return: list of DetImage
            List of bias-subtracted science images.
        """
        ## call lazy_run and collect all results
        results = []
        for batch in self.lazy_run(images, batch_size=len(images)):
            results.extend(batch)
        return results

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Convenience non-Prefect path for raw arrays.
        """
        return image - self.master_bias_data.values


########### Scan Subtraction Task ###########
class ScanSubtraction(LazyTask):
    required_keys = ["which_scan"]

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize the ScanSubtraction task. kwargs must include 'which_scan' indicating which scan to subtract.
        :param name: Optional[str]
        :param kwargs:
            which_scan: str, required
                One of 'serial_prescan', 'serial_overscan', 'parallel_prescan', 'parallel_overscan'.
            method: str, optional
                Method to use for subtraction. Default is 'simple_median'.
            skip_rows: int, optional
                Number of rows to skip from the start of the scan region. Default is 0.
            skip_cols: int, optional
                Number of columns to skip from the start of the scan region. Default is 0.
        """
        super().__init__(name=name, **kwargs)
        self.which_scan = kwargs["which_scan"]
        if self.which_scan not in ["serial_prescan", "serial_overscan", "parallel_prescan", "parallel_overscan"]:
            raise ValueError(f"Invalid which_scan value: {self.which_scan}")

        self.method = kwargs.get("method", "simple_median")
        if self.method not in self.methods:
            self.print_methods()
            raise NotImplementedError(f"Method {self.method} not implemented for ScanSubtraction.")
        self.method = globals()[self.method]


    def _subtract_scan_per_output(self, output: Output, skip_rows: int = 0, skip_cols: int = 0):
        axis, scan = self.which_scan.split("_")
        getfunc = getattr(output, "get_" + scan)

        scan_data = getfunc(kind=axis) # xr.DataArray
        trimmed_scan_data = scan_data.isel({'y': slice(skip_rows, None), 'x': slice(skip_cols, None)})

        axisint = 1 if axis == "serial" else 0

        subtracted_scan, subtract_value = subtract_from_image(
            image=output.data.values,
            subtract_object=trimmed_scan_data.values,
            method=self.method,
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
        new_image.image_type = f'scan_sub_{image.image_type}'
        return new_image


    def lazy_run(
        self,
        images: Iterable[DetImage],
        batch_size: int = 1,
    ) -> Iterator[list[DetImage]]:
        """
        Lazy execution: yield processed batches as they complete.
        - images: an iterable/stream of DetImage
        - batch_size: number of images to process per batch
        """
        batch: list[DetImage] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)
                batch = []

        if batch:
            yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)


    def run(self, images: list[DetImage]) -> list[DetImage]:
        """
        Subtract the specified scan from the given images.
        :param images: list of DetImage
            List of images to be processed.
        :return: list of DetImage
            List of processed images with scans subtracted.
        """
        ## call lazy_run and collect all results
        results = []
        for batch in self.lazy_run(images, batch_size=len(images)):
            results.extend(batch)
        return results

    @property
    def methods(self):
        """
        Return a dictionary of available methods for creating master bias and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {
            'simple_median': 'core.image_operations.simple_median(data: np.ndarray) -> float[Any]',
            'median_by_axis': 'core.image_operations.median_by_axis(data: np.ndarray, axis: int) -> np.ndarray',
            'simple_mean': 'core.image_operations.simple_mean(data: np.ndarray) -> float[Any]',
        }

########### Cosmic Ray/Bad Pixel Masking ###########
class SigmaClipMasking(LazyTask):
    required_keys = []

    def __init__(self, name: Optional[str] = "sigma_clip_masking", sigma_clip_args: dict[str, Any] = None, **kwargs):
        """
        Initialize the SigmaClipMasking task.
        :param name: Optional[str]
        :param sigma_clip_args: Dict[str, Any]
            Arguments to pass to astropy.stats.sigma_clip function.
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.sigma_clip_args = sigma_clip_args or {"sigma": 5.0, "axis": None, "masked": True, "copy": True, "grow": 10.0}


    def _sigma_clip_per_output(self, output: Output) -> xr.DataArray:
        """
        Apply sigma clipping to a single output to create a mask.
        :param output: Output
            The output to be processed.
        :return: xr.DataArray
            The mask created from sigma clipping.
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
        im_yslc = slice(None, output.parallel_overscan.start)
        im_xslc = slice(None, output.serial_overscan.start)
        image_data = output.data.isel(y=im_yslc, x=im_xslc).values
        image_data_clipped = sigma_clip_image(image_data, **self.sigma_clip_args)
        # combine masks
        combined_clipped = np.ma.MaskedArray(output.data.values, mask=np.zeros_like(output.data.values))
        combined_clipped.mask[im_yslc, im_xslc] = image_data_clipped.mask
        combined_clipped.mask[output.serial_overscan, :] |= serial_overscan_clipped.mask
        combined_clipped.mask[:, output.parallel_overscan] |= parallel_overscan_clipped.mask
        return xr.DataArray(combined_clipped, dims=["y", "x"])


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
        for output, clipped_data in zip(new_image.outputs.values(), results):
            output.set_data_in_parent(clipped_data)
        new_image.meta["bad_pixel_masked"] = True
        new_image.image_type = f'bpm_{img.image_type}'
        return new_image


    def lazy_run(
        self,
        images: Iterable[DetImage],
        batch_size: int = 1,
    ) -> Iterator[list[DetImage]]:
        """
        Lazy execution: yield processed batches as they complete.
        - images: an iterable/stream of DetImage
        - batch_size: number of images to process per batch
        """
        batch: list[DetImage] = []
        for img in images:
            batch.append(img)
            if len(batch) >= batch_size:
                yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)
                batch = []

        if batch:
            yield Parallel(n_jobs=self.n_jobs)(delayed(self._process_single_image)(i) for i in batch)


    def run(self, images: list[DetImage]) -> list[DetImage]:
        """
        Apply sigma clipping to create bad pixel masks for the given images.
        :param images: list of DetImage
            List of images to be processed.
        :return: list of DetImage
            List of processed images with updated masks.
        """
        ## call lazy_run and collect all results
        results = []
        for batch in self.lazy_run(images, batch_size=len(images)):
            results.extend(batch)
        return results





