from functools import partial, wraps
import inspect
import os
import glob2
import time
import warnings
import numpy as np
from typing import Iterator, Generator, Callable, Iterable, Optional, Any
from pydantic import model_validator, ConfigDict
from joblib import Parallel, delayed
from itertools import batched

from eregion.utils import load_image_fits, parse_list_of_files, guess_image_type_from_header, load_class
from eregion.configs import DetectorConfig
from eregion.datamodels import TaskResult, ImageBundle, DetImage, FocalPlaneImage, FPImageBundle
from eregion.tasks import LazyTask, Task

##################### Class to handle image generation from configuration files ####################################
class ImageResult(TaskResult):
    """
    A dataclass to hold the results of an image generation task.

    Attributes
    ----------
    data: ImageBundle | FPImageBundle
        Bundle of generated images (DetImage or FocalPlaneImage) from the task.
    """
    data: ImageBundle | FPImageBundle
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def parse_result(cls, kwargs):
        payload_fields = cls.payload_field_names()
        for key, val in kwargs.items():
            if key not in payload_fields:
                continue
            if isinstance(val, list):
                if all(isinstance(x, DetImage) for x in val):
                    kwargs[key] = ImageBundle(val)
                elif all(isinstance(x, FocalPlaneImage) for x in val):
                    kwargs[key] = FPImageBundle(val)
                else:
                    raise ValueError(f"Invalid input. Value of {key} must be an ImageBundle, FPImageBundle, or a list "
                                     f"of DetImage or FocalPlaneImage objects.")
        return kwargs

    def save(self, filepath: str, **kwargs) -> None:
        for attr, value in self.payload_dict().items():
            if isinstance(value, ImageBundle):
                if len(value)==0:
                    warnings.warn(f"ImageBundle for attribute {attr} is empty. Not saving.")
                else:
                    value.save(os.path.join(filepath, f"{attr}"), **kwargs)
        super().save(filepath)

    @classmethod
    def load(cls, filepath: str):
        attrs = {}
        for attr in cls.payload_field_names():
            if isinstance(cls.model_fields[attr].annotation, type(ImageBundle | FPImageBundle)):
                attrs[attr] = ImageBundle.load(os.path.join(filepath, f"{attr}"))
        metadata = cls.load_metadata(filepath)
        return cls(**attrs, **metadata)

class ImageCreator(LazyTask):
    """
    Task to generate DetImage objects from input FITS files or in-memory arrays based on a detector configuration.
    """
    task_result = ImageResult

    def __init__(self, detector_config, name='image_creator',
                 watch_mode=False, poll_interval=0, max_batch_size=10, **kwargs):
        super().__init__(name, watch_mode, poll_interval, max_batch_size, **kwargs)

        # Load detector configuration
        self.det_config = DetectorConfig(config_input=detector_config)

        # Store a custom function for identification
        self._identifier_task: Optional[Callable] = guess_image_type_from_header

        # Store a custom function for image loading
        self._fileloader_task: Optional[Callable] = load_image_fits

        # Store processed filenames
        self._seen_files: set[str] = set()

    def set_detector_config(self, detector_config):
        """
        Set or update the detector configuration.
        :param detector_config: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        self.det_config = DetectorConfig(config_input=detector_config)

    def set_identifier(self, func: str | Callable, **kwargs) -> None:
        """
        Register a custom image-type identification function. It should return a dict with image identifiers as keys and
        their corresponding values, and must have either 'filename' or 'headers' as a required argument.

        Expected signature::

            func(headers: list[astropy.io.fits.Header | dict], **kwargs) -> dict
            func(filename: str, **kwargs) -> dict
        """
        if not callable(func):
            try:
                func = load_class(func)
            except Exception as e:
                raise ValueError(f"Error loading identifier function '{func}': {e}")
        # check if func has 'filename' or 'headers' as arguments
        sig = inspect.signature(func)
        if 'filename' not in sig.parameters.keys() and 'headers' not in sig.parameters.keys():
            raise ValueError("Identifier function must have either 'filename' or 'headers' as an argument")
        self._identifier_task = partial(func, **kwargs)

    def set_fileloader(self, func: str | Callable, **kwargs) -> None:
        """
        Register a custom FITS/file loading function. It should return a tuple of (data_list, header_list) and
        must have filename as a required argument.

        In case of loading data from memory instead of files, this custom function can be used to just supply headers,
        with filename as None.

        Expected signature::

            func(filename: str, **kwargs) -> tuple(list[Any], list[astropy.io.fits.Header | dict])
        """
        if not callable(func):
            try:
                func = load_class(func)
            except Exception as e:
                raise ValueError(f"Error loading FITS loader function '{func}': {e}")
        sig = inspect.signature(func)
        if 'filename' not in sig.parameters.keys():
            raise ValueError("File loader function must have 'filename' as an argument")
        self._fileloader_task = partial(func, **kwargs)


    # ----------------- Input sources -----------------
    def from_files(self, input_path: str | list[str]) -> Iterator[list[str]]:
        """
        Yields batches of file paths.
        - If watch_mode=False: yields a single batch with all discovered files.
        - If watch_mode=True: watches a directory/glob and yields newly discovered files.
        """
        def _discover_once() -> list[str]:
            if isinstance(input_path, list):
                self.logger.info("Provided input_path is a list, checking each item")
                list_to_process = []
                for ipath in input_path:
                    if '*' in ipath:
                        self.logger.info(f"Item {ipath} is a glob pattern")
                        list_to_process.extend(glob2.glob(ipath, recursive=True))
                    else:
                        self.logger.info(f"Item {ipath} is a regular path string")
                        list_to_process.append(ipath)
                return parse_list_of_files(list_to_process)

            elif isinstance(input_path, str):
                if '*' in input_path:
                    self.logger.info(f"Item {input_path} is a glob pattern")
                    return parse_list_of_files(glob2.glob(input_path, recursive=True))
                else:
                    self.logger.info(f"Item {input_path} is a regular path string")
                    return parse_list_of_files([input_path])

            else:
                self.logger.error(f"Invalid input")
                raise ValueError("input_path must be a string or list of strings representing file paths, directories, or glob patterns.")

        if not self.watch_mode:
            batch = _discover_once()
            yield batch
            return
        # Watch mode
        while True:
            batch = _discover_once()
            new_files = [f for f in batch if f not in self._seen_files]
            if new_files:
                self._seen_files.update(new_files)
                yield new_files
            time.sleep(self.poll_interval)

    def from_arrays(self, input_arrays: Iterable[np.ndarray]) -> Iterator[list[np.ndarray]]:
        """
        Yields batches of input data arrays.
        - If watch_mode=False: yields a single batch with all arrays collected
        - If watch_mode=True: yields arrays one by one as they are provided.
        """
        if not self.watch_mode:
            buffer = list(input_arrays)
            yield buffer
            return
        # Watch mode
        for array in input_arrays:
            yield [array]

    # ----------------- Image object builders -----------------
    def _build_image_objects(self, input_data, data_on_demand=False) -> list[Any]:
        """
        Internal method to parallely build DetImage objects from a given filename or in-memory array.
        :param input_data: str or array
        :param data_on_demand: If True, the actual data loading into memory is done when DetImage data is accessed,
                            instead of during image object creation, to save memory usage.
        :return: list of DetImage objects
        """
        if isinstance(input_data, str):
            input_data_array, input_headers = self._fileloader_task(filename=input_data)
            filename = input_data
            if data_on_demand:
                del input_data_array
                input_data_array = []
                self.logger.debug("Data will be loaded on demand for file %s", input_data)
            else:
                self.logger.debug("Loaded %d HDU from file %s", len(input_data_array), input_data)
        elif isinstance(input_data, np.ndarray):
            data_on_demand = False
            filename = None
            _, input_headers = self._fileloader_task(filename=filename)
            input_data_array = input_data
            self.logger.debug("Data provided as array input, got %d HDU worth of headers", len(input_headers))
        else:
            raise ValueError("Must provide either input filename or array")

        if (not data_on_demand) and (len(input_data_array) == 0):
            self.logger.error("No data found in file %s, skipping this input.")
            return []

        cfg = self.det_config.config
        ## use joblib to parallelize building image objects
        images = []
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_single_image_object)(
                obj, cfg['detector_output_class'], input_data_array, input_headers, filename
            )
            for obj in cfg['objects']
        )
        for img in results:
            if img is not None:
                images.append(img)

        # free up memory
        del input_data_array, input_headers

        return images

    def _build_single_image_object(self,
                                   obj,
                                   output_class,
                                   input_data_array,
                                   input_headers,
                                   filename=None):
        # If a filename is given, check if it matches the filename format
        if filename is not None:
            if not glob2.fnmatch.fnmatch(os.path.basename(filename), obj['filename_format']):
                self.logger.info("Skipping file %s as it does not match filename format for this object (%s)",
                            filename, obj['filename_format'])
                return None

        # Build outputs and image
        image_class = obj.pop('class')
        if "datamodels" not in output_class:
            output_class = "datamodels."+output_class
        if "datamodels" not in image_class:
            image_class = "datamodels."+image_class
        OutputClass = load_class(output_class)
        ImageClass = load_class(image_class)

        outputs = obj.pop('outputs')
        self.logger.info("Building object %s with %s %s outputs from file %s", image_class, len(outputs),
                         output_class, filename or 'array input')

        # instantiate image object
        image = ImageClass(**obj, filename=filename or 'none')

        image_data_size = [0] * image.ndim
        primary_hdr = input_headers[obj['header_index']] if len(input_headers) > 0 and 'header_index' in obj else {}
        image.meta.update(primary_hdr)
        headers = [primary_hdr]

        for output in outputs:
            output_obj = OutputClass(**output)
            if len(input_headers) > 0:
                output_obj.header = input_headers[output_obj.input_array_axis]
                headers.append(output_obj.header)
            # Determine full image size
            for i in range(len(image_data_size)):
                image_data_size[i] = max(image_data_size[i], output_obj.output_slice[i].stop)
            image.add_output(output_obj)

        # verify that calculated image size is consistent with set size in obj properties (if given)
        if 'properties' in obj and 'x_size' in obj['properties'] and 'y_size' in obj['properties']:
            if image_data_size != [obj['properties']['y_size'], obj['properties']['x_size']]:
                self.logger.error(f"Calculated image size {image_data_size} does not match specified size in config"
                                  f" {obj['properties']['y_size'], obj['properties']['x_size']} for {obj['name']}")
                raise ValueError("Calculated image size does not match specified size in config")
        # add image size to meta
        image.meta['shape'] = tuple(image_data_size)

        # Determine image meta (type, exptime, etc.) using identifier task
        sig = inspect.signature(self._identifier_task)
        args = {}
        if 'filename' in sig.parameters.keys():
            args['filename'] = filename
        if 'headers' in sig.parameters.keys():
            args['headers'] = headers
        imtype = self._identifier_task(**args)
        image.meta.update({'image_type': imtype})
        image.image_type = imtype
        self.logger.debug("Identified image type as %s", imtype)

        # Assemble full image data from outputs if not data_on_demand
        if len(input_data_array) > 0:
            image_data = np.zeros(image_data_size)
            for output_id, output_obj in image.outputs.items():
                image_data[*output_obj.output_slice] = (
                    input_data_array)[output_obj.input_array_axis][*output_obj.input_slice]
        else:
            image_data = self._fileloader_task
        image.set_data(image_data)

        return image

    # ----------------- Main lazy run method -----------------
    def lazy_run(self,
                 input_source: str | list[str] | Iterable[np.ndarray],
                 identifier_func: Optional[str | Callable] = None,
                 identifier_kwargs: Optional[dict[str, Any]] = None,
                 fileloader_func: Optional[str | Callable] = None,
                 fileloader_kwargs: Optional[dict[str, Any]] = None,
                 data_on_demand: bool = False,
                 require_data: bool = True,
                 **kwargs
    ) -> Generator[ImageResult, None, None]:
        """
        :param input_source: str or list of str or Iterable of np.ndarray
            Input source can be a path to FITS files (file, directory, glob pattern),
            a list of FITS file paths, or an iterable of numpy arrays.
        :param identifier_func: str (from pipeline config) or Callable (if using directly), optional
            Custom image type identification function.
        :param identifier_kwargs: dict, optional
            Additional keyword arguments for the identifier function.
        :param fileloader_func: str (from pipeline config) or Callable (if using directly), optional
            Custom FITS loading function.
        :param fileloader_kwargs: dict, optional
            Additional keyword arguments for the FITS loader function.
        :param data_on_demand: bool
            If True, the actual data loading from FITS is done when DetImage objects are being used, to save memory usage.
        :param require_data: bool
            If True, raises an error if no files are found in the input source.
        :return: {"images": list of DetImage}
            Generator yielding lists of DetImage objects, stored under the key 'images' in the yielded dict.
        """
        if self.watch_mode:
            self.logger.info("Running in watch mode, will monitor input source for new data, setting require_data=False")
            require_data = False

        identifier_kwargs = identifier_kwargs or {}
        if identifier_func is not None:
            self.set_identifier(identifier_func, **identifier_kwargs)

        fileloader_kwargs = fileloader_kwargs or {}
        if fileloader_func is not None:
            self.set_fileloader(fileloader_func, **fileloader_kwargs)

        _file_input = False
        if isinstance(input_source, (str, list[str])):
            input_batches = self.from_files(input_source)
            _file_input = True
        elif isinstance(input_source, Iterable):
            input_batches = self.from_arrays(input_source)
        else:
            raise TypeError("Input source must be either a path to FITS files or an iterable of numpy arrays")

        for input_batch in input_batches:
            if len(input_batch) == 0:
                self.logger.warn("Empty input source. Skipping.")
                if require_data:
                    if _file_input:
                        raise FileNotFoundError("No files found in the input source.")
                    else:
                        raise ValueError("Empty iterable input source.")
                continue

            ## split batch into smaller batches of max_batch_size
            for batch in batched(input_batch, self.max_batch_size):
                images_batch = []
                for inp in batch:
                    if _file_input:
                        self.logger.debug("Processing file %s", inp)
                    # Build image objects
                    images = self._build_image_objects(inp, data_on_demand=data_on_demand)
                    images_batch.extend(images)
                yield self.task_result(data=images_batch)

    @wraps(lazy_run)
    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)


class AssembleFocalPlane(Task):
    """
    From an ImageBundle or path to ImageBundle, load the bundle and identify images that belong to the same exposure
    by matching observation time, and assemble them into FocalPlaneImage objects.
    The assembled objects are returned as an ImageBundle wrapped in ImageResult.
    """
    task_result = ImageResult

    def __init__(self,
                 num_detectors: int,
                 dim: tuple[float, ...] = None,
                 name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.bundle = None
        self.num_detectors = num_detectors
        self.FPClass = partial(FocalPlaneImage, num_detectors=num_detectors, dim=dim)

    def run(self,
            from_path: str = None,
            from_images: ImageBundle | list[DetImage] = None,
            groupby_keys: list[str] = None,
            **kwargs)-> ImageResult:

        if from_path is not None:
            self.bundle = ImageBundle.load(from_path)
        elif from_images is not None:
            self.bundle = from_images if isinstance(from_images, ImageBundle) else ImageBundle(from_images)
        else:
            raise ValueError("Must provide either from_path or from_images to assemble focal plane images.")

        fp_images = []
        for unique_keys, group in self.bundle.groupby(by=groupby_keys):
            if not (len(group) <= self.num_detectors and group['det_id'].is_unique):
                self.logger.warning(f"Group with groupby column values {unique_keys} has incorrect number of detectors."
                                    f"Check that the grouping columns are correct to uniquely identify the images."
                                    f"Skipping this group.")
                continue
            imbundle = ImageBundle.from_dataframe(group)
            fp_images.append(self.FPClass(det_images=imbundle))
        return self.task_result(data=FPImageBundle(images=fp_images))
