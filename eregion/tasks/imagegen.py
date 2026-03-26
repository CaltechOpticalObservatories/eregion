import os
import glob2
import time
import importlib
from typing import Iterator, Generator, Callable, Iterable
from joblib import Parallel, delayed

from datamodels.image import *
from utils.image_utils import ensure_dataarray
from configs.config import DetectorConfig
from tasks.task import LazyTask
from utils.io_utils import load_image_fits, parse_list_of_files, guess_image_type_from_header


## Classes to handle image generation from configuration files
class ImageCreator(LazyTask):
    def __init__(self, detector_config, name='image_creator', watch_mode=False, poll_interval=10, **kwargs):
        super().__init__(name, watch_mode, poll_interval, **kwargs)

        # Load detector configuration
        self.det_config = DetectorConfig(config_input=detector_config)

        # Store a custom function for identification
        self._identifier_task: Optional[Callable[..., str]] = None

        # Store a custom function for image loading
        self._fitsloader_task: Optional[Callable[..., np.ndarray]] = None

        # Store processed filenames
        self._seen_files: set[str] = set()

    def set_detector_config(self, detector_config):
        """
        Set or update the detector configuration.
        :param detector_config: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        self.det_config = DetectorConfig(config_input=detector_config)

    def set_identifier(self, func: str | Callable[..., str]) -> None:
        """
        Register a custom image-type identification function.
        Expected signature: func(filename: str, **kwargs) -> str
        """
        if not callable(func):
            try:
                module, cls = func.rsplit('.', 1)
                func = getattr(importlib.import_module(module), cls)
            except Exception as e:
                raise ValueError(f"Error loading identifier function '{func}': {e}")
        self._identifier_task = func

    def set_fitsloader(self, func: str | Callable[..., str]) -> None:
        """
        Register a custom FITS loading function. It should return a tuple of
        (data_list, header_list).
        Must have filename as a required argument.
        Expected signature: func(filename: str, **kwargs) -> Tuple(List[Any], List[astropy.io.fits.Header])
        """
        if not callable(func):
            try:
                module, cls = func.rsplit('.', 1)
                func = getattr(importlib.import_module(module), cls)
            except Exception as e:
                raise ValueError(f"Error loading FITS loader function '{func}': {e}")
        self._fitsloader_task = func


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
    def _build_image_objects(self, input_data_array, input_headers, image_type, filename=None):
        cfg = self.det_config.config
        ## use joblib to parallelize building image objects
        images = []
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_single_image_object)(
                obj, cfg['detector_output_class'], input_data_array, input_headers, image_type, filename
            )
            for obj in cfg['objects']
        )
        for img in results:
            if img is not None:
                images.append(img)
        return images

    def _build_single_image_object(self,
                                   obj,
                                   output_class,
                                   input_data_array,
                                   input_headers,
                                   image_type,
                                   filename=None):
        # If a filename is given, check if it matches the filename format
        if filename is not None:
            if not glob2.fnmatch.fnmatch(os.path.basename(filename), obj['filename_format']):
                self.logger.info("Skipping file %s as it does not match filename format for this object (%s)",
                            filename, obj['filename_format'])
                return None


        # pop outputs and class from object
        outputs = obj.pop('outputs')
        self.logger.info("Building object %s with %s %s outputs and type %s from file %s", obj['class'], len(outputs),
                         output_class, image_type, filename or 'array input')
        ImageClass = globals()[obj.pop('class')]
        OutputClass = globals()[output_class]

        # instantiate image object
        image = ImageClass(image_type=image_type, **obj, filename=filename or 'none')
        image_data_size = [0] * image.ndim
        for output in outputs:
            output_obj = OutputClass(**output)
            output_obj.fits_header = input_headers[output_obj.input_array_axis]
            # Determine full image size
            image_data_size[0] = max(image_data_size[0], output_obj.output_slice[0].stop)
            image_data_size[1] = max(image_data_size[1], output_obj.output_slice[1].stop)
            image.add_output(output_obj)

        # Assemble full image data from outputs
        image_data = np.zeros(image_data_size)
        for output_id in image.outputs:
            output_obj = image.outputs[output_id]
            image_data[*output_obj.output_slice] = (
                input_data_array)[output_obj.input_array_axis][*output_obj.input_slice]
        image.data = ensure_dataarray(image_data)
        return image

    # ----------------- Main lazy run method -----------------
    def lazy_run(self,
                 input_source: str | list[str] | Iterable[np.ndarray],
                 identifier_func: Optional[Callable[..., str]] = None,
                 identifier_kwargs: Optional[dict[str, Any]] = None,
                 fitsloader_func: Optional[Callable[..., str]] = None,
                 fitsloader_kwargs: Optional[dict[str, Any]] = None,
                 require_data: bool = True,
    ) -> Generator[dict[str, list], None, None]:
        """
        Main lazy run method to generate image objects from input source.
        :param input_source: str or list of str or Iterable of np.ndarray
            Input source can be a path to FITS files (file, directory, glob pattern),
            a list of FITS file paths, or an iterable of numpy arrays.
        :param identifier_func: str (from pipeline config) or Callable (if using directly), optional
            Custom image type identification function.
        :param identifier_kwargs: dict, optional
            Additional keyword arguments for the identifier function.
        :param fitsloader_func: str (from pipeline config) or Callable (if using directly), optional
            Custom FITS loading function.
        :param fitsloader_kwargs: dict, optional
            Additional keyword arguments for the FITS loader function.
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
            self.set_identifier(identifier_func)

        fitsloader_kwargs = fitsloader_kwargs or {}
        if fitsloader_func is not None:
            self.set_fitsloader(fitsloader_func)

        if isinstance(input_source, (str, list[str])):
            file_batches = self.from_files(input_source)
            for file_batch in file_batches:
                ## if file_batch is empty, print a warning
                if len(file_batch) == 0:
                    self.logger.warn("Empty input source. Skipping.")
                    if require_data:
                        raise FileNotFoundError("No FITS files found in the input source.")
                    continue

                images_batch = {}
                for filename in file_batch:
                    self.logger.info("Processing file %s", filename)
                    # Load FITS data
                    if self._fitsloader_task is not None:
                        input_data_array, input_headers = self._fitsloader_task(filename=filename, **fitsloader_kwargs)
                    else:
                        input_data_array, input_headers = load_image_fits(filename)
                    self.logger.debug("Loaded %d HDU from file %s", len(input_data_array), filename)
                    if len(input_data_array) == 0:
                        self.logger.warn("No data found in file %s, skipping.", filename)
                        continue

                    # Determine image type
                    if self._identifier_task is not None:
                        image_type = self._identifier_task(filename=filename, **identifier_kwargs)
                    else:
                        image_type = guess_image_type_from_header(input_headers[0], **identifier_kwargs)
                    self.logger.debug("Identified image type as %s for file %s", image_type, filename)

                    # Build image objects
                    images = self._build_image_objects(input_data_array, input_headers, image_type, filename=filename)
                    images_batch[image_type] = images_batch.get(image_type, []) + images
                yield images_batch

        else:
            array_batches = self.from_arrays(input_source)
            for array_batch in array_batches:
                ## if array_batch is empty, print a warning
                if len(array_batch) == 0:
                    self.logger.warn("Empty input source. Skipping.")
                    if require_data:
                        raise ValueError("Empty input source.")
                    continue

                images_batch = {}
                for input_data_array in array_batch:
                    if self._identifier_task is not None:
                        image_type = self._identifier_task()
                    else:
                        image_type = 'unknown'
                    images = self._build_image_objects(input_data_array, image_type, filename=None)
                    images_batch[image_type] = images_batch.get(image_type, []) + images
                yield images_batch