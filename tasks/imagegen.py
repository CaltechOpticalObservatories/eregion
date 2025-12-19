from datamodels.image import *
from datamodels.image_utils import ensure_dataarray
from datamodels.detector_config import DetectorConfig
from tasks.task import Task, LazyTask

import logging
import os, glob2, time
from astropy.io import fits
from typing import Iterator, Generator, Callable, Optional, Iterable, Union

from prefect import task, flow
from prefect.futures import wait

logger = logging.getLogger(__name__)


def load_image_fits(filename: str) -> list[Any]:
    """
    Load FITS file and return the data as a numpy array with hdu extensions as the first axis.
    :param filename: str
        Path to the FITS file.
    :return: np.ndarray
        Numpy array with shape (n_hdus, y_size, x_size).
    """
    input_data_array = []
    with fits.open(filename) as hdulist:
        for hdu in hdulist:
            input_data_array.append(hdu.data)
    return input_data_array

## Classes to handle image generation from configuration files
class ImageCreator(LazyTask):
    def __init__(self, detector_config=None, name='image_creator', watch_mode=False, poll_interval=10, **kwargs):
        super().__init__(name, watch_mode, poll_interval, **kwargs)

        # TODO: if no detector_config is provided, try to generate from a FITS file later
        if detector_config is not None:
            self.set_detector_config(detector_config)
        else:
            raise ValueError("Detector configuration must be provided.")

        # Store a Prefect task for identification; wrap callables on registration.
        self._identifier_task: Optional[Callable[..., str]] = None

        # Store processed filenames
        self._seen_files: set[str] = set()

    def set_detector_config(self, detector_config):
        """
        Set or update the detector configuration.
        :param detector_config: str or dict
            Path to a YAML config file or config data as a string or dictionary.
        """
        self.config = DetectorConfig(config_input=detector_config).config

    def set_identifier(self, func: Callable[..., str]) -> None:
        """
        Register a custom image-type identification function.
        Accepts only a plain callable to be wrapped as a Prefect task.
        Expected signature: func(filename: str, **kwargs) -> str
        """
        if not callable(func):
            raise TypeError("identifier must be callable.")
        self._identifier_task = task(func, name="custom_identifier")

    @task
    def _guess_image_type_from_header(self, filename, hdu=0, keywords=None):
        """
        Default image type guessing logic based on FITS header.
        Parameters
        ----------
        filename : str
            Path to the FITS file.
        hdu : int, optional
            HDU index to read the header from. Default is 0.
        keywords : list of str, optional
            List of header keywords to check for image type.
        """

        hdulist = fits.open(filename)
        header = hdulist[hdu].header

        # Check given keywords first
        if keywords:
            for key in keywords:
                if key in header:
                    return header[key].lower()

        # Check common header keywords for image type
        if 'IMAGETYP' in header:
            return header['IMAGETYP'].lower()
        elif 'OBSTYPE' in header:
            return header['OBSTYPE'].lower()
        elif 'OBJECT' in header:
            obj_name = header['OBJECT'].lower()
            if 'bias' in obj_name:
                return 'bias'
            elif 'flat' in obj_name:
                return 'flat'
            elif 'dark' in obj_name:
                return 'dark'
            elif 'science' in obj_name or 'object' in obj_name:
                return 'science'
        return 'unknown'  # Default assumption

    # ----------------- Input sources -----------------
    def from_files(self, input_path: str | list[str]) -> Iterator[List[str]]:
        """
        Yields batches of file paths.
        - If watch_mode=False: yields a single batch with all discovered files.
        - If watch_mode=True: watches a directory/glob and yields newly discovered files.
        """
        def _discover_once() -> List[str]:
            if isinstance(input_path, list):
                filenames = [f for f in input_path if os.path.isfile(f) and '.fits' in f]
            elif os.path.isdir(input_path):
                filenames = glob2.glob(os.path.join(input_path, '*.fits*'))
            elif '*' in input_path:
                filenames = [f for f in glob2.glob(input_path) if os.path.isfile(f) and '.fits' in f]
            elif os.path.isfile(input_path) and '.fits' in input_path:
                filenames = [input_path]
            else:
                raise ValueError(
                    "Input path must be a FITS file, a list of FITS files, "
                    "a directory with FITS files, or a glob pattern for FITS files.")
            return filenames

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

    def from_arrays(self, input_arrays: Iterable[np.ndarray]) -> Iterator[List[np.ndarray]]:
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
    @flow
    def _build_image_objects(self, input_data_array, image_type, filename=None):
        images = []
        for obj in self.config['objects']:
            image = self._build_single_image_object.submit(obj, input_data_array, image_type, filename=filename)
            images.append(image)
        wait(images)
        images = [img.result() for img in images if img.result() is not None]
        return images

    @task
    def _build_single_image_object(self, obj, input_data_array, image_type, filename=None):
        # If a filename is given, check if it matches the filename format
        if filename is not None:
            if not glob2.fnmatch.fnmatch(os.path.basename(filename), obj['filename_format']):
                logger.info("Skipping file %s as it does not match filename format for this object (%s)",
                            filename, obj['filename_format'])
                return None

        # pop outputs and class from object
        outputs = obj.pop('outputs')
        ImageClass = globals()[obj.pop('class')]
        OutputClass = globals()[self.config['detector_output_class']]

        # instantiate image object
        image = ImageClass(image_type=image_type, **obj, filename=filename)
        image_data_size = [0] * image.ndim
        for output in outputs:
            output_obj = OutputClass(**output)
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
                 input_source: Union[str, list[str], Iterable[np.ndarray]],
                 identifier_func: Optional[Callable[..., str]] = None,
                 identifier_kwargs: Optional[Dict[str, Any]] = None
    ) -> Generator[List, None, None]:
        """
        Main lazy run method to generate image objects from input source.
        :param input_source: str or list of str or Iterable of np.ndarray
            Input source can be a path to FITS files (file, directory, glob pattern),
            a list of FITS file paths, or an iterable of numpy arrays.
        :param identifier_func: Callable, optional
            Custom image type identification function.
        :param identifier_kwargs: dict, optional
            Additional keyword arguments for the identifier function.
        :return: Generator yielding lists of DetImage objects.
        """
        identifier_kwargs = identifier_kwargs or {}
        if identifier_func is not None:
            self.set_identifier(identifier_func)

        if isinstance(input_source, (str, list)):
            file_batches = self.from_files(input_source)
            for file_batch in file_batches:
                images_batch = []
                for filename in file_batch:
                    input_data_array = load_image_fits(filename)
                    # Determine image type
                    if self._identifier_task is not None:
                        image_type = self._identifier_task(filename=filename, **identifier_kwargs)
                    else:
                        image_type = self._guess_image_type_from_header(filename=filename, **identifier_kwargs)
                    images = self._build_image_objects(input_data_array, image_type, filename=filename)
                    images_batch.extend(images)
                yield images_batch
        else:
            array_batches = self.from_arrays(input_source)
            for array_batch in array_batches:
                images_batch = []
                for input_data_array in array_batch:
                    if self._identifier_task is not None:
                        image_type = self._identifier_task()
                    else:
                        image_type = 'unknown'
                    images = self._build_image_objects(input_data_array, image_type, filename=None)
                    images_batch.extend(images)
                yield images_batch

    def run(self,
            input_source: Union[str, list[str], Iterable[np.ndarray]],
            identifier_func: Optional[Callable[..., str]] = None,
            identifier_kwargs: Optional[Dict[str, Any]] = None
    ) -> List:
        """
        Eager run method to generate image objects from input source.
        :param input_source: str or list of str or Iterable of np.ndarray
            Input source can be a path to FITS files (file, directory, glob pattern),
            a list of FITS file paths, or an iterable of numpy arrays.
        :param identifier_func: Callable, optional
            Custom image type identification function.
        :param identifier_kwargs: dict, optional
            Additional keyword arguments for the identifier function.
        :return: List of DetImage objects.
        """
        all_images = []
        for images_batch in self.lazy_run(input_source, identifier_func, identifier_kwargs):
            all_images.extend(images_batch)
        return all_images