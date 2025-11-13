from datamodels.image import *
from datamodels.detector_config import DetectorConfig
from tasks.task import IOTask
import logging
import os, glob2, time
from astropy.io import fits
from typing import Iterator, Generator


logger = logging.getLogger(__name__)

## Classes to handle image generation from configuration files
class ImageCreator(IOTask):
    def __init__(self, input_path: str | list[str], detector_config=None, watch_mode=False, poll_interval=10, **kwargs):
        super().__init__(input_path=input_path, **kwargs)

        self.watch_mode = watch_mode
        self.poll_interval = poll_interval
        self.filenames = set()

        if os.path.isfile(input_path) and '.fits' in input_path:
            self.input_dir = os.path.dirname(input_path)
            self.filenames.update([input_path]) if not self.watch_mode else None
        elif isinstance(input_path, list) and all([os.path.isfile(f) and '.fits' in f for f in input_path]):
            self.input_dir = os.path.dirname(os.path.commonprefix(input_path))
            self.filenames.update(input_path) if not self.watch_mode else None
        elif os.path.isdir(input_path):
            self.input_dir = input_path
            self.filenames.update(glob2.glob(os.path.join(input_path, '*.fits*'))) if not self.watch_mode else None
        elif '*' in input_path:
            files = glob2.glob(input_path)
            files_to_add = []
            for file in files:
                if os.path.isfile(file) and '.fits' in file:
                    files_to_add.append(file)
            self.input_dir = os.path.dirname(os.path.commonprefix(files_to_add))
            self.filenames.update(files_to_add) if not self.watch_mode else None
        else:
            raise ValueError(
                "Input path must be a FITS file, a list of FITS files, a directory with FITS files, or a glob pattern for FITS files.")

        if detector_config is not None:
            self.config = DetectorConfig(config_path=detector_config).config
        else:
            if self.filenames:
                logger.info("Detector configuration file not provided, generating from FITS headers")
                self.config = DetectorConfig(fits_path=self.filenames[0], output_path=os.path.dirname(self.filenames[0])).config
            else:
                self.config = None

        # Initialize filelist DataFrame with dtype specified for each column
        self.filelist = pd.DataFrame(columns=["filename", "image_type", "processed", "images"], dtype=object)
        if self.filenames:
            imtypes = self.discovery(list(self.filenames))
            entries = pd.DataFrame({
                "filename": list(self.filenames),
                "image_type": imtypes,
                "processed": [False] * len(self.filenames),
                "images": [None] * len(self.filenames)
            })
            self.filelist = pd.concat([self.filelist, entries], ignore_index=True)

    def discovery(self, filenames: list[str]) -> list[str]:
        """
        Discover image types for specific files.
        Override this method for custom image type guessing logic.
        Returns a list of image types.
        """
        imtypes = []
        for filename in filenames:
            try:
                with fits.open(filename) as hdulist:
                    header = hdulist[0].header
                    imtypes.append(self._guess_image_type_from_header(header))
            except Exception as e:
                logger.warning(f"Could not determine image type for {filename}: {e}")
                imtypes.append('unknown')
        return imtypes

    def _guess_image_type_from_header(self, header):
        """
        Default image type guessing logic based on FITS header.
        Override discovery() method for custom logic.
        """
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

    def process_files(self, filelist: pd.DataFrame)-> pd.DataFrame:
        """
        Process files from the filelist and build image objects.
        """
        for idx, row in filelist.iterrows():
            if not row['processed']:
                images = self._build_image_objects(row['filename'], row['image_type'])
                filelist.at[idx, 'images'] = images
                filelist.at[idx, 'processed'] = True
        return filelist

    def _build_image_objects(self, filename, image_type):
        images = []

        with fits.open(filename) as hdulist:
            for obj in self.config['objects']:
                # Check if filename matches the filename format
                if not glob2.fnmatch.fnmatch(os.path.basename(filename), obj['filename_format']):
                    continue
                # Instantiate the image class
                ImageClass = globals()[obj['class']]

                # Check if obj has outputs defined
                if 'outputs' in obj:
                    full_data_size = [0, 0]
                    outputs = obj.pop('outputs')
                    image = ImageClass(image_type=image_type, **obj, filename=filename)
                    for output in outputs:
                        full_data_size[0] = max(full_data_size[0], output["ext_slice"][0].stop)
                        full_data_size[1] = max(full_data_size[1], output["ext_slice"][1].stop)
                        args = {k: output.pop(k) for k in ['id', 'ext_id', 'ext_slice', 'data_slice', 'serial_prescan',
                                                           'serial_overscan', 'parallel_prescan', 'parallel_overscan',
                                                           'parallel_axis', 'readout_pixel'] if k in output.keys()}
                        args['ext_slice'] = tuple(args['ext_slice'])
                        args['data_slice'] = tuple(args['data_slice'])
                        if 'readout_pixel' in output.keys():
                            args['readout_pixel'] = tuple(output['readout_pixel'])
                        output_obj = Output(filename=filename, **args, meta=output)
                        image.add_output(output_obj)
                    # Assemble full image data from outputs
                    image.data = np.zeros(full_data_size)
                    for output in image.outputs:
                        image.data[*output.data_slice] = hdulist[output.ext_id].data[*output.ext_slice]
                else:
                    # If no outputs defined, check if ext_id is defined for obj
                    if 'ext_id' in obj.keys():
                        ext_id = obj['ext_id']
                    else:
                        logger.info("No outputs defined individually, and no ext_id specified for object. Assuming ext_id=0.")
                        ext_id = 0
                    image = ImageClass(data=hdulist[ext_id].data, image_type=image_type, **obj)

                images.append(image)
        return images

    def watch_for_new_files(self) -> Generator[pd.DataFrame, None, None]:
        """
        Watch directory for new FITS files and yield discovered files as they appear.
        """
        while True:
            self.filenames.update(glob2.glob(os.path.join(self.input_dir, '*.fits')))
            new_files = self.filenames - set(self.filelist['filename'])

            if new_files:
                # Initialize config if not set
                if self.config is None and new_files:
                    first_file = next(iter(new_files))
                    logger.info("Detector configuration file not provided, generating from FITS headers")
                    self.config = DetectorConfig(fits_path=first_file, output_path=self.input_dir).config

                # Discover new files
                imtypes = self.discovery(list(new_files))
                filelist = pd.DataFrame({
                    "filename": list(new_files),
                    "image_type": imtypes,
                    "processed": [False] * len(new_files),
                    "images": [None] * len(new_files)
                })
                yield filelist

            time.sleep(self.poll_interval)

    def lazy_process_files(self) -> Iterator[pd.DataFrame]:
        """
        Lazily process files as they are discovered in watch mode.
        Yields the updated filelist after processing new files.
        """
        for new_filelist in self.watch_for_new_files():
            processed_filelist = self.process_files(new_filelist)
            self.filelist = pd.concat([self.filelist, processed_filelist], ignore_index=True)
            yield processed_filelist

    def run(self):
        if not self.watch_mode:
            return self.process_files(self.filelist)
        else:
            return self.lazy_process_files()

    def __call__(self):
        return self.run()
