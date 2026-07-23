import os
import glob2
from typing import Any
from astropy.io import fits
import shutil

from utils.misc_utils import configure_logger
logger = configure_logger(__name__)

def search_directory_for_fits_files(directory: str) -> list[str]:
    fits_files = sorted(glob2.glob(os.path.join(directory, '**/*.fits*'), recursive=True))
    logger.info(f"Found {len(fits_files)} FITS files in directory {directory} and its sub-directories.")
    return fits_files


def is_fits_file(path: str) -> bool:
    if os.path.exists(path):
        if os.path.isfile(path) and '.fits' in path:
            return True
        else:
            return False
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def is_archive_file(path: str) -> bool:
    if os.path.exists(path):
        if os.path.isfile(path) and (
            '.zip' in path or '.tar' in path or '.gz' in path or '.bz2' in path or '.xz' in path) and not is_fits_file(path):
            return True
        else:
            return False
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def is_directory(path: str) -> bool:
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        else:
            return False
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def parse_list_of_files(items: list[str]) -> list[str]:
    new_items = []
    for item in items:
        # if item is a compressed archive, unpack it and search for fits files within
        if is_archive_file(item):
            logger.info(f"Found archive file {item}, unpacking and searching for FITS files within.")
            # unpack archive to parent directory and search for fits files within
            extraction_dir = os.path.join(os.path.dirname(item), str(os.path.basename(item).split('.')[0]))
            shutil.unpack_archive(item, extraction_dir)
            new_items.extend(search_directory_for_fits_files(extraction_dir))
        elif is_fits_file(item):
            logger.debug(f"Found FITS file {item}.")
            new_items.append(item)
        elif is_directory(item):
            new_items.extend(search_directory_for_fits_files(item))
        else:
            logger.warning(f"Unrecognized item: {item}, not a FITS file, archive or directory, skipping.")
    return sorted(new_items)


def load_image_fits(filename: str | None, **kwargs) -> tuple[list[Any], list[fits.Header]]:
    """
    Load FITS file and return the data as a list with hdu extensions as the first axis.
    :param filename: str | None
        Path to the FITS file.
    :return: Tuple[List[Any], List[fits.Header]]
        A tuple containing a list of data arrays (image/table/...) for each HDU and a list of corresponding FITS headers.
    """
    if filename:
        if ".fits.fz" in filename:
            logger.info(f"Loading compressed FITS file {filename} using fits.open() with in-memory decompression enabled.")
        input_data_array, input_headers = [], []
        try:
            with fits.open(filename, decompress_in_memory=True) as hdulist:
                for hdu in hdulist:
                    input_data_array.append(hdu.data)
                    input_headers.append(hdu.header)
        except Exception as e:
            logger.error(f"Error loading FITS file {filename}: {e}")

        return input_data_array, input_headers
    return [], []

def guess_image_type_from_header(headers: list[fits.Header | dict], keywords=None) -> dict[str, Any]:
    """
    Default image type guessing logic based on FITS header.
    Parameters
    ----------
    headers: list[fits.Header | dict]
        List of FITS headers to check for image type.
    keywords : dict[str, list[str]] optional
        Dictionary specifying keywords to check for different image identifiers
    """
    imtype = {'type': 'unknown', 'exptime': None}
    for header in headers:
        # Check given keywords first
        if keywords is None:
            keywords = {'type':['IMAGETYP', 'OBSTYPE', 'OBJECT'], 'exptime': ['EXPTIME', 'EXPOSURE']}
        for key, hkeys in keywords.items():
            for hkey in hkeys:
                if hkey in header:
                    imtype[key] = header[hkey].lower()
                    break
    return imtype