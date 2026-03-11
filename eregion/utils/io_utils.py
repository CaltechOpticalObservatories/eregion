import os
import glob2
from typing import Any
from astropy.io import fits
import shutil

from utils.misc_utils import configure_logger
logger = configure_logger(__name__)

def search_directory_for_fits_files(directory: str) -> list[str]:
    fits_files = glob2.glob(os.path.join(directory, '**/*.fits*'), recursive=True)
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
            logger.info(f"Found FITS file {item}.")
            new_items.append(item)
        elif is_directory(item):
            new_items.extend(search_directory_for_fits_files(item))
        else:
            logger.info(f"Unrecognized item: {item}, not a FITS file, archive or directory, skipping.")
    return sorted(new_items)


def load_image_fits(filename: str) -> tuple[list[Any], list[fits.Header]]:
    """
    Load FITS file and return the data as a list with hdu extensions as the first axis.
    :param filename: str
        Path to the FITS file.
    :return: Tuple[List[Any], List[fits.Header]]
        A tuple containing a list of data arrays (image/table/...) for each HDU and a list of corresponding FITS headers.
    """
    if ".fits.fz" in filename:
        logger.info(f"Loading compressed FITS file {filename} using fits.open() with in-memory decompression enabled.")
    input_data_array, input_headers = [], []
    try:
        with fits.open(filename, decompress_in_memory=True) as hdulist:
            for hdu in hdulist:
                input_data_array.append(hdu.data)
                input_headers.append(hdu.header)
    except Exception as e:
        logger.info(f"Error loading FITS file {filename}: {e}")

    return input_data_array, input_headers


def guess_image_type_from_header(header, keywords=None):
    """
    Default image type guessing logic based on FITS header.
    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to inspect for image type.
    keywords : list of str, optional
        List of header keywords to check for image type.
    """
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