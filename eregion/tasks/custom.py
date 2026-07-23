### File for custom input functions/tasks defined by the user. ###
import os
import numpy as np
from typing import Any
from astropy.io import fits

from utils import load_image_fits

def guess_image_type_from_filename_DEIMOS(filename: str) -> str:
    """
    Custom function to guess the image type for DEIMOS based on filename patterns.
    :param filename: str
        The name of the FITS file.
    :return: str
        The guessed image type (e.g., 'bias', 'flat', 'science', etc.)
    """
    filename_lower = os.path.basename(filename).lower()
    if 'bias' in filename_lower:
        return 'bias'
    elif 'flat' in filename_lower:
        return 'flat'
    elif 'arc' in filename_lower or 'lamp' in filename_lower:
        return 'arc'
    elif 'science' in filename_lower or 'obj' in filename_lower:
        return 'science'
    else:
        return 'unknown'

def load_image_fits_DEIMOS(filename: str) -> tuple[list[Any], list[fits.Header]]:
    input_data_array, input_headers = load_image_fits(filename)
    for i, hdr in enumerate(input_headers):
        if "TAPOFFS" in hdr:
            input_data_array[i] = (input_data_array[i].astype(np.int64) >> 12) - hdr["TAPOFFS"]
    return input_data_array, input_headers