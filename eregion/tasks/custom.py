### File for custom input functions/tasks defined by the user. ###
import os
import numpy as np
from typing import Any
from astropy.io import fits

from utils import load_image_fits, configure_logger
logger = configure_logger(__name__)

def guess_image_type_from_filename_DEIMOS(filename: str) -> dict[str, Any]:
    """
    Custom function to guess the image type for DEIMOS based on filename patterns.
    :param filename: str
        The name of the FITS file.
    :return: dict
        Dictionary containing guessed image type metadata.
    """
    imtype = {'type':'unknown', 'exptime':0.}
    try:
        f = os.path.basename(filename).split('DTU_DT-')[1].split('_')
        imtype['type'] = f[2]
        imtype['exptime'] = float(f[3]) if f[2]!='bias' else 0.0
    except:
        logger.warning("Could not guess image type for DEIMOS file %s", filename)
    return imtype

def load_image_fits_DEIMOS(filename: str) -> tuple[list[Any], list[fits.Header]]:
    input_data_array, input_headers = load_image_fits(filename)
    for i, hdr in enumerate(input_headers):
        if "TAPOFFS" in hdr:
            input_data_array[i] = (input_data_array[i].astype(np.int64) >> 12) - hdr["TAPOFFS"]
    return input_data_array, input_headers