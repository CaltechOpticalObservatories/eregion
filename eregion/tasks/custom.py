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
        f = os.path.basename(filename).split('DTU_DT-')[1].replace('bias_','bias_0.000_').split('_')
        imtype['type'] = f[2]
        imtype['exptime'] = float(f[3])
        imtype['seqnum'] = int(f[4])
        imtype['obstime'] = f[5].replace('.fits','')
    except:
        logger.warning("Could not guess image type for DEIMOS file %s", filename)
    return imtype

def process_raw_numbers(data: np.ndarray, offs) -> np.ndarray:
    bit16: int = (1<<16) - 1
    bit32to16: int = ( (1 <<32) -1) ^ bit16
    quotient = np.bitwise_and(data, bit32to16)
    quotient = np.right_shift(quotient, 16)
    remainder = np.bitwise_and(data, bit16)
    #NOTE: should this be divided by (2**16-1)
    fldat = quotient.astype(np.float64) + (remainder / 2**16)
    return fldat - offs

def load_image_fits_DEIMOS(filename: str) -> tuple[list[Any], list[fits.Header]]:
    input_data_array, input_headers = load_image_fits(filename)
    for i, hdr in enumerate(input_headers):
        if "TAPOFFS" in hdr:
            input_data_array[i] = process_raw_numbers(input_data_array[i], hdr["TAPOFFS"])
            logger.debug("Processed raw numbers for DEIMOS HDU %d", i)
        else:
            logger.debug("TAPOFFS not found in header for file %s HDU %d", filename, i)
    return input_data_array, input_headers