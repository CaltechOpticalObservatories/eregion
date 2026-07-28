import os
import glob2
from typing import Any
from astropy.io import fits
import shutil
import numpy as np
import pandas as pd
import uncertainties as unc
import pint

from .misc_utils import configure_logger
logger = configure_logger(__name__)

def search_directory_for_fits_files(directory: str) -> list[str]:
    fits_files = sorted(glob2.glob(os.path.join(directory, '**/*.fits*'), recursive=True))
    logger.info(f"Found {len(fits_files)} FITS files in directory {directory} and its sub-directories.")
    return fits_files


def is_fits_file(path: str) -> bool:
    if os.path.exists(path):
        return True if (os.path.isfile(path) and '.fits' in path) else False
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def is_archive_file(path: str) -> bool:
    if os.path.exists(path):
        return True if (os.path.isfile(path)
                        and ('.zip' in path or '.tar' in path or '.gz' in path or '.bz2' in path or '.xz' in path)
                        and not is_fits_file(path)) else False
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def is_directory(path: str) -> bool:
    if os.path.exists(path):
        return True if os.path.isdir(path) else False
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

######################### Saving and loading PTC table with FITS ######################################

def save_ptc_table_fits(table: pd.DataFrame, filepath: str) -> None:
    """
    Save the PTC stats table to a FITS binary table.

    Array-valued cells are stored as fixed-shape vector columns so they round-trip
    as homogeneous NumPy arrays per cell.
    """
    columns = []
    for name in table.columns:
        columns.append(_dataframe_column_to_fits_column(name, table[name]))

    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)])
    hdul.writeto(filepath, overwrite=True)

def load_ptc_table_fits(filepath: str) -> pd.DataFrame:
    """
    Load a PTC stats table previously written by ``save_ptc_table_fits``.
    """
    with fits.open(filepath, memmap=False) as hdul:
        data = hdul[1].data
        rows = {}
        for name in data.names:
            column = data[name]
            if column.ndim == 1:
                values = column.tolist()
                rows[name] = [value.decode("utf-8").rstrip() if isinstance(value, (bytes, np.bytes_)) else value
                             for value in values]
            else:
                rows[name] = [np.array(value) for value in column]
    return pd.DataFrame(rows)


def _quantity_column_to_fits_column(name: str, series: pd.Series) -> list[fits.Column]:
    #already assume len >0, else how did anything else work??
    unit_the_first = series[0].units

    outcol = [_.to(unit_the_first).magnitude for _ in series]
    #TODO: proper translation between pint strings and FITS strings...ARGH!


    #if it's an uncertainties ufloat, split into two columns
    if isinstance(series[0], unc.UFloat):



    fmt = _fits_format_code(type(outcol[0]))

    #HACK: for now, just use the pint string
    return fits.Column(name=name, array=outcol, unit=str(unit_the_first))



def _dataframe_column_to_fits_column(name: str, series: pd.Series) -> fits.Column:
    values = list(series)
    non_null = [value for value in values if value is not None]
    if not non_null:
        return fits.Column(name=name, array=np.array(values, dtype="U1"), format="1A")

    if all(_is_array_cell(value) for value in non_null):
        arrays = [np.asarray(value) for value in values]
        _validate_homogeneous_array_column(name, arrays)
        stacked = np.stack(arrays)
        code = _fits_format_code(stacked.dtype)
        flat_size = int(np.prod(stacked.shape[1:])) if stacked.ndim > 1 else 1
        dim = None
        if stacked.ndim > 1:
            dim = "(" + ",".join(str(size) for size in stacked.shape[1:]) + ")"
        return fits.Column(name=name, array=stacked, format=f"{flat_size}{code}", dim=dim)

    if all(isinstance(value, (str, bytes, np.str_, np.bytes_)) for value in non_null):
        strings = ["" if value is None else str(value) for value in values]
        width = max(1, max(len(value) for value in strings))
        return fits.Column(name=name, array=np.array(strings, dtype=f"U{width}"), format=f"{width}A")

    if all(isinstance(value, (bool, np.bool_)) for value in non_null):
        return fits.Column(name=name, array=np.asarray(values, dtype=np.bool_), format="L")

    if all(isinstance(value, (int, np.integer, bool, np.bool_)) for value in non_null):
        return fits.Column(name=name, array=np.asarray(values, dtype=np.int64), format="K")

    if all(isinstance(value, (float, np.floating, int, np.integer, bool, np.bool_)) for value in non_null):
        return fits.Column(name=name, array=np.asarray(values, dtype=np.float64), format="D")

    if all(isinstance(value, (pint.Quantity)) for value in non_null):
        return _quantity_column_to_fits_column(name, non_null)

    raise TypeError(f"Unsupported PTC table column '{name}' with values of type {type(non_null[0]).__name__}.")

def _is_array_cell(value: Any) -> bool:
    return isinstance(value, (np.ndarray, list, tuple))

def _validate_homogeneous_array_column(name: str, arrays: list[np.ndarray]) -> None:
    first_shape = arrays[0].shape
    first_dtype = arrays[0].dtype
    for array in arrays[1:]:
        if array.shape != first_shape:
            raise ValueError(
                f"Column '{name}' contains arrays with different shapes: {first_shape} vs {array.shape}."
            )
        if array.dtype != first_dtype:
            raise ValueError(
                f"Column '{name}' contains arrays with different dtypes: {first_dtype} vs {array.dtype}."
            )

def _fits_format_code(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return "E" if dtype.itemsize <= 4 else "D"
    if np.issubdtype(dtype, np.signedinteger):
        if dtype.itemsize <= 2:
            return "I"
        if dtype.itemsize <= 4:
            return "J"
        return "K"
    if np.issubdtype(dtype, np.bool_):
        return "L"
    raise TypeError(f"Unsupported array dtype for FITS serialization: {dtype}.")
