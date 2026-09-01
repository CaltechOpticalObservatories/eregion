### Collection of utility functions for image processing tasks.
from typing import Callable
import numpy as np
from astropy.stats import sigma_clip

def median_combine(images: list[np.ndarray]) -> np.ndarray:
    """
    Combine a list of images by computing the median across them.

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D numpy arrays representing images to be combined.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the median-combined image.
    """
    stacked_images = np.stack(images, axis=0)
    return np.median(stacked_images, axis=0)

def mean_combine(images: list[np.ndarray]) -> np.ndarray:
    """
    Combine a list of images by computing the mean across them.

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D numpy arrays representing images to be combined.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the mean-combined image.
    """
    stacked_images = np.stack(images, axis=0)
    return np.mean(stacked_images, axis=0)

def sigma_clip_image(image: np.ndarray | np.ma.MaskedArray, sigma: float, axis: int | None=None, **kwargs) -> np.ma.MaskedArray:
    """
    Apply sigma clipping (astropy.stats.sigma_clip) to an image.

    Parameters
    ----------
    image : np.ndarray or np.ma.MaskedArray
        2D numpy array representing the image.
    sigma : float
        The sigma threshold for clipping.
    axis : int or None
        Axis along which to perform the sigma clipping. If None, the entire array is treated as a single entity.
    kwargs : dict
        Additional keyword arguments to pass to astropy.stats.sigma_clip.

    Returns
    -------
    np.ma.MaskedArray
        The sigma-clipped image.
    """
    masked = sigma_clip(image, sigma=sigma, axis=axis, **kwargs)
    return masked

def flip_and_rotate(image: np.ndarray, angle: float, flip_x: bool=False, flip_y: bool=False) -> np.ndarray:
    """
    Flip and rotate an image. Rotation angle is assumed to be in degrees and positive for counter-clockwise direction,
    and has to be a multiple of 90.
    :param image: 2D numpy array
    :param angle: in degrees
    :param flip_x: True to flip left-right
    :param flip_y: True to flip up-down
    :return: flipped and rotated image
    """
    if image.ndim != 2:
        raise ValueError('Input image is not a 2D array.')
    if flip_y:
        image = np.flipud(image)
    if flip_x:
        image = np.fliplr(image)

    if angle:
        if angle % 90 != 0:
            raise ValueError('Angle must be a multiple of 90 degrees.')
        else:
            k = (angle // 90) % 4
            image = np.rot90(image, int(k))
    return image


def do_digital_binning(data: np.ndarray, binsizes: list[int], binaxis: int = 0) -> np.ndarray:
    """
    Perform digital binning on the provided data. Assumes that readout is towards the 0th index of the binning axis,
    i.e. the first row of the data is the first row read out from the CCD. If that's not true, pre-flip your data in
    the correct order.

    NOTE: Eregion's data loading (ImageCreator + DetectorConfig) slicing options can set the readout direction correctly.

    :param data: np.ndarray,
        The input data to be binned (2D image).
    :param binsizes: list[int]
        Number of rows to sum per binning iteration. Each bin size should be an integer.
    :param binaxis: int, optional
        The axis along which to perform the binning. Default is 0.
    :return: np.ndarray
        The digitally binned data.
    """
    assert np.sum(binsizes) == data.shape[binaxis], "Sum of binsizes must equal the size of the data along the binning axis."
    binaxis = int(binaxis)
    assert binaxis < data.ndim, "binaxis is out of bounds."
    binned_data = np.zeros_like(data)
    src_idx = [slice(None)] * data.ndim
    dst_idx = [slice(None)] * data.ndim
    ind = 0
    # compute start indices for each bin from binsizes
    starts = np.concatenate(([0], np.cumsum(binsizes)[:-1])).astype(int)
    # sum each bin along binaxis efficiently
    binned = np.add.reduceat(data, starts, axis=binaxis)
    # preserve original output buffer shape: write summed bins into the leading indices
    dst_idx[binaxis] = slice(0, len(binsizes))
    binned_data[tuple(dst_idx)] = binned
    return binned_data