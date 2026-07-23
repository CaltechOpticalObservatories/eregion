### Collection of utility functions for image processing tasks.
import numpy as np
from typing import Callable, Any, Optional
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

def subtract_from_image(image: np.ndarray, subtract_object: np.ndarray | float, method: Optional[Callable]=None, **kwargs):
    """
    Subtract a given object (array or scalar) from an image.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the image.
    subtract_object : np.ndarray or float
        The object to subtract from the image. Can be an array of any size from which the object to subtract is derived.
    method : Optional[Callable]
        A function that takes the subtract_object and returns a scalar/array to subtract from the image.
    kwargs :
        Additional keyword arguments to pass to the method function.
    Returns
    -------
    np.ndarray
        The resulting image after subtraction.
    """
    if method:
        subtract_object = method(subtract_object, **kwargs)
    return image - subtract_object, subtract_object

def simple_median(data: np.ndarray, *args) -> Any:
    return np.median(data)

def simple_mean(data: np.ndarray, *args) -> Any:
    return np.mean(data)

def median_by_axis(data: np.ndarray, axis: int, *args) -> np.ndarray:
    return np.median(data, axis=axis, keepdims=True)

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
    if angle % 90 != 0:
        raise ValueError('Angle must be a multiple of 90 degrees.')
    else:
        k = (angle // 90) % 4
        image = np.rot90(image, int(k))
    return image