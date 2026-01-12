### Collection of utility functions for image processing tasks.
import numpy as np
from typing import Union, Callable, Any
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

def subtract_from_image(image: np.ndarray, subtract_object: Union[np.ndarray, float], method: Callable, *args):
    """
    Subtract a given object (array or scalar) from an image.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the image.
    subtract_object : np.ndarray or float
        The object to subtract from the image. Can be an array of any size from which the value to subtract is derived.
    method : Callable
        A function that takes the subtract_object and returns a scalar/array to subtract from the image.
    Returns
    -------
    np.ndarray
        The resulting image after subtraction.
    """
    value_to_subtract = method(subtract_object, *args)
    return image - value_to_subtract, value_to_subtract

def simple_median(data: np.ndarray, *args) -> Any:
    return np.median(data)

def simple_mean(data: np.ndarray, *args) -> Any:
    return np.mean(data)

def median_by_axis(data: np.ndarray, axis: int, *args) -> np.ndarray:
    return np.median(data, axis=axis, keepdims=True)

def sigma_clip_image(image: Union[np.ndarray, np.ma.MaskedArray], sigma: float, axis: Union[int, None]=None, **kwargs) -> np.ma.MaskedArray:
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