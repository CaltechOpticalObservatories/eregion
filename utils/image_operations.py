### Collection of utility functions for image processing tasks.
import numpy as np

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