### Collection of utility functions for image processing tasks.
from typing import Callable
import numpy as np
from astropy.stats import sigma_clip
from scipy import ndimage

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

def _disk_structure(radius: int) -> np.ndarray:
    """
    Boolean disk-shaped structuring element for binary morphology.
    :param radius: disk radius in pixels
    :return: 2D boolean array of shape (2*radius+1, 2*radius+1)
    """
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (yy ** 2 + xx ** 2) <= radius ** 2


def find_vignetted_region(
        image: np.ndarray,
        threshold: float = 0.8,
        smooth_size: int = 15,
        reference_percentile: float = 90.0,
        morph_radius: int = 5,
        min_area: int = 100,
        edge_connected: bool = True,
) -> np.ndarray:
    """
    Find vignetted (under-illuminated) regions in an illuminated, zero-level corrected (bias/overscan subtracted) image.
    Pixels whose median-smoothed value is below threshold * reference level are flagged, where the reference level is
    a high percentile of the image (the fully illuminated level). Mask borders are smoothed with binary opening, closing
    and hole filling using a disk structuring element.

    :param image: 2D numpy array of the illuminated image region (no prescan/overscan).
    :param threshold: fraction of the reference level below which a pixel is vignetted. Must be in (0, 1).
    :param smooth_size: median filter size used to suppress noise, cosmics and bad pixels before thresholding.
    :param reference_percentile: percentile of finite pixels used as the fully illuminated reference level.
    :param morph_radius: radius in pixels of the disk used for opening/closing. 0 disables opening/closing.
    :param min_area: connected regions with fewer pixels than this are dropped.
    :param edge_connected: if True, keep only regions touching the image border (vignetting comes in from the edges).
    :return: 2D boolean array, True where vignetted.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got {image.ndim}D.")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
    if not 0 < reference_percentile <= 100:
        raise ValueError(f"reference_percentile must be in (0, 100], got {reference_percentile}.")
    if smooth_size < 1:
        raise ValueError(f"smooth_size must be >= 1, got {smooth_size}.")
    if morph_radius < 0:
        raise ValueError(f"morph_radius must be >= 0, got {morph_radius}.")
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}.")

    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("image has no finite pixels.")
    reference = np.percentile(image[finite], reference_percentile)
    if reference <= 0:
        raise ValueError(f"Reference illumination level is {reference} (<= 0). "
                         f"Input must be an illuminated, bias/overscan subtracted image.")

    # non-finite pixels are set to the reference level so they are never flagged
    filled = np.where(finite, image, reference)
    smoothed = ndimage.median_filter(filled, size=smooth_size, mode="nearest")
    mask = smoothed < threshold * reference

    if morph_radius > 0:
        # pad by edge replication so regions touching the border are not eroded by the array boundary
        structure = _disk_structure(morph_radius)
        padded = np.pad(mask, morph_radius, mode="edge")
        padded = ndimage.binary_opening(padded, structure=structure)
        padded = ndimage.binary_closing(padded, structure=structure)
        mask = padded[morph_radius:-morph_radius, morph_radius:-morph_radius]
    mask = ndimage.binary_fill_holes(mask)

    labels, nlabels = ndimage.label(mask)
    if nlabels == 0:
        return mask
    areas = np.bincount(labels.ravel(), minlength=nlabels + 1)
    keep = areas >= min_area
    if edge_connected:
        on_edge = np.zeros(nlabels + 1, dtype=bool)
        on_edge[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = True
        keep &= on_edge
    keep[0] = False  # background label
    return keep[labels]


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
    if np.sum(binsizes) != data.shape[binaxis]:
        raise ValueError("Sum of binsizes must equal the size of the data along the binning axis.")

    binaxis = int(binaxis)
    if not 0 <= binaxis < data.ndim:
        raise ValueError("binaxis is out of bounds.")

    dst_idx = [slice(None)] * data.ndim
    # compute start indices for each bin from binsizes
    starts = np.concatenate(([0], np.cumsum(binsizes)[:-1])).astype(int)
    # sum each bin along binaxis efficiently
    binned = np.add.reduceat(data, starts, axis=binaxis)
    # preserve original output buffer shape: write summed bins into the leading indices
    dst_idx[binaxis] = slice(0, len(binsizes))
    binned_data = np.zeros(data.shape, dtype=np.promote_types(data.dtype, binned.dtype))
    binned_data[tuple(dst_idx)] = binned
    return binned_data