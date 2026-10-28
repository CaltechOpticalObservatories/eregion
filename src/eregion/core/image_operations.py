### Collection of utility functions for image processing tasks.
from collections.abc import Sequence
import numpy as np
from astropy.stats import sigma_clip
from eregion.core.image_stats import do_statistics
from scipy import ndimage
from lacosmic import remove_cosmics


def median_combine(images: Sequence[np.ndarray | np.ma.MaskedArray]) -> dict[str, np.ndarray]:
    """
    Combine a list of images by computing the median across them. If a masked array is passed, median is computed
    only on the unmasked pixels.

    Parameters
    ----------
    images : list of np.ndarray or np.ma.MaskedArray
        List of 2D numpy arrays representing images to be combined.

    Returns
    -------
    dict
        A dictionary containing the median-combined image, standard deviation, and count of pixels used in the combination.
    """
    stacked_images = np.ma.stack(images, axis=0)
    stats = do_statistics(data=stacked_images, which={'data': np.ma.median, 'std': np.ma.std, 'count': np.ma.count},
                          axis=0)
    return stats

def mean_combine(images: Sequence[np.ndarray | np.ma.MaskedArray]) -> dict[str, np.ndarray]:
    """
    Combine a list of images by computing the mean across them. If a masked array is passed, mean is computed only on
    the unmasked pixels.

    Parameters
    ----------
    images : list of np.ndarray or np.ma.MaskedArray
        List of 2D numpy arrays representing images to be combined.

    Returns
    -------
    dict
        A dictionary containing the mean-combined image, standard deviation, and count of pixels used in the combination.
    """
    stacked_images = np.ma.stack(images, axis=0)
    stats = do_statistics(data=stacked_images, which={'data': np.ma.mean, 'std': np.ma.std, 'count': np.ma.count},
                          axis=0)
    return stats

def _check_same_shape(**arrays: np.ndarray) -> None:
    shapes = {name: np.shape(arr) for name, arr in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All arrays must have the same shape, got {shapes}")

def welford_update(count: np.ndarray, mean: np.ndarray, m2: np.ndarray,
                   image: np.ndarray, valid: np.ndarray) -> None:
    """
    Add one image to running per-pixel Welford accumulators. Arrays are updated in place to keep memory use flat
    when streaming many images.

    :param count: np.ndarray, number of valid samples per pixel so far
    :param mean: np.ndarray (float), running mean per pixel
    :param m2: np.ndarray (float), running sum of squared deviations from the mean per pixel
    :param image: np.ndarray, new image to add
    :param valid: np.ndarray (bool), True for pixels of image that should be added (i.e. not masked)
    :return: None
    """
    _check_same_shape(count=count, mean=mean, m2=m2, image=image, valid=valid)
    count += valid
    delta = np.where(valid, image - mean, 0.0)
    mean += np.divide(delta, count, out=np.zeros_like(mean), where=count > 0)
    m2 += delta * np.where(valid, image - mean, 0.0)

def welford_finalize(count: np.ndarray, mean: np.ndarray, m2: np.ndarray) -> dict[str, np.ndarray]:
    """
    Turn Welford accumulators into mean, sample standard deviation (ddof=1) and count images.
    Pixels with no samples get NaN mean, pixels with fewer than 2 samples get NaN std.

    :param count: np.ndarray, number of valid samples per pixel
    :param mean: np.ndarray, running mean per pixel
    :param m2: np.ndarray, running sum of squared deviations from the mean per pixel
    :return: dict with 'data' (mean), 'std' and 'count' arrays
    """
    _check_same_shape(count=count, mean=mean, m2=m2)
    data = np.where(count > 0, mean, np.nan)
    std = np.sqrt(np.divide(m2, count - 1, out=np.full_like(m2, np.nan, dtype=float), where=count > 1))
    return {'data': data, 'std': std, 'count': count.copy()}

def stochastic_median_update(median: np.ndarray, count: np.ndarray, std: np.ndarray,
                             image: np.ndarray, valid: np.ndarray) -> None:
    """
    Robbins-Monro streaming approximation of the per-pixel median, updated in place:
    median += sqrt(pi/2) * std / count * sign(image - median). sqrt(pi/2) is 1/f(median) for a Gaussian,
    which gives the optimal step size. Error falls like std/sqrt(count), the estimate is mildly sensitive
    to image order if the signal drifts over the series.

    :param median: np.ndarray (float), running median estimate per pixel
    :param count: np.ndarray, number of valid samples per pixel including the current image
    :param std: np.ndarray, running standard deviation per pixel including the current image
    :param image: np.ndarray, new image to add
    :param valid: np.ndarray (bool), True for pixels of image that should be added (i.e. not masked)
    :return: None
    """
    _check_same_shape(median=median, count=count, std=std, image=image, valid=valid)
    first = valid & (count == 1)
    median[first] = image[first]
    update = valid & (count > 1)
    step = np.sqrt(np.pi / 2) * np.divide(std, count, out=np.zeros_like(median), where=update)
    median += np.where(update, step * np.sign(image - median), 0.0)


def sigma_clip_image(image: np.ndarray | np.ma.MaskedArray,
                     sigma: float,
                     axis: int | None=None,
                     **kwargs):
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


def find_cosmic_rays(
        image: np.ndarray,
        contrast: float = 2.0,
        cr_threshold: float = 5.0,
        neighbor_threshold: float = 3.0,
        effective_gain: float | None = None,
        readnoise: float | None = None,
        error: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        maxiter: int = 4,
        border_mode: str = "mirror",
) -> np.ndarray:
    """
    Find cosmic rays in an image with the L.A.Cosmic algorithm (van Dokkum 2001) via lacosmic.remove_cosmics.
    Non-finite pixels, and pixels with non-finite or non-positive error, are excluded from detection and replaced by
    the finite median so they do not spread through the median filters.

    :param image: 2D numpy array, ideally zero-level corrected (bias/overscan subtracted).
    :param contrast: contrast threshold between the Laplacian and fine-structure images (~2 for critically sampled data).
    :param cr_threshold: Laplacian signal-to-noise threshold for cosmic ray detection.
    :param neighbor_threshold: Laplacian signal-to-noise threshold for pixels neighboring detected cosmic rays.
    :param effective_gain: ratio of electrons to image units (e-/ADU). Required with readnoise if error is not given.
    :param readnoise: read noise in electrons. Required with effective_gain if error is not given.
    :param error: optional 2D 1-sigma error array (e.g. a 'std' plane), same shape as image. Overrides effective_gain
        and readnoise.
    :param mask: optional 2D boolean array, same shape as image. True pixels are ignored and never flagged.
    :param maxiter: maximum number of detection iterations.
    :param border_mode: scipy.ndimage border mode for convolution and median filtering.
    :return: 2D boolean array, True where a cosmic ray was found.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got {image.ndim}D.")
    if error is None and (effective_gain is None or readnoise is None):
        raise ValueError("effective_gain and readnoise must both be given when error is not given.")
    if error is not None:
        error = np.asarray(error, dtype=float)
        if error.shape != image.shape:
            raise ValueError(f"error shape {error.shape} does not match image shape {image.shape}.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError(f"mask shape {mask.shape} does not match image shape {image.shape}.")
    for pname, pval in (("contrast", contrast), ("cr_threshold", cr_threshold),
                        ("neighbor_threshold", neighbor_threshold)):
        if pval <= 0:
            raise ValueError(f"{pname} must be > 0, got {pval}.")
    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}.")

    finite = np.isfinite(image)
    if error is not None:
        good_error = np.isfinite(error) & (error > 0)
        finite &= good_error
        if good_error.any():
            error = np.where(good_error, error, np.median(error[good_error]))
    if not finite.any():
        raise ValueError("image has no finite pixels with finite, positive error.")
    if not finite.all():
        image = np.where(finite, image, np.median(image[finite]))
        mask = ~finite if mask is None else (mask | ~finite)

    _, cr_mask = remove_cosmics(image, contrast, cr_threshold, neighbor_threshold, error=error, mask=mask,
                                effective_gain=effective_gain, readnoise=readnoise, maxiter=maxiter,
                                border_mode=border_mode)
    return cr_mask


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


def do_digital_binning(data: np.ndarray, binsizes: Sequence[int], binaxis: int = 0) -> np.ndarray:
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