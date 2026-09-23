import numpy as np
from typing import Callable, Any
from scipy.signal.windows import boxcar, get_window
from scipy.signal import welch

def split_trim_array_squares(array: np.ndarray, target_size: int) -> list[np.ndarray]:
    """
    Trim an array centrally and split it into equal square sub-arrays.

    The function removes edge pixels as needed so each dimension becomes an
    integer multiple of ``target_size``. Trimming is symmetric (centered) and
    the trimmed array is then split along each axis into non-overlapping
    ``target_size x target_size`` blocks.

    :param array: Input 2D (or N-D) NumPy array to trim and split.
    :param target_size: Side length (in pixels) for each square block.
    :return: List of equally sized sub-arrays extracted from the trimmed input.
    """
    sections = tuple(_ // target_size for _ in array.shape)
    trims = tuple(_ % target_size if _ % target_size else None for _ in array.shape)

    trimage_slc = tuple(slice(_//2, -_//2) if _ is not None else slice(None, None) for _ in trims)

    trimage = array[trimage_slc]

    splitarrs = [trimage]
    for ind, sec in enumerate(sections):
        splitarrs_new = []
        for arr in splitarrs:
            splitarrs_new.extend(np.split(arr, sec, axis=ind))
        splitarrs = splitarrs_new

    return splitarrs


def welch2d(array: np.ndarray,
            target_shape: int,
            subtract_dc: bool = False,
            window_func: Callable[...,Any] = boxcar,
            averaging_func: Callable[..., np.ndarray] = np.nanmean,
            stdev_func: Callable[..., np.ndarray] = np.nanstd) -> tuple[np.ndarray, ...]:
    """
    Estimate a 2D power spectral density using Welch-style block averaging.

    The input image is centrally trimmed so each axis is divisible by ``target_shape`` and split into non-overlapping
    square blocks. For each block, it is windowed with ``wdw_func(target_shape)``, optionally mean-subtracted, and its
    2D FFT is computed. The power spectral density (PSD) is calculated as the squared magnitude of the FFT, normalized
    by the variance of the block. Finally, the PSDs from all blocks are averaged using ``averaging_func`` and the
    standard error is estimated using ``stdev_func`` divided by the square root of the number of blocks.

    :param array: Input NumPy array to analyze (typically 2D image data).
    :param target_shape: Side length of each square block in pixels.
    :param subtract_dc: If ``True``, subtract the mean from each windowed block before FFT.
    :param window_func: Window generator callable that accepts ``target_shape`` and returns a 1D window; used to build
                        a separable 2D window via outer product.
    :param averaging_func: Reduction function used to average block PSDs along the block axis (default ``numpy.mean``).
    :param stdev_func: Reduction function used to compute the spread of block PSDs along the block axis
                        (default ``numpy.std``).
    :return: Tuple ``(avpsds, errpsds)`` where ``avpsds`` is the averaged PSD and ``errpsds`` is the standard error
            estimate ``stdev_func(outpsds, axis=0) / sqrt(n_blocks)``.
    """
    wdwnd = np.outer(*(window_func(target_shape) for i in range(len(array.shape))))
    splits = split_trim_array_squares(array, target_shape)

    outpsds = []
    for subarr in splits:
        prepim = wdwnd * subarr
        if subtract_dc:
            prepim -= np.mean(prepim)
        spec = np.fft.fft2(prepim)
        psd = np.abs(spec * spec.conjugate())
        outpsds.append(psd / np.var(subarr))

    outpsds = np.array(outpsds)
    avpsds = averaging_func(outpsds, axis=0)
    errpsds = stdev_func(outpsds, axis=0) / np.sqrt(len(outpsds))

    return avpsds, errpsds


def spatial_to_temporal_freq(fx, fy, hz_x, hz_y):
    """
    Convert spatial frequency coordinates from PSD to temporal frequency.

    :param fx: Spatial frequency along the x-axis (in cycles per pixel).
    :param fy: Spatial frequency along the y-axis (in cycles per pixel).
    :param hz_x: Temporal frequency corresponding to the x-axis (in Hz), e.g. pixel readout rate.
    :param hz_y: Temporal frequency corresponding to the y-axis (in Hz),  e.g. 1/(line time).
    :return: Apparent temporal frequency (Hx)
    """
    return np.abs(fx * hz_x + fy * hz_y)


def raveled_welch(array: np.ndarray,
                  target_shape: int,
                  window_func: str | Callable[..., Any] = 'boxcar',
                  **kwargs) -> tuple[np.ndarray, ...]:
    """
    Ravel a 2D array into a 1D array and compute the PSD using scipy.signal.welch.
    :param array: Input 2D array to analyze (typically image data).
    :param target_shape: Segment length for Welch's method (in pixels), passes to `nperseg` kwarg.
    :param window_func: string or callable, window function to use for Welch's method. If a string, it should be a
                        valid window name from `scipy.signal.get_window`. If a callable, it should accept the segment
                        length and return a 1D window array of that length. Default is 'boxcar'.
    :param kwargs: Additional kwargs to pass to `scipy.signal.welch`, such as `fs` (sampling frequency), `nfft`, etc.
    :return: Tuple `(f, Pxx)` where `f` is the array of sample frequencies and `Pxx` is the power spectral density of
            the raveled array.
    """
    # Ravel the 2D array into a 1D array
    raveled_array = array.ravel()

    # Determine the window function
    if isinstance(window_func, str):
        window = get_window(window_func, target_shape)
    elif callable(window_func):
        window = window_func(target_shape)
    else:
        raise ValueError("window_func must be a string or a callable function.")

    # Compute the PSD using scipy.signal.welch
    kwargs['noverlap'] = kwargs.get('noverlap', 0)  # set to 0 if not provided
    kwargs['return_onesided'] = kwargs.get('return_onesided', True)  # set to True if not provided
    f, Pxx = welch(raveled_array, window=window, nperseg=target_shape, **kwargs)

    return f, Pxx
