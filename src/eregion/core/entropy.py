"""
Shannon entropy of a histogram, and an entropy-based algorithm for choosing a
well-justified histogram bin width for continuous 1-D data.

Reference: S. J. Watts, L. Crow, "The Shannon Entropy of a Histogram",
https://arxiv.org/pdf/2210.02848
"""
import warnings
import numpy as np
import numpy.typing as npt
from typing import Any, Optional


def _as_float_array(data: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Convert input to a numpy array, casting to float64 only when the input
    isn't already some floating-point dtype (float16/32/64/...). This avoids
    an unnecessary full-array copy and the memory doubling that
    comes with it for already-floating input (e.g. float32 image data).
    """
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def shannon_entropy_histogram(counts: np.ndarray, base: float = 2) -> float:
    """
    Shannon entropy of a histogram from its bin counts (Watts & Crow, Eq. 2).

    :param counts: np.ndarray
        Number of entries per bin, n_i. Empty bins contribute zero entropy (the
        standard 0*log(0) = 0 convention)
    :param base: float
        Logarithm base for the entropy. base=2 (default) gives entropy in bits
    :return: float
        H = -sum(p_i * log(p_i)), where p_i = n_i / N
    """
    # Bin counts are naturally integers
    counts = np.asarray(counts)
    n = counts.sum()
    if n <= 0:
        raise ValueError("Histogram must contain at least one entry.")
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log(p)) / np.log(base))


def histogram_efficiency(entropy_bits: float, n_bins: int) -> float:
    """
    Histogram efficiency (Watts & Crow, Eq. 3): the ratio of the number of bins
    "expected" from the entropy, 2**H, to the number of bins actually used.
    e=1 for a uniform distribution; e<1 indicates over-binning (e.g. Poisson
    noise) or that the histogram carries less information than its bin count
    suggests

    :param entropy_bits: float
        Shannon entropy of the histogram, in bits (base-2)
    :param n_bins: int
        Number of bins used in the histogram
    :return: float
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    return float(2 ** entropy_bits / n_bins)


def differential_entropy_knn(data: npt.NDArray[np.floating], dither: bool = False,
                              rng: Optional[np.random.Generator] = None) -> float:
    """
    Nearest-neighbour (Kozachenko-Leonenko) estimator of the differential entropy
    of a 1-D continuous distribution, in bits (Watts & Crow, Eq. 11):

        h [bits] = gamma/ln(2) + (1/N) * sum(log2(2 * N * l_i))

    where l_i is the distance from point i to its nearest neighbour in the
    sample, and gamma is the Euler-Mascheroni constant

    :param data: np.ndarray[np.floating]
        1-D array of N samples
    :param dither: bool
        If True, add small uniform noise to break ties from quantized/discrete
        data before estimating
    :param rng: np.random.Generator, optional
        Random generator used for dithering. Defaults to np.random.default_rng()
    :return: float
        Estimated differential entropy, in bits
    """
    data = _as_float_array(data).ravel()
    n = data.size
    if n < 2:
        raise ValueError("Need at least 2 data points to estimate nearest-neighbour distances.")

    if dither:
        rng = rng or np.random.default_rng()
        gaps = np.diff(np.sort(data))
        nonzero_gaps = gaps[gaps > 0]
        # Use the median (not min) nonzero gap as the dither scale
        scale = np.median(nonzero_gaps) if nonzero_gaps.size else 1.0
        data = data + rng.uniform(-scale / 2, scale / 2, size=n)

    sorted_data = np.sort(data)
    gaps = np.diff(sorted_data)

    if np.any(gaps == 0):
        if not dither:
            raise ValueError(
                "Data contains duplicate/tied values, giving zero nearest-neighbour "
                "distances; the differential entropy estimator is undefined for "
                "these. Pass dither=True to break ties, or de-duplicate/jitter the "
                "data first."
            )
        # A residual handful of exact ties after dithering
        # floor them to a tiny positive value
        n_residual = int(np.sum(gaps == 0))
        warnings.warn(
            f"{n_residual} residual tied value(s) remained after dithering "
            f"(out of {n} points); flooring their nearest-neighbour distance to "
            "a negligible value. This is expected at very large N due to "
            "float64 precision limits and has a bounded, negligible effect on "
            "the entropy estimate.",
            stacklevel=2,
        )
        gaps = np.where(gaps == 0, np.finfo(float).tiny, gaps)

    # Nearest-neighbour distance for an interior point is the smaller of its two
    # adjacent gaps in the sorted array; the two endpoints only have one neighbour
    nn_dist = np.empty(n)
    nn_dist[0] = gaps[0]
    nn_dist[-1] = gaps[-1]
    nn_dist[1:-1] = np.minimum(gaps[:-1], gaps[1:])

    return float(np.euler_gamma / np.log(2) + np.mean(np.log2(2 * n * nn_dist)))


def optimal_bin_width(data: npt.NDArray[np.floating], M: float = 2.5, **knn_kwargs) -> float:
    """
    Entropy-based optimal histogram bin width for 1-D continuous data
    (Watts & Crow, Eqs. 9-10):

        Delta = 2**h(data) * N**(-1/M)

    where h is the nearest-neighbour differential entropy estimate of the data,
    and M is a free parameter controlling the number of histogram bins via
    H_M = (1/M) * log2(N). The paper finds 2 <= M <= 3 avoids both over-binning
    (Poisson noise dominates, M<2) and under-binning (loses distribution shape, M>3);
    M=2.5 is a reasonable default within that range.

    :param data: np.ndarray[np.floating]
        1-D array of N samples
    :param M: float
        Entropy scaling parameter, expected in [2, 3]
    :param knn_kwargs:
        Extra keyword arguments
    :return: float
        Recommended fixed bin width, Delta
    """
    if not (1 < M <= 3):
        raise ValueError("M should be in (1, 3]; recommend 2 <= M <= 3.")
    data = _as_float_array(data).ravel()
    n = data.size
    h = differential_entropy_knn(data, **knn_kwargs)
    return float(2 ** h * n ** (-1 / M))


def entropy_optimal_histogram(data: npt.NDArray[np.floating], M: float = 2.5, **knn_kwargs) -> dict[str, Any]:
    """
    Build a histogram of 1-D data using the entropy-based optimal bin width
    (Watts & Crow), and report the diagnostics.

    :param data: np.ndarray[np.floating]
        1-D array of N samples
    :param M: float
        Entropy scaling parameter passed to optimal_bin_width; see its docstring
    :param knn_kwargs:
        Extra keyword arguments
    :return: dict
        counts: np.ndarray, histogram bin counts
        bin_edges: np.ndarray, histogram bin edges (len(counts) + 1)
        bin_width: float, the bin width Delta used
        entropy_bits: float, Shannon entropy H_B of the resulting histogram (Eq. 2)
        efficiency: float, histogram efficiency e_H of the resulting histogram (Eq. 3)
        M_bin: float, estimate of M from the histogram's own entropy (Eq. 23)
    """
    data = _as_float_array(data).ravel()
    n = data.size
    delta = optimal_bin_width(data, M=M, **knn_kwargs)

    x_start = data.min()
    # small relative margin to avoid dropping the max value due to floating-point
    # rounding when computing the bin count
    n_bins = max(1, int(np.ceil((data.max() - x_start) / delta * (1 + 1e-9))))
    edges = x_start + delta * np.arange(n_bins + 1)
    counts, bin_edges = np.histogram(data, bins=edges)

    entropy_bits = shannon_entropy_histogram(counts)
    efficiency = histogram_efficiency(entropy_bits, n_bins)
    m_bin = np.log2(n) / entropy_bits if entropy_bits > 0 else float("inf")

    return {
        "counts": counts,
        "bin_edges": bin_edges,
        "bin_width": delta,
        "entropy_bits": entropy_bits,
        "efficiency": efficiency,
        "M_bin": float(m_bin),
    }
