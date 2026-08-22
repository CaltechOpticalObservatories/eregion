import numpy as np
from typing import Any, Callable
from functools import partial
from scipy.stats import median_abs_deviation, skew, skewtest, kurtosis, kurtosistest

def median_by_axis(data: np.ndarray, axis: int, **kwargs) -> np.ndarray:
    """
    Calculate the median of the input data along the specified axis, keeping the dimensions same as the input data.
    """
    return np.median(data, axis=axis, keepdims=True, **kwargs)

def wrap_scipy_statfun(fun):
    def wrapper(dat, **kwargs):
        if "axis" not in kwargs:
            kwargs["axis"] = None
        if not isinstance(dat, np.ma.MaskedArray):
            return fun(dat, **kwargs)
        else:
            filledarr = dat.filled(np.nan)
            return fun(filledarr, nan_policy="omit", **kwargs)
    return wrapper

def ma_skew(data: np.ndarray | np.ma.MaskedArray, **kwargs) -> Any:
    return wrap_scipy_statfun(skew)(data, **kwargs)

def ma_kurt(data: np.ndarray | np.ma.MaskedArray, **kwargs) -> Any:
    return wrap_scipy_statfun(kurtosis)(data, **kwargs)

def ma_skewtest(data: np.ndarray | np.ma.MaskedArray, **kwargs) -> Any:
    return wrap_scipy_statfun(skewtest)(data, **kwargs)

def ma_kurttest(data: np.ndarray | np.ma.MaskedArray, **kwargs) -> Any:
    return wrap_scipy_statfun(kurtosistest)(data, **kwargs)

def ma_mad(data: np.ndarray | np.ma.MaskedArray, **kwargs) -> Any:
    return wrap_scipy_statfun(median_abs_deviation)(data, **kwargs)

################### Helper functions for statistics calculations ###################
STATFUNCS = {
    'mean' : np.ma.mean,
    'median': np.ma.median,
    'std' : np.ma.std,
    'mad': partial(ma_mad, scale="normal"),
    'skew': ma_skew,
    'kurt': ma_kurt,
    'skewtest': ma_skewtest,
    'kurttest': ma_kurttest,
}

def do_statistics(data: np.ndarray | np.ma.MaskedArray,
                  which: dict[str, Callable] = STATFUNCS,
                  axis: int | None = None,
                  prepend_kw: str = "",
                  ) -> dict[str, Any]:
    """
    Calculate statistics for the given data.
    :param data: np.ndarray or np.ma.MaskedArray
        Input data array.
    :param which: dict[str, Callable]
        Dictionary of statistics functions to apply. Keys are the names of the statistics, and values are the corresponding functions.
    :param axis: int or None
        Axis along which to calculate the statistics. If None, the entire array is used.
    :param prepend_kw: str
        Prefix to prepend to the keys in the output dictionary.
    :return: dict[str, Any]
        Dictionary containing calculated statistics.
    """
    if len(which) == 0 or which is None:
        raise ValueError("No statistics functions provided in 'which' parameter.")

    stats = {}
    for kw, operation in which.items():
        if kw in ["skewtest", "kurttest"]:
            stat, pval = operation(data, axis=axis)
            stats[f"{prepend_kw}{kw}"] = float(stat)
            stats[f"{prepend_kw}{kw}p"] = float(pval)
        else:
            val = operation(data, axis=axis)
            val = np.nan if isinstance(val, np.ma.core.MaskedConstant) else val.filled(np.nan) if isinstance(val, np.ma.MaskedArray) else val
            stats[f"{prepend_kw}{kw}"] = val
    return stats