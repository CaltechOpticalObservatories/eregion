import numpy as np
from typing import Any
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