import numpy as np
from typing import Any, Callable
from functools import partial
from scipy.stats import median_abs_deviation, skew, skewtest, kurtosis, kurtosistest

STATKEYS = {
    'mean'
    'med',
    'std',
    'mad',
    'skew',
    'kurt',
    'skewtest',
    'kurttest',
    'eper_med',
    'eper_mean'
}

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

def basic_stats(data: np.ndarray | np.ma.MaskedArray, prepend_kw: str = "") -> dict[str, Any]:
    """
    Calculate basic statistics for the given data.
    :param data: np.ndarray or np.ma.MaskedArray
        Input data array.
    :param prepend_kw: str
        Prefix to prepend to the keys in the output dictionary.
    :return: dict[str, Any]
        Dictionary containing calculated statistics.
    """
    _operations: dict[str, Callable] = {"med": np.ma.median,
                                        "mean": np.ma.mean,
                                        "std": np.ma.std,
                                        "mad": partial(ma_mad, scale="normal")}

    out = {f"{prepend_kw}{kw}": float(op(data)) for kw, op in _operations.items()}
    return out

def stats_tests(data: np.ndarray | np.ma.MaskedArray, prepend_kw: str = "") -> dict[str, Any]:
    """
    Calculate skewness and kurtosis for the given data.
    :param data: np.ndarray or np.ma.MaskedArray
        Input data array.
    :param prepend_kw: str
        Prefix to prepend to the keys in the output dictionary.
    :return: dict[str, Any]
    """

    _operations: dict[str, Callable] = {"skew" : ma_skew,
                                       "kurt" : ma_kurt}
    out = {f"{prepend_kw}{kw}" : float(op(data)) for kw, op in _operations.items()}

    tests: dict[str,Callable] = {"skewtest" : ma_skewtest,
                                 "kurttest": ma_kurttest}

    for teststr, testop in tests.items():
        stat, pval = testop(data)
        k = f"{prepend_kw}{teststr}"
        out[k] = float(stat)
        out[f"{k}p"] = float(pval)
    return out

def calc_eper_trail(data: np.ndarray | np.ma.MaskedArray, axis: int, prepend_kw: str = "") -> dict[str, Any]:
    eper_med = np.median(data, axis=axis)
    eper_mean = np.mean(data, axis=axis)
    return {f"{prepend_kw}eper_med" : eper_med,
            f"{prepend_kw}eper_mean" : eper_mean}