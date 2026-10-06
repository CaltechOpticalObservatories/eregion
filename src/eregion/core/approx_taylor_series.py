from typing import Callable
from numbers import Complex, Real, Integral
import numpy as np

def calc_approx_taylor_series_coeffs(func: Callable[np.ndarray[Complex], [np.ndarray[Complex]]], N: Integral) -> np.ndarray[Real]:
    """calculate the Taylor series coefficients of a function approximately, by using the Fourier Transform
    of the complex roots of unity approach

    parameters
    =========

    :param func : Callable
        function that should take complex argument and return complex argument. The function must be able to evaluate
        complex arguments correctly. The function must be vectorised in a numpy sense (i.e. able to accept numpy arrays
        and return numpy arrays)

    :param N : Integer
        number of coefficients to calculate

    """

    #using the fact that the derivative of a function in Fourier space is just 1/N times the function
    zz = np.exp(2j * np.pi * np.arange(N) / N)
    #Taylor coefficients are just given by the roots of unity divided by the order of the derivative. Neat
    coeffs = np.real(np.fft.fft(func(zz)) / N)
    return coeffs


