import numpy as np
from eregion.core.approx_taylor_series import calc_approx_taylor_series_coeffs


def test_approx_taylor_series_exponential():

    def fun(z):
        return np.exp(z)

    #Taylor coefficients of exp(x) up to order 6
    ref_coeffs = [1., 1., 1./2., 1./6., 1./24., 1./120.]

    coeffs = calc_approx_taylor_series_coeffs(fun, 6)
    diff = coeffs - np.array(ref_coeffs)
    assert np.all(np.isclose(diff, 0, atol=2E-3))
