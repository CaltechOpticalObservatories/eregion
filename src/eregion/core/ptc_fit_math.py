import numpy as np
from typing import Optional
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import umath
from numpy.polynomial import Polynomial, polynomial

def polynomial_fit_covariance_matrix(poly, xdat: np.ndarray, ydat: np.ndarray, deg: int) -> np.ndarray:
    """calculate the covariance matrix of a polynomial fit by computing the design matrix"""
    coefs = poly.convert().coef
    vander = polynomial.polyvander(xdat, deg)
    dof = len(xdat) - vander.shape[1]
    resid = ydat - np.dot(vander, coefs)
    chi2 = np.sum(resid**2)
    inv_v = np.linalg.inv(np.dot(vander.T, vander))
    cov = inv_v * (chi2 / dof)
    return cov

def do_polynomial_fit(xdat: np.ndarray, ydat: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """fit a polynomial to the data and return the coefficients and covariance matrix"""
    poly = Polynomial.fit(xdat, ydat, deg)
    ft = poly.convert().coef
    cov = polynomial_fit_covariance_matrix(poly, xdat, ydat, deg)
    errs = np.sqrt(np.diag(cov))
    return ft, errs

def find_adc_sat_index(flux: np.ndarray, mean: np.ndarray, sigma: float = 5.0, return_spline: bool = False) -> Optional[int]:
    """Find the location of an ADC saturation point in a flux vs mean graph.
    The method is to fit an interpolating spline, get the analytic 2nd derivative of that,
    then find the most negative point of the 2nd derivative.
    :param flux: np.ndarray
        array of flux values (or exposure times)
    :param mean: np.ndarray
        array of mean counts from the flat images
    :param sigma: float
        significance threshold for the 2nd derivative to be considered a valid saturation point. Default is 5.0
    :param return_spline: bool
        if True, return the spline object along with the index of the saturation point. Default is False
    :return: Optional[int]
        index of the saturation point if found, otherwise None. If return_spline is True, returns a tuple of (index, spline object)
    """
    spl = make_interp_spline(flux, mean)
    derv = spl.derivative(2)
    derv_yy = derv(flux)
    satidx = np.argmin(derv_yy)

    derv_significance = (derv_yy[satidx] - np.mean(derv_yy)) / np.std(derv_yy)
    fwvalid: bool = abs(derv_significance) > sigma
    out = satidx if fwvalid else None

    if return_spline:
        return out, spl
    return out

def find_rough_full_well(mean: np.ndarray, noise: np.ndarray, fwfact: float = 0.9) -> tuple[int, int]:
    """
    Find index and mean value of full well point, by simple location of the maximum in the mean vs variance graph
    :param mean: np.ndarray
        array of mean counts in the flat images
    :param noise: np.ndarray
        array of standard deviation values of the differenced images
    :param fwfact: float
        factor to multiply the full well value by. Default is 0.9
    :return: tuple[int, int]
        index of the full well point and the full well value
    """

    #TODO: check that there's an actual maximum here
    am = np.argmax(noise)
    fwloc = fwfact * mean[am]

    #find nearest index to full well fact times that
    fwfactloc = np.argmin(np.abs(fwloc - mean))
    return fwfactloc, am


def trad_ptc_shot_noise_fit(mean: np.ndarray, noise: np.ndarray, limitidx: int, brighterfatter: bool=True
                            ) -> tuple[unc.ufloat, unc.ufloat] | tuple[unc.ufloat, unc.ufloat, unc.ufloat]:
    """fit a PTC by the traditional (Janesick) method, with optional brighter-fatter modifications

    Parameters
    ----------

    :param mean : np.ndarray
       array representing mean counts in the flat images
    :param noise : np.ndarray
       array representing standard deviation values of differenced images
    :param limitidx : int
       integer index up to which to do the fit (should be below full well, and in the case
       of detectors with significant brighter-fatter effect, should be substantially below (e.g. 80% of full well)
    :param brighterfatter : bool
       if True, fit a quadratic to somewhat compensate for brighter-fatter effect. If False, do a traditional, linear fit

    Returns
    -------

    tuple[unc.ufloat, unc.ufloat]
       if brighterfatter is False, return estimates of the camera gain K with error, and the noise n with error

    tuple[unc.ufloat, unc.ufloat]
       if brighterfatter is True, return estimates of camera gain K with error, noise n with error, and
       brighter fatter total curvature coefficient (roughly a00) wth error

    all errors are estimated from fit covariance matrix

    """

    deg: int = 2 if brighterfatter else 1
    xdat = mean[:limitidx]
    ydat = noise[:limitidx]**2

    ft, errs = do_polynomial_fit(xdat, ydat, deg)

    Kest = 1 / unc.ufloat(ft[1], errs[1])
    noiseest = np.sign(ft[0]) * umath.sqrt(unc.ufloat(abs(ft[0]), errs[0]))

    if brighterfatter:
        a00est = unc.ufloat(ft[2], errs[2])
        return Kest, noiseest, a00est

    return Kest, noiseest

def astier_approx_fun(mu, g, a00, n):
    """Astier's approximate PTC shape function. Equation (15) in Astier et al (2019)"""
    return  (np.exp(2* a00 * mu * g) - 1) / (2*g**2*a00) + n / g**2

def astier_approx_eval_std(mu, K, a00, noise):
    """evaluate Astier's function but with gain and noise in Janesick scaling
       In Astier, gain g is the gain between mean and variance. This is the same as Janesick's K constant
       Likewise in Astier the noise term is given in electrons**2, in Janesick it's in electrons
       Our PTC classes return Janesick's quantities, even when fitting with Astier method, to avoid confusion
       since Janesick's definitions are much more common in the community
    """
    g =  K
    n = (noise*K)**2
    return astier_approx_fun(mu, g, a00, n)

def astier_approx_one_param_fit(mean: np.ndarray, noise: np.ndarray, Kguess: float, aguess: float, noiseguess: float,
                                fitlim: Optional[int] = None) -> tuple[unc.ufloat, unc.ufloat, unc.ufloat]:
    """Fit photon transfer curve data with Astier's approximate one parameter function

    Parameters
    ----------

    mean: np.ndarray
        array of mean counts in the flat images
    noise: np.ndarray
        data containing standard deviation values. NOTE Astier's fit uses variance, but this function takes std deviation
        to keep all our fitting functions consistent
    fitlim: Optional[int]
        convenience function to trim off some data below full well, if provided data is only fitted up to this index

    Returns
    -------
    tuple[unc.ufloat, unc.ufloat, unc.ufloat]

    estimates of K, a00 and n including error estimates (from fit covariance matrix)

    """

    if fitlim is not None:
        xdat = mean[:fitlim]
        ydat = noise[:fitlim]**2
    else:
        xdat = mean
        ydat = noise**2

    p0 = [Kguess, aguess, (noiseguess/Kguess)**2]
    bounds = ([-np.inf, -np.inf, 0], [np.inf, 0, np.inf])

    popt, pcov = curve_fit(astier_approx_fun, xdat, ydat,  p0=p0, bounds=bounds)
    errs = np.sqrt(np.diag(pcov))

    print(f"astier popt: {popt}, pcov: {pcov}")
    K = unc.ufloat(popt[0], errs[0])
    a00 = unc.ufloat(popt[1], errs[1])
    n = umath.sqrt(unc.ufloat(abs(popt[2]), errs[2]))
    print(f"Astier popt noise number: {popt[2]}, errn: {errs[2]}")
    print(f"Astier noise term: {n}")
    return K, a00, n

def linearity_fit(flux: np.ndarray, mean: np.ndarray, fitlim: Optional[int]):
    """Fit the linearity curve (mean vs exposure time) with a linear curve

    Parameters
    ----------

    flux: np.ndarray
        array of flux values (or exposure times)

    mean: np.ndarray
        array of mean values

    fitlim: Optional[int]
        if supplied, trim the data beyond this point

    Returns
    -------

    tuple[np.ndarray, np.ndarray]
    tuple containing the array of coefficients (in new numpy order, lowest coefficient first),
    and the array of errors estimated from fit covariance
    """


    if fitlim is not None:
        xdat = flux[:fitlim]
        ydat = mean[:fitlim]
    else:
        xdat = flux
        ydat = mean

    ft, errs = do_polynomial_fit(xdat, ydat, 1)
    return ft, errs
