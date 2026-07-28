import numpy as np
from typing import Optional
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
import uncertainties as unc
from uncertainties import umath
from numpy.polynomial import Polynomial, polynomial


def find_adc_sat_index(etime_dat: np.ndarray, mndat: np.ndarray, sigma: float = 5.0, return_spline: bool = False) -> Optional[int]:
    """find the location of an ADC saturation point in a flux vs mean graph.
    The method is to fit an interpolating spline, get the analytic 2nd derivative of that,
    then find the most negative point of the 2nd derivative. """
    spl = make_interp_spline(etime_dat, mndat)
    derv = spl.derivative(2)
    dervyy = derv(etime_dat)
    satidx = np.argmin(dervyy)

    dervsigniff = (dervyy[satidx] - np.mean(dervyy)) / np.std(dervyy)
    fwvalid: bool = abs(dervsigniff) > sigma
    out = satidx if fwvalid else None

    if return_spline:
        return out, spl
    return out

def find_rough_full_well(mndat: np.ndarray, stddat: np.ndarray, fwfact: float = 0.9) -> tuple[int, int]:
    """find index and mean value of full well point, by simple location of the maximum in the mean vs variance graph"""

    #TODO: check that there's an actual maximum here
    am = np.argmax(stddat)
    fwloc = fwfact * mndat[am]

    #find nearest index to full well fact times that
    fwfactloc = np.argmin(np.abs(fwloc - mndat))
    return fwfactloc, am


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



def trad_ptc_shot_noise_fit(mndat: np.ndarray, sddat: np.ndarray, limitidx: int, brighterfatter: bool=True
                            ) -> tuple[unc.ufloat, unc.ufloat] | tuple[unc.ufloat, unc.ufloat, unc.ufloat]:
    """fit a PTC by the traditional (Janesick) method, with optional brighter-fatter modifications

    Parameters
    ----------
    :param mndat : np.ndarray
       array representing mean values
    :param sddat : np.ndarray
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
    xdat = mndat[:limitidx]
    ydat = sddat[:limitidx]**2

    poly = Polynomial.fit(xdat, ydat, deg)
    ft = poly.convert().coef

    #TODO: we will want this a lot, pull it into a convenience math function
    #compute covariance matrix

    cov = polynomial_fit_covariance_matrix(poly, xdat, ydat, deg)
    errs = np.sqrt(np.diag(cov))

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


def astier_approx_one_param_fit(mndat: np.ndarray, sddat: np.ndarray, Kguess: float, aguess: float, noiseguess: float,
                                fitlim: Optional[int] = None) -> tuple[unc.ufloat, unc.ufloat, unc.ufloat]:
    """Fit photon transfer curve data with Astier's approximate one parameter function

    Parameters
    ----------

    mndat: np.ndarray
        data containing mean values

    sddat: np.ndarray
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
        xdat = mndat[:fitlim]
        ydat = sddat[:fitlim]**2
    else:
        xdat = mndat
        ydat = sddat**2

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


def linearity_fit(etimedat: np.ndarray, mndat: np.ndarray, fitlim: Optional[int]):
    """Fit the linearity curve (mean vs exposure time) with a linear curve

    Parameters
    ----------

    etimedat: np.ndarray
        array of exposure times (or flux)

    mndat: np.ndarray
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
        xdat = etimedat[:fitlim]
        ydat = mndat[:fitlim]
    else:
        xdat = etimedat
        ydat = mndat

    poly = Polynomial.fit(xdat, ydat, 1)
    ft = poly.convert().coef
    cov = polynomial_fit_covariance_matrix(poly, xdat, ydat, 1)
    errs = np.sqrt(np.diag(cov))
    return ft, errs
