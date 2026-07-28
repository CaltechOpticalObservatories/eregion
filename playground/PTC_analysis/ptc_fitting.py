import numpy as np
from numpy.polynomial import Polynomial, polynomial

from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from typing import Optional
import logging
import uncertainties as unc
from uncertainties import umath


logger = logging.getLogger(__name__)

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

    deg: int = 2 if brighterfatter else 1
    xdat = mndat[:limitidx]
    ydat = sddat[:limitidx]**2

    poly = Polynomial.fit(xdat, ydat, deg)
    ft = poly.convert().coef

    #TODO: we will want this a lot, pull it into a convenience math function
    #compute covariance matrix

    cov = polynomial_fit_covariance_matrix(poly, xdat, ydat, deg)
    errs = np.sqrt(np.diag(cov))

    Kest = 1 / umath.sqrt(unc.ufloat(ft[1], errs[1]))
    noiseest = np.sign(ft[0]) * umath.sqrt(unc.ufloat(abs(ft[0]), errs[0]))

    if brighterfatter:
        a00est = unc.ufloat(ft[2], errs[2])
        return Kest, noiseest, a00est

    return Kest, noiseest



def astier_approx_fun(mu, g, a00, n):
    return  (np.exp(2* a00 * mu * g) - 1) / (2*g**2*a00) + n / g**2

def astier_approx_eval_std(mu, K, a00, noise):
    g =  K **2
    n = noise**2 * K**2
    return astier_approx_fun(mu, g, a00, n)


def astier_approx_one_param_fit(mndat: np.ndarray, sddat: np.ndarray, fitlim: int,  Kguess: float, aguess: float, noiseguess: float):

    xdat = mndat[:fitlim]
    ydat = sddat[:fitlim]**2

    p0 = [1./Kguess**2, aguess, (noiseguess*Kguess)**2]
    bounds = ([-np.inf, -np.inf, 0], [np.inf, 0, np.inf])

    popt, pcov = curve_fit(astier_approx_fun, xdat, ydat,  p0=p0, bounds=bounds)
    errs = np.sqrt(np.diag(pcov))

    K = umath.sqrt(unc.ufloat(popt[0], errs[0]))
    a00 = unc.ufloat(popt[1], errs[1])
    n = umath.sqrt(unc.ufloat(abs(popt[2]), errs[2]))
    return K, a00, n

def linearity_fit(etimedat: np.ndarray, mndat: np.ndarray, fitlim: Optional[int]):
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



if __name__ == "__main__":
#    DATA_DIR="/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260721-095716/" # blue no back bias
#    DATA_DIR="/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260721-174626/" # blue -50.0 VBB
#    DATA_DIR="/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260721-135158//" # blue -30.0 VBB
    DATA_DIR="/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260718-001407/" # green -50.0 VBB
#    DATA_DIR="/dettest_data/DTU_detreduce/DTU_fullfp_bringup/PTC/SCI/20260717-161253//" # green no back bias

    from astropy.io import fits
    import os
    import matplotlib.pyplot as plt

    hdul = fits.open(os.path.join(DATA_DIR, "ptc_table.fits"))

    #prepare data manually for now
    dets = set(hdul[1].data["det_id"])
    outputs = set(hdul[1].data["output"])

    def select_data(dat, det, op) -> np.ndarray:
        selarr = np.logical_and(dat["det_id"] == det, dat["output"] == op)
        #selarr = np.logical_and(dat["diff"] == diff, selarr)
        return dat[selarr]

    def preproc_data(dat, det, op):
        seldat  = select_data(dat, det, op)

        etime = seldat["exptime"]
        med = seldat["med"]
                     
        mean = seldat["mean"]
        sd = (seldat["std_0"] + seldat["std_1"])/2.
        diffsd = seldat["std_diff"]
        mad = (seldat["mad_0"] + seldat["mad_1"]) / 2.

        return {"etime_dat" : etime, "mean_dat" : mean,
                "std_dat" : sd, "diff_std_dat" : diffsd,
                "med_dat" : med, "mad_dat" : mad}



    DET = "det_2"
    OP = "E"
    FWFACT: float = 0.8

    # for det in dets:
    #     for op in outputs:
    #         dstr: str = f"{det}[{op}]"
    #         print(f"------------processing: {dstr}----------")
    #         ppdat = preproc_data(hdul[1].data, det, op)
    #         satpnt = find_adc_sat_index(ppdat["etime_dat"], ppdat["mean_dat"])
    #         sddat = ppdat["diff_std_dat"] / np.sqrt(2)

    #         fwfactloc, fwloc = find_rough_full_well(ppdat["mean_dat"], sddat)
    #         Kguess, nguess, a00guess = trad_ptc_shot_noise_fit(ppdat["mean_dat"], sddat, fwfactloc)

    #         print(f" rough estimate camera gain: {Kguess} (e-/DN), noise upper bound: {nguess}  (e-)")
    #         print(f" rough a00 estimate: {a00guess}")

    #         Kast, a00ast, nast = astier_approx_one_param_fit(ppdat["mean_dat"], sddat, fwloc, Kguess.nominal_value, a00guess.nominal_value, nguess.nominal_value)
    #         print(f" Astier one-param fit, gain: {Kast}, a00: {a00ast}, noise: {nast}")

    #         if satpnt is not None:
    #             satmn = ppdat["mean_dat"][satpnt]
    #             print(f"ADC saturation index: {satpnt}, value: {satmn} fpDN, {satmn*Kguess.nominal_value} e-")
    #         else:
    #             print("no ADC saturation point found")
    #         fwguess = ppdat["mean_dat"][fwloc]
    #         print(f"rough full well: {fwguess } fpDN, {fwguess * Kguess.nominal_value } e-")



    pd = preproc_data(hdul[1].data, DET, OP)


    plt.close("all")
    fig = plt.figure(figsize=(10,7), constrained_layout=True)
    fig.suptitle(f"{DET}[{OP}]")
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3,1])
    linresidax = fig.add_subplot(gs[1,0])
    linresidax.set_xlabel("exposure time (s)")
    linax = fig.add_subplot(gs[0,0], sharex=linresidax)
    linax.set_ylabel("$\mu$ (fDN)")
    linax.tick_params(labelbottom=False)
    mvresidax = fig.add_subplot(gs[1,1])
    mvresidax.set_xlabel("$\mu$ (fDN)")
    mvax = fig.add_subplot(gs[0,1], sharex = mvresidax)
    mvax.tick_params(labelbottom=False)
    mvax.set_ylabel("$\\sigma$ (fDN)")



    sddat = pd["diff_std_dat"] 

    satpnt = find_adc_sat_index(pd["etime_dat"], pd["mean_dat"])
    fwfactloc, fwloc = find_rough_full_well(pd["mean_dat"], sddat, FWFACT)

    K, n,  a00 = trad_ptc_shot_noise_fit(pd["mean_dat"], sddat, fwfactloc)
    Ka, aa, na = astier_approx_one_param_fit(pd["mean_dat"], sddat, fwloc, K.nominal_value, a00.nominal_value, abs(n.nominal_value))
    linax.plot(pd["etime_dat"], pd["mean_dat"], "x", label="data")
    if satpnt is not None:
        linax.axvline(pd["etime_dat"][satpnt], c="red")
        linresidax.axvline(pd["etime_dat"][satpnt], c="red")


    fwguess = pd["mean_dat"][fwloc]
    fwfactguess = pd["mean_dat"][fwfactloc]

    astprop = {"c" : "purple", "linestyle" : "--"}
    classprop = {"c" : "green", "linestyle" : "-"}

    mvax.plot(pd["mean_dat"], pd["std_dat"], "x", label="data (single)")
    mvax.plot(pd["mean_dat"],  sddat, ".", label="data (diff pair)")
    mvax.axvline(fwguess, c="red")
    mvax.axvline(fwfactguess, c="grey", ls="--")
    mvy = np.sqrt(pd["mean_dat"]) / K.nominal_value #+ abs(n.nominal_value) / K.nominal_value
    shotprop = mvax.plot(pd["mean_dat"],  mvy , "--", label="PTC shot noise fit", **classprop)
    print(f"n : {n}")
    yy = np.sqrt(astier_approx_eval_std(pd["mean_dat"], Ka.nominal_value, a00.nominal_value, 0.0))
    astrprop = mvax.plot(pd["mean_dat"], yy, "--", label="Astier approx 1-param fit", **astprop)

    linfit, linerrs = linearity_fit(pd["etime_dat"], pd["mean_dat"], satpnt)
    print(f"linearity fit: {linfit}")
    liny = pd["etime_dat"] * linfit[1] + linfit[0]

    linfitprop = {"c" : "red", "linestyle" : "--"}
    linax.plot(pd["etime_dat"], liny, label="linear fit", **linfitprop)

    mvax.loglog()

    linresid =   pd["mean_dat"] / liny - 1
    linresidax.plot(pd["etime_dat"], linresid, **linfitprop)
    linresidax.set_ylim(-0.02, 0.02)

    linax.legend()
    mvax.legend()


    mvresid1 = sddat / mvy - 1
    mvresid2 = sddat / yy - 1

    mvresidax.plot(pd["mean_dat"], mvresid1,  **classprop)
    mvresidax.plot(pd["mean_dat"], mvresid2,  **astprop)
    mvresidax.axvline(fwguess, c="red")
    mvresidax.axvline(fwfactguess, c="grey", ls="--")

    mvresidax.set_ylim(-0.1,0.05)
