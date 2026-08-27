import pytest
import numpy as np

from eregion.core.expsum_fit_math import ExpSumFitter, Expsumfun

# NOTE: don't think any reason this needs to be a pytest fixture TBH


def gen_data():
    testks = [0.02, 0.3, 0.7]
    testas = [2.1, 4.1, 4.9]

    NPTS: int = 200
    nn = np.arange(NPTS)
    xs = np.linspace(0, 12, NPTS)

    dU = xs[1] - xs[0]

    yy = np.zeros_like(xs)
    for k, a in zip(testks, testas):
        yy += a * np.exp(-k * xs)

    return nn, xs, dU, yy, testks, testas


def test_eval_expsum_array():

    nn, xs, dU, yy, ks, aas = gen_data()
    thetas = np.exp(-1 * np.array(ks) * dU)

    ev = Expsumfun(nn, aas, thetas)

    assert isinstance(ev, np.ndarray)
    assert len(ev.shape) == 1
    assert len(ev) == len(yy)

    diffarr = ev - yy
    assert np.all(np.isclose(ev, yy))


def test_eval_expsum_single():
    nn, xs, dU, yy, ks, aas = gen_data()
    thetas = np.exp(-1 * np.array(ks) * dU)

    IDX = 7
    ytest = yy[IDX]
    xtest = xs[IDX]

    print(f"x: {xtest}, y: {ytest}")
    evy = Expsumfun(nn[IDX], aas, thetas)
    print(f"ey: {evy}")

    assert isinstance(evy, float)
    assert np.isclose(evy, ytest)


def test_noiseless_fit_fixedM():
    nn, xs, dU, yy, ks, aas = gen_data()

    M = 3

    # add seed just to check it works. Not verifying
    # reproducibility of nonlinear fit at this point.
    efit = ExpSumFitter(data=yy, dU=dU, seed=42)
    iters = efit.run_fit(M=M)

    print(f"iterations: {iters}")
    print(f"thetafit: {efit.thetas}, afit: {efit.a}, kfit: {efit.ks}")

    citer = efit.coalesce_to_fixedM(M=M)
    print(f"coalescing iterations: {citer}")
    print(f"after coalescing...")
    print(f"thetass: {efit.thetas}, afit: {efit.a}, kfit: {efit.ks}")

    assert len(efit.thetas) == len(ks)
    roundk = np.round(efit.ks, 2)
    s = np.argsort(roundk)
    s_in = np.argsort(efit.ks)

    rounda = np.round(efit.a, 2)
    sa = np.argsort(rounda)
    sa_in = np.argsort(efit.a)

    # oof look at those errors
    # best we can do with simple coalescence, guvnor
    assert np.all(np.isclose(roundk[s], np.array(efit.ks)[s_in], rtol=0.05, atol=0.05))

    assert np.all(np.isclose(rounda[sa], np.array(efit.a)[sa_in], rtol=0.05, atol=0.05))


def test_noiseless_fit_convergence():
    nn, xs, dU, yy, ks, aas = gen_data()

    efit = ExpSumFitter(data=yy, dU=dU, seed=0xDEADBEEF)
    iters = efit.run_fit()

    print(f"iterations: {iters}")
    print(f"thetafit: {efit.thetas}, afit: {efit.a}, kfit: {efit.ks}")

    citer = efit.coalesce_to_tol()

    print(f"coalescing iterations: {citer}")

    rounda = np.round(efit.a, 2)
    sa = np.argsort(rounda)
    sa_in = np.argsort(aas)

    # choose the fitted coefficients up to the lenght of ground truth
    proca = rounda[sa[::-1]][: len(aas)]
    print(f"proca: {proca}")

    # sort amplitudes in descending order
    inpac = np.array(aas)[sa_in[::-1]]
    print(f"compar_input a: {inpac}")
    diffa = proca - inpac
    print(f"diffa: {diffa}")
    assert np.all(np.isclose(diffa, 0.0, rtol=0.05, atol=0.05))

    # no need to check ks, above constrained M proves appropriate scaling when
    # amplitudes are correct
