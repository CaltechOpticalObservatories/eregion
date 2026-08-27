"""This module implements the robust exponential-sum fitting algorithm
found in the paper:
Wiscombe, W. J., and J. W. Evans. “Exponential-sum fitting of radiative transmission functions.” Journal of Computational Physics 24.4 (1977)

I was not able to find a maintained implementation that is shipped in a modern python numerical analysis library. If we ever do find one, please delete this code

"""

from typing import TypeVar, Optional, Generator
import warnings
import logging
from math import log, inf, nan, copysign

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import dual_annealing


_logger = logging.getLogger(__name__)


Fl = TypeVar("float")
NDArrF = np.ndarray[np.floating]
NDArrI = np.ndarray[np.integer]
Int = int | np.integer

def _safediv(x, y):
    return x/y if y else (copysign(x*inf, x*y) if x else nan)


def Expsumfun(n: NDArrI | Int, a: NDArrF, thetas: NDArrF) -> float | NDArrF:
    """
    Evaluate an exponential sum function (equation 4a in Wisombe & Evans).

    $$ sum_i a_i \theta_i ^n $$

    parameters
    ----------
    n: Int | np.ndarray[Int]
        power to raise to (either single or numpy array of integer powers)
    a: np.ndarray[np.floating]
        array of coefficients
    theta: np.ndarray[np.floating]
        array of exponential terms


    returns
    -------
    float | NDArrF

    returns either a single float, or a numpy array of floats, depending on the type of n
    """

    assert len(a) == len(thetas), "theta must be same length as a"

    if isinstance(n, Int):
        # single power, just evaluate one sum
        outt = np.dot(a, thetas**n)
        return float(outt)

    # construct an array of powers
    assert len(n.shape) == 1, "n must be a 1D array"
    powarr = n.reshape((-1, 1)) * np.ones_like(n, shape=(1, len(thetas)))
    # should be same shape as n
    out = np.sum(a * thetas**powarr, axis=1)
    assert len(out.shape) == 1, "logic error, out should be 1D"
    assert len(out) == len(n), "logic error, should be same length as power array"
    return out


class ExpSumFitter:
    def __init__(
        self,
        data: NDArrF,
        dU: float = 1.0,
        weights: Optional[NDArrF] = None,
        R0_epsilon: float = 1e-20,
        P0_epsilon=1e-20,
        seed: Optional = None,
    ):
        """

        parameters
        ----------
        data: np.ndarray[np.floating]
            y data points (evenly spaced in x) to fit to
        dU: float
            spacing of x data points (default = 1.0)
        weights: Optional[np.ndarray[np.floating]]
            least squares weights. If None, weights will be set to unity
        R0_epsilon: float
            convergence criteria on the R0 least squares residual sum
        P0_epsilon: float
            convergence criteria on the P0 minimum function
        seed: Optional
            something to seed the internal RNG with (used in the non-linear part of the fit)
            anything accepted by np.random.default_rng is acceptable. Supply to get deterministic behaviour for the exact same input data


        """
        self.data = data
        if weights is None:
            self.weights = np.ones_like(data)
        else:
            self.weights = weights
            assert len(weights.shape) == 1, "weights array must be 1D"
            assert len(weights) == len(data), "weights must be same length as data"

        self._R0_epsilon = R0_epsilon
        self._P0_epsilon = P0_epsilon

        self.R0 = None

        # TODO: add U, delta U
        self.n = np.arange(0, len(data))
        self.dU = dU
        self.a: list[float] = list()
        self.thetas: list[float] = list()
        self.minPtheta = None

        self._rng = np.random.default_rng(seed=seed)

    def _calc_pn(self):
        """calculate the array p_n (defined in equation 7 of Wiscombe & Evans), without the weight prefactor"""
        pn: np.ndarray = (
            Expsumfun(self.n, np.asarray(self.a), np.asarray(self.thetas)) - self.data
        )
        return pn

    def residpoly(self, theta: float | np.floating) -> float:
        """Calculate the "residual polynomial" (equation 6 in Wiscombe & Evans).

        parameters
        ----------
        theta: float | np.floating
            value of theta for which to evaluate the residual polynomial

        """
        pn = self.weights * self._calc_pn()
        return 2 * float(np.sum(pn * theta**self.n))

    def R0_resid(self):
        """calculate the R0 residual (equation 3 in Wiscombe & Evans)"""

        pn = self._calc_pn()
        out = pn**2 * self.weights
        return np.sum(out)

    def find_min_theta(self) -> tuple[float, float]:
        """Do the non-linear portion of the fit. Find the minimum of the residual polynomial
           P(theta). The location of this theta is then added to the fit terms. Unlike the original paper (which was written in 1977 when computers had somewhat less power available) we use a hideously modern and expensive annealing method to find the global minimum. This is slow but seems to work robustly

        returns
        -------

        tuple[float, float]

        tuple containing the value of theta which minimises the residual polynomial, and the value of the residual  polynomial respectively

        """

        # theta must by construction be between 0 and 1
        res = dual_annealing(self.residpoly, bounds=[(0, 1.0)], rng=self._rng)

        if not res.success:
            warnings.warn("residual poly minimization did not succeed")
            # TODO: debug message here

        _logger.debug(f"minimum theta value at: {res.x}, value {res.fun} ")
        return float(res.x[0]), float(res.fun)

    def check_convergence(self, M: Optional[int] = None) -> bool:
        """Check whether numerical convergence has been achieved, as defined in the paper. Briefly:
        a) whether the relative update of the L2 norm fit residual stops decreasing on another iteration - this means the fit will not get any better with more iterations
        b) if not, whether the residual polynomial of the fit is always above 0. Roughly speaking, this implies that adding another trial fitted exponential will not be able to fit the data better (see the paper section 6 for excruciating detail)
        """

        # first check numerical convergence
        R0 = self.R0_resid()
        _logger.debug(f"R0 residual: {R0}")

        # check first convergence criterion (update of residual is tiny)
        if self.R0 is None:
            self.R0 = R0
        elif R0 == 0.0 or self.R0 == 0.0:
            _logger.debug("model fits data exactly to machine precision. Unlikely, but it happened. Converged")
            return True
        else:
            #NOTE: if R0 is close to 0, this safely return infinity, which will compare larger than epsilon
            E = _safediv(self.R0 - R0, self.R0)
            _logger.debug(f"E update value: {E}")
            self.R0 = R0
            if E < self._R0_epsilon:
                _logger.debug("convergence achieved via R0 value")
                return True

        # minimum of P(theta) is >= 0 (2nd convergence criteria)
        if self.minPtheta is not None and self.minPtheta >= self._P0_epsilon:
            _logger.debug("convergence achieved via P0 minimum value")
            return True

        # we demanded a specific number of terms, carry on if that's not met
        if M is not None:
            if len(self.thetas) <= M:
                return False

        return False

    def do_linear_fit(self) -> list[float]:
        """Calculate the least-squares inversion part of the fit. For details see the paper"""
        powmat = np.tile(self.n, (len(self.thetas), 1))
        A = (np.asarray(self.thetas)[:, np.newaxis] ** powmat).T

        x, resid, rank, s = lstsq(A, self.data)
        # TODO: check conditioning here!
        _logger.debug(f"x in do_linear_fit: {x}")
        return list(float(_) for _ in x)

    def linear_fit_stage(self):
        """Run the whole linear portion of the fit iteration. For details see the paper"""
        amps = self.do_linear_fit()
        dropidx, amptrim = self.drop_zero_term(self.a, amps)

        ndrops: int = 0
        if dropidx is None:
            self.a = amps
        while dropidx is not None:
            del self.thetas[dropidx]
            amps = self.do_linear_fit()
            dropidx, amptrim = self.drop_zero_term(self.a, amps)
            self.a = amps
            ndrops += 1

        _logger.debug(f"n drops: {ndrops}")

    def drop_zero_term(
        self, amps_old: list[float], amps_new: list[float]
    ) -> tuple[Optional[int], Optional[list[float]]]:
        """perform "zero-trimming" procedure on amplitude coefficients as described in section 2(g) of Wiscombe & Evans

        parameters
        ----------

        amps_old: list[float]
            the previous set of amplitude coefficients
        amps_new: list[float]
            the proposed new set of amplitude coefficients (from a linear fit)


        returns
        -------

        Optional[int], Optional[list[float]]

        returns None if iterations have ended (indicated by all coeffs being higher than 0),
        otherwise list of new coefficients after shuffling and zero-dropping is complete, and the index of the zero term that was dropped.


        """
        if all(_ > 0 for _ in amps_new):
            return None, amps_new

        min_idx = -1
        min_beta = float(np.inf)

        # ensure last term of amps_old is 0 as required by algorithm
        if len(amps_old) == 0 or amps_old[-1] != 0:
            amps_old.append(0)

        for idx, (ao, an) in enumerate(zip(amps_old, amps_new)):
            if an < 0:
                beta: float = ao / (ao - an)
                if beta < min_beta:
                    min_beta = beta
                    min_idx = idx

        amps_out = [
            (1 - min_beta) * ao + min_beta * an for (ao, an) in zip(amps_old, amps_new)
        ]
        del [amps_out[min_idx]]

        return min_idx, amps_out

    def run_fit(self, M: Optional[int] = None, max_iters: int = 200) -> int:
        """Run the fit procedure until convergence or a maximum number of iterations is reached

        parameters
        ----------

        :param M: Optional[int]
            The number of desired terms in the fit. If not supplied, fit procedes until the numberical
            convergence criteria descibed in the paper are reached (see docstring for check_convergence

        :param max_iters: int
            maximum number of iterations to run.

        returns
        -------

        int

        the number of fit iterations that were actually run

        """

        niter: int = 0
        genfit = self.iterate_fit()
        while not self.check_convergence(M) and niter < max_iters:
            _logger.debug(f"niter: {niter}")
            niter += 1
            thetas, aas = next(genfit)

        if niter >= max_iters:
            _logger.info("max_iters reached, iterations halted!")

        return niter

    @property
    def ks(self) -> list[float]:
        """convenience property to calculate the exponential decay rates corresponding to the
        currently held theta values"""
        kout = [-1.0 * log(float(_)) / float(self.dU) for _ in self.thetas]
        return kout

    def iterate_fit(self) -> Generator:
        """perform an iteration of the fit procedure, yielding updated values of theta and a. Can be used manually, but intended to be run from one of the overall fit running routines

        returns
        -------

        Generator
           when iterated over, this generator performs both the non-linear and linear stages of the fit,
           and yields copies of the current fitted values of theta and amplitude

        """
        while True:
            new_theta, minPtheta = self.find_min_theta()
            self.minPtheta = minPtheta
            self.thetas.append(new_theta)
            self.linear_fit_stage()

            _logger.debug(f"thetas: {self.thetas}")
            _logger.debug(f"as: {self.a}")
            yield self.thetas.copy(), self.a.copy()

    def _find_closest_k_pair(self):
        """Locate the closest pair of k values in the current fit

        returns
        -------

        tuple[float, float]

        returns the values of theta (NOTE: not the values of k!!! despite the name)
        corresponding to the k values closest together in the current fit

        """
        s = np.argsort(self.thetas)
        minidx = np.argmin(np.diff(np.array(self.thetas)[s]))
        return s[minidx], s[minidx + 1]

    def coalesce_to_tol(self, tol: float = 0.25, maxiter: int = 200) -> int:
        """repeatedly  coalesce (remove terms from) the fit until a tolerance condition is met, or maximum number of iterations is reached

         The tolerance condition is trivial: that no two k values differ in ratio more than a specified amount.
        Note: the original paper specifies another non-linear fiut to optimise the coalesced terms. At the moment we do not implement that second fit. HOWEVER, the initial guesses for that fit are pretty good. If only % level accuracy is required (rather than ppm level) this should be sufficient. In future we may implement that second fitting procedure here. This implementation is intended to fit EPER curves of CCD detectors, where that level of accuracy is absolutely pointless, and impossible anyway.


        parameters
        ----------

        :param tol: float
            the tolerance ratio of the k values (not theta values, you normally care about k in the end!!!)

        :param maxiter: int
            the maximum number of iterations to do before giving up, even if tolerance condition isn't met

        returns
        -------

        int

        the number of iterations actually run during coalescence

        """

        gencol = self.simple_coalesce_iter(tol)

        niter: int = 0
        for it in gencol:
            niter += 1
            if niter >= maxiter:
                break

        return niter

    def coalesce_to_fixedM(self, M: int, maxiter: int = 200) -> int:
        """repeatedly coalesce (remove terms from) the fit until only a specified number remain, or maximum number of iterations is reached

        parameters
        ----------

        :param M: int
            number of desired terms in the final fit

        :param maxiter: int
            maximum number of iterations to do (extremely unlikely to occur except for enormous fits)

        """

        gencol = self.simple_coalesce_iter(float(np.inf))
        niter: int = 0
        for thet, aa in gencol:
            niter += 1
            if niter >= maxiter:
                break
            if len(thet) == M:
                break
        return niter

    def simple_coalesce_iter(self, tol: float = 0.25) -> Generator:
        """implement trivial term coalescence (just using initial guesses combining nearby terms).
        The full term coalescence fit from the paper is a future implementation goal

        Each iteration first removes the two k terms closest to each other, then replaces those with
        a single k term in the middle. This trivial algorithm works well enough for our current purposes


        parameters
        ----------

        :param tol: float
           the relative k term tolerance ratio to check. Stop iterations if it is met

        returns
        -------

        Generator

        a generator which does one coalescence iteration per yield

        """

        # remove any theta ==1.0 or 0.0
        def remove_theta_val(val):
            while val in self.thetas:
                idx = self.thetas.index(val)
                del self.thetas[idx]
                del self.a[idx]

        remove_theta_val(0.0)
        remove_theta_val(1.0)

        # k1 is the bigger of the two terms by construction
        while True:
            s1, s2 = self._find_closest_k_pair()
            k1 = -1.0 * log(self.thetas[s1]) / self.dU
            k2 = -1.0 * log(self.thetas[s2]) / self.dU

            assert k1 > k2, "logic error, k2 should be bigger than k1"

            if (k1 / k2) > (1.0 + tol):
                _logger.debug("coalescent iteration finished")
                break

            thet1 = self.thetas[s1]
            thet2 = self.thetas[s2]

            self.thetas[s1] = 0.0
            self.thetas[s2] = 0.0

            a1 = self.a[s1]
            a2 = self.a[s2]

            remove_theta_val(0.0)

            self.thetas.append(0.5 * (thet1 + thet2))
            self.a.append(a1 + a2)

            yield self.thetas.copy(), self.a.copy()
