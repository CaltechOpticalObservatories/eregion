"""This module implements the robust exponential-sum fitting algorithm
found in the paper:
Wiscombe, W. J., and J. W. Evans. “Exponential-sum fitting of radiative transmission functions.” Journal of Computational Physics 24.4 (1977)

I was not able to find a maintained implementation that is shipped in a modern python numerical analysis library. If we ever do find one, please delete this code

"""

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize_scalar
from typing import Sequence, TypeVar, Optional, Generator
import warnings
import logging


_logger = logging.getLogger(__name__)


Fl = TypeVar("float")
NDArrF = np.ndarray[np.floating]
NDArrI = np.ndarray[np.integer]
Int = int | np.integer


def Expsumfun(n: NDArrI | Int , a: NDArrF, thetas: NDArrF) -> float | NDArrF:
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
        #single power, just evaluate one sum
        outt = np.dot(a, thetas**n)
        return float(outt)

    #construct an array of powers
    assert len(n.shape) == 1, "n must be a 1D array"
    powarr = n.reshape((-1,1)) * np.ones_like(n, shape=(1, len(thetas)))
    #should be same shape as n
    out = np.sum(a * thetas**powarr, axis=1)
    assert len(out.shape) == 1, "logic error, out should be 1D"
    assert len(out) == len(n), "logic error, should be same length as power array"
    return out


class ExpSumFitter:
    def __init__(self, data: NDArrF, dU: float = 1.0,  weights: Optional[NDArrF] = None,
                 R0_epsilon: float = 1E-20, P0_epsilon = 1E-20):
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

        #TODO: add U, delta U
        self.n = np.arange(0, len(data))

        self.a: list[float] = list()
        self.thetas: list[float] = list()
        self.minPtheta = None

    def _calc_pn(self):
        """calculate the array p_n (defined in equation 7 of Wiscombe & Evans), without the weight prefactor"""
        pn: np.ndarray = (Expsumfun(self.n, np.asarray(self.a), np.asarray(self.thetas)) - self.data)
        return pn

    
    def residpoly(self, theta: float | np.floating) -> float:
        """    Calculate the "residual polynomial" (equation 6 in Wiscombe & Evans).
        
            parameters
            ----------
            theta: float | np.floating
                value of theta for which to evaluate the residual polynomial

        """
        pn = self.weights * self._calc_pn()
        return 2 * float(np.sum(pn * theta ** self.n))

    def R0_resid(self):
        """ calculate the R0 residual (equation 3 in Wiscombe & Evans)
        """

        pn = self._calc_pn()
        out = pn**2 * self.weights
        return np.sum(out)

    def find_min_theta(self) -> float:
        #theta must by construction be between 0 and 1
        res = minimize_scalar(self.residpoly, bounds=(0, 1.0))

        if not res.success:
            warnings.warn("residual poly minimization did not succeed")
            #TODO: debug message here

        _logger.debug(f"minimum theta value at: {res.x}, value {res.fun} ")
        return float(res.x), float(res.fun)

    def check_convergence(self, M: Optional[int] = None) -> bool:

        #first check numerical convergence
        R0 = self.R0_resid()
        _logger.debug(f"R0 residual: {R0}")

        #check first convergence criterion (update of residual is tiny)
        if self.R0 is None:
            self.R0 = R0
        else:
            E = (self.R0 - R0) / self.R0
            _logger.debug(f"E update value: {E}")
            self.R0 = R0
            if E < self._R0_epsilon:
                _logger.debug("convergence achieved via R0 value")
                return True

        #minimum of P(theta) is >= 0 (2nd convergence criteria)
        if self.minPtheta is not None and self.minPtheta >= self._P0_epsilon:
            _logger.debug("convergence achieved via P0 minimum value")
            return True

        #we demanded a specific number of terms, carry on if that's not met
        if M is not None:
            if len(self.thetas) <= M:
                return False

        return False

    def do_linear_fit(self) -> list[float]:
        powmat = np.tile(self.n, (len(self.thetas), 1))
        A = (np.asarray(self.thetas)[:, np.newaxis] ** powmat).T
    
        x, resid, rank, s = lstsq(A, self.data)
        #TODO: check conditioning here!
        _logger.debug(f"x in do_linear_fit: {x}")
        return list(float(_) for _ in x)


    def linear_fit_stage(self):
        amps = self.do_linear_fit()
        dropidx, amptrim = self.drop_zero_term(self.a, amps)

        ndrops: int = 0
        if dropidx is None:
            self.a = amptrim
        while dropidx is not None:
            del self.thetas[dropidx]
            amps = self.do_linear_fit()
            dropidx, amptrim = self.drop_zero_term(self.a, amps)
            self.a = amps
            ndrops +=1

        _logger.debug(f"n drops: {ndrops}")
        
    
    def drop_zero_term(self, amps_old: list[float], amps_new: list[float]) -> tuple[Optional[int],Optional[list[float]]]:
        """ perform "zero-trimming" procedure on amplitude coefficients as described in section 2(g) of Wiscombe & Evans

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

        #ensure last term of amps_old is 0 as required by algorithm
        if len(amps_old) == 0 or amps_old[-1] != 0:
            amps_old.append(0)
        
        for idx, (ao, an) in  enumerate(zip(amps_old, amps_new)):
            if an < 0:
                beta: float = ao / (ao - an)
                if beta < min_beta:
                    min_beta = beta
                    min_idx = idx
        
        amps_out = [(1 - min_beta) * ao + min_beta * an for (ao, an) in zip(amps_old, amps_new)]
        del[amps_out[min_idx]]

        return min_idx, amps_out


    def iterate_fit(self, M: Optional[int] = None, max_iters: int = 200) -> Generator:
        niter: int = 0
        while not self.check_convergence(M) and niter < max_iters:
            new_theta, minPtheta = self.find_min_theta()
            self.minPtheta = minPtheta
            self.thetas.append(new_theta)
            self.linear_fit_stage()

            _logger.debug(f"thetas: {self.thetas}")
            _logger.debug(f"as: {self.a}")

            yield self.thetas, self.a
            niter +=1
            _logger.debug(f"niter: {niter}")

        if niter >= max_iters:
            _logger.debug("max_iters reached!")

