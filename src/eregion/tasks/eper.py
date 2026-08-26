from typing import Optional

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task

from eregion.core.expsum_fit_math import ExpSumFitter

from pydantic import Field

Number = int | float

class EPERFitUncalibratedResult(TaskResult):
    """results of a single EPER trail fit.  All units are uncalibrated."""
    TDC: float = Field(description = "total deferred chargee")
    relTDC: float = Field(description = "relative TDC")
    CTI_jan: float = Field(description = "CTI, Janesick approximation")
    CTI: float = Field(description = "CTI, full equation")
    signal_level: float = Field(description = "signal level, from input data")
    trap_rates: list[float] | None = Field(description = "fitted trap decay rates")
    trap_amplitudes: list[float] | None = Field(description = "fitted trap amplitudes")
    eper_baseline: Number | None  = Field(description = "EPER trail baseline subtracted")


class SingleEPERTrailFitter(Task):
    task_result = EPERFitUncalibratedResult

    def __init__(self, N_transfers: int, do_trapfit: bool=True, N_trap_species: Optional[int] = None, decay_rate_tolerance: float = 0.25, name: Optional[str] = None, subtract_eper_zeros: Optional[int] = None,  **kwargs):
        """Task that fits a single EPER trail, reporting total deferred charge, CTI estimates, and fitted trap decay curves

        parameters
        ----------

        :param N_transfers: int
           number of transfers the charge has gone through in this EPER measurement (e.g image size in appropriate direction for flatfield, half image size for split TDC bright line frame, etc)

        :param do_trapfit: bool
           whether to do the fit of summed exponential decays to extract trap parameters
        
        :param N_trap_species: Optional[int]
           how many trap species to fit. If supplied, limits exponential fit to this number of trap species. If unspecified, fit continues until convergence
        
        :param decay_rate_tolerance: float
            when N_trap_species is None, exponential fit terms will be coalesced until there are no decay rates closer together in ratio than this number. This is useful because the trap fit tends to produce many species close together in decay rate when unconstrained in number. When N_trap_species is specified, this value does nothing

        :param name: Optional[str]
            name of the task. Default is current class name

        :param subtract_eper_zeros: Optional[int]
            number of values at the end of the EPER trail to average as the zero level. If not supplied, assume the EPER trails are already properly pre-processed (e.g. median line-by-line bias subtracted or similar)

        """

        if name is None:
            name = type(self).__name__

        super().__init__(name=name, **kwargs, N_trap_species=N_trap_species, decay_rate_tolerance=decay_rate_tolerance, subtract_eper_zeros=subtract_eper_zeros, N_transfers=N_transfers)

        self.decay_rate_tolerance = decay_rate_tolerance
        self.N_trap_species = N_trap_species
        self.subtract_eper_zeros = subtract_eper_zeros
        self.N_transfers = N_transfers

    def run(self, siglevel: float, eper_trail: np.ndarray, n_transfers: int) -> EPERFitUncalibratedResult:
        results = dict()

        assert len(eper_trail.shape) == 1, "eper_trail should be 1D"
        self.logger.debug("calculating TDC and CTI estimates...")

        
        if self.subtract_eper_zeros is not None:
            self.logger.debug("subtracting baseline from EPER trail")
            eper_offset = np.mean(eper_trail[self.subtract_eper_zeros:])
            eper_trail -= eper_offset
            results["eper_baseline"] = eper_offset

        results["signal_level"] = siglevel
        TDC = float(np.sum(eper_trail))
        relTDC = TDC / (siglevel + TDC)
        results["TDC"] = TDC
        results["relTDC"] = relTDC

        #basic CTI estimates from TDC
        results["CTI_jan"] = TDC / (siglevel * self.N_transfers)
        results["CTI"] = 1 - np.exp(np.log(1-relTDC) / self.N_transfers)
                                    
        #run exponential sum fit
        if self.do_trapfit:
            efitter = ExpSumFitter(eper_trail, dU=1.0)
            self.logger.info("running exponential decay sum fit")
            iters = efitter.run_fit(M=self.N_trap_species)

            self.logger.debug(f"fit iterations: {iters}")
            self.logger.debug("fit values before coalescence...")
            self.logger.debug(f"thetas: {efitter.thetas}, k: {efitter.ks}, a: {efitter.a}")

            if self.N_trap_species is None:
                self.logger.info("coalescing fit terms to tolerance limit")
                citer = efitter.coalesce_to_tol(self.decay_rate_tolerance)
                self.logger.debug(f"coalescence iterations: {citer}")
            else:
                self.logger.info("coalescing fit to fixed number of terms")
                citer = efitter.coalesce_to_fixedM(M=self.N_trap_species)
                self.logger.debug(f"coalescence iterations: {citer}")

            results["trap_rates"] = efitter.ks
            results["trap_amplitudes"] = efitter.a
        return EPERFitUncalibratedResult(**results)

