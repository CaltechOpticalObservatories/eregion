from typing import Optional

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task

from eregion.core.expsum_fit_math import ExpSumFitter

from pydantic import Field

class EPERFitUncalibratedResult(TaskResult):
    """results of a single EPER trail fit.  All units are uncalibrated."""
    TDC: float | None = Field(description = "total deferred chargee")
    CTI_jan: float | None = Field(description = "CTI, Janesick approximation")
    CTI: float | None = Field(description = "CTI, full equation")
    signal_level: float = Field(description = "signal level, from input data")
    trap_rates: list[float] = Field(description = "fitted trap decay rates")
    trap_amplitudes: list[float] = Field(description = "fitted trap amplitudes")


class FlatFieldEPERTrailFitter(Task):
    task_result = EPERFitUncalibratedResult

    def __init__(self, N_trap_species: Optional[int] = None, decay_rate_tolerance: float = 0.25, name: Optional[str] = None, subtract_eper_zeros: Optional[int] = None,  **kwargs):
        """Task that fits a single EPER trail, reporting total deferred charge, CTI estimates, and fitted trap decay curves

        parameters
        ----------

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

        super().__init__(name=name, **kwargs, N_trap_species=N_trap_species, decay_rate_tolerance=decay_rate_tolerance, subtract_eper_zeros=subtract_eper_zeros)

        self.decay_rate_tolerance = decay_rate_tolerance
        self.N_trap_species = N_trap_species
        self.subtract_eper_zeros = subtract_eper_zeros

    def run(self, siglevel: float, eper_trail: np.ndarray) -> EPERFitUncalibratedResult:
        results = dict()

        assert len(eper_trail.shape) == 1, "eper_trail should be 1D"
        self.logger.debug("calculating TDC and CTI estimates...")
        TDC = np.sum(eper_trail
