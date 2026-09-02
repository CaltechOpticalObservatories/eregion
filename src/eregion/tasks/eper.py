from typing import Optional, Any, Generator, Callable

import numpy as np
import pandas as pd

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task, LazyTask
from eregion.tasks.ptc import PTCResult

from eregion.core.expsum_fit_math import ExpSumFitter

from pydantic import Field
from eregion.utils.dangerous_magic import pack_argument_helper

Number = int | float

class EPERFitUncalibratedResult(TaskResult):
    """results of a single EPER trail fit.  All units are uncalibrated."""
    TDC: float = Field(description = "total deferred chargee")
    relTDC: float = Field(description = "relative TDC")
    CTI_jan: float = Field(description = "CTI, Janesick approximation")
    CTI: float = Field(description = "CTI, full equation")
    signal_level: float = Field(description = "signal level, from input data")
    trap_rates: list[float] | None = Field(description = "fitted trap decay rates", default=None)
    trap_amplitudes: list[float] | None = Field(description = "fitted trap amplitudes", default=None)
    eper_baseline: Number | None  = Field(description = "EPER trail baseline subtracted", default=None)


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

        superkwargs = pack_argument_helper(selfarg=self)
        super().__init__(**superkwargs)

        self.decay_rate_tolerance = decay_rate_tolerance
        self.N_trap_species = N_trap_species
        self.subtract_eper_zeros = subtract_eper_zeros
        self.N_transfers = N_transfers
        self.do_trapfit = do_trapfit

    def run(self, siglevel: float, eper_trail: np.ndarray) -> EPERFitUncalibratedResult:
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


class PTCEPERFitResult(TaskResult):
    """results of an EPER fit to a whole PTC dataset"""
    eper_table: pd.DataFrame
    

_EPERFitterType = dict[str, Any] | SingleEPERTrailFitter
_Extractor = str | Callable[[tuple], Any]
    
class PTCEPERFitter(LazyTask):
    task_result = PTCEPERFitResult

    def __init__(self, pass_columns: list[str], siglevelcol: _Extractor, ser_eper_col: _Extractor, llel_eper_col: _Extractor, ser_settings: _EPERFitterType, llel_settings: _EPERFitterType, name: Optional[str]=None):
        """Task that fits EPER trails from the result of a PTC.

        parameters
        ----------

        :param pass_columns: list[str]
            the names of columns which should be passed through to the output. For example, if the input PTC data has two columns ["det_id", "output"] representing CCD output names, and a column "exptime" representing flux level, we might choose pass_columns=["det_id", "output", "exptime"]

        :param siglevelcol: str | Callable[[tuple], Any]
            the name of the  column in the PTC table to base the calculation of signal levels on, or a callable that accepts a namedtuple as its only argument. In that case the signal level is calculated by calling this callable on the current namedtuple representing the current row of PTC data

        :param ser_eper_col: str | Callable[[tuple], Any]
            the name of the  column in the PTC table which contains serial EPER traces, or a callable that accepts a namedtuple as its only argument. In that case the signal level is calculated by calling this callable on the current namedtuple representing the current row of PTC data

        :param llel_eper_col: str | Callable[[tuple], Any]
            

        :param ser_settings: dict[str, Any] | SingleEPERTrailFitter
            dictionary of settings arguments that will be passed to the serial EPER fitter, or an instance of an EPER trail fitter already configured. See documentation for SingleEPERTrailFitter for details

        :param llel_settings: dict[str, Any] | SingleEPERTrailFitter
            dictionary of settings arguments that will be passed to the parallel EPER fitter, or an instance of an EPER trail fitter already configured. See documentation for SingleEPERTrailFitter for details

        :param name: Optional[str]
            the name of the task. Default is current class name

        """

        superkwargs = pack_argument_helper(selfarg=self)
        super().__init__(**superkwargs)
        
        self.pass_columns = pass_columns
        self.siglevelcol = siglevelcol
        self.ser_eper_col = ser_eper_col
        self.llel_eper_col = llel_eper_col
        
        if name is None:
            name = type(self).__name__


        ftrs = {"ser" : ser_settings,
                "llel": llel_settings}

        for ftrname, ftr in ftrs.items():
            match ftr:
                case SingleEPERTrailFitter():
                    setattr(self, f"{ftrname}_fitter", ftr)
                case dict():
                    outftr = SingleEPERTrailFitter(**ftr)
                    setattr(self, f"{ftrname}_fitter", outftr)
                case _:
                    raise TypeError(f"type of supplied argument for {ftrname} fitter is invalid.")

    def _extract_from_row(self, row: tuple, col: _Extractor) -> Any:
        match col:
            case str():
                return getattr(row, col)
            case callable():
                return col(row)
            case _:
                raise TypeError("invalid column spec, cannot extract data from row")
                
    def lazy_run(self, ptc_results: PTCResult) -> Generator[PTCEPERFitResult]:

        outcols = { k : list() for k in self.pass_columns}
        outcols |= {"siglevel" : list()}

        colbasenames = ["TDC", "relTDC", "CTI_jan", "CTI", "trap_rates", "trap_amplitudes", "_baseline"]
        outcols |= {f"ser{k}" : list() for k in colbasenames}
        outcols |= {f"llel{k}" : list() for k in colbasenames}


        for row in  ptc_results.ptc_table.itertuples(index=False):
            ident = []
            for pcol in self.pass_columns:
                v = getattr(row, pcol)
                ident.append(v)
                outcols[pcol].append(v)

            siglevel = self._extract_from_row(row, self.siglevelcol)
            outcols["siglevel"].append(siglevel)

            self.logger.info(f"running serial EPER fit on row with id {ident} and signal level: {siglevel}")
            seper = self._extract_from_row(row, self.ser_eper_col)
            seper_result = self.ser_fitter.run(siglevel, seper)

            self.logger.debug("seper_result: %s" , seper_result)
            
            self.logger.info(f"running parallel EPER fit on row with id {ident} and signal level: {siglevel}")
            peper = self._extract_from_row(row, self.llel_eper_col)
            peper_result = self.llel_fitter.run(siglevel, peper)

            self.logger.debug("peper_result: %s", peper_result)
            

        outres = PTCEPERFitResult(eper_table=pd.DataFrame())
        yield outres

            

            


