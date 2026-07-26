

from datamodels import TaskResult
from ..tasks.task import Task
from ..tasks.ptc import PTCResult
from ..utils.pydantic import generate_iterable_model
from ..core.ptc_fit_math  import (find_adc_sat_index, find_rough_full_well,
                                  trad_ptc_shot_noise_fit, astier_approx_one_param_fit,
                                  linearity_fit)
from typing import Optional, Annotated, Iterable, TypeVar, Generator, Any
from pydantic import Field
import numpy as np
import pandas as pd
from astropy import units as un
import uncertainties as unc
from uncertainties import umath
from uncertainties.core import Variable as uVar
import pint
from pydantic_pint import PydanticPintQuantity, set_registry
from itertools import product
from enum import Enum
from pydantic import field_serializer
import os




_Table = TypeVar("Table")

#unit stuff setup
_ureg = pint.get_application_registry()
_Q = _ureg.Quantity
if "DN" not in _ureg:
    _ureg.define("DN = ")
if "elec" not in _ureg:
    _ureg.define("@alias e = elec")

set_registry(_ureg)


_PPQ = PydanticPintQuantity

class CCDPTCFitResult(TaskResult):
    sat_val: Annotated[_Q, _PPQ("DN", ureg=_ureg)] | None = Field(description="estimated ADC saturation value, if found")
    full_well: Annotated[_Q, _PPQ("DN", ureg=_ureg)] = Field(description="estimated full well value")

    camera_gain_classic: Annotated[_Q, _PPQ("elec / DN", ureg=_ureg)] = Field(description="camera gain from classic PTC fit")
    camera_gain_BFE: Annotated[_Q, _PPQ("elec / DN", ureg=_ureg)] | None = Field(description="camera gain from more advanced BFE fit", default=None)
    ptc_noise_classic: Annotated[_Q, _PPQ("elec")] = Field(description="noise estimated from classical PTC fit")
    ptc_noise_BFE: Annotated[_Q, _PPQ("elec")] | None = Field(description="noise estimated from more advanced BFE fit", default=None)
    ptc_a00_classic: uVar | None = Field(description="estimate of brighter-fatter a00 from classic PTC fit, if requested", default= None)
    ptc_a00_astier: uVar | None = Field(description="estimate of brighter-fatter a00 from Astier's approximate fit, if requested", default=None)

    bfe_aij: np.ndarray | None = Field(description="brighter-fatter a matrix from full Astier fit, if requested", default=None)
    bfe_bij: np.ndarray | None = Field(description="brighter-fatter b matrix from full Astier fit, if requested", default=None)

    flux_rate: Annotated[_Q, _PPQ("DN / s", ureg=_ureg)] = Field(description="flux rate estimated from linearity fit")
    linearity_offset: Annotated[_Q, _PPQ("DN", ureg=_ureg)] = Field(description="zero offset of the linearity fit")
    #linearity_resid_norm: float = Field(description="L2 norm of the linearity residuals, a measure of linearity in some sense", deafult=None)
    #ptc_mad_resid_norm: float = Field(description="L2 norm of the residuals between PTC and median-MAD curve, diagnostic for tearing or vignetting not removed by sigma clipping", default=None)

    @field_serializer("ptc_a00_classic", "ptc_a00_astier")
    def serialize(self, value: uVar) -> str:
        #TODO: serialize to proper float when in python mode
        return f"{value:.2u}"


class CCDPTCFitResultCollection(TaskResult):
    fits: dict[tuple[str,...], CCDPTCFitResult] = Field(description="dictionary mapping detector channel identifiers to PTC results")

    def save(self, filepath: str) -> None:
        os.makedirs(filepath, exist_ok=True)
        outres = os.path.join(filepath, "ptc_fit_results.json")

        with open(outres, "w+") as f:
            jsondat = self.model_dump_json(indent=2)
            f.write(jsondat)

        super().save(filepath)


_idxfld = Field(description="detector output ID")
_idxtp = list[str]
TabularCCDPTCFitResultCollection = generate_iterable_model(CCDPTCFitResult, "TabularCCDPTCFitResultCollection",
                                                           "output_id", _idxtp, _idxfld, arbitrary_types_allowed=True)



class BrighterFatterFitTypes(Enum):
    NO_FIT = 0
    ASTIER_ONE_PARAM = 1
    ASTIER_FULL = 2


class CCDPTCFit(Task):
    task_result = CCDPTCFitResultCollection

    def __init__(self, selection_columns: list[str], brighter_fatter: BrighterFatterFitTypes, name: Optional[str] = None,
                 fwfact: float = 0.8, lincoff: float = 0.2,  **kwargs):
        """Fit results of a PTC reduction in a CCD specific manner.

        Calculates usual results of a PTC test, namely:
            - camera gain
            - noise *(note 1)
            - full well *(note 2)
            - ADC saturation point
            - linearity *(note 3)
            - (optionally) brighter fatter coefficients *(note 4)
            - illumination rate
            - divergence of mean-variance from median-MAD curve

        note 1: this class only calculates noise as fitted with PTC functions. For some cases, estimation from overscans
        is a better method. This will be covered by a separate class

        note 2: This is a rough full well estimate based on the turnover point of the mean-variance curve.
        It is an estimate only and various more advanced methods will be covered in other tasks

        note 3: This is the classic "Janesick" linearity. Better results can be obtained from a specific "linbin" test

        note 4: currently only Astier's approximate one-parameter fit is implemented, which supplies an estimate for the
        dominant brighter-fatter coefficien a_00. In future, this task will also implement Astier's full covariance matrix
        fitting method to estimate the whole brighter fatter a and b matrices.

        Parameters
        ----------

        :param selection_columns: list[str]
            the names of columns which should be selected for separating PTC curves. For example, to iindividually process curves
            on columns labelled "det_id" and "output", use selection_columns=["det_id", "output"]

        :param brighter_fatter: BrigherFatterFitTypes
           choose what type of brighter fatter fit to do.
           NO_FIT: no brighter fatter fit, do linear fit only (not recommended on thick detectors)
           ASTIER_ONE_PARAM: do Astier's simplified one parameter fit
           ASTIER_FULL: do Astier's full multi-covariance fit
                   - NOT IMPLEMENTED YET
                   - requires power spectral density column in the PTC result


        :param name: Optional[str]
            the name of the task. Default is the current class name

        :param fwfact: float
            proportion of the full well up to which to do the classic and Astier PTC fits. Necessary because PTC models do not
            work above full well. Usually a value of 0.8 or 0.9 is a good compromise between accuracy and robustness for "clean" PTC data
        :param lincoff: float
           proportion of the total flux range over which to evaluate the linearity. Most modern CCDs are highly linear but
           exhibit large nonlinearities at low values, so to assess in one number it is sensible to only consider residuals
           above a certain proportion.

        :param return_resids: bool
           return the full residual arrays. Adds a lot of data to the result, but very useful for plotting

        """

        if name is None:
            name = type(self).__name__

        super().__init__(name=name, **kwargs)
        self.selection_columns = selection_columns
        self.brighter_fatter = brighter_fatter
        self.fwfact = fwfact


    @classmethod
    def subselect_data_from_table(cls, table: _Table, **selkeyvals: dict[str, str]) -> _Table:
        out = np.ones(len(table), dtype=np.bool)

        for keyname, keyval in selkeyvals.items():
            out = np.logical_and(table[keyname] == keyval, out)

        return table[out]

    def iter_ptc_curves(self, table) -> Generator[_Table, None, None]:
        selkeysets: dict[str, list[str]] = dict()
        for selkeyname in self.selection_columns:
            keyvals = set(table[selkeyname])
            selkeysets[selkeyname] = keyvals

        self.logger.info(f"selection key values: {selkeysets}")

        for selkeyvals in product(*selkeysets.values()):
            kd = {k: v for k, v in zip(selkeysets.keys(), selkeyvals)}
            self.logger.debug(f" selecting curve with key values: {kd}")

            seldat = self.subselect_data_from_table(table, **kd)
            yield seldat, selkeyvals

    def run(self, inp: PTCResult) -> CCDPTCFitResultCollection:

        results = dict()

        dat: pd.DataFrame
        for dat, skvs in self.iter_ptc_curves(inp.ptc_table):
            resultdct: dict[str, Any]  = dict()
            #TODO: should these be configurable? proabbly not needed

            self.logger.debug("finding ADC saturation and full well")
            SAT_SIGMA: float = 5.0 #hardcode for now, not sure it'll ever be different

            satidx  = find_adc_sat_index(dat["exptime"], dat["mean"], SAT_SIGMA)
            if satidx is None:
                satval = None
                self.logger.info("couldn't find an ADC saturation value")
            else:
                satval = _Q(float(dat["mean"].iloc[satidx]) * _ureg.DN)
                self.logger.debug(f"saturation value: {satval:.0f}")

            resultdct["sat_val"] = satval

            #NOTE: dat["std_diff"] is already divided by sqrt(2)
            fwfactloc, fwloc = find_rough_full_well(dat["mean"].array, dat["std_diff"].array, self.fwfact)
            resultdct["full_well"]  = _Q(float(dat["mean"].iloc[fwloc]) *_ureg.DN)

            self.logger.info("doing PTC fits...")
            match self.brighter_fatter:
                case BrighterFatterFitTypes.NO_FIT:
                    self.logger.debug("doing PTC shot noise fit with no brighter-fatter correction")
                    Kest, nest = trad_ptc_shot_noise_fit(dat["mean"].array, dat["std_diff"].array, fwfactloc, False)

                    resultdct["camera_gain_classic"]  = _Q(Kest * _ureg.elec / _ureg.DN)
                    resultdct["ptc_noise_classic"]  = _Q(nest * _ureg.elec)
                case BrighterFatterFitTypes.ASTIER_ONE_PARAM:
                    Kest, nest, a00est, Kast, a00ast, nast = self.astier_fit_bootstrap(dat["mean"].array, dat["std_diff"].array,
                                                                                       fwfactloc, fwloc)
                    resultdct["camera_gain_classic"] = _Q(Kest * _ureg.elec / _ureg.DN)
                    resultdct["ptc_noise_classic"] = _Q(nest * _ureg.elec)
                    resultdct["ptc_a00_classic"] = a00est
                    resultdct["camera_gain_BFE"]  = _Q(Kast * _ureg.elec / _ureg.DN)
                    resultdct["ptc_noise_BFE"] = _Q(nast * _ureg.elec)
                    resultdct["ptc_a00_astier"] = a00ast
                case _:
                    raise NotImplementedError("requested fit type is not implemented")


            self.logger.info("doing linearity fit...")
            linfit, linerr = linearity_fit(dat["exptime"].array, dat["mean"].array, satidx)
            resultdct["flux_rate"]  = _Q(unc.ufloat(linfit[1], linerr[1]) * _ureg.DN / _ureg.s)
            resultdct["linearity_offset"] = _Q(unc.ufloat(linfit[0], linerr[0]) * _ureg.DN)

            results[skvs] = CCDPTCFitResult(**resultdct)

        return CCDPTCFitResultCollection(fits=results)



    def astier_fit_bootstrap(self, mndat, sddat, fwfactloc, fwloc) -> tuple[uVar]:
        self.logger.debug("doing PTC shot noise fit with brighter-fatter correction")
        Kest, nest, a00est = trad_ptc_shot_noise_fit(mndat, sddat, fwfactloc, True)
        self.logger.debug("doing Astier approximated one-parameter brighter-fatter fit")
        Kast, a00ast, nast = astier_approx_one_param_fit(mndat, sddat,
                                                         Kest.nominal_value, a00est.nominal_value,
                                                         nest.nominal_value, fwloc)

        return Kest, nest, a00est, Kast, a00ast, nast



class CCDPTCFitTabular(CCDPTCFit):
    task_result = TabularCCDPTCFitResultCollection
    def run(self, inp: PTCResult) -> TabularCCDPTCFitResultCollection:
        self.logger.debug("running base class task")
        base_results = super().run(inp)

        #make an empty tabular result
        tabular_results = self.task_result.model_construct()

        tabular_results.output_ids = list(base_results.fits.keys())
        for result in list(base_results.fits.values()):
            tabular_results.add_result_item(result)

        return tabular_results
