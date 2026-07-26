

from ..datamodels import TaskResult
from ..tasks.task import Task
from ..tasks.ptc import PTCResult
from typing import Optional, Annotated, Iterable, TypeVar, Generator
from pydantic import Field
import numpy as np
from astropy import units as un
import uncertainties as unc
from uncertainties import umath
import pint
from pydantic_pint import PydanticPintQuantity, set_registry
from itertools import product


_Table = TypeVar("Table")

#unit stuff setup
_ureg = pint.get_application_registry()
_Q = _ureg.Quantity
if "DN" not in _ureg:
    _ureg.define("DN = dimensionless")
if "elec" not in _ureg:
    _ureg.define("@alias e = elec")

set_registry(_ureg)


_PPQ = PydanticPintQuantity

class CCDPTCFitResult(TaskResult):
    camera_gain: Annotated[_Q, _PPQ("e")] = Field(description="camera gain")
    ptc_noise_classical: Annotated[_Q, _PPQ("e")] = Field(description="noise estimated from classical PTC fit")
    ptc_noise_astier_oneparam: Annotated[_Q, _PPQ("e")] = Field(description="noise estimated from Astier one-param fit")


class CCDPTCFit(Task):
    task_result = PTCResult

    def __init__(self, selection_columns: list[str], name: Optional[str] = None, do_full_brighter_fatter_fit: bool = False ,
                 fwfact: float = 0.8,  **kwargs):
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

        :param selection_columns: list[str]
            the names of columns which should be selected for separating PTC curves. For example, to iindividually process curves
            on columns labelled "det_id" and "output", use selection_columns=["det_id", "output"]

        :param name: Optional[str]
            the name of the task. Default is the current class name

        :param do_full_brigter_fatter_fit: bool
            Whether to do the full (quite expensive) Astier fit to obtain full a_ij and b_ij brighter-fatter matrix
            Currently unimplemented. Requires existence of appropriate correlation power spectral density column in the PTC dataset

        :param fwfact: float
            proportion of the full well up to which to do the classic and Astier PTC fits. Necessary because PTC models do not
            work above full well. Usually a value of 0.8 or 0.9 is a good compromise between accuracy and robustness for "clean" PTC data

        """

        if name is None:
            name = type(self).__name__

        super().__init__(name=name, **kwargs)
        self.selection_columns = selection_columns
        self.do_full_brighter_fatter_fit = do_full_brighter_fatter_fit
        self.fwfact = fwfact


    def subselect_data_from_table(self, table: _Table, *selkeyvals: Iterable[str]) -> _Table:
        out = np.ones(len(table), dtype=np.bool)

        for keyname, keyval  in zip(self.selection_columns, selkeys):
            out = np.logical_and(table[keyname] == keyval, out)

        return out

    def iter_ptc_curves(self, table) -> Generator[_Table, None, None]:
        print("hello")
        selkeysets: dict[str, set[str]] = dict()
        for selkeyname in self.selection_columns:
            keyvals = set(table[selkeyname])
            selkeysets[selkeyname] = keyvals

        self.logger.debug(f"selection key values: {selkeysets}")
        allvals = (_ for _ in selkeysets.values())


        for selkeyvals in product(allvals):
            kd = {k : v for k,v in zip(selkeysets.keys(), selkeyvals)}
            self.logger.debug(f" selecting curve with key values: {kd}")
            print(kd)



    def run(self, inp: PTCResult) -> CCDPTCFitResult:
        pass
