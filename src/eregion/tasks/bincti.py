from typing import Optional, Literal

import numpy as np
from eregion.tasks import LazyTask, Task
from eregion.datamodels import TaskResult
from eregion.datamodels.image import DetImage
from eregion.utils.dangerous_magic import pack_argument_helper
from astropy.stats import sigma_clipped_stats
from collections import defaultdict
import pandas as pd

class CTIEPERExtractResult(TaskResult):
    axis: Literal["parallel", "serial"]
    fluxes: list[float]
    EPER_trails: list[np.ndarray]


class CTIEPERExtractor(Task):
    task_result = CTIEPERExtractResult
    _axis: Literal["parallel", "serial"] | None = None

    def __init__(self, first_bright_line: int, bright_line_spacing: int,
                 N_bright_lines: int,
                 axis: Literal["parallel", "serial"],
                 N_before_bright: int = 0,
                  trim_trace: int = 0,
                  trace_type: Literal["mean", "median"] = "mean",
                  do_sigma_clip: bool = True,
                  sigma_clip_args: Optional[dict] = None,
                  name: str = None,
                  **kwargs):
        """Extract EPER (and full well evidence) trails from binned CTI data plots.

        parameters
        ----------

        :param first_bright_line: int
            location (along readout direction) of the first bright CTI line in the image

        :param bright_line_spacing: int
            the spacing between bright CTI lines

        :param N_bright_lines: int
            The number of bright lines to extract

        :param axis: Literal["parallel", "serial"]
            which axis to measure the CTI along (e.g. if you choose "parallel", it means you have  done a binned parallel CTI measurement, profiles will be summed along the
            serial axis to measure parallel CTI.
        
        :param N_before_bright: int
            the number of lines before the bright line to include in EPER trails (e.g. for assessing full well by forward bleed)

        :param trim_trace: int
            how many excess values to trim from the end of the trace.

        :param trace_type: Literal["mean", "median"]
            the method to use to extract traces. Default is mean after sigma clipping

        :param sigma_clip_args: dict
            extra arguments to pass to sigma clipping function. The default is
            {"sigma" : 5.0, "grow" : 10, "cenfunc" : "mean", "stdfunc" : "std"}

        """

        if name is None:
            name = type(self).__name__

        kwargs_sup = pack_argument_helper(selfarg=self)
        super().__init__(**kwargs_sup)

        if axis not in ["parallel", "serial"]:
            raise ValueError("require axis be the literal value 'parallel' or 'serial'")

        if trace_type not in ["mean", "median"]:
            raise ValueError("require the trace type to be the literal value 'mean' or 'median'")

        self.N_bright_lines = N_bright_lines
        self.bright_line_spacing = bright_line_spacing
        self.first_bright_line = first_bright_line

        self.N_before_bright = N_before_bright
        self.trim_trace = trim_trace
        self.trace_type = trace_type
        self.axis = axis
        upd_sc_args = sigma_clip_args if sigma_clip_args is not None else dict()

        self.sigma_clip_args = {"sigma": 5.0, "grow": 10,  "cenfunc": "mean",
                                "stdfunc": "std"} | upd_sc_args


    def run(self, img: DetImage) -> CTIEPERExtractResult:
        outdat = defaultdict(list)
        for opid, output in img.outputs.items():
            imgdat, imgmask = output.get_image_region(return_masks=True)

            # sigma clipped stats along correct axis
            # axis is opposite to the one we want CTI measured from
            ax = output.serial_axint if self.axis == "parallel" else output.parallel_axint
            mean, median, std = sigma_clipped_stats(imgdat, **self.sigma_clip_args, axis=ax)


