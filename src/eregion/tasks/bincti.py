from typing import Optional, Literal

import numpy as np
from eregion.tasks import LazyTask
from eregion.datamodels import TaskResult
from eregion.datamodels.image import DetImage
from eregion.utils.dangerous_magic import pack_argument_helper

class CTIEPERExtractResult(TaskResult):
    pass


class CTIEPERExtractor(LazyTask):
    task_result = CTIEPERExtractResult
    _axis: Literal["parallel", "serial"] | None = None

    def __init__(self, first_bright_line: int, bright_line_spacing: int,
                 N_bright_lines: int,
                 axis: Literal["parallel", "serial"],
                 n_before_bright: int = 0,
                  trim_trace: int = 0,
                  trace_type: Literal["mean", "median"] = "mean",
                  do_sigma_clip: bool = True,
                  sigma_clip_args: Optional[dict] = None,
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
        
        :param n_before_bright: int
            the number of lines before the bright line to include in EPER trails (e.g. for assessing full well by forward bleed)

        :param trim_trace: int
            how many excess values to trim from the end of the trace.

        :param trace_type: Literal["mean", "median"]
            the method to use to extract traces. Default is mean after sigma clipping

        :param sigma_clip_args: dict
            extra arguments to pass to sigma clipping function. The default is
            {"sigma" : 5.0, "grow" : 10, "stdfunc" : "mad_std"}

        """
        
        if name is None:
            name = type(self).__name__

        kwargs = pack_argument_helper(selfarg=self)
        super().__init__(name=name, **kwargs)

        if axis not in ["parallel", "serial"]:
            raise ValueError("require axis be the literal value 'parallel' or 'serial'")

        if trace_type not in ["mean", "median"]:
            raise ValueError("require the trace type to be the literal value 'mean' or 'median'")

        self.N_bright_lines = N_bright_lines
        self.bright_line_spacing = bright_line_spacing
        self.first_bright_line = first_bright_line


    def run(self, img: DetImage) -> CTIEPERExtractResult:
        for opid, output in img.outputs.items():
            imgdat, imgmask = output.get_image_region(return_masks=True)
            #sum along the 
