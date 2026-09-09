from typing import Optional, Literal

import numpy as np
from eregion.tasks import LazyTask
from eregion.datamodels import TaskResult
from eregion.datamodels.image import DetImage
from eregion.utils.dangerous_magic import pack_argument_helper

class CTIEPERExtractResult(TaskResult):
    pass


class CTIEPERExtractor(Task):
    task_result = CTIEPERExtractResult
    _axis: Literal["parallel", "serial"] | None = None

    def __init__(self, first_bright_line: int, bright_line_spacing: int,
                 axis: Literal["parallel", "serial"], trim_trace: int = 0, N_bright_lines: int = 0, name: Optional[str] = None):
        if name is None:
            name = type(self).__name__

        kwargs = pack_argument_helper(selfarg=self)
        super().__init__(name=name, **kwargs)


class ParallelCTIEPERExtractor(CTIEPERExtractor):

    def run(self, img: DetImage) -> CTIEPERExtractResult:
        for opid, output in img.outputs.items():
            imgdat, imgmask = output.get_image_region(return_masks=True)
