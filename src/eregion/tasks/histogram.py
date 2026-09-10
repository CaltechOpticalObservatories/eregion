from typing import Any, Optional
import numpy as np
from pydantic import Field, field_validator

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task

def numpy_histogram(data: np.ndarray, bins: int | str | np.ndarray = "auto", **kwargs) -> dict[str, Any]:
    """
    Default histogram-binning method (wrapper around numpy.histogram)

    :param data: np.ndarray
        1-D sample of values to histogram
    :param bins: int | str | np.ndarray
        Default 'auto' picks a bin count heuristically (see numpy docs)
    :param kwargs:
        Additional keyword arguments 
    :return: dict
        counts: np.ndarray, histogram bin counts
        bin_edges: np.ndarray, histogram bin edges (len(counts) + 1)
    """
    data = np.asarray(data).ravel()
    counts, bin_edges = np.histogram(data, bins=bins, **kwargs)
    return {"counts": counts, "bin_edges": bin_edges}


class HistogramResult(TaskResult):
    counts: np.ndarray = Field(description="Histogram bin counts")
    bin_edges: np.ndarray = Field(description="Histogram bin edges (len(counts) + 1)")
    label: Optional[str] = Field(default=None, description="Optional label describing what was histogrammed, "
                                                             "e.g. for use as a plot title")
    diagnostics: dict[str, Any] = Field(default_factory=dict,
                                         description="Any extra outputs from the binning method beyond "
                                                      "counts/bin_edges, e.g. bin_width, entropy_bits, efficiency "
                                                      "for entropy-optimal binning.")

    @field_validator("counts", "bin_edges", mode="before")
    @classmethod
    def coerce_array(cls, inp):
        if isinstance(inp, np.ndarray):
            return inp
        return np.asarray(inp)


class HistogramTask(Task):
    """
    Compute histogram data for image data or a 1-D series
        - 'numpy': plain numpy histogram
          (default 'auto')
        - 'entropy_optimal': Watts & Crow's Shannon-entropy-based optimal bin width
    This task only computes histogram data
    To draw the resulting HistogramResult, use eregion.plotting.HistogramPlotter
    """
    task_result = HistogramResult

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        :param name: Optional[str]
            Name of the task. Default is the current class name.

        kwargs:

        - method: str
            Name of the binning method to use. Default is 'numpy'
        - any other kwargs are forwarded to the chosen method as binning options
        """
        if "method" not in kwargs:
            kwargs["method"] = "numpy"
        super().__init__(name=name, **kwargs)

    @property
    def methods(self):
        """
        Return a dictionary of available histogram binning methods
        :return: dict
            Dictionary with method names as keys and function signatures as values
        """
        return {
            "numpy": numpy_histogram,
            "entropy_optimal": "core.entropy.entropy_optimal_histogram",
        }

    def run(self, data: np.ndarray, label: Optional[str] = None, **method_kwargs) -> HistogramResult:
        """
        Compute histogram data for the given data using the task's selected method
        :param data: np.ndarray
            Image data or 1-D series to histogram
        :param label: Optional[str]
            Optional label describing what was histogrammed
        :param method_kwargs:
            Additional keyword arguments
        :return: HistogramResult
        """
        data = np.asarray(data).ravel()

        hist_out = dict(self.method(data, **method_kwargs))
        counts = hist_out.pop("counts")
        bin_edges = hist_out.pop("bin_edges")

        self.logger.info(f"Computed histogram with {len(counts)} bins from {data.size} samples "
                          f"using method '{self.method_name}'.")
        return self.task_result(counts=counts, bin_edges=bin_edges, label=label, diagnostics=hist_out)
