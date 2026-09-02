from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt

from eregion.tasks.histogram import HistogramResult
from eregion.plotting.base import Plotter
from eregion.plotting.descriptor import PlotDescriptor


@dataclass
class HistogramPlotDescriptor(PlotDescriptor):
    xlabel: Optional[str] = "Value"
    ylabel: Optional[str] = "Counts"


class HistogramPlotter(Plotter[HistogramResult]):
    """
    Draw a HistogramResult (computed by eregion.tasks.histogram.HistogramTask) as a
    step/stairs plot.
    """
    result_cls = HistogramResult
    descriptor_cls = HistogramPlotDescriptor

    def plot(self, ax=None, **kwargs):
        """
        :param ax: Optional[matplotlib.axes.Axes]
        :param kwargs:
            Forwarded to matplotlib.axes.Axes.stairs. Default is fill=True.
        :return: matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)

        stairs_kwargs = {"fill": True}
        stairs_kwargs.update(kwargs)
        ax.stairs(self.result.counts, self.result.bin_edges, **stairs_kwargs)

        ax.set_xlabel(self.descriptor.xlabel)
        ax.set_ylabel(self.descriptor.ylabel)
        ax.set_xscale(self.descriptor.xscale)
        ax.set_yscale(self.descriptor.yscale)

        # HistogramResult.label is task data (what was histogrammed), not plot
        # metadata -- it's a reasonable default title but the descriptor wins if set.
        title = self.descriptor.resolve_title(self.result) or self.result.label
        if title:
            ax.set_title(title)
        return ax
