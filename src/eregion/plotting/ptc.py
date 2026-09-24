from dataclasses import dataclass
from typing import Optional

import seaborn as sns

from eregion.tasks.ptc import PTCResult
from eregion.plotting.base import Plotter
from eregion.plotting.descriptor import PlotDescriptor


@dataclass
class PTCCurvePlotDescriptor(PlotDescriptor):
    xlabel: Optional[str] = "Mean signal (DN)"
    ylabel: Optional[str] = "Variance (DN$^2$)"
    xscale: str = "log"
    yscale: str = "log"


class PTCCurvePlotter(Plotter[PTCResult]):
    """
    Draw a PTCResult's photon transfer curve: variance vs. mean signal, one
    point per (det_id, output, exptime) row of PTCResult.ptc_table.

    A PTCResult commonly holds rows for several CCD outputs (and/or det_ids)
    at once, so this is drawn with seaborn (sns.scatterplot) rather than raw
    matplotlib: passing the grouping column as `hue` tells the series apart by
    color and builds the legend, instead of a hand-rolled loop over groups.
    Seaborn's axes-level functions accept `ax=` like matplotlib does, so this
    is still just a Plotter -- one TaskResult drawn onto one Axes.
    """
    result_cls = PTCResult
    descriptor_cls = PTCCurvePlotDescriptor
    # Column in ptc_table that separates this result into multiple series,
    # e.g. distinct CCD outputs read out from the same detector.
    hue: str = "output"

    def plot(self, ax=None, **kwargs):
        """
        :param ax: Optional[matplotlib.axes.Axes]
        :param kwargs:
            Forwarded to seaborn.scatterplot.
        :return: matplotlib.axes.Axes
        """
        if ax is None:
            ax = self.new_axes()

        table = self.result.ptc_table
        variance = table["std_diff"] ** 2

        # Relabel the hue column through the descriptor up front
        hue_data = table[self.hue].map(self.descriptor.resolve_legend_label) if self.hue in table.columns else None

        sns.scatterplot(x=table["mean"], y=variance, hue=hue_data, ax=ax, **kwargs)

        self.descriptor.apply(ax, self.result)
        return ax
