import matplotlib.pyplot as plt

from eregion.tasks.histogram import HistogramResult
from eregion.plotting.base import Plotter


class HistogramPlotter(Plotter):
    """
    Draw a HistogramResult (computed by eregion.tasks.histogram.HistogramTask) as a
    step/stairs plot.
    """
    result_cls = HistogramResult

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

        ax.set_xlabel("Value")
        ax.set_ylabel("Counts")
        if self.result.label:
            ax.set_title(self.result.label)
        return ax
