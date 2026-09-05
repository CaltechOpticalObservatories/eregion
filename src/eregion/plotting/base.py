from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Optional, TypeVar

from eregion.datamodels import TaskResult
from eregion.plotting.descriptor import PlotDescriptor
from eregion.utils import configure_logger

# Bound, not the result_cls attribute name itself
TResult = TypeVar("TResult", bound=TaskResult)


class Plotter(ABC, Generic[TResult]):
    """
    Base class for drawing one TaskResult (a task's computed output) onto one
    matplotlib Axes.

    Tasks compute data, and Plotters draw it. Keeping the two separate means a
    Task's output stays plain data; it can be saved, reloaded, and run
    without ever needing matplotlib.

    This class only knows how to draw one result onto one set of axes. Larger
    plots meaning several panels side by side (e.g. a linearity plot next to a PTC
    curve with residuals underneath), or several labelled data series sharing
    one legend, are meant to be built by combining a few Plotters together.
    """
    # Which TaskResult subclass this Plotter draws, e.g. HistogramResult. Checked
    # when a Plotter is created, to catch being handed the wrong kind of data.
    # Each subclass sets its own value, e.g. HistogramPlotter sets
    # `result_cls = HistogramResult`.
    result_cls: type[TResult] = TaskResult  # type: ignore[assignment]
    # Default look of this Plotter's plot: axis labels, title, scale, etc. Each
    # subclass can set its own sensible defaults here. Anyone creating a Plotter
    # can also pass in their own `descriptor` to change these without writing a
    # new subclass.
    descriptor_cls: ClassVar[type[PlotDescriptor]] = PlotDescriptor

    def __init__(self, result: TResult, descriptor: Optional[PlotDescriptor] = None, **kwargs):
        if not isinstance(result, self.result_cls):
            raise TypeError(f"{type(self).__name__} expects a {self.result_cls.__name__}, "
                             f"got {type(result).__name__}.")
        self.result = result
        self.descriptor = descriptor if descriptor is not None else self.descriptor_cls()
        self.meta = kwargs
        self.logger = configure_logger(type(self).__name__)

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        """
        Draw self.result onto ax, creating a new Axes if one isn't given.
        :param ax: Optional[matplotlib.axes.Axes]
        :param kwargs:
            Subclass-specific styling options, forwarded to matplotlib
        :return: matplotlib.axes.Axes
        """

    def show(self, ax=None, save: Optional[str] = None, **kwargs):
        """
        Convenience wrapper around plot(): draw self.result, optionally save the figure.
        :param ax: Optional[matplotlib.axes.Axes]
        :param save: Optional[str]
            If given, path to save the resulting figure to
        :param kwargs:
            Forwarded to plot()
        :return: matplotlib.axes.Axes
        """
        ax = self.plot(ax=ax, **kwargs)
        if save is not None:
            ax.figure.savefig(save)
        return ax
