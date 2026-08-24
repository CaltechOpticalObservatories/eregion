from abc import ABC, abstractmethod
from typing import Optional

from eregion.datamodels import TaskResult
from eregion.utils import configure_logger


class Plotter(ABC):
    """
    Base class for rendering a TaskResult onto a matplotlib Axes.

    Plotter is the presentation half of eregion's compute/plot split: a Task
    produces a plain TaskResult with no plotting dependency, so it stays
    serializable and can run headlessly in a pipeline
    """
    result_cls: type[TaskResult] = TaskResult

    def __init__(self, result: TaskResult, **kwargs):
        if not isinstance(result, self.result_cls):
            raise TypeError(f"{type(self).__name__} expects a {self.result_cls.__name__}, "
                             f"got {type(result).__name__}.")
        self.result = result
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
