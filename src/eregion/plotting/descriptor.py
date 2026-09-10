from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional, Union

from eregion.datamodels import TaskResult


@dataclass
class PlotDescriptor:
    """
    Presentation metadata for a particular (TaskResult, Plotter) pairing.

    A TaskResult describes data only; a PlotDescriptor describes how one
    particular kind of plot over that data should look (axis labels, title,
    scale, preferred display units, legend text). The two are kept separate
    because a single TaskResult is often drawn by more than one Plotter -
    e.g. a CCDPTCFitResult feeds a linearity plot, a PTC curve, and a
    residuals panel, each wanting different labels/scales/titles. Baking all
    of that onto the TaskResult would mean every result carries plot
    metadata for every plot anyone might ever make from it, most of it
    unused for any given plot.

    A Plotter subclass sets a class-level default via `descriptor_cls`, so
    each concrete Plotter ships defaults for the plot it draws.
    Callers can override per-instance by passing their own PlotDescriptor
    into the Plotter constructor, e.g. to change display units, scale, or
    title without subclassing.
    """
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[Union[str, Callable[[TaskResult], str]]] = None
    xscale: str = "linear"
    yscale: str = "linear"
    # field_name -> preferred display unit string
    display_units: dict[str, str] = field(default_factory=dict)
    # Formats a multi-series key (e.g. a (det_id, output) tuple) into legend text
    legend_label: Optional[Union[str, Callable[[Hashable], str]]] = None

    def resolve_title(self, result: TaskResult) -> Optional[str]:
        """Resolve title against a concrete result, supporting a callable title."""
        if callable(self.title):
            return self.title(result)
        return self.title

    def apply(self, ax, result: TaskResult, default_title: Optional[str] = None) -> None:
        """
        Apply this descriptor's axis labels, scale, and title to ax.

        Centralizes the handful of ax.set_*() calls a Plotter.plot()
        implementation would otherwise end with, so a new basic formatting
        default only needs to be added here once.
        :param ax: matplotlib.axes.Axes
        :param result: TaskResult
            Passed through to a callable title, via resolve_title.
        :param default_title: Optional[str]
            Used if this descriptor doesn't resolve to a title of its own,
            e.g. a Plotter falling back to a title carried on the result itself.
        """
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        title = self.resolve_title(result) or default_title
        if title:
            ax.set_title(title)

    def resolve_legend_label(self, key: Hashable) -> str:
        """Resolve legend text for a multi-series key."""
        if callable(self.legend_label):
            return self.legend_label(key)
        if self.legend_label is not None:
            return self.legend_label
        return str(key)
