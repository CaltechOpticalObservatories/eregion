"""
Builds entropy-optimal histograms for a few example distributions 
    pytest tests/test_entropy_visualization.py -m viz -v

Output PNGs are written to tests/output/
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest

from eregion.tasks.histogram import HistogramTask
from eregion.plotting import HistogramPlotter

OUTPUT_DIR = Path(__file__).parent / "output"


def _plot_and_save(data, title, filename):
    task = HistogramTask(method="entropy_optimal")
    result = task.run(data, M=2.5)

    diag = result.diagnostics
    result.label = (
        f"{title}\n"
        f"$\\Delta$={diag['bin_width']:.4g}, H={diag['entropy_bits']:.2f} bits, "
        f"e={diag['efficiency']:.2f}, M$_{{bin}}$={diag['M_bin']:.2f}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / filename
    ax = HistogramPlotter(result).show(save=str(out_path))
    plt.close(ax.figure)
    print(f"Saved {out_path}")
    return out_path


@pytest.mark.viz
def test_plot_entropy_histogram_normal():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 20_000)
    out_path = _plot_and_save(sample, "Standard normal (N=20,000)", "normal.png")
    assert out_path.exists()


@pytest.mark.viz
def test_plot_entropy_histogram_exponential():
    rng = np.random.default_rng(1)
    sample = rng.exponential(1.0, 20_000)
    out_path = _plot_and_save(sample, "Exponential, mean=1 (N=20,000)", "exponential.png")
    assert out_path.exists()


@pytest.mark.viz
def test_plot_entropy_histogram_bimodal():
    rng = np.random.default_rng(2)
    sample = np.concatenate([rng.normal(-3, 1, 10_000), rng.normal(3, 1, 10_000)])
    out_path = _plot_and_save(sample, "Bimodal normal mixture (N=20,000)", "bimodal.png")
    assert out_path.exists()
