"""
Draws a synthetic photon transfer curve to check PTCCurvePlotter's seaborn
integration (multi-series hue/legend, log-log scaling).

    pytest tests/test_ptc_visualization.py -m viz -v

Output PNGs are written to tests/output/
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from eregion.datamodels import ImageBundle
from eregion.tasks.ptc import PTCResult
from eregion.plotting import PTCCurvePlotter

OUTPUT_DIR = Path(__file__).parent / "output"


def _make_ptc_result(outputs, gains, read_noise, n_points=15):
    """Build a PTCResult with a synthetic ptc_table, without running the PTC task."""
    rng = np.random.default_rng(0)
    mean = np.logspace(1, 4.5, n_points)

    rows = []
    for output, gain, noise in zip(outputs, gains, read_noise):
        shot_noise_variance = mean / gain
        true_variance = shot_noise_variance + noise ** 2
        # std_diff is halved in quadrature by the task (see PTC.make_ptc_table),
        # so undo that here to land back on true_variance.
        std_diff = np.sqrt(true_variance * 2) * rng.normal(1.0, 0.02, n_points)
        rows.append(pd.DataFrame({
            "det_id": "TEST_DET",
            "output": output,
            "exptime": np.linspace(0.1, 100, n_points),
            "mean": mean,
            "std_diff": std_diff / np.sqrt(2),
        }))

    ptc_table = pd.concat(rows, ignore_index=True)
    return PTCResult(ptc_table=ptc_table, diff_images=ImageBundle())


@pytest.mark.viz
def test_plot_ptc_curve_single_output():
    result = _make_ptc_result(outputs=["A"], gains=[2.0], read_noise=[5.0])

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "ptc_single_output.png"
    ax = PTCCurvePlotter(result).show(save=str(out_path))
    plt.close(ax.figure)
    assert out_path.exists()


@pytest.mark.viz
def test_plot_ptc_curve_multi_output():
    result = _make_ptc_result(
        outputs=["A", "B", "C", "D"],
        gains=[2.0, 1.8, 2.2, 1.9],
        read_noise=[5.0, 4.5, 5.5, 6.0],
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "ptc_multi_output.png"
    ax = PTCCurvePlotter(result).show(save=str(out_path))
    assert ax.get_legend() is not None
    plt.close(ax.figure)
    assert out_path.exists()
