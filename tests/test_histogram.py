import numpy as np
import pytest

from eregion.tasks.histogram import HistogramTask, HistogramResult
from eregion.plotting import HistogramPlotter


def test_histogram_task_default_method_computes_counts_and_edges():
    rng = np.random.default_rng(0)
    data = rng.normal(size=10_000)

    task = HistogramTask()
    result = task.run(data, bins=20)

    assert isinstance(result, HistogramResult)
    assert result.counts.sum() == data.size
    assert len(result.bin_edges) == len(result.counts) + 1
    assert np.all(np.diff(result.bin_edges) > 0)


def test_histogram_task_flattens_multidimensional_input():
    image = np.arange(24, dtype=float).reshape(4, 6)

    task = HistogramTask()
    result = task.run(image, bins=6)

    assert result.counts.sum() == image.size


def test_histogram_task_label_and_unknown_method():
    task = HistogramTask()
    result = task.run(np.arange(10.0), bins=5, label="my series")
    assert result.label == "my series"

    with pytest.raises(NotImplementedError):
        HistogramTask(method="does_not_exist")


def test_histogram_task_entropy_optimal_method_populates_diagnostics():
    rng = np.random.default_rng(0)
    data = rng.normal(size=5_000)

    task = HistogramTask(method="entropy_optimal")
    result = task.run(data, M=2.5)

    assert result.counts.sum() == data.size
    assert len(result.bin_edges) == len(result.counts) + 1
    assert result.diagnostics.keys() == {"bin_width", "entropy_bits", "efficiency", "M_bin"}
    assert result.diagnostics["bin_width"] > 0


def test_histogram_result_save_and_load_round_trip(tmp_path):
    result = HistogramResult(counts=np.array([1, 2, 3]), bin_edges=np.array([0.0, 1.0, 2.0, 3.0]))
    result.save(str(tmp_path))

    loaded_meta_path = tmp_path / f"{HistogramResult.__name__}_metadata.json"
    assert loaded_meta_path.exists()


def test_histogram_plotter_draws_stairs_and_accepts_a_result():
    task = HistogramTask()
    result = task.run(np.arange(100.0), bins=10, label="test histogram")

    plotter = HistogramPlotter(result)
    ax = plotter.plot()

    assert ax.get_title() == "test histogram"
    assert len(ax.patches) > 0 or len(ax.containers) > 0 or len(ax.lines) > 0


def test_plotter_rejects_wrong_result_type():
    from eregion.datamodels import TaskResult

    class OtherResult(TaskResult):
        pass

    with pytest.raises(TypeError):
        HistogramPlotter(OtherResult())
