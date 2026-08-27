import time
import numpy as np

from eregion.datamodels import CCDOutput, DetImage
from eregion.tasks.linbin import LinBin

DET_SIZE = 8192
PRESCAN = 4
OVERSCAN = 4


def _make_flat_det_image(name: str, seed: int) -> DetImage:
    rng = np.random.default_rng(seed)
    data = rng.random((DET_SIZE, DET_SIZE), dtype=np.float32)

    output = CCDOutput(
        id="A",
        ext_id=0,
        ext_slice=(slice(0, DET_SIZE), slice(0, DET_SIZE)),
        data_slice=(slice(0, DET_SIZE), slice(0, DET_SIZE)),
        parallel_axis="y",
        parallel_prescan=slice(0, PRESCAN),
        parallel_overscan=slice(DET_SIZE - OVERSCAN, DET_SIZE),
        serial_prescan=slice(0, PRESCAN),
        serial_overscan=slice(DET_SIZE - OVERSCAN, DET_SIZE),
        readout_pixel= (0, 0)
    )
    return DetImage(data=data, output_objects={"A": output}, name=name)


def _expected_incremental_bins(parallel_size: int) -> list[int]:
    bins = []
    total = 0
    nxt = 1
    while total < parallel_size:
        bins.append(nxt)
        total += nxt
        nxt += 1
    if total > parallel_size:
        bins[-1] -= total - parallel_size
    return bins


def test_linbin_runs_on_8k_by_8k_image_with_incremental_binning():
    normal_flat = _make_flat_det_image("D1", seed=0)
    linbin_flat = _make_flat_det_image("D1", seed=1)

    task = LinBin(binsizes=1, n_jobs=1)

    start = time.perf_counter()
    result = task.run([normal_flat], [linbin_flat])
    elapsed = time.perf_counter() - start
    # print elapsed time in pytest output
    print(f"Elapsed time: {elapsed:.2f}s")

    parallel_size = DET_SIZE - PRESCAN - OVERSCAN
    expected_bins = _expected_incremental_bins(parallel_size)

    assert task.bins == expected_bins
    assert sum(task.bins) == parallel_size

    stats = result.stats
    assert len(stats) == 1
    assert stats.loc[0, "det_id"] == "D1"
    assert stats.loc[0, "output"] == "A"
    assert stats.loc[0, "bins"] == expected_bins

    # one binned-stat value per bin row, for both the digital and analog binned flats
    for suffix in ("digital_0", "analog_0"):
        assert len(stats.loc[0, f"mean_{suffix}"]) == len(expected_bins)
        assert len(stats.loc[0, f"median_{suffix}"]) == len(expected_bins)

    assert elapsed < 60.0, f"LinBin.run took too long on {DET_SIZE}x{DET_SIZE} data: {elapsed:.2f}s"

