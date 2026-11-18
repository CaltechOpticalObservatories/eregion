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



def _make_small_masked_image() -> DetImage:
    rng = np.random.default_rng(1)
    output = CCDOutput(
        id="A", ext_id=0,
        ext_slice=(slice(0, 12), slice(0, 6)), data_slice=(slice(0, 12), slice(0, 6)),
        parallel_axis="y",
        parallel_prescan=slice(0, 1, 1), parallel_overscan=slice(11, 12, 1),
        serial_prescan=slice(0, 1, 1), serial_overscan=slice(5, 6, 1),
    )
    img = DetImage(data=rng.random((12, 6)), output_objects={"A": output}, name="D1")
    for key, pixel in (("cr", (2, 2)), ("vignette", (5, 3))):
        mask = np.zeros((12, 6), dtype=bool)
        mask[pixel] = True
        output.set_mask(key, output.data.copy(data=mask))
    return img


def test_linbin_bins_every_mask_with_the_default_mask_key():
    img = _make_small_masked_image()
    # image rows 1..10 are binned into rows of 1, 2, 3 and 4 summed rows
    LinBin(binsizes=1)._bin_image(img)
    masks = img.outputs["A"].masks
    # row 2 is summed into the second bin (rows 2-3 -> row 2), row 5 into the third (rows 4-6 -> row 3)
    assert bool(masks["cr"].sel(y=2, x=2)) and int(masks["cr"].sum()) == 1
    assert bool(masks["vignette"].sel(y=3, x=3)) and int(masks["vignette"].sum()) == 1


def test_linbin_stats_use_the_mask_key():
    task = LinBin(binsizes=1, mask_key="cr")
    normal, linbin = _make_small_masked_image(), _make_small_masked_image()
    stats = next(task.lazy_run(normal_flats=[normal], linbin_flats=[linbin])).stats
    assert set(stats["n_masked_digital_0"]) == {1} and set(stats["n_masked_analog_0"]) == {1}
