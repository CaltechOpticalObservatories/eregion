import numpy as np
import pytest

from eregion.tasks.noise import AtomicNoiseResult, NoiseResult


def _atomic(det_id, region, nbins, seed):
    rng = np.random.default_rng(seed)
    return AtomicNoiseResult(det_id=det_id, output="amp1", region=region, noise=rng.random(),
                             histogram=(np.linspace(0, 1, nbins + 1), rng.integers(0, 100, nbins)),
                             psd=(np.linspace(0, 0.5, 257), rng.random(257)),
                             mean=rng.random(), filename=f"{det_id}.fits", time=60725.5 + seed)


def _assert_same(loaded, expected):
    assert len(loaded.data) == len(expected.data)
    for col in ("det_id", "output", "region", "filename"):
        assert loaded.data[col].tolist() == expected.data[col].tolist()
    for col in ("noise", "mean", "time"):
        np.testing.assert_allclose(loaded.data[col].astype(float), expected.data[col].astype(float))
    for col in ("histogram", "psd"):
        for got, want in zip(loaded.data[col], expected.data[col]):
            for g, w in zip(got, want):
                np.testing.assert_array_equal(g, w)


def test_data_has_no_metadata_columns():
    result = NoiseResult(data=[_atomic("green", "full", 10, 0)])
    assert not set(AtomicNoiseResult.metadata_field_names()) & set(result.data.columns)


def test_save_load_ragged_histograms(tmp_path):
    # ragged across rows, and edges are one longer than counts within a row
    result = NoiseResult(data=[_atomic("green", "full", 432, 0), _atomic("green", "overscan", 26219, 1)])
    result.save(str(tmp_path / "img"))
    _assert_same(NoiseResult.load(str(tmp_path / "img")), result)


def test_save_load_equal_length_arrays(tmp_path):
    result = NoiseResult(data=[_atomic("green", "full", 50, 0), _atomic("red", "full", 50, 1)])
    result.save(str(tmp_path / "img"))
    _assert_same(NoiseResult.load(str(tmp_path / "img")), result)


def test_load_run_combines_in_name_order(tmp_path):
    first = NoiseResult(data=[_atomic("green", "full", 20, 0)])
    second = NoiseResult(data=[_atomic("red", "full", 30, 1), _atomic("red", "overscan", 40, 2)])
    second.save(str(tmp_path / "b_image"))
    first.save(str(tmp_path / "a_image"))
    (tmp_path / "not_an_image").mkdir()
    _assert_same(NoiseResult.load_run(tmp_path), first.combine(second))


def test_load_run_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        NoiseResult.load_run(tmp_path)
