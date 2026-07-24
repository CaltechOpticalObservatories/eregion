import numpy as np
import pandas as pd

from eregion.datamodels import ImageBundle
from eregion.tasks.ptc import PTCResult
from eregion.utils import load_ptc_table_fits, save_ptc_table_fits


def test_ptc_table_fits_round_trip_preserves_scalar_and_array_columns(tmp_path):
    table = pd.DataFrame(
        {
            "det_id": ["D1", "D2"],
            "output": ["A", "B"],
            "exptime": [10.0, 20.0],
            "diff": [False, True],
            "n_masked": [1, 2],
            "llel_eper_med": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            "llel_eper_mean": [np.array([1.5, 2.5, 3.5]), np.array([4.5, 5.5, 6.5])],
            "PSD": [
                np.arange(9.0).reshape(3, 3),
                np.arange(9.0, 18.0).reshape(3, 3),
            ],
            "dPSD": [
                np.arange(9.0, 18.0).reshape(3, 3),
                np.arange(18.0, 27.0).reshape(3, 3),
            ],
        }
    )

    path = tmp_path / "ptc_table.fits"
    save_ptc_table_fits(table, str(path))
    loaded = load_ptc_table_fits(str(path))

    assert list(loaded.columns) == list(table.columns)
    assert loaded["det_id"].tolist() == table["det_id"].tolist()
    assert loaded["output"].tolist() == table["output"].tolist()
    assert loaded["diff"].tolist() == table["diff"].tolist()
    assert loaded["n_masked"].tolist() == table["n_masked"].tolist()
    assert loaded["exptime"].tolist() == table["exptime"].tolist()
    assert np.array_equal(loaded.loc[0, "llel_eper_med"], table.loc[0, "llel_eper_med"])
    assert np.array_equal(loaded.loc[1, "llel_eper_mean"], table.loc[1, "llel_eper_mean"])
    assert np.array_equal(loaded.loc[0, "PSD"], table.loc[0, "PSD"])
    assert np.array_equal(loaded.loc[1, "dPSD"], table.loc[1, "dPSD"])


def test_ptcresult_save_and_load_round_trip(tmp_path):
    table = pd.DataFrame(
        {
            "det_id": ["D1"],
            "output": ["A"],
            "exptime": [10.0],
            "diff": [False],
            "n_masked": [1],
            "llel_eper_med": [np.array([1.0, 2.0])],
            "llel_eper_mean": [np.array([1.5, 2.5])],
            "PSD": [np.arange(4.0).reshape(2, 2)],
            "dPSD": [np.arange(4.0, 8.0).reshape(2, 2)],
        }
    )
    result = PTCResult(ptc_table=table, diff_images=ImageBundle())

    result.save(str(tmp_path))
    loaded = PTCResult.load(str(tmp_path))

    assert loaded.ptc_table["det_id"].tolist() == ["D1"]
    assert np.array_equal(loaded.ptc_table.loc[0, "PSD"], np.arange(4.0).reshape(2, 2))
    assert isinstance(loaded.diff_images, ImageBundle)
