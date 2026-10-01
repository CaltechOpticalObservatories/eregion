import numpy as np
import pandas as pd
import pytest

from eregion.datamodels import ImageBundle
from eregion.tasks.ptc import PTCResult
from eregion.tasks.ptc_fitting import (
    BrighterFatterFitTypes,
    CCDPTCFit,
)



@pytest.fixture
def gen_sample_PTC_fit_table(monkeypatch):
    fit = CCDPTCFit(
        selection_columns=["det_id", "output"],
        brighter_fatter=BrighterFatterFitTypes.NO_FIT,
    )
    table = pd.DataFrame(
        {
            "det_id": ["D1", "D2"],
            "output": ["A", "B"],
            "exptime": [1.0, 2.0],
            "mean": [1.0, 2.0],
            "std": [0.1, 0.2],
        }
    )
    monkeypatch.setattr("eregion.tasks.ptc_fitting.find_adc_sat_index", lambda *_, **__: None)
    monkeypatch.setattr("eregion.tasks.ptc_fitting.find_rough_full_well", lambda *_, **__: (0, 0))
    monkeypatch.setattr("eregion.tasks.ptc_fitting.trad_ptc_shot_noise_fit", lambda *_, **__: (1.0, 2.0))
    monkeypatch.setattr(
        "eregion.tasks.ptc_fitting.linearity_fit",
        lambda *_: (np.array([0.0, 1.0]), np.array([0.1, 0.1])),
    )

    result = fit.run(PTCResult(ptc_table=table, diff_images=ImageBundle()))
    return result


def test_run_groups_by_selection_columns(gen_sample_PTC_fit_table):
    result = gen_sample_PTC_fit_table
    assert set(result.fits) == {("D1", "A"), ("D2", "B")}


def test_run_rejects_missing_selection_columns():
    fit = CCDPTCFit(
        selection_columns=["det_id"],
        brighter_fatter=BrighterFatterFitTypes.NO_FIT,
    )

    with pytest.raises(KeyError, match="det_id"):
        fit.run(PTCResult(ptc_table=pd.DataFrame(), diff_images=ImageBundle()))


def test_get_df_from_results(gen_sample_PTC_fit_table):
    result = gen_sample_PTC_fit_table

    df = result.as_dataframe()
    print(result)

    #if the join happened correctly, then the column for e.g. camera-gain_classic is identical between before and after
    # dataframe transformation

    for k, v in result.fits.items():
        row = df[ (df[k._fields[0]] == k[0]) & ( df[k._fields[1]] == k[1]) ]
        #should only find one, keys were unique after all
        assert len(row) == 1
        for kk, vv in v.items():
            assert row[kk].iloc[0] == vv

