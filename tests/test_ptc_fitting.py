import numpy as np
import pandas as pd
import pytest

from eregion.core.ptc_fit_math import find_rough_full_well


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


def test_fw_finding():
    #trivial example
    mnvals = np.array([ 0, 1, 2, 3, 4])
    stdvals = np.array([ 0, 10, 20, 30, 5])

    fwlocguess, fw = find_rough_full_well(mnvals, stdvals)

    assert fwlocguess == 3
    assert fw == 30

def test_fw_finding_hideous_but_commonly_occurring_messed_up_data_case():
    mnvals = np.array([0, 1, 2, 3, 4])
    stdvals = np.array([2000000, 10, 20, 30, 5])

    # should fail, empty array
    with pytest.raises(ValueError):
        fwlocguess, fw = find_rough_full_well(mnvals, stdvals)

    fwlocguess, fw = find_rough_full_well(mnvals, stdvals, n_candidates = 3)

    assert fwlocguess == 3
    assert fw == 30
    
    
                    
    
    
