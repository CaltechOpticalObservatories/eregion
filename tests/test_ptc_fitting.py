import numpy as np
import pandas as pd
import pytest

from eregion.datamodels import ImageBundle
from eregion.tasks.ptc import PTCResult
from eregion.tasks.ptc_fitting import (
    BrighterFatterFitTypes,
    CCDPTCFit,
)


def test_run_groups_by_selection_columns(monkeypatch):
    fit = CCDPTCFit(
        selection_columns=["det_id", "output"],
        brighter_fatter=BrighterFatterFitTypes.NO_FIT,
    )
    table = pd.DataFrame(
        {
            "det_id": ["D1", "D2"],
            "output": ["A", "B"],
            "exptime": [1.0, 1.0],
            "mean": [1.0, 2.0],
            "std": [0.1, 0.2],
        }
    )
    monkeypatch.setattr("eregion.tasks.ptc_fitting.find_adc_sat_index", lambda *_: None)
    monkeypatch.setattr("eregion.tasks.ptc_fitting.find_rough_full_well", lambda *_: (0, 0))
    monkeypatch.setattr("eregion.tasks.ptc_fitting.trad_ptc_shot_noise_fit", lambda *_: (1.0, 2.0))
    monkeypatch.setattr(
        "eregion.tasks.ptc_fitting.linearity_fit",
        lambda *_: (np.array([0.0, 1.0]), np.array([0.1, 0.1])),
    )

    result = fit.run(PTCResult(ptc_table=table, diff_images=ImageBundle()))

    assert set(result.fits) == {("D1", "A"), ("D2", "B")}


def test_run_rejects_missing_selection_columns():
    fit = CCDPTCFit(
        selection_columns=["det_id"],
        brighter_fatter=BrighterFatterFitTypes.NO_FIT,
    )

    with pytest.raises(KeyError, match="det_id"):
        fit.run(PTCResult(ptc_table=pd.DataFrame(), diff_images=ImageBundle()))
