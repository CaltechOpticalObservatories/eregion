import numpy as np
import pandas as pd
import pytest


from eregion.tasks.ptc import PTCResult
from eregion.tasks.eper import SingleEPERTrailFitter, PTCEPERFitter
from eregion.datamodels import ImageBundle

_gen = np.random.default_rng(0xDEADBEEF)

_example_EPER_data = [12.0, 2.0, 1.0, 0.0] + [0.0] * 16
_example_EPER_data = np.array(_example_EPER_data) + _gen.normal(
    loc=0.0, scale=0.2, size=20
)
_siglevel = 10000.0


def test_single_EPER_fit():
    # NOTE: see test_expsumfitting.py for more stuff related to the do_trapfit option
    ftr = SingleEPERTrailFitter(N_transfers=1024, do_trapfit=False)
    res = ftr.run(_siglevel, _example_EPER_data)
    assert res.TDC == np.sum(_example_EPER_data)


def test_single_EPER_fit_subtraction():
    ftr = SingleEPERTrailFitter(
        N_transfers=1024, do_trapfit=False, subtract_eper_zeros=10
    )
    testdat = _example_EPER_data - np.mean(_example_EPER_data[-10:])
    res = ftr.run(_siglevel, _example_EPER_data)
    assert res.TDC == np.sum(testdat)


@pytest.fixture
def sample_PTC_table_EPER_data():
    tab = pd.DataFrame(
        {
            "meablorp": [_siglevel],
            "speeeepertrail": [_example_EPER_data],
            "badaboopeper": [_example_EPER_data],
        }
    )
    ptcres = PTCResult(ptc_table=tab, diff_images=ImageBundle())
    settings = {"do_trapfit": False, "N_transfers": 1024}

    return ptcres, settings


def test_PTC_EPER_fit(sample_PTC_table_EPER_data):

    ptcres, settings = sample_PTC_table_EPER_data

    ftr = PTCEPERFitter(
        pass_columns=[],
        siglevelcol="meablorp",
        ser_eper_col="speeeepertrail",
        llel_eper_col="badaboopeper",
        ser_settings=settings,
        llel_settings=settings,
    )
    res = ftr.run(ptcres)

    assert res.eper_table["ser_TDC"].iloc()[0] == np.sum(_example_EPER_data)
    assert res.eper_table["llel_TDC"].iloc()[0] == np.sum(_example_EPER_data)


def test_PTC_EPER_fit_selector(sample_PTC_table_EPER_data):
    ptcres, settings = sample_PTC_table_EPER_data

    def ser_eper_select(rowtpl):
        return rowtpl.speeeepertrail

    ftr = PTCEPERFitter(
        pass_columns=[],
        siglevelcol="meablorp",
        ser_eper_col=ser_eper_select,
        llel_eper_col="badaboopeper",
        ser_settings=settings,
        llel_settings=settings,
    )

    res = ftr.run(ptcres)
    assert res.eper_table["ser_TDC"].iloc()[0] == np.sum(_example_EPER_data)
    assert res.eper_table["llel_TDC"].iloc()[0] == np.sum(_example_EPER_data)
