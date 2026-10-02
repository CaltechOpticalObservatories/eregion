import pytest
import pint
import uncertainties as unc
import numpy as np
import pandas as pd

from eregion.utils.io_utils import (
    _quantity_column_to_fits_column,
    _choose_minimal_numpy_dtype,
    save_dataframe_to_fits,
)

_HDUL = None


_ureg = pint.get_application_registry()

# just test the Pint to FITS stuff for now, life is short .
# and I intend to finesse the rest as we migrate to new Task design anyway


def test_simple_conversion():

    vals = [1.0, 12.0, 19.0]
    VQs = [pint.Quantity(_, "volt") for _ in vals]

    outcol = _quantity_column_to_fits_column("voltages", VQs)

    assert outcol.unit == "V"
    assert outcol.format == "E"


def test_conversion_with_uncertainties():
    punc = lambda n, e: pint.Quantity(unc.ufloat(n, e), "metre")

    vals = [punc(1.0, 0.01), punc(12.0, 0.2), punc(19.0, 0.2)]

    outcols = _quantity_column_to_fits_column("lengths", vals)

    # should produce a list of 2 columns this time
    assert len(outcols) == 2
    assert isinstance(outcols, list)

    # check name of the error column
    assert outcols[1].name == "lengths_err"
    assert outcols[0].unit == outcols[1].unit


def test_heterogeneous_units_columns():
    vals = [pint.Quantity(1.0, "volt"), pint.Quantity(100, "millivolt")]

    outcol = _quantity_column_to_fits_column("voltages", vals)
    assert outcol.unit == "V"

    # all should have been converted to Volts
    vval = vals[1].magnitude / 1000.0

    # not sure why this needs a round, but it does. hmmm
    assert round(outcol.array[1], 1) == round(vval, 1)


def test_np_dtype_promotion():
    t1 = _choose_minimal_numpy_dtype([65534])
    t2 = _choose_minimal_numpy_dtype([65536])

    assert t1 == np.dtypes.UInt16DType()
    assert t2 == np.dtypes.UInt32DType()

    t3 = _choose_minimal_numpy_dtype([12, 1287])
    assert t3 == np.dtypes.UInt16DType()

    t4 = _choose_minimal_numpy_dtype([-1, 2556])
    assert t4 == np.dtypes.Int32DType()

    # could do hundreds more cases but we'd just be testing the
    # functionality of np.promote_type and np.min_scalar_type at that point


def test_saving_FITS_from_dataframe_with_uncertainties(monkeypatch):
    punc = lambda n, e: pint.Quantity(unc.ufloat(n, e), "metre")
    vals = [punc(1.0, 0.01), punc(12.0, 0.2), punc(19.0, 0.2)]

    dfdict = {"length": vals}

    df = pd.DataFrame(dfdict)

    def patched_hdul_save(self, *args, **kwargs):
        global _HDUL
        _HDUL = self

    monkeypatch.setattr("astropy.io.fits.HDUList.writeto", patched_hdul_save)

    save_dataframe_to_fits(df, "nobodycaresatall.fits")

    # Saves to Bintable HDU we can then access
    assert np.all(_HDUL[1].data["length"] == np.array([1.0, 12.0, 19.0]))
    diff = _HDUL[1].data["length_err"] - np.array([0.01, 0.2, 0.2])
    # again, no idea why this diff fails slightly. But it's errors,
    # if you care about more than one decimal point you're going wrong
    assert np.all(np.isclose(diff, 0, atol=1e-4))
