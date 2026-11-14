from __future__ import annotations
from functools import singledispatch
import pandas as pd
from astropy.table import Table
from typing import Any, Optional


type TableAndHeader = tuple[Table, dict[str, Any]]

@singledispatch
def to_astropy_table(intbl, extra_mdat: Optional[dict[str,Any]] = None, **kwargs) -> TableAndHeader:
    """dispatch overload to convert input table type into an Astropy Table

    parameters
    ----------

    intbl
    the input table. Will use singledispatch to call implementation function based on the actual type of table passed in

    extra_mdat: Optional[dict[str, Any]]
    extra metadata to pass through to the resulting header

    **kwargs:
    extra keyword arguments will be passed through to the underlying dispatcher


    """
    tpname = type(intbl).__name__
    raise TypeError(f"do not know how to convert table with type {tpname} to an Astropy Table. ")


@to_astropy_table.register
def _(intbl: np.recarray, extra_mdat: Optional[dict[str, Any]] = None, **kwargs) -> TableAndHeader:
    """dispatch overload to convert a numpy recarray into an Astropy Table and headers"""
    raise NotImplementedError("cannot convert numpy recarrays to astropy tables yet!")

@to_astropy_table.register
def _(intbl: pd.DataFrame, extra_mdat: Optional[dict[str, Any]] = None, **kwargs) -> TableAndHeader:
    """dispatch overload to convert a pandas DataFrame into an Astropy Table and headers"""
    raise NotImplementedError("cannot convert pandas Dataframe to astropy tables yet!")


    
    





