from __future__ import annotations
from typing import Optional, Any, Iterable
from .binary_artifact import BinaryArtifact, LazyLoadedDataContainer, PathType
from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
import pathlib

#astropy.fits.connect.is_fits exists, but seems very Janky IMO. Internally it just checks these extensions
#if you pass it a location on disk
_FITS_EXTENSIONS = [".fits", ".fits.gz", ".fit", ".fit.gz", ".fts", ".fts.gz"]

class FITSHDULazyLoad(LazyLoadedDataContainer[_BaseHDU]):
    pass



def _hdu_ensure_name(hdu: _BaseHDU, idx: int) -> str:
    return hdu.name if hdu.name != "" else idx


class FITSArtifact(BinaryArtifact, friendly_name="FITS"):

    @classmethod
    def load(cls, location: PathType, readonly: bool=False, light_load: bool=True, **kwargs) -> FITSArtifact:

        fitsargs = { "mode" : "readonly" if readonly else "update",
                     "lazy_load_hdus" : light_load}

        hdul = fits.open(location, **fitsargs)
        return cls(hdul, **kwargs)

    @classmethod
    def make_empty(cls) -> FITSArtifact:
        hdul = fits.HDUList()
        return cls(hdul)

    def __init__(self, hdul = fits.HDUList):
        self._hdul = hdul

    def save(self, location: Optional[PathType], allow_overwrite: bool=False, **saveopts) -> None:
        #if no location, try saving it to where we got it from
        if location is None and allow_overwrite == False:
            raise ValueError("Either specify new location for saving file, or allow overwriting!")
        elif location is None:
            self._hdul.flush()
        else:
            default_saveopts = {"checksum" : True}
            default_saveopts |= saveopts
            self._hdul.writeto(location, **default_saveopts)

    def identify_from_location(cls, location: PathType) -> bool:
        pth = pathlib.Path(location)
        extn = pth.suffix
        if extn == "":
            return False
        elif extn  in _FITS_EXTENSIONS:
            return True
        return False

    def identify(cls, location: PathType) -> bool:
        try:
            temphdul = fits.open(location, mode="readonly", lazy_load_hdus=True)
            return True
        except OSError as err:
            return False

    def identify_and_load(cls, location: PathType, **loadkwargs) -> Optional[FITSArtifact]:
        try:
            defaultkwargs = {"lazy_load_hdus" : True}
            defaultkwargs |= loadkwargs
            temphdul = fits.open(location, **defaultkwargs)
            return temphdul
        except OSError as err:
            return None

    def __getitem__(self, idx: str | int):
        #FITS HDUL can be indexed by either integer or str, which looks up EXTNAME, so that's good
        hdu = self._hdul[idx]
        return FITSHDULazyLoad(_parent_artifact=self, name=_hdu_ensure_name(hdu, idx), _payload=hdu)

    def __iter__(self) -> Generator[LazyLoadedDataContainer]:
        for idx, hdu in enumerate(self._hdul):
            yield FITSHDULazyLoad(_parent_artifact=self, name=_hdu_ensure_name(hdu, idx), payload=hdu)


    
