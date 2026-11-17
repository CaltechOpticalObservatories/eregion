from __future__ import annotations

from .binary_artifact import BinaryArtifact, LazyLoadedDataContainer
from astropy.io import fits

class FITSHDULazyLoad(LazyLoadedDataContainer):
    pass


class FITSArtifact(BinaryArtifact):

    @classmethod
    def load(cls, location: PathType, readonly: bool=False, light_load: bool=True, **kwargs) -> FITSArtifact:

        fitsargs = { "mode" : "readonly" if readonly else "update",
                     "lazy_load_hdus" : light_load}

        hdul = fits.open(location, **fitsargs)
        return cls(hdul, **kwargs)

    def __init__(self, hdul = fits.HDUList):
        self._hdul = hdul

    
        
