from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional, Any, MutableMapping, Mapping, Generator
from os import PathLike
from functools import singledispatch, wraps
from astropy.table import Table
from dataclasses import dataclass
import weakref
from enum import IntEnum

class DataItemRep(metaclass=ABCMeta):
    """base class for representation of a DataItem within a binary artifact"""



type PathType = str | bytes | PathLike
type MDatDct = dict[str, Any]

class BinaryDatasetType(IntEnum):
    TABLE,
    DENSE_ARRAY


class NoPayloadError(Exception):
    pass


def ensure_payload(fun):
    @wraps(fun)
    def wrapper(selfarg, *args, **kwargs):
        if selfarg._payload is None:
            raise NoPayloadError(f"No payload present, cannot pass through command")
        return run(selfarg, *args, **kwargs)
    return wrapper

@dataclass
class LazyLoadedDataContainer[DataT](metaclass=ABCMeta):
    """base class for individual lazily loaded data items (individual arrays or tables)"""
    _parent_artifact: BinaryArtifact | weakref.ref[BinaryArtifact]
    name: str
    metadata: MDatDct
    _payload: Optional[DataT] = None

    @abstractmethod
    def load(self) -> None:
        """load the payload from disk"""

    def detach(self) -> None:
        """load the payload from disk and remove strong references to the parent artifact. Effectively this makes the Data Container outlive the opened file it came from"""
        self.load()
        self._parent_artifact = weakref.ref(self._parent_artifact)

    @property
    def payload(self) -> Optional[DataT]:
        """get the data payload. If it it not already loaded, this will cause it to load from disk"""
        if self._payload is None:
            self.load()
        return self._payload

    @property
    def is_loaded(self) -> bool:
        return self._payload is not None

    @property
    def parent(self) -> Optional[BinaryArtifact]:
        match self._parent_artifact:
            case BinaryArtifact():
                return self._parent_artifact
            case weakref.ref():
                return  self._parent_artifact()
            case _:
                raise TypeError("could not match type of parent artifact")
    
    #basic wrapping of attributes on the payload for convenience
    #this may turn out to not be enough, in which case we will eventually want to use
    #something like wrapt (external dependency)
    @ensure_payload
    def __getattr__(self, name):
        """pass through attribute access directly to the payload object, for convenience"""
        return getattr(self._payload, name)

    @ensure_payload
    def __setattr__(self, name, val):
        """pass through attribute setting directly to the payload object, for conveience"""
        if hasattr(self._payload, name):
            setattr(self._payload, name, val)
        super().__setattr__(name, val)

    @ensure_payload
    def __getitem__(self, idx):
        """pass through subscripting to the proxy object"""
        return self._payload[idx]

    @ensure_payload
    def __setitem__(self, idx, val):
        self._payload[idx] = val

    @ensure_payload
    def __len__(self):
        return len(self._payload)

    @ensure_payload
    def __repr__(self):
        return f"{type(self).__name__}({repr(self._payload)})"

    @ensure_payload
    def __str__(self):
        return f"{type(self).__name__}({str(self._payload)})"


class BinaryArtifact(metaclass=ABCMeta):
    """base class for a binary (usually on-disk) artifact within Eregion results storage system. These binary artifacts must support the following features:

       - ability to store tabular data retaining datatypes in a round-trippable way
       - ability to store multi-D dense arrays in a round-trippable way

       Examples which could be implemented pretty easily include:
         - FITS files
         - HDF5 files
         - Zarr files
         - ASDF files

    """

    location: PathType = None
    """The location on disk (or cloud, etc) of this artifact. None is perfectly valid for a loaded artifact, it may be constructed e.g. in RAM"""

    @classmethod
    @abstractmethod
    def load(cls, location: PathType, readonly: bool=False, lazy_load: bool=True, **kwargs) -> BinaryArtifact:
        """load a Binary artifact of this type from disk, or RAM."""

    @classmethod
    @abstractmethod
    def identify_from_location(cls, location: PathType) -> Optional[bool]:
        """class method which attempts to identify whether the targeted artifact is of this specific BinaryArtifact type
           by only checking the location (for example, by checking file extension names). This method MAY NOT assume
        the ability to open the location"""

    @classmethod
    @abstractmethod
    def identify(cls, location: PathType) -> Optional[bool]:
        """class method which attempts to identify whether the targeted artifact is of this specific BinaryArtifact type
           by using "lightweight" methods (i.e. short of fully obtaining and opening the whole artifact in memory). For example, may check the
           magic  number in a binary file header, or attempt to load just the primary HDU of a FITS file, etc
        """

    @classmethod
    @abstractmethod
    def make_empty(cls) -> BinaryArtifact:
        """make an empty Binary artifact (i.e. to be stored in RAM until saved)"""

        
    @abstractmethod
    def save(self, location: Optional[PathType] = None, allow_overwrite: bool=False):
        """save in-memory Binary artifact to a location"""

        
    @abstractmethod
    def add_table(self, tbl: astropy.table, name: str, mdat: MDatDct) -> None:
        """Add a table to this binary artifact. It must be first converted to an astropy.Table object"""

    @abstractmethod
    def add_dense_array(self, arry: np.ndarray, name: str, mdat: MdatDct) -> None:
        """Add a dense array to this binary artifact. It must first be converted to a numpy NDArray"""

    @abstractmethod
    def add_data(self, dat, name: str, mdat: MdatDct) -> None:
        """Add binary data to the container (arrays, tables, whatever). """

    @abstractmethod
    @property
    def toplevel_metadata(self) -> MutableMapping[str, Any]:
        """access the "top level" metadata of the artifact (e.g. for a FITS file the Primary HDU header, for an HDF5 file the top level attributes"""
        return self.get_metadata(allow_full_load=False, recursive=False, flatten=False)

    @abstractmethod
    @property
    def full_metadata_nested(self) -> MutableMapping[str, Any]:
        """access fully all the metadata for all data items. Metadata for child items comes out as a nested mapping. When accessed from this property it may imply a full deep load of the entire dataset"""
        return self.get_metadata(allow_full_load=True, recursive=True, flatten=False)

    @abstractmethod
    @property
    def full_metadata_flat(self) -> MutableMapping[str, Any]:
        """access fully all the metadata for all data items. Flat dictionary with no hierarchy. Keys are renamed in a path style format for child items"""
        return self.get_metadata(allow_full_load=True, recursive=True, flatten=True)
    
    @abstractmethod
    def get_metadata(self, allow_full_load: bool, recursive: bool, flatten: bool) -> MutableMapping[str, Any]:
        """metadata access function which the convenience properties are implemented in terms of. """
        pass

    @abstractmethod
    def get_metadata_for(self, name: str, recursive: bool=False) -> MutableMapping[str, Any]:
        """metadata access function for a single data item"""

    def __getitem__(self, idx: str | int) -> LazyLoadedDataContainer:
        """access an array or table by its name"""
        pass

    def __iter__(self) -> tuple[str, LazyLoadedDataContainer]:
        """iterate over all binary data items in the artifact"""
        pass

    def tables(self) -> Generator[tuple[str, LazyLoadedDataContainer]]:
        """iterate over binary tables in the artifact"""

    def dense_arrays(self) -> Generator[tuple[str, LazyLoadedDataContainer]]:
        """iterate over dense arrays in the artifact"""
