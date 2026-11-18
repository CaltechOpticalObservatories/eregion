from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional, Any, MutableMapping, Mapping, Generator, Type
from os import PathLike
from functools import singledispatch, wraps
from astropy.table import Table
from dataclasses import dataclass
import weakref


type PathType = str | bytes | PathLike
type MDatDct = dict[str, Any]


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
class LazyLoadedDataContainer[UnderlyingT](metaclass=ABCMeta):
    """base class for individual lazily loaded data items (individual arrays or tables)"""
    _parent_artifact: BinaryArtifact | weakref.ref[BinaryArtifact]
    name: str
    _payload: Optional[UnderlyingT] = None

    @abstractmethod
    def load(self) -> None:
        """load the payload from disk"""

    def detach(self) -> None:
        """load the payload from disk and remove strong references to the parent artifact. Effectively this makes the Data Container outlive the opened file it came from. Note for some implementations this may result in a deep copy of the actual data happening"""
        self.load()
        self._parent_artifact = weakref.ref(self._parent_artifact)
        self._payload_detach()

    def _payload_detach(self) -> None:
        pass
        
    @property
    def payload(self) -> UnderlyingT:
        """get the data payload (in native, underlying type, e.g. FITS HDU or HDF5 Dataset. If it it not already loaded, this will cause it to load from disk"""
        if not self.is_loaded():
            self.load()
        return self._payload

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """check whether the payload has already been loaded."""

    @property
    def parent(self) -> Optional[BinaryArtifact]:
        """get the artifact that produced this data container. Note it may have been unloaded if the container is detached"""
        match self._parent_artifact:
            case BinaryArtifact():
                return self._parent_artifact
            case weakref.ref():
                return  self._parent_artifact()
            case _:
                raise TypeError("could not match type of parent artifact")

    @property
    @abstractmethod
    def metadata(self) -> MutableMapping[str, Any]:
        """return the (native, underlying type) metadata mapping for this item. Should not be recursively loaded for children,
           should also be implemented so as not to load the payload if it is not already loaded"""

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

    _type_registry: dict[str, Type[BinaryArtifact]] = dict()
    
    def __init_subclass__(cls, friendly_name: Optional[str] = None, **kwargs) -> None:
        """register subclasses automatically, to allow for discovery and such"""
        super().__init_subclass__(**kwargs)

        name = cls.__name__ if not friendly_name else friendly_name
        cls._type_registry[name] = cls
        cls.friendly_name = name


    location: PathType = None
    """The location on disk (or cloud, etc) of this artifact. None is perfectly valid for a loaded artifact, it may be constructed e.g. in RAM"""

    @classmethod
    @abstractmethod
    def load(cls, location: PathType, readonly: bool=False, lazy_load: bool=True, **kwargs) -> BinaryArtifact:
        """load a Binary artifact of this type from disk, or remote storage."""

    @classmethod
    def identify_from_location(cls, location: PathType) -> Optional[bool]:
        """class method which attempts to identify whether the targeted artifact is of this specific BinaryArtifact type
           by only checking the location (for example, by checking file extension names). This method MAY NOT assume
        the ability to open the location. An implementation returning None indicates that such a check isn't possible, or is ambiguous. More heavyweight checks should be attempted to determine if the file is actually of the correct BinaryArtifact type"""
        return None

    @classmethod
    @abstractmethod
    def identify(cls, location: PathType) -> bool:
        """class method which attempts to identify whether the targeted artifact is of this specific BinaryArtifact type
           by using "lightweight" methods if possible (i.e. short of fully obtaining and opening the whole artifact in memory). For example, may check the  magic  number in a binary file header, or attempt to load just the primary HDU of a FITS file, etc. An implementation may not return None, this check must be unambiguous. However, the test should NOT check the _validity_ of the full file (e.g. via checksums or similar) which would imply a full "heavyweight" check of the artifact.
        """

    @classmethod
    @abstractmethod
    def identify_and_load(cls, location: PathType, **loadkwargs) -> Optional[BinaryArtifact]:
        """class method which identifies then loads an artifact of this type. It is not redundant with the above because it can save some
        (small) amount of i/o for example by re-using the open file object used in the identification check."""

        
    @classmethod
    @abstractmethod
    def make_empty(cls) -> BinaryArtifact:
        """make an empty Binary artifact (i.e. to be stored in RAM until saved)"""


    @abstractmethod
    def save(self, location: Optional[PathType] = None, allow_overwrite: bool=False, **saveopts) -> None:
        """save in-memory Binary artifact to a location"""

    def add_table(self, tbl: astropy.table, name: str, mdat: MDatDct) -> None:
        """Add a table to this binary artifact. It must be first converted to an astropy.Table object"""

    def add_dense_array(self, arry: np.ndarray, name: str, mdat: MdatDct) -> None:
        """Add a dense array to this binary artifact. It must first be converted to a numpy NDArray"""

    @abstractmethod
    def add_data(self, dat, name: str, mdat: MdatDct) -> None:
        """Add binary data to the container (arrays, tables, whatever). """

    @property
    def toplevel_metadata(self) -> MutableMapping[str, Any]:
        """access the "top level" metadata of the artifact (e.g. for a FITS file the Primary HDU header, for an HDF5 file the top level attributes"""
        return self.get_metadata(allow_full_load=False, recursive=False, flatten=False)

    @property
    def full_metadata_nested(self) -> MutableMapping[str, Any]:
        """access fully all the metadata for all data items. Metadata for child items comes out as a nested mapping. When accessed from this property it may imply a full deep load of the entire dataset"""
        return self.get_metadata(allow_full_load=True, recursive=True, flatten=False)

    @property
    def full_metadata_flat(self) -> MutableMapping[str, Any]:
        """access fully all the metadata for all data items. Flat dictionary with no hierarchy. Keys are renamed in a path style format for child items"""
        return self.get_metadata(allow_full_load=True, recursive=True, flatten=True)

    @abstractmethod
    def get_metadata(self, allow_full_load: bool, recursive: bool, flatten: bool) -> MutableMapping[str, Any]:
        """metadata access function which the convenience properties are implemented in terms of. """
        pass

    def get_metadata_for(self, idx: str | int, recursive: bool=False) -> MutableMapping[str, Any]:
        """metadata access function for a single data item"""
        ditem = self[idx]
        return ditem.metadata

    @abstractmethod
    def __getitem__(self, idx: str | int) -> LazyLoadedDataContainer:
        """access an array or table by its name"""
        pass

    @abstractmethod
    def __iter__(self) -> tuple[str, LazyLoadedDataContainer]:
        """iterate over all binary data items in the artifact"""
        pass

    def tables(self) -> Generator[tuple[str, LazyLoadedDataContainer]]:
        """iterate over binary tables in the artifact"""

    def dense_arrays(self) -> Generator[tuple[str, LazyLoadedDataContainer]]:
        """iterate over dense arrays in the artifact"""


def get_available_artifact_plugins() -> dict[str, Type[BinaryArtifact]]:
    return BinaryArtifact._type_registry

def identify_and_load_artifact(location: PathType, location_only_checks: bool=True) -> BinaryArtifact:
    pass
