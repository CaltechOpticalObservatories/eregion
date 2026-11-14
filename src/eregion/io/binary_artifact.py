from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional, Any, MutableMapping, Mapping
from os import PathLike
from functools import singledispatch
from astropy.table import Table
from dataclasses import dataclass
import weakref

class DataItemRep(metaclass=ABCMeta):
    """base class for representation of a DataItem within a binary artifact"""



type PathType = str | bytes | PathLike
type MDatDct = dict[str, Any]


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

    @abstractmethod
    def detach(self) -> None:
        """load the payload from disk and remove strong references to the parent artifact. Effectively this makes the Data Container outlive the opened file it came from"""

    @abstractmethod
    @property
    def payload(self) -> Optional[DataT]:
        


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

    @classmethod
    @abstractmethod
    def load(cls, location: PathType, readonly: bool=False, light_load: bool=True, **kwargs) -> DataItemRep:
        """load a Binary artifact of this type from disk. Should perform a full load of the entire artifact into memory"""

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

    @abstractmethod
    def add_table(self, tbl: astropy.table, name: str, mdat: MDatDct) -> None:
        """Add a table to this binary artifact. It must be first converted to an astropy.Table object"""

    @abstractmethod
    def add_dense_array(self, arry: np.ndarray, name: str, mdat: MdatDct) -> None:
        """Add a dense array to this binary artifact. It must first be converted to a numpy NDArray"""

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


    @property
    def tables(self, idx: name) -> Mapping[str, LazyLoadedDataContainer]:
        pass
    
        
    
