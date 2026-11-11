from abc import ABCMeta, abstractmethod
from typing import Optional
from os import PathLike

class DataItemRep(metaclass=ABCMeta):
    """base class for representation of a DataItem within a binary artifact"""



type PathType = str | bytes | PathLike
    

class BinaryArtifact(metaclass=ABCMeta):
    """base class for a binary (usually on-disk) artifact within Eregion results storage system. These binary artifacts must support the following features:

       - ability to store tabular data retaining datatypes in a round-trippable way
    

    """

    @classmethod
    @abstractmethod
    def load(cls, location: PathType, readonly: bool=False, **kwargs) -> DataItemRep:
        """load a Binary artifact of this type from disk"""

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

    
