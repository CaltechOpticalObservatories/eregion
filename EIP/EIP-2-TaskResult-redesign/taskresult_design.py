from typing import TypeVar, Generic, NamedTuple
from pydantic import BaseModel, field_validator
import pandas as pd


class TaskResult(BaseModel):
    pass

class TaskResultKeyedCollection[R: TaskResult, NT: NamedTuple](BaseModel):
    """represents a dict-like collection of TaskResults.
       Unlike a python dict, all those TaskResults must be of the same type,
       and they must be keyed with a named tuple"""

    payload: dict[NT, R]

    @field_validator("payload")
    @classmethod
    def check_is_named_tuple(cls, v: dict[NT, R]) -> Any:
        """example way of pydantic enforcing NamedTuple types for the dict key"""
        if len(v) > 0:
            keyval = next(iter(v.keys()))
            if not all(isinstance(_, type(keyval)) for _ in v.keys()):
                raise TypeError("mixture of named tupele types found in results collection, not allowed")
            if not isinstance(keyval, tuple) and hasattr(keyval, "_fields"):
                raise TypeError("require key type to be a named tuple")


    def as_dataframe(self) -> pd.DataFrame:
        """convert the tuple keyed dict to a dataframe. Expensive, memory intensive,
           only do this after building the collection up.
        """
        #get the namedtuple list
        keysl = list(self.payload.keys())

        #this implicitly splits the named tuple key values into separate columns
        keysdf = pd.DataFrame(data=lkeys, index=lkeys)

        
        
        

class TaskResultSequence[R: TaskResult]:
    """represents a non-keyed collection of TaskResults.
       Should _ONLY_ be used to contain a whole pile of TaskResults which
       represent otherwise completely indistinguishable but repeated tests
       (for example, extractions of Fe-55 events out of a single image)"""

