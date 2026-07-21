from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from pydantic import BaseModel, ConfigDict


class Mappable(BaseModel, Mapping):
    """
    Base class for pydantic mappable basemodel. Combines Pydantic validation with the full
    Python Mapping protocol, so instances behave like dicts while remaining
    typed models.

    Inheriting from both BaseModel and Mapping works because Pydantic's
    ModelMetaclass already derives from ABCMeta, so there is no metaclass
    conflict.  The three abstract methods required by Mapping (__getitem__,
    __iter__, __len__) are implemented here; everything else (keys(), values(),
    items(), get(), __contains__, __eq__) is provided for free by Mapping.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # --- Mapping protocol implementation ---------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise KeyError(key)

    def __iter__(self):
        # iterate over keys present in the model dump
        return iter(self.model_dump().keys())

    def __len__(self) -> int:
        return len(self.model_dump())

    def to_dict(self) -> dict:
        """Return a plain dict snapshot of the current model state."""
        return self.model_dump()