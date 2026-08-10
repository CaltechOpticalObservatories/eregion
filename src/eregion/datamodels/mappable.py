from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator, SerializationInfo
import numpy as np


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

    # --- JSON serialization / deserialization ----------------------------

    @model_serializer(when_used='json', mode='plain')
    def _serialize_for_json(self, info: SerializationInfo):
        """
        Recursively convert non-JSON-native types before serialization:
          - slice          → {"start": ..., "stop": ..., "step": ...}
          - np.ndarray     → nested list
          - np.integer     → int
          - np.floating    → float
          - fits.Header    → dict  (when astropy is available)
        Traverses tuples, lists, and dicts to handle nested structures.
        """
        try:
            from astropy.io import fits as _fits
            _fits_header = _fits.Header
        except ImportError:
            _fits_header = None

        def _convert(value):
            if isinstance(value, slice):
                return {"start": value.start, "stop": value.stop, "step": value.step}
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if _fits_header is not None and isinstance(value, _fits_header):
                return dict(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                converted = [_convert(v) for v in value]
                return type(value)(converted)
            return value

        data = self.model_dump(
            mode="python",
            include=info.include,
            exclude=info.exclude,
            by_alias=info.by_alias,
            exclude_unset=info.exclude_unset,
            exclude_defaults=info.exclude_defaults,
            exclude_none=info.exclude_none,
            round_trip=info.round_trip,
        )
        return _convert(data)

    @model_validator(mode='before')
    @classmethod
    def _parse_from_json(cls, kwargs):
        """
        Recursively revive JSON representations of slices back to slice objects.
        A dict with exactly the keys {"start", "stop", "step"} is treated as a
        serialized slice. Traverses dicts and lists to handle nested structures.
        """

        def _revive(value):
            if isinstance(value, dict) and {"start", "stop", "step"}.issubset(value):
                return slice(value["start"], value["stop"], value["step"])
            if isinstance(value, dict):
                return {k: _revive(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_revive(v) for v in value]
            return value

        if isinstance(kwargs, dict):
            return {k: _revive(v) for k, v in kwargs.items()}
        return kwargs