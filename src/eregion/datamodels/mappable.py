from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, ClassVar
from pydantic import BaseModel, ConfigDict
import json
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

    def update(self, other: Mapping[str, Any]) -> None:
        """
        Recursively merge a mapping into this model in place.

        :param other: Values to merge into this model.
        :return: None.
        """
        if not isinstance(other, Mapping):
            raise TypeError("Mappable.update() requires a Mapping.")

        for key, value in other.items():
            current_value = getattr(self, key, None)
            if isinstance(current_value, Mappable) and isinstance(value, Mapping):
                current_value.update(value)
            elif isinstance(current_value, MutableMapping) and isinstance(value, Mapping):
                current_value.update(value)
            else:
                self[key] = value

    # --- JSON persistence --------------------------------------------------

    _JSON_TYPE_KEY: ClassVar[str] = "__eregion_json_type__"
    _JSON_SLICE: ClassVar[str] = "slice"
    _JSON_ARRAY: ClassVar[str] = "ndarray"

    @classmethod
    def _json_fallback(cls, value: Any) -> Any:
        if isinstance(value, slice):
            return {
                cls._JSON_TYPE_KEY: cls._JSON_SLICE,
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if isinstance(value, np.ndarray):
            return {
                cls._JSON_TYPE_KEY: cls._JSON_ARRAY,
                "dtype": value.dtype.str,
                "shape": value.shape,
                "data": value.tolist(),
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"{cls.__name__} cannot serialize {type(value).__name__} to JSON.")

    @classmethod
    def _decode_json_value(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._decode_json_value(item) for item in value]
        if not isinstance(value, dict):
            return value
        value = {key: cls._decode_json_value(item) for key, item in value.items()}
        if value.get(cls._JSON_TYPE_KEY) == cls._JSON_SLICE:
            return slice(value["start"], value["stop"], value["step"])
        if value.get(cls._JSON_TYPE_KEY) == cls._JSON_ARRAY:
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
        return value

    def to_json(self, *, indent: int | None = None, **kwargs) -> str:
        """Serialize this model using Pydantic field serializers."""
        return self.model_dump_json(
            indent=indent,
            fallback=self._json_fallback,
            **kwargs,
        )

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray) -> "Mappable":
        """Decode tagged values and validate the result as this concrete model."""
        return cls.model_validate(cls._decode_json_value(json.loads(json_data)))