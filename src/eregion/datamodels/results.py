"""
A Mappable base class for task outputs, replacing the older dict[str, Any].

This module provides a base class that combines the benefits of dataclasses (typed fields,
IDE autocomplete, refactor safety) with the flexibility of dict-like access (str keys,
dynamic wiring in pipelines).

The Mapping-facing API intentionally exposes only task payload fields. Execution metadata
(``params``, ``upstream``, and ``timestamp``) remains available as attributes and through
``metadata_dict()``, while payload-only snapshots are available through ``payload_dict()``.

Each Task subclass defines a concrete Result class that:
  1. Inherits from TaskResult
  2. Defines typed fields for each output
  3. Returns instances instead of raw dicts
  4. Enables early validation of pipeline wiring

Design rationale:
  - Mapping protocol (__getitem__, keys(), items(), etc.) allows dict-like access for pipelines
  - Pydantic BaseModel gives runtime validation, serialization, and field constraints
  - Explicit fields (vs Any) enable IDE autocompletion and type safety
  - get_schema() class method enables discovery of available keys before pipeline runs
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from collections.abc import Mapping
from typing import Any, Iterator
from pydantic import ConfigDict, Field
from astropy.time import Time
import os
import json

from . import ImageBundle
from .mappable import Mappable


class TaskResult(Mappable):
    """
    Base class for task outputs. Inherits from ``Mappable`` which combines Pydantic validation with the full
    Python Mapping protocol, so instances behave like dicts while remaining
    typed models.

    Mapping-style access is payload-only by design: ``result["field"]``,
    ``result.keys()``, and ``result.to_dict()`` expose only the task outputs that
    should be wired downstream. Execution metadata is still available via the
    ``params``, ``upstream``, and ``timestamp`` attributes, or explicitly through
    ``metadata_dict()``.

    Examples:

        class MyTaskOutput(TaskResult):
            image: DetImage = Field(..., description="Processed image")
            stats: dict[str, float] = Field(default_factory=dict, description="Processing statistics")

        # In a task:

        class MyTask(Task):
            task_result = MyTaskOutput

            def run(self, images):
                result = MyTaskOutput(image=processed_img, stats={"mean": 100.5})
                return result
    """
    params: dict[str, Any] = Field(default_factory=dict)
    upstream: list[str] = Field(default_factory=list)
    timestamp: Time | list[Time] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @classmethod
    def _metadata_field_names(cls) -> set[str]:
        return {"params", "upstream", "timestamp"}

    @classmethod
    def _payload_field_names(cls) -> list[str]:
        return [field_name for field_name in cls.model_fields if field_name not in cls._metadata_field_names()]

    @classmethod
    def metadata_field_names(cls) -> tuple[str, ...]:
        """Return the names of metadata fields that are excluded from mapping-style access."""
        return tuple(field_name for field_name in cls.model_fields if field_name in cls._metadata_field_names())

    @classmethod
    def payload_field_names(cls) -> tuple[str, ...]:
        """Return the names of payload fields exposed through the Mapping interface."""
        return tuple(cls._payload_field_names())

    # ==================== Mapping ABC – three required methods ===============
    def __getitem__(self, key: str) -> Any:
        """Access field values like a dict: result['image']"""
        if key not in self.__class__._payload_field_names():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names (keys)."""
        return iter(self.__class__._payload_field_names())

    def __len__(self) -> int:
        """Number of declared fields."""
        return len(self.__class__._payload_field_names())

    def __eq__(self, other: object) -> bool:
        """Compare payload mappings rather than full model metadata."""
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    # keys(), values(), items(), get(), __contains__, __eq__ are all provided
    # for free by collections.abc.Mapping using the three methods above.

    # ==================== Optional convenience extras ========================
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a field value like a dict: result['image'] = new_image"""
        if key not in self.__class__._payload_field_names():
            raise KeyError(key)
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert the payload fields to a plain dict, excluding TaskResult metadata."""
        return self.payload_dict()

    def payload_dict(self) -> dict[str, Any]:
        """Return only the task payload fields, excluding TaskResult metadata."""
        model_data = self.model_dump()
        return {
            field_name: model_data[field_name]
            for field_name in self.__class__._payload_field_names()
        }

    def metadata_dict(self) -> dict[str, Any]:
        """Return only TaskResult metadata fields such as params, upstream, and timestamp."""
        model_data = self.model_dump()
        return {
            field_name: model_data[field_name]
            for field_name in self.__class__.metadata_field_names()
        }

    def combine(self, other: "TaskResult") -> "TaskResult":
        """
            Combine this result with another result of the same concrete class.

            This is primarily useful for lazy task execution where each iteration yields
            a partial result that should be accumulated across iterations.

            Payload fields are combined according to their type:
              - ``ImageBundle`` / subclasses : images lists concatenated via ``type(a)(images=...)``
              - ``pd.DataFrame``             : ``pd.concat(..., ignore_index=True)``
              - ``np.ndarray``               : ``np.concatenate`` along axis 0 when trailing
                                               dimensions are compatible, otherwise ``[a, b]``
              - ``dict``                     : merged; duplicate keys are combined recursively
              - ``list``                     : extended
              - ``None``                     : non-``None`` value is kept; both ``None`` → ``None``
              - anything else (scalar, str,
                tuple, pint.Quantity, etc.)  : promoted to / appended into a list

            Metadata fields are merged as follows:
              - ``params``    : values from ``other`` override values from ``self``
              - ``upstream``  : concatenated and deduplicated while preserving order
              - ``timestamp`` : accumulated into a list in iteration order

            Notes:
                Combined payload fields may no longer match the original field annotations
                exactly (e.g. a scalar field may become a list after accumulation). This is
                intentional for lazy accumulation; the combined instance is constructed
                without re-validating the merged payload.
            """
        if self.__class__ is not other.__class__:
            raise ValueError(
                "Can only combine TaskResult instances of the same concrete class. "
                f"Got {self.__class__.__name__} and {other.__class__.__name__}."
            )

        def combine_values(current_value: Any, other_value: Any) -> Any:
            match (current_value, other_value):
                case None, other:
                    return other
                case other, None:
                    return other
                case ImageBundle(), ImageBundle():
                    return type(current_value)(images=current_value.images + other_value.images)
                case pd.DataFrame(), pd.DataFrame():
                    return pd.concat([current_value, other_value], ignore_index=True)
                case np.ndarray() as a, np.ndarray() as b if (
                    a.ndim > 0 and b.ndim > 0 and a.shape[1:] == b.shape[1:]
                ):
                    return np.concatenate([a, b], axis=0)
                case np.ndarray() as a, np.ndarray() as b:
                    return [a, b]
                case dict() as a, dict() as b:
                    combined_dict = a.copy()
                    for key, value in b.items():
                        combined_dict[key] = (
                            combine_values(combined_dict[key], value)
                            if key in combined_dict
                            else value
                        )
                    return combined_dict
                case list() as a, list() as b:
                    return a + b
                case list() as a, other:
                    return a + [other]
                case other, list() as b:
                    return [other] + b
                case _:
                    return [current_value, other_value]

        combined_payload = {
            field_name: combine_values(getattr(self, field_name), getattr(other, field_name))
            for field_name in self.__class__._payload_field_names()
        }

        combined_params = {**self.params, **other.params}
        combined_upstream = list(dict.fromkeys([*self.upstream, *other.upstream]))
        current_timestamps = self.timestamp if isinstance(self.timestamp, list) else [self.timestamp]
        other_timestamps = other.timestamp if isinstance(other.timestamp, list) else [other.timestamp]
        combined_timestamp = current_timestamps + other_timestamps

        return self.__class__.model_construct(
            **combined_payload,
            params=combined_params,
            upstream=combined_upstream,
            timestamp=combined_timestamp,
        )

    # ==================== Schema & Discovery Methods ==========================
    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """
        Return the output schema (field names, types, descriptions, defaults).

        Useful for pipeline builders to:
          - Validate task wiring before execution
          - Show users available output keys
          - Detect typos in key names early

        Only payload fields are included. Use ``metadata_field_names()`` or
        ``metadata_dict()`` if you need to inspect execution metadata separately.

        Returns:
            dict with keys: 'fields', 'required', 'defaults', 'descriptions'

        Example:
            A subclass such as ``BiasOutput`` can call ``get_schema()`` to expose
            its payload field names, required fields, descriptions, and defaults.
        """
        schema = {
            'fields': {},
            'required': [],
            'defaults': {},
            'descriptions': {},
        }

        for field_name, field_info in cls.model_fields.items():
            if field_name in cls._metadata_field_names():
                continue

            # Field type
            annotation = field_info.annotation
            type_str = getattr(annotation, '__name__', str(annotation))
            schema['fields'][field_name] = type_str

            # Check if required (no default)
            if field_info.is_required():
                schema['required'].append(field_name)
            else:
                schema['defaults'][field_name] = field_info.default

            # Description
            if field_info.description:
                schema['descriptions'][field_name] = field_info.description

        return schema

    @classmethod
    def get_empty_instance(cls) -> TaskResult:
        """
        Return a dummy/empty instance of this result class.

        Useful for pipeline introspection (e.g., "what keys does this task output?")
        without running the task. Required fields get type-appropriate defaults,
        optional ones use their declared defaults.

        The returned instance supports both ``payload_dict()`` and ``metadata_dict()``
        if callers want to inspect those categories separately.

        Returns:
            An instance with placeholder values.

        Example:
            A subclass can call ``get_empty_instance()`` to obtain a placeholder
            instance for schema inspection, and ``list(dummy.keys())`` will show
            the available payload keys.
        """
        def get_type_default(annotation):
            """Get a sensible default for a type."""
            try:
                # Check the type annotation
                type_name = getattr(annotation, '__name__', str(annotation))

                if 'float' in type_name:
                    return 0.0
                elif 'int' in type_name:
                    return 0
                elif 'str' in type_name:
                    return ""
                elif 'bool' in type_name:
                    return False
                elif 'list' in type_name.lower():
                    return []
                elif 'dict' in type_name.lower():
                    return {}
                else:
                    return None
            except:
                return None

        kwargs = {}
        for field_name, field_info in cls.model_fields.items():
            if field_info.is_required():
                # Get a type-appropriate default
                kwargs[field_name] = get_type_default(field_info.annotation)
            else:
                # Use the field's default or default_factory
                if field_info.default is not None:
                    kwargs[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    kwargs[field_name] = field_info.default_factory()
                else:
                    # Fall back to None or type default
                    kwargs[field_name] = get_type_default(field_info.annotation)

        try:
            instance = cls(**kwargs)
            return instance
        except Exception as e:
            raise ValueError(
                f"Cannot create empty instance of {cls.__name__}. "
                f"Override get_empty_instance() in your Result class for custom logic. "
                f"Error: {e}"
            ) from e

    @classmethod
    def print_schema(cls, verbose: bool = False) -> None:
        """
        Pretty-print the payload schema for user reference.

        Args:
            verbose: If True, show descriptions and defaults.

        Example:
            ``MyTaskOutput.print_schema(verbose=True)`` prints the payload fields,
            whether each field is required, and optionally their descriptions and
            defaults. Metadata fields are intentionally omitted; inspect them with
            ``metadata_field_names()`` if needed.
        """
        schema = cls.get_schema()
        print(f"\n{cls.__name__} Output Schema:")
        for field_name in schema['fields'].keys():
            type_str = schema['fields'][field_name]
            is_required = field_name in schema['required']
            req_str = "required" if is_required else "optional"
            print(f"  - {field_name}: {type_str} ({req_str})")

            if verbose:
                if field_name in schema['descriptions']:
                    print(f"      Description: {schema['descriptions'][field_name]}")
                if field_name in schema['defaults']:
                    print(f"      Default: {schema['defaults'][field_name]}")
        print()

    def save(self, filepath: str) -> None:
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, f"{self.__class__.__name__}_metadata.json"), "w") as f:
            f.write(self.to_json(indent=2, exclude=set(self.__class__.payload_field_names())))

    @classmethod
    def load_metadata(cls, filepath: str) -> dict[str, Any]:
        """Load decoded metadata for a result whose payload is stored separately."""
        with open(os.path.join(filepath, f"{cls.__name__}_metadata.json"), "r") as f:
            return cls._decode_json_value(json.load(f))

    @classmethod
    def load(cls, filepath: str):
        pass
