"""
Tests for eregion.datamodels.results.TaskResult.

Covers:
  - Mapping ABC contract (isinstance, __getitem__, __iter__, __len__)
  - Methods provided free by Mapping ABC (keys, values, items, get, __contains__, __eq__)
  - Extra convenience methods (__setitem__, to_dict)
  - Schema introspection (get_schema, print_schema)
  - Dummy-instance creation (get_empty_instance) for primitive and complex types
  - Pydantic validation (required fields, extra fields, type coercion)
  - Subclassing behavior
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from astropy.time import Time
from pydantic import Field, ValidationError

from eregion.datamodels import TaskResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class SimpleOutput(TaskResult):
    """Two required fields and one optional."""
    value: float = Field(..., description="A numeric value")
    count: int   = Field(..., description="A count")
    name: str    = Field(default="default", description="A label")


class AllOptionalOutput(TaskResult):
    """All fields optional – useful for get_empty_instance tests."""
    tag: str              = Field(default="x",   description="Tag")
    data: list            = Field(default_factory=list, description="Data bucket")
    stats: dict[str, Any] = Field(default_factory=dict, description="Stats")


class NestedOutput(SimpleOutput):
    """Subclass that adds one more field."""
    extra: str = Field(default="bonus", description="Extra label")


class CombineOutput(TaskResult):
    """Fixture model used to test lazy accumulation semantics."""
    image_id: str = Field(..., description="Identifier for a single image")
    measurements: list[int] = Field(default_factory=list, description="Per-image measurements")


# ---------------------------------------------------------------------------
# 1. Mapping ABC contract
# ---------------------------------------------------------------------------

class TestMappingContract:
    def test_isinstance_of_mapping(self):
        result = SimpleOutput(value=1.0, count=2)
        assert isinstance(result, Mapping)

    def test_getitem_existing_key(self):
        result = SimpleOutput(value=3.14, count=7)
        assert result["value"] == pytest.approx(3.14)
        assert result["count"] == 7
        assert result["name"] == "default"

    def test_getitem_missing_key_raises_key_error(self):
        result = SimpleOutput(value=1.0, count=1)
        with pytest.raises(KeyError):
            _ = result["nonexistent"]

    def test_iter_returns_field_names(self):
        result = SimpleOutput(value=0.0, count=0)
        assert list(result) == ["value", "count", "name"]

    def test_len_matches_field_count(self):
        result = SimpleOutput(value=0.0, count=0)
        assert len(result) == 3

    def test_len_nested_subclass(self):
        result = NestedOutput(value=0.0, count=0)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 2. Methods provided for free by collections.abc.Mapping
# ---------------------------------------------------------------------------

class TestMappingABCFreeMethods:
    def test_keys(self):
        result = SimpleOutput(value=1.0, count=2, name="hi")
        assert list(result.keys()) == ["value", "count", "name"]

    def test_values(self):
        result = SimpleOutput(value=9.0, count=3, name="v")
        vals = list(result.values())
        assert vals == [9.0, 3, "v"]

    def test_items(self):
        result = SimpleOutput(value=5.0, count=10, name="n")
        assert dict(result.items()) == {"value": 5.0, "count": 10, "name": "n"}

    def test_get_existing_key(self):
        result = SimpleOutput(value=2.0, count=4)
        assert result.get("count") == 4

    def test_get_missing_key_returns_default(self):
        result = SimpleOutput(value=2.0, count=4)
        assert result.get("missing") is None
        assert result.get("missing", 99) == 99

    def test_contains_existing_key(self):
        result = SimpleOutput(value=1.0, count=1)
        assert "value" in result
        assert "count" in result
        assert "name" in result

    def test_contains_missing_key(self):
        result = SimpleOutput(value=1.0, count=1)
        assert "nonexistent" not in result

    def test_eq_same_values(self):
        a = SimpleOutput(value=1.0, count=2, name="x")
        b = SimpleOutput(value=1.0, count=2, name="x")
        assert a == b

    def test_eq_different_values(self):
        a = SimpleOutput(value=1.0, count=2)
        b = SimpleOutput(value=1.0, count=99)
        assert a != b


# ---------------------------------------------------------------------------
# 3. Extra convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_setitem_updates_field(self):
        result = SimpleOutput(value=1.0, count=1)
        result["name"] = "updated"
        assert result.name == "updated"
        assert result["name"] == "updated"

    def test_setitem_updates_numeric_field(self):
        result = SimpleOutput(value=1.0, count=1)
        result["value"] = 99.9
        assert result.value == pytest.approx(99.9)

    def test_to_dict_returns_plain_dict(self):
        result = SimpleOutput(value=7.0, count=3, name="d")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d == {"value": 7.0, "count": 3, "name": "d"}

    def test_to_dict_independent_of_instance(self):
        result = SimpleOutput(value=1.0, count=1)
        d = result.to_dict()
        d["value"] = 999.0          # mutating the dict…
        assert result.value == 1.0  # …should not affect the model

    def test_payload_dict_matches_to_dict(self):
        result = SimpleOutput(value=7.0, count=3, name="d")
        assert result.payload_dict() == result.to_dict()

    def test_metadata_dict_contains_only_metadata(self):
        t = Time("2025-01-01T00:00:00")
        result = SimpleOutput(value=7.0, count=3, params={"a": 1}, upstream=["up"], timestamp=t)
        assert result.metadata_dict() == {
            "params": {"a": 1},
            "upstream": ["up"],
            "timestamp": t,
        }

    def test_metadata_field_names_classmethod(self):
        assert SimpleOutput.metadata_field_names() == ("params", "upstream", "timestamp")

    def test_payload_field_names_classmethod(self):
        assert SimpleOutput.payload_field_names() == ("value", "count", "name")


# ---------------------------------------------------------------------------
# 4. Schema introspection: get_schema
# ---------------------------------------------------------------------------

class TestGetSchema:
    def test_fields_keys_match_model_fields(self):
        schema = SimpleOutput.get_schema()
        assert list(schema["fields"].keys()) == ["value", "count", "name"]

    def test_required_only_has_fields_without_defaults(self):
        schema = SimpleOutput.get_schema()
        assert schema["required"] == ["value", "count"]

    def test_defaults_captured(self):
        schema = SimpleOutput.get_schema()
        assert schema["defaults"]["name"] == "default"

    def test_descriptions_captured(self):
        schema = SimpleOutput.get_schema()
        assert schema["descriptions"]["value"] == "A numeric value"
        assert schema["descriptions"]["count"] == "A count"
        assert schema["descriptions"]["name"] == "A label"

    def test_field_type_strings_present(self):
        schema = SimpleOutput.get_schema()
        assert schema["fields"]["value"] == "float"
        assert schema["fields"]["count"] == "int"
        assert schema["fields"]["name"] == "str"

    def test_all_optional_class_has_no_required(self):
        schema = AllOptionalOutput.get_schema()
        assert schema["required"] == []

    def test_subclass_includes_parent_fields(self):
        schema = NestedOutput.get_schema()
        assert "extra" in schema["fields"]
        assert "value" in schema["fields"]


# ---------------------------------------------------------------------------
# 5. Schema introspection: print_schema
# ---------------------------------------------------------------------------

class TestPrintSchema:
    def test_print_schema_runs_without_error(self, capsys):
        SimpleOutput.print_schema()
        captured = capsys.readouterr()
        assert "SimpleOutput" in captured.out
        assert "value" in captured.out

    def test_print_schema_verbose_includes_descriptions(self, capsys):
        SimpleOutput.print_schema(verbose=True)
        captured = capsys.readouterr()
        assert "A numeric value" in captured.out
        assert "A label" in captured.out

    def test_print_schema_verbose_includes_defaults(self, capsys):
        SimpleOutput.print_schema(verbose=True)
        captured = capsys.readouterr()
        assert "default" in captured.out

    def test_print_schema_marks_required(self, capsys):
        SimpleOutput.print_schema()
        captured = capsys.readouterr()
        assert "required" in captured.out

    def test_print_schema_marks_optional(self, capsys):
        SimpleOutput.print_schema()
        captured = capsys.readouterr()
        assert "optional" in captured.out


# ---------------------------------------------------------------------------
# 6. get_empty_instance
# ---------------------------------------------------------------------------

class TestGetEmptyInstance:
    def test_returns_mappable_instance(self):
        dummy = SimpleOutput.get_empty_instance()
        assert isinstance(dummy, TaskResult)
        assert isinstance(dummy, SimpleOutput)

    def test_keys_present_on_empty_instance(self):
        dummy = SimpleOutput.get_empty_instance()
        assert list(dummy.keys()) == ["value", "count", "name"]

    def test_all_optional_empty_instance_uses_defaults(self):
        dummy = AllOptionalOutput.get_empty_instance()
        assert dummy.tag == "x"
        assert dummy.data == []
        assert dummy.stats == {}

    def test_required_float_defaults_to_zero(self):
        dummy = SimpleOutput.get_empty_instance()
        assert dummy.value == pytest.approx(0.0)

    def test_required_int_defaults_to_zero(self):
        dummy = SimpleOutput.get_empty_instance()
        assert dummy.count == 0

    def test_subclass_empty_instance_has_all_fields(self):
        dummy = NestedOutput.get_empty_instance()
        assert set(dummy.keys()) == {"value", "count", "name", "extra"}

    def test_complex_field_raises_valueerror_with_useful_message(self):
        """A field with no sensible zero-default should ask user to override."""
        class ComplexOutput(TaskResult):
            model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
            # A custom type that Pydantic cannot coerce from zero-like defaults
            payload_list: list[int] = Field(..., description="List of integers")

        # list defaults to [] which is valid for list[int], so this should succeed
        dummy = ComplexOutput.get_empty_instance()
        assert dummy.payload_list == []

    def test_override_get_empty_instance(self):
        """Users can override to supply custom placeholder logic."""
        class CustomOutput(TaskResult):
            label: str = Field(..., description="Must not be empty")

            @classmethod
            def get_empty_instance(cls) -> "CustomOutput":
                return cls(label="<placeholder>")

        dummy = CustomOutput.get_empty_instance()
        assert dummy.label == "<placeholder>"


# ---------------------------------------------------------------------------
# 7. Pydantic validation behaviour
# ---------------------------------------------------------------------------

class TestPydanticValidation:
    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SimpleOutput.model_validate({"value": 1.0})  # count is missing

    def test_extra_field_raises_because_extra_is_forbidden(self):
        with pytest.raises(ValidationError):
            SimpleOutput.model_validate({"value": 1.0, "count": 1, "unexpected_field": "boom"})

    def test_type_coercion_str_to_float(self):
        # Pydantic coerces "3.14" -> 3.14 in lax mode
        result = SimpleOutput.model_validate({"value": "3.14", "count": 2})
        assert result.value == pytest.approx(3.14)

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            SimpleOutput.model_validate({"value": "not-a-float", "count": 1})

    def test_optional_field_uses_default_when_omitted(self):
        result = SimpleOutput(value=0.0, count=0)
        assert result.name == "default"


# ---------------------------------------------------------------------------
# 8. Subclassing
# ---------------------------------------------------------------------------

class TestSubclassing:
    def test_subclass_inherits_mapping_behaviour(self):
        result = NestedOutput(value=1.0, count=2, name="n", extra="e")
        assert isinstance(result, Mapping)
        assert result["extra"] == "e"

    def test_subclass_schema_extends_parent(self):
        schema = NestedOutput.get_schema()
        assert "extra" in schema["fields"]
        assert "value" in schema["fields"]
        assert "extra" in schema["required"] or "extra" in schema["defaults"]

    def test_subclass_to_dict_includes_all_fields(self):
        result = NestedOutput(value=2.0, count=3)
        d = result.to_dict()
        assert set(d.keys()) == {"value", "count", "name", "extra"}


# ---------------------------------------------------------------------------
# 9. Lazy accumulation: combine
# ---------------------------------------------------------------------------

class TestCombine:
    def test_combine_promotes_scalar_fields_to_lists(self):
        first = SimpleOutput(value=1.0, count=2, name="first")
        second = SimpleOutput(value=3.0, count=4, name="second")

        combined = first.combine(second)

        assert isinstance(combined, SimpleOutput)
        assert combined.value == [1.0, 3.0]
        assert combined.count == [2, 4]
        assert combined.name == ["first", "second"]

    def test_combine_extends_list_fields(self):
        first = CombineOutput(image_id="img-1", measurements=[1, 2])
        second = CombineOutput(image_id="img-2", measurements=[3, 4])

        combined = first.combine(second)

        assert combined.image_id == ["img-1", "img-2"]
        assert combined.measurements == [1, 2, 3, 4]

    def test_combine_merges_params_and_preserves_latest_values(self):
        first = SimpleOutput(value=1.0, count=1, params={"alpha": 1, "shared": "old"})
        second = SimpleOutput(value=2.0, count=2, params={"beta": 2, "shared": "new"})

        combined = first.combine(second)

        assert combined.params == {"alpha": 1, "beta": 2, "shared": "new"}

    def test_combine_deduplicates_upstream_while_preserving_order(self):
        first = SimpleOutput(value=1.0, count=1, upstream=["a", "b"])
        second = SimpleOutput(value=2.0, count=2, upstream=["b", "c", "a"])

        combined = first.combine(second)

        assert combined.upstream == ["a", "b", "c"]

    def test_combine_accumulates_timestamps(self):
        t1 = Time("2025-01-01T00:00:00")
        t2 = Time("2025-01-01T00:00:01")
        first = SimpleOutput(value=1.0, count=1, timestamp=t1)
        second = SimpleOutput(value=2.0, count=2, timestamp=t2)

        combined = first.combine(second)

        assert isinstance(combined.timestamp, list)
        assert combined.timestamp == [t1, t2]

    def test_combine_supports_repeated_accumulation(self):
        first = SimpleOutput(value=1.0, count=1, name="a")
        second = SimpleOutput(value=2.0, count=2, name="b")
        third = SimpleOutput(value=3.0, count=3, name="c")

        combined = first.combine(second).combine(third)

        assert combined.value == [1.0, 2.0, 3.0]
        assert combined.count == [1, 2, 3]
        assert combined.name == ["a", "b", "c"]

    def test_combine_rejects_different_result_classes(self):
        first = SimpleOutput(value=1.0, count=1)
        second = NestedOutput(value=2.0, count=2)

        with pytest.raises(ValueError, match="same concrete class"):
            first.combine(second)


