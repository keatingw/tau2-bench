"""Tests for numeric-type canonicalization in hashing and tool-arg coercion.

Guards against the grader bug where `33` vs `33.0` in tool arguments produced
different DB hashes (and different deterministic record IDs), scoring
semantically correct runs as failures.
"""

import pytest

from tau2.environment.toolkit import ToolKitBase, ToolType, coerce_numeric_args, is_tool
from tau2.utils.utils import canonicalize_json_numbers, get_dict_hash


class TestCanonicalizeJsonNumbers:
    def test_integral_float_becomes_int(self):
        assert canonicalize_json_numbers(33.0) == 33
        assert isinstance(canonicalize_json_numbers(33.0), int)

    def test_int_unchanged(self):
        assert canonicalize_json_numbers(33) == 33
        assert isinstance(canonicalize_json_numbers(33), int)

    def test_non_integral_float_unchanged(self):
        assert canonicalize_json_numbers(33.5) == 33.5
        assert isinstance(canonicalize_json_numbers(33.5), float)

    def test_bool_untouched(self):
        assert canonicalize_json_numbers(True) is True
        assert canonicalize_json_numbers(False) is False

    def test_recurses_containers(self):
        obj = {"a": [1.0, 2.5, {"b": 3.0}], "c": (4.0,)}
        assert canonicalize_json_numbers(obj) == {"a": [1, 2.5, {"b": 3}], "c": [4]}

    def test_strings_untouched(self):
        assert canonicalize_json_numbers({"a": "33.0"}) == {"a": "33.0"}


class TestGetDictHashNumericEquivalence:
    def test_int_and_integral_float_hash_equal(self):
        assert get_dict_hash({"amount": 33}) == get_dict_hash({"amount": 33.0})

    def test_nested_structures_hash_equal(self):
        a = {"records": {"r1": {"amount": 98, "apy": 5}}, "list": [1, 2.0]}
        b = {"records": {"r1": {"amount": 98.0, "apy": 5.0}}, "list": [1.0, 2]}
        assert get_dict_hash(a) == get_dict_hash(b)

    def test_different_values_hash_differently(self):
        assert get_dict_hash({"amount": 33}) != get_dict_hash({"amount": 34})

    def test_string_vs_number_hash_differently(self):
        assert get_dict_hash({"amount": "33"}) != get_dict_hash({"amount": 33})

    def test_bool_vs_int_hash_differently(self):
        assert get_dict_hash({"flag": True}) != get_dict_hash({"flag": 1})


def _float_tool(amount: float, note: str = "") -> str:
    return "ok"


def _int_tool(count: int) -> str:
    return "ok"


def _optional_float_tool(amount: float | None = None) -> str:
    return "ok"


def _unannotated_tool(amount) -> str:
    return "ok"


class TestCoerceNumericArgs:
    def test_int_to_float_for_float_param(self):
        out = coerce_numeric_args(_float_tool, {"amount": 33})
        assert out["amount"] == 33.0 and isinstance(out["amount"], float)

    def test_integral_float_to_int_for_int_param(self):
        out = coerce_numeric_args(_int_tool, {"count": 5.0})
        assert out["count"] == 5 and isinstance(out["count"], int)

    def test_non_integral_float_not_coerced_to_int(self):
        out = coerce_numeric_args(_int_tool, {"count": 5.5})
        assert out["count"] == 5.5 and isinstance(out["count"], float)

    def test_optional_float_param(self):
        out = coerce_numeric_args(_optional_float_tool, {"amount": 33})
        assert out["amount"] == 33.0 and isinstance(out["amount"], float)

    def test_string_never_coerced(self):
        out = coerce_numeric_args(_float_tool, {"amount": "33"})
        assert out["amount"] == "33" and isinstance(out["amount"], str)

    def test_bool_never_coerced(self):
        out = coerce_numeric_args(_float_tool, {"amount": True})
        assert out["amount"] is True

    def test_unannotated_untouched(self):
        out = coerce_numeric_args(_unannotated_tool, {"amount": 33})
        assert out["amount"] == 33 and isinstance(out["amount"], int)

    def test_unknown_param_passes_through(self):
        out = coerce_numeric_args(_float_tool, {"bogus": 1})
        assert out == {"bogus": 1}


class TestUseToolCoercion:
    def test_use_tool_coerces_via_annotation(self):
        captured = {}

        class Kit(ToolKitBase):
            def __init__(self):
                pass

            @is_tool(ToolType.WRITE)
            def pay(self, amount: float) -> str:
                captured["amount"] = amount
                return "ok"

        Kit().use_tool("pay", amount=33)
        assert captured["amount"] == 33.0
        assert isinstance(captured["amount"], float)

    def test_use_tool_unknown_tool_still_raises(self):
        class Kit(ToolKitBase):
            def __init__(self):
                pass

        with pytest.raises(ValueError):
            Kit().use_tool("nope", amount=1)
