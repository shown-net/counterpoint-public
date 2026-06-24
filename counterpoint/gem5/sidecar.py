"""Thin gem5 sidecar binding FDL counters to source-backed raw stats."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


_ROOT_KEYS = {
    "model_id",
    "module_id",
    "fdl_path",
    "gem5_revision",
    "source_refs",
    "counter_bindings",
}
_BANNED_KEYS = {
    "actions",
    "canonical",
    "event_group",
    "flow",
    "g_group",
    "metric",
    "pmu",
    "population_unit",
}
_SOURCE_REF_KEYS = {"path", "symbol", "snippets"}
_COUNTER_BINDING_KEYS = {"stat_selector", "declaration_ref", "increment_refs"}


def _exact_keys(node: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(node))
    unknown = sorted(set(node) - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch: missing={missing}, unknown={unknown}")


def _reject_banned_keys(node: Mapping[str, Any], context: str) -> None:
    banned = sorted(_BANNED_KEYS & set(node))
    if banned:
        raise ValueError(f"{context} contains banned keys: {banned}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SourceReference:
    """Source symbol and snippets used as evidence for one binding."""

    path: str
    symbol: str
    snippets: tuple[str, ...]


@dataclass(frozen=True)
class CounterBinding:
    """Logical counter to raw gem5 stat selector binding."""

    stat_selector: str
    declaration_ref: str
    increment_refs: tuple[str, ...]


@dataclass(frozen=True)
class Gem5Sidecar:
    """FDL-first gem5 model sidecar."""

    path: Path
    model_id: str
    module_id: str
    fdl_path: Path
    gem5_revision: str
    source_refs: Mapping[str, SourceReference]
    counter_bindings: Mapping[str, CounterBinding]


def _parse_source_refs(raw: Any) -> dict[str, SourceReference]:
    if not isinstance(raw, dict):
        raise ValueError("source_refs must be an object")
    parsed: dict[str, SourceReference] = {}
    for ref_id, node in raw.items():
        _nonempty_string(ref_id, "source reference id")
        if not isinstance(node, dict):
            raise ValueError(f"source_refs.{ref_id} must be an object")
        _reject_banned_keys(node, f"source_refs.{ref_id}")
        _exact_keys(node, _SOURCE_REF_KEYS, f"source_refs.{ref_id}")
        snippets = node["snippets"]
        if not isinstance(snippets, list) or any(
            not isinstance(item, str) or not item for item in snippets
        ):
            raise ValueError(f"source_refs.{ref_id}.snippets must be a string list")
        parsed[str(ref_id)] = SourceReference(
            path=_nonempty_string(node["path"], f"source_refs.{ref_id}.path"),
            symbol=_nonempty_string(node["symbol"], f"source_refs.{ref_id}.symbol"),
            snippets=tuple(snippets),
        )
    return parsed


def _parse_counter_bindings(
    raw: Any,
    source_refs: Mapping[str, SourceReference],
) -> dict[str, CounterBinding]:
    if not isinstance(raw, dict):
        raise ValueError("counter_bindings must be an object")
    parsed: dict[str, CounterBinding] = {}
    for counter, node in raw.items():
        _nonempty_string(counter, "counter id")
        if not isinstance(node, dict):
            raise ValueError(f"counter_bindings.{counter} must be an object")
        _reject_banned_keys(node, f"counter_bindings.{counter}")
        _exact_keys(node, _COUNTER_BINDING_KEYS, f"counter_bindings.{counter}")
        declaration_ref = _nonempty_string(
            node["declaration_ref"],
            f"counter_bindings.{counter}.declaration_ref",
        )
        increment_refs = node["increment_refs"]
        if not isinstance(increment_refs, list) or any(
            not isinstance(item, str) or not item for item in increment_refs
        ):
            raise ValueError(
                f"counter_bindings.{counter}.increment_refs must be a string list"
            )
        refs = (declaration_ref, *increment_refs)
        unknown_refs = sorted(set(refs) - set(source_refs))
        if unknown_refs:
            raise ValueError(
                f"counter_bindings.{counter} has unknown source refs: {unknown_refs}"
            )
        parsed[str(counter)] = CounterBinding(
            stat_selector=_nonempty_string(
                node["stat_selector"],
                f"counter_bindings.{counter}.stat_selector",
            ),
            declaration_ref=declaration_ref,
            increment_refs=tuple(increment_refs),
        )
    return parsed


def load_sidecar(path: Path) -> Gem5Sidecar:
    """Load a thin sidecar without interpreting the FDL model."""
    sidecar_path = path.resolve()
    raw = json.loads(
        sidecar_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(raw, dict):
        raise ValueError("sidecar root must be an object")
    _reject_banned_keys(raw, "sidecar")
    _exact_keys(raw, _ROOT_KEYS, "sidecar")

    fdl_path = sidecar_path.parent / _nonempty_string(raw["fdl_path"], "fdl_path")
    fdl_path = fdl_path.resolve()
    if fdl_path.parent != sidecar_path.parent:
        raise ValueError("fdl_path must stay in the sidecar directory")
    if not fdl_path.is_file():
        raise ValueError(f"fdl_path does not exist: {fdl_path}")

    source_refs = _parse_source_refs(raw["source_refs"])
    counter_bindings = _parse_counter_bindings(raw["counter_bindings"], source_refs)

    return Gem5Sidecar(
        path=sidecar_path,
        model_id=_nonempty_string(raw["model_id"], "model_id"),
        module_id=_nonempty_string(raw["module_id"], "module_id"),
        fdl_path=fdl_path,
        gem5_revision=_nonempty_string(raw["gem5_revision"], "gem5_revision"),
        source_refs=source_refs,
        counter_bindings=counter_bindings,
    )
