"""Export CounterPoint-native stat features and count relations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable

import pandas as pd

from .audit import LoadedGem5Model, require_fully_bound_model


Q_ATOM_COLUMNS = (
    "q_atom_id",
    "module_id",
    "counter",
    "gem5_stat",
)
COUNT_RELATION_COLUMNS = (
    "relation_id",
    "sense",
    "rhs",
)
COUNT_RELATION_TERM_COLUMNS = (
    "relation_id",
    "q_atom_id",
    "coefficient",
)


@dataclass(frozen=True)
class QAtomArtifact:
    q_atoms: pd.DataFrame


@dataclass(frozen=True)
class CountRelationArtifact:
    count_relation: pd.DataFrame
    count_relation_terms: pd.DataFrame


@dataclass(frozen=True)
class CountPointerArtifacts:
    q_atoms: pd.DataFrame
    count_relation: pd.DataFrame
    count_relation_terms: pd.DataFrame


def _q_atom_id(module_id: str, counter: str, gem5_stat: str) -> str:
    digest = hashlib.sha1(f"{module_id}\0{counter}\0{gem5_stat}".encode("utf-8")).hexdigest()[:16]
    return f"q_atom:{module_id}:{digest}"


def _relation_id(module_id: str, source_kind: str, index: int) -> str:
    return f"cp_relation:{module_id}:{source_kind}:{index}"


def _require_raw_candidate_columns(raw_stat_candidates: pd.DataFrame) -> None:
    required = {
        "gem5_stat",
        "anchor_module_id",
        "anchor_counter",
        "anchor_gem5_stat",
    }
    missing = sorted(required - set(raw_stat_candidates.columns))
    if missing:
        raise ValueError(f"raw_stat_candidates missing columns: {missing}")


def build_q_atoms(
    models: Iterable[LoadedGem5Model],
    raw_stat_candidates: pd.DataFrame,
) -> QAtomArtifact:
    """Build CounterPoint stat-backed q atoms for downstream H1 models."""
    loaded = tuple(models)
    if not loaded:
        raise ValueError("at least one loaded gem5 model is required")
    _require_raw_candidate_columns(raw_stat_candidates)
    candidates: set[tuple[str, str, str]] = set()
    for raw in raw_stat_candidates.itertuples(index=False):
        module_id = str(raw.anchor_module_id)
        counter = str(raw.anchor_counter)
        gem5_stat = str(raw.gem5_stat)
        anchor_gem5_stat = str(raw.anchor_gem5_stat)
        if gem5_stat != anchor_gem5_stat:
            raise ValueError(
                "q_atom binding candidate gem5_stat must match anchor_gem5_stat: "
                f"{module_id} {counter} {gem5_stat} != {anchor_gem5_stat}"
            )
        candidates.add((module_id, counter, gem5_stat))

    rows: list[dict[str, Any]] = []
    for model in loaded:
        require_fully_bound_model(model)
        for counter, binding in sorted(model.sidecar.counter_bindings.items()):
            module_id = str(model.sidecar.module_id)
            counter_id = str(counter)
            gem5_stat = str(binding.gem5_stat)
            key = (module_id, counter_id, gem5_stat)
            if key not in candidates:
                raise ValueError(
                    "raw stat candidates missing q_atom binding: "
                    f"{module_id} {counter_id} {gem5_stat}"
                )
            rows.append(
                {
                    "q_atom_id": _q_atom_id(module_id, counter_id, gem5_stat),
                    "module_id": module_id,
                    "counter": counter_id,
                    "gem5_stat": gem5_stat,
                }
            )
    frame = pd.DataFrame(rows, columns=Q_ATOM_COLUMNS)
    if frame.empty:
        raise ValueError("q_atoms are empty")
    sort_columns = ["module_id", "counter", "gem5_stat"]
    return QAtomArtifact(frame.sort_values(sort_columns).reset_index(drop=True))


def _load_cone_frames(model: LoadedGem5Model) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    counters = tuple(model.mfd.Counters())
    try:
        cone = model.mfd.counter_cone
        return cone.equalities, cone.inequalities, "ok"
    except ValueError:
        if len(counters) != 1:
            raise
        counter = counters[0]
        return (
            pd.DataFrame(columns=counters, dtype=float),
            pd.DataFrame([{counter: 1.0}], columns=counters, dtype=float),
            "degenerate_1d_cone",
        )


def _terms_from_items(items: Iterable[tuple[str, Any]]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for counter, coefficient in items:
        value = float(coefficient)
        if not math.isclose(value, 0.0):
            terms[str(counter)] = value
    return terms


def build_count_relations(
    models: Iterable[LoadedGem5Model],
    q_atoms: pd.DataFrame,
) -> CountRelationArtifact:
    """Build CounterPoint-native count relations and participating stat terms."""
    loaded = tuple(models)
    if not loaded:
        raise ValueError("at least one loaded gem5 model is required")
    missing = sorted(set(Q_ATOM_COLUMNS) - set(q_atoms.columns))
    if missing:
        raise ValueError(f"q_atoms missing columns: {missing}")

    q_atoms_by_counter: dict[tuple[str, str], dict[str, str]] = {}
    for row in q_atoms.itertuples(index=False):
        key = (str(row.module_id), str(row.counter))
        if key in q_atoms_by_counter:
            raise ValueError(f"duplicate q_atom counter binding: {key}")
        q_atoms_by_counter[key] = {
            "q_atom_id": str(row.q_atom_id),
        }

    relation_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    for model in loaded:
        require_fully_bound_model(model)
        equalities, inequalities, cone_status = _load_cone_frames(model)
        for source_kind, frame in (("equality", equalities), ("inequality", inequalities)):
            counter_columns = tuple(str(column) for column in frame.columns)
            for index, row in enumerate(frame.itertuples(index=False, name=None)):
                terms = _terms_from_items(zip(counter_columns, row, strict=True))
                if not terms:
                    continue
                relation_id = _relation_id(
                    model.sidecar.module_id,
                    source_kind,
                    index,
                )
                if cone_status == "degenerate_1d_cone":
                    sense = "ge"
                    rhs = 0.0
                else:
                    sense = "eq" if source_kind == "equality" else "le"
                    rhs = 0.0
                relation_rows.append(
                    {
                        "relation_id": relation_id,
                        "sense": sense,
                        "rhs": rhs,
                    }
                )
                for counter in sorted(terms):
                    mapped = q_atoms_by_counter.get((str(model.sidecar.module_id), counter))
                    if mapped is None:
                        raise ValueError(
                            "q_atoms missing binding for relation term: "
                            f"{model.sidecar.module_id} {counter}"
                        )
                    coefficient = float(terms[counter])
                    term_rows.append(
                        {
                            "relation_id": relation_id,
                            "q_atom_id": mapped["q_atom_id"],
                            "coefficient": coefficient,
                        }
                    )

    return CountRelationArtifact(
        count_relation=pd.DataFrame(
            relation_rows,
            columns=COUNT_RELATION_COLUMNS,
        ),
        count_relation_terms=pd.DataFrame(
            term_rows,
            columns=COUNT_RELATION_TERM_COLUMNS,
        ),
    )


def build_countpointer_artifacts(
    models: Iterable[LoadedGem5Model],
    raw_stat_candidates: pd.DataFrame,
) -> CountPointerArtifacts:
    """Build minimal H1 countpointer artifacts for a model set."""
    loaded = tuple(models)
    if not loaded:
        raise ValueError("at least one loaded gem5 model is required")
    q_atoms = build_q_atoms(loaded, raw_stat_candidates)
    relations = build_count_relations(
        loaded,
        q_atoms.q_atoms,
    )
    return CountPointerArtifacts(
        q_atoms=q_atoms.q_atoms,
        count_relation=relations.count_relation,
        count_relation_terms=relations.count_relation_terms,
    )
