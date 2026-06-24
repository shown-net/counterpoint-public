"""Exact audit for FDL-first source-backed gem5 observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from upath.fdl import FdlToGraph
from upath.mfd.finaldiagram import FinalMicroFlowDiagram
from upath.solver import ModelFeasibilityProblem, ModelFeasibilitySolution
from upath.solver.cvconstraints import ExactValues

from .raw_stats import (
    WINDOW_KEY_COLUMNS,
    iter_window_stats,
    numeric_stat_value,
    read_raw_stat_long,
)
from .sidecar import Gem5Sidecar, load_sidecar


@dataclass(frozen=True)
class LoadedGem5Model:
    """FDL model plus thin sidecar bindings."""

    sidecar: Gem5Sidecar
    mfd: FinalMicroFlowDiagram


@dataclass(frozen=True)
class Gem5AuditResult:
    """Exact feasibility result for one gem5 stats observation."""

    values: Mapping[str, float]
    solution: ModelFeasibilitySolution
    window: Mapping[str, Any] | None = None

    @property
    def status(self) -> str:
        """LP solver feasibility status."""
        return self.solution.Status()


def load_fdl_model(sidecar_path: Path) -> LoadedGem5Model:
    """Load a sidecar and finalize its FDL as the primary model."""
    sidecar = load_sidecar(sidecar_path)
    graph = FdlToGraph.build_graph_from_fdl(str(sidecar.fdl_path))
    fdl_counters = set(graph.Counters())
    unknown_bindings = sorted(set(sidecar.counter_bindings) - fdl_counters)
    if unknown_bindings:
        raise ValueError(f"sidecar binds counters not present in FDL: {unknown_bindings}")
    mfd = FinalMicroFlowDiagram(
        graph,
        model_id=sidecar.model_id,
        module_id=sidecar.module_id,
        metadata={
            "gem5_revision": sidecar.gem5_revision,
            "counter_bindings": counter_selectors(sidecar),
        },
    )
    return LoadedGem5Model(sidecar=sidecar, mfd=mfd)


def counter_selectors(sidecar: Gem5Sidecar) -> dict[str, str]:
    """Return logical counter to raw stat selector mapping."""
    return {
        counter: binding.stat_selector
        for counter, binding in sidecar.counter_bindings.items()
    }


def unbound_model_counters(model: LoadedGem5Model) -> tuple[str, ...]:
    """Return FDL counters without raw stat bindings."""
    return tuple(sorted(set(model.mfd.Counters()) - set(model.sidecar.counter_bindings)))


def require_fully_bound_model(model: LoadedGem5Model) -> None:
    """Require every FDL counter to have one raw gem5 stat binding."""
    selectors = counter_selectors(model.sidecar)
    if not selectors:
        raise ValueError(
            f"model {model.sidecar.model_id} has no raw stat bindings; "
            "use validate-model for skeleton FDLs"
        )
    unbound = unbound_model_counters(model)
    if unbound:
        raise ValueError(f"FDL counters missing raw stat bindings: {list(unbound)}")


def selector_coverage(
    model: LoadedGem5Model,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Report sidecar selector presence for the whole table and each window."""
    selectors = counter_selectors(model.sidecar)
    windows = tuple(iter_window_stats(table))
    all_seen_selectors = {selector for _, stats in windows for selector in stats}
    rows: list[dict[str, Any]] = []
    for counter, selector in selectors.items():
        missing_windows = tuple(
            tuple(metadata[column] for column in WINDOW_KEY_COLUMNS)
            for metadata, stats in windows
            if selector not in stats
        )
        rows.append(
            {
                "logical_counter": counter,
                "stat_selector": selector,
                "present_in_table": selector in all_seen_selectors,
                "window_count": len(windows),
                "missing_window_count": len(missing_windows),
                "missing_windows": missing_windows,
            }
        )
    return pd.DataFrame(rows)


def bind_counter_values(
    model: LoadedGem5Model,
    stats: Mapping[str, float],
) -> dict[str, float]:
    """Bind FDL logical counters to exact values from flattened gem5 stats."""
    require_fully_bound_model(model)
    selectors = counter_selectors(model.sidecar)
    missing = [selector for selector in selectors.values() if selector not in stats]
    if missing:
        raise ValueError(f"missing gem5 stat selectors: {missing}")
    return {
        counter: numeric_stat_value(stats[selector], selector)
        for counter, selector in selectors.items()
    }


def audit_stat_values(
    model: LoadedGem5Model,
    stats: Mapping[str, float],
    *,
    window: Mapping[str, Any] | None = None,
) -> Gem5AuditResult:
    """Run exact feasibility for one flattened gem5 stat observation."""
    values = bind_counter_values(model, stats)
    solution = ModelFeasibilityProblem(model.mfd, ExactValues(values)).Solve()
    return Gem5AuditResult(values=values, solution=solution, window=window)


def audit_raw_stat_long(
    model: LoadedGem5Model,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Run exact audit for every window in a raw gem5 stat long table."""
    require_fully_bound_model(model)
    selectors = counter_selectors(model.sidecar)
    selector_to_counter = {selector: counter for counter, selector in selectors.items()}
    rows: list[dict[str, Any]] = []
    for metadata, stats in iter_window_stats(table):
        missing_selectors = tuple(
            selector for selector in selectors.values() if selector not in stats
        )
        if missing_selectors:
            status = "MissingStat"
            counter_values = {}
        else:
            result = audit_stat_values(model, stats, window=metadata)
            status = result.status
            counter_values = dict(result.values)
        missing_counters = tuple(
            selector_to_counter[selector] for selector in missing_selectors
        )
        rows.append(
            {
                **metadata,
                "model_id": model.sidecar.model_id,
                "module_id": model.sidecar.module_id,
                "audit_status": status,
                "bound_counters": tuple(counter_values),
                "counter_values": counter_values,
                "missing_selectors": missing_selectors,
                "missing_counters": missing_counters,
                "unbound_counters": (),
            }
        )
    return pd.DataFrame(rows)


def audit_raw_stat_long_path(
    model: LoadedGem5Model,
    path: Path,
) -> pd.DataFrame:
    """Read and audit a cpu_microarchitecture raw gem5 stat long parquet file."""
    return audit_raw_stat_long(model, read_raw_stat_long(path))
