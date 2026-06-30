"""Exact audit for FDL counters bound to gem5 stat observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from upath.fdl import FdlToGraph
from upath.mfd.finaldiagram import FinalMicroFlowDiagram
from upath.solver import ModelFeasibilityProblem, ModelFeasibilitySolution
from upath.solver.cvconstraints import ExactValues

from .raw_stats import (
    WINDOW_KEY_COLUMNS,
    iter_window_stats,
    numeric_gem5_value,
    read_raw_stat_long,
)
from .sidecar import Gem5Sidecar, load_sidecar


COUNTER_OBSERVATION_COLUMNS = (
    *WINDOW_KEY_COLUMNS,
    "model_id",
    "module_id",
    "counter",
    "counter_value",
    "audit_status",
)
COUNTER_BINDING_COLUMNS = (
    "model_id",
    "module_id",
    "counter",
    "gem5_stat",
)


@dataclass(frozen=True)
class LoadedGem5Model:
    """FDL model plus thin gem5 stat bindings."""

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
    mfd = FinalMicroFlowDiagram(graph)
    return LoadedGem5Model(sidecar=sidecar, mfd=mfd)


def counter_gem5_stats(sidecar: Gem5Sidecar) -> dict[str, str]:
    """Return FDL counter to gem5 stat mapping."""
    return {
        counter: binding.gem5_stat
        for counter, binding in sidecar.counter_bindings.items()
    }


def unbound_model_counters(model: LoadedGem5Model) -> tuple[str, ...]:
    """Return FDL counters without raw stat bindings."""
    return tuple(sorted(set(model.mfd.Counters()) - set(model.sidecar.counter_bindings)))


def require_fully_bound_model(model: LoadedGem5Model) -> None:
    """Require every FDL counter to have one raw gem5 stat binding."""
    gem5_stats = counter_gem5_stats(model.sidecar)
    if not gem5_stats:
        raise ValueError(
            f"model {model.sidecar.model_id} has no raw stat bindings; "
            "use validate-model for skeleton FDLs"
        )
    unbound = unbound_model_counters(model)
    if unbound:
        raise ValueError(f"FDL counters missing raw stat bindings: {list(unbound)}")


def gem5_stat_coverage(
    model: LoadedGem5Model,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Report sidecar gem5 stat presence for the whole table and each window."""
    gem5_stats = counter_gem5_stats(model.sidecar)
    windows = tuple(iter_window_stats(table))
    all_seen_stats = {gem5_stat for _, stats in windows for gem5_stat in stats}
    rows: list[dict[str, Any]] = []
    for counter, gem5_stat in gem5_stats.items():
        missing_windows = tuple(
            tuple(metadata[column] for column in WINDOW_KEY_COLUMNS)
            for metadata, stats in windows
            if gem5_stat not in stats
        )
        rows.append(
            {
                "counter": counter,
                "gem5_stat": gem5_stat,
                "present_in_table": gem5_stat in all_seen_stats,
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
    """Bind FDL counters to exact values from flattened gem5 stats."""
    require_fully_bound_model(model)
    gem5_stats = counter_gem5_stats(model.sidecar)
    missing = [gem5_stat for gem5_stat in gem5_stats.values() if gem5_stat not in stats]
    if missing:
        raise ValueError(f"missing gem5 stats: {missing}")
    return {
        counter: numeric_gem5_value(stats[gem5_stat], gem5_stat)
        for counter, gem5_stat in gem5_stats.items()
    }


def audit_gem5_stats(
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
    gem5_stats = counter_gem5_stats(model.sidecar)
    gem5_stat_to_counter = {
        gem5_stat: counter for counter, gem5_stat in gem5_stats.items()
    }
    rows: list[dict[str, Any]] = []
    for metadata, stats in iter_window_stats(table):
        missing_gem5_stats = tuple(
            gem5_stat for gem5_stat in gem5_stats.values() if gem5_stat not in stats
        )
        if missing_gem5_stats:
            status = "MissingStat"
            counter_values = {}
        else:
            result = audit_gem5_stats(model, stats, window=metadata)
            status = result.status
            counter_values = dict(result.values)
        missing_counters = tuple(
            gem5_stat_to_counter[gem5_stat] for gem5_stat in missing_gem5_stats
        )
        rows.append(
            {
                **metadata,
                "model_id": model.sidecar.model_id,
                "module_id": model.sidecar.module_id,
                "audit_status": status,
                "bound_counters": tuple(counter_values),
                "counter_values": counter_values,
                "missing_gem5_stats": missing_gem5_stats,
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


def build_counter_observations(
    models: Iterable[LoadedGem5Model],
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Build counter observations from raw gem5 stat observations."""
    loaded_models = tuple(models)
    if not loaded_models:
        raise ValueError("at least one loaded gem5 model is required")
    windows = tuple(iter_window_stats(table))
    rows: list[dict[str, Any]] = []

    for model in loaded_models:
        require_fully_bound_model(model)
        bindings = model.sidecar.counter_bindings
        gem5_stats = counter_gem5_stats(model.sidecar)
        for metadata, stats in windows:
            missing_gem5_stats = frozenset(
                gem5_stat for gem5_stat in gem5_stats.values() if gem5_stat not in stats
            )
            if missing_gem5_stats:
                audit_status = "MissingStat"
            else:
                audit_status = audit_gem5_stats(
                    model,
                    stats,
                    window=metadata,
                ).status

            for counter, binding in sorted(bindings.items()):
                gem5_stat = binding.gem5_stat
                rows.append(
                    {
                        **metadata,
                        "model_id": model.sidecar.model_id,
                        "module_id": model.sidecar.module_id,
                        "counter": counter,
                        "counter_value": (
                            float("nan")
                            if gem5_stat in missing_gem5_stats
                            else numeric_gem5_value(stats[gem5_stat], gem5_stat)
                        ),
                        "audit_status": audit_status,
                    }
                )

    return pd.DataFrame(rows, columns=COUNTER_OBSERVATION_COLUMNS)


def build_counter_binding_table(
    models: Iterable[LoadedGem5Model],
) -> pd.DataFrame:
    """Build the sidecar-derived counter to gem5 stat binding table."""
    rows: list[dict[str, Any]] = []
    for model in models:
        require_fully_bound_model(model)
        for counter, binding in sorted(model.sidecar.counter_bindings.items()):
            rows.append(
                {
                    "model_id": model.sidecar.model_id,
                    "module_id": model.sidecar.module_id,
                    "counter": counter,
                    "gem5_stat": binding.gem5_stat,
                }
            )
    return pd.DataFrame(rows, columns=COUNTER_BINDING_COLUMNS)
