"""FDL-first helpers for source-backed gem5 raw stat audits."""

from .audit import (
    Gem5AuditResult,
    LoadedGem5Model,
    audit_raw_stat_long,
    audit_raw_stat_long_path,
    audit_stat_values,
    bind_counter_values,
    counter_selectors,
    load_fdl_model,
    require_fully_bound_model,
    selector_coverage,
    unbound_model_counters,
)
from .raw_stats import (
    WINDOW_KEY_COLUMNS,
    iter_window_stats,
    numeric_stat_value,
    read_raw_stat_long,
    validate_raw_stat_long,
)
from .sidecar import (
    CounterBinding,
    Gem5Sidecar,
    SourceReference,
    load_sidecar,
)
from .source_index import SourceEvidence, validate_source_references

__all__ = [
    "CounterBinding",
    "Gem5AuditResult",
    "Gem5Sidecar",
    "LoadedGem5Model",
    "SourceEvidence",
    "SourceReference",
    "WINDOW_KEY_COLUMNS",
    "audit_raw_stat_long",
    "audit_raw_stat_long_path",
    "audit_stat_values",
    "bind_counter_values",
    "counter_selectors",
    "iter_window_stats",
    "load_fdl_model",
    "load_sidecar",
    "numeric_stat_value",
    "read_raw_stat_long",
    "require_fully_bound_model",
    "selector_coverage",
    "unbound_model_counters",
    "validate_raw_stat_long",
    "validate_source_references",
]
