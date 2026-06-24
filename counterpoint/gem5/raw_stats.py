"""Reader for cpu_microarchitecture raw gem5 stat long tables."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


WINDOW_KEY_COLUMNS = ("workload_id", "parent_window_id", "window_id")
RAW_STAT_REQUIRED_COLUMNS = (*WINDOW_KEY_COLUMNS, "stat_selector", "stat_value")


def numeric_stat_value(value: Any, context: str) -> float:
    """Convert one raw stat value to float without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        raise ValueError(f"{context} must be finite")
    return result


def validate_raw_stat_long(table: pd.DataFrame) -> None:
    """Validate the raw gem5 stat long table contract."""
    missing = sorted(set(RAW_STAT_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"raw stat table missing columns: {missing}")
    duplicate_mask = table.duplicated([*WINDOW_KEY_COLUMNS, "stat_selector"])
    if duplicate_mask.any():
        duplicate = table.loc[duplicate_mask].iloc[0]
        key = tuple(str(duplicate[column]) for column in WINDOW_KEY_COLUMNS)
        raise ValueError(
            f"duplicate stat_selector in window {key}: {duplicate['stat_selector']}"
        )
    for index, value in table["stat_value"].items():
        numeric_stat_value(value, f"stat_value at row {index}")


def read_raw_stat_long(path: Path) -> pd.DataFrame:
    """Read a cpu_microarchitecture raw gem5 stat long parquet file."""
    table = pd.read_parquet(path)
    validate_raw_stat_long(table)
    return table


def iter_window_stats(
    table: pd.DataFrame,
) -> Iterable[tuple[dict[str, Any], dict[str, float]]]:
    """Yield minimal per-window keys and selector-value maps."""
    validate_raw_stat_long(table)
    for key, group in table.groupby(list(WINDOW_KEY_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(WINDOW_KEY_COLUMNS, key, strict=True))
        stats = {
            str(row.stat_selector): numeric_stat_value(
                row.stat_value,
                str(row.stat_selector),
            )
            for row in group.itertuples(index=False)
        }
        yield metadata, stats
