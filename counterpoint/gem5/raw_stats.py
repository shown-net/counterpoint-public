"""Reader for raw gem5 stat observation long tables."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


WINDOW_KEY_COLUMNS = ("workload_id", "parent_window_id", "window_id")
RAW_STAT_REQUIRED_COLUMNS = (*WINDOW_KEY_COLUMNS, "gem5_stat", "gem5_value")


def numeric_gem5_value(value: Any, context: str) -> float:
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
    duplicate_mask = table.duplicated([*WINDOW_KEY_COLUMNS, "gem5_stat"])
    if duplicate_mask.any():
        duplicate = table.loc[duplicate_mask].iloc[0]
        key = tuple(str(duplicate[column]) for column in WINDOW_KEY_COLUMNS)
        raise ValueError(
            f"duplicate gem5_stat in window {key}: {duplicate['gem5_stat']}"
        )
    for index, value in table["gem5_value"].items():
        numeric_gem5_value(value, f"gem5_value at row {index}")


def read_raw_stat_long(path: Path) -> pd.DataFrame:
    """Read a cpu_microarchitecture raw gem5 stat long parquet file."""
    table = pd.read_parquet(path)
    validate_raw_stat_long(table)
    return table


def iter_window_stats(
    table: pd.DataFrame,
) -> Iterable[tuple[dict[str, Any], dict[str, float]]]:
    """Yield minimal per-window keys and gem5 stat-value maps."""
    validate_raw_stat_long(table)
    for key, group in table.groupby(list(WINDOW_KEY_COLUMNS), sort=True, dropna=False):
        metadata = dict(zip(WINDOW_KEY_COLUMNS, key, strict=True))
        stats = {
            str(row.gem5_stat): numeric_gem5_value(
                row.gem5_value,
                str(row.gem5_stat),
            )
            for row in group.itertuples(index=False)
        }
        yield metadata, stats
