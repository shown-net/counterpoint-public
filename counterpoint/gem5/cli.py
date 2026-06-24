"""Command-line helpers for FDL-first gem5 audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import (
    audit_raw_stat_long,
    load_fdl_model,
    selector_coverage,
    unbound_model_counters,
)
from .raw_stats import read_raw_stat_long


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def audit_raw(model_path: Path, raw_path: Path, out_path: Path) -> Path:
    """Audit one raw stat long parquet and write parquet plus JSON summary."""
    model = load_fdl_model(model_path)
    table = read_raw_stat_long(raw_path)
    report = audit_raw_stat_long(model, table)
    coverage = selector_coverage(model, table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(out_path)
    summary = {
        "model_id": model.sidecar.model_id,
        "module_id": model.sidecar.module_id,
        "fdl_path": model.sidecar.fdl_path,
        "raw_path": raw_path,
        "audit_path": out_path,
        "window_count": int(len(report)),
        "audit_status_counts": report["audit_status"].value_counts().to_dict(),
        "counter_bindings": {
            counter: binding.stat_selector
            for counter, binding in model.sidecar.counter_bindings.items()
        },
        "selector_coverage": coverage.to_dict(orient="records"),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def validate_model(model_path: Path) -> dict[str, Any]:
    """Load one FDL-first model and report binding completeness."""
    model = load_fdl_model(model_path)
    bindings = {
        counter: binding.stat_selector
        for counter, binding in model.sidecar.counter_bindings.items()
    }
    return {
        "model_id": model.sidecar.model_id,
        "module_id": model.sidecar.module_id,
        "fdl_path": model.sidecar.fdl_path,
        "counter_count": len(model.mfd.Counters()),
        "path_count": len(model.mfd.Paths()),
        "bound_counter_count": len(bindings),
        "unbound_counters": unbound_model_counters(model),
        "counter_bindings": bindings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m counterpoint.gem5.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-raw")
    audit.add_argument("--model", required=True, type=Path)
    audit.add_argument("--raw", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)
    validate = subparsers.add_parser("validate-model")
    validate.add_argument("--model", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit-raw":
        summary_path = audit_raw(args.model, args.raw, args.out)
        print(json.dumps({"summary": str(summary_path)}, indent=2))
        return
    if args.command == "validate-model":
        print(
            json.dumps(
                _json_ready(validate_model(args.model)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
