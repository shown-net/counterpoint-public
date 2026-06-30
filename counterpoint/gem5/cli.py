"""Command-line helpers for FDL-first gem5 audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import (
    audit_raw_stat_long,
    build_counter_binding_table,
    build_counter_observations,
    gem5_stat_coverage,
    load_fdl_model,
    unbound_model_counters,
)
from .countpointer import build_countpointer_artifacts
from .raw_stats import read_raw_stat_long


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
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
    coverage = gem5_stat_coverage(model, table)
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
            counter: binding.gem5_stat
            for counter, binding in model.sidecar.counter_bindings.items()
        },
        "gem5_stat_coverage": coverage.to_dict(orient="records"),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _model_paths_from_args(
    *,
    model_paths: list[Path] | None,
    model_dir: Path | None,
) -> tuple[Path, ...]:
    if model_paths and model_dir is not None:
        raise ValueError("--model and --model-dir are mutually exclusive")
    if not model_paths and model_dir is None:
        raise ValueError("one of --model or --model-dir is required")
    if model_paths:
        return tuple(model_paths)
    assert model_dir is not None
    paths = tuple(
        sorted(
            path
            for path in model_dir.glob("*.json")
        )
    )
    if not paths:
        raise ValueError(f"model directory contains no sidecar JSON files: {model_dir}")
    return paths


def build_counts(
    *,
    model_paths: list[Path] | None,
    model_dir: Path | None,
    raw_path: Path,
    out_path: Path,
) -> Path:
    """Build counter observations parquet plus binding and summary files."""
    paths = _model_paths_from_args(model_paths=model_paths, model_dir=model_dir)
    models = tuple(load_fdl_model(path) for path in paths)
    table = read_raw_stat_long(raw_path)
    counts = build_counter_observations(models, table)
    bindings = build_counter_binding_table(models)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(out_path)
    bindings_path = out_path.with_name(f"{out_path.stem}.bindings{out_path.suffix}")
    bindings.to_parquet(bindings_path)
    summary = {
        "model_count": len(models),
        "window_count": int(
            counts[list(("workload_id", "parent_window_id", "window_id"))]
            .drop_duplicates()
            .shape[0]
        ),
        "row_count": int(len(counts)),
        "audit_status_counts": counts["audit_status"].value_counts().to_dict(),
        "models": [
            {
                "model_id": model.sidecar.model_id,
                "module_id": model.sidecar.module_id,
                "sidecar_path": model.sidecar.path,
            }
            for model in models
        ],
        "counters": sorted(counts["counter"].dropna().unique().tolist()),
        "bindings_path": bindings_path,
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
        counter: binding.gem5_stat
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


def export_countpointer(
    *,
    model_paths: list[Path] | None,
    model_dir: Path | None,
    raw_stat_candidates: Path,
    out_dir: Path,
) -> Path:
    """Export q atoms and count relations."""
    paths = _model_paths_from_args(model_paths=model_paths, model_dir=model_dir)
    models = tuple(load_fdl_model(path) for path in paths)
    candidates = pd.read_parquet(raw_stat_candidates)
    artifacts = build_countpointer_artifacts(models, candidates)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "q_atoms": out_dir / "q_atoms.parquet",
        "count_relation": out_dir / "count_relation.parquet",
        "count_relation_terms": out_dir / "count_relation_terms.parquet",
    }
    for path in (*out_dir.glob("*.parquet"), out_dir / "h1_countpointer_manifest.json"):
        if path.exists():
            path.unlink()

    artifacts.q_atoms.to_parquet(output_paths["q_atoms"])
    artifacts.count_relation.to_parquet(output_paths["count_relation"])
    artifacts.count_relation_terms.to_parquet(output_paths["count_relation_terms"])

    manifest = {
        "model_count": len(models),
        "raw_stat_candidates_path": raw_stat_candidates,
        "models": [
            {
                "model_id": model.sidecar.model_id,
                "module_id": model.sidecar.module_id,
                "sidecar_path": model.sidecar.path,
            }
            for model in models
        ],
        "artifact_paths": output_paths,
        "artifact_rows": {
            "q_atoms": int(artifacts.q_atoms.shape[0]),
            "count_relation": int(artifacts.count_relation.shape[0]),
            "count_relation_terms": int(artifacts.count_relation_terms.shape[0]),
        },
    }
    manifest_path = out_dir / "h1_countpointer_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m counterpoint.gem5.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-raw")
    audit.add_argument("--model", required=True, type=Path)
    audit.add_argument("--raw", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)
    build = subparsers.add_parser("build-counts")
    build.add_argument("--model", action="append", default=None, type=Path)
    build.add_argument("--model-dir", default=None, type=Path)
    build.add_argument("--raw", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    validate = subparsers.add_parser("validate-model")
    validate.add_argument("--model", required=True, type=Path)
    countpointer = subparsers.add_parser("countpointer-export")
    countpointer.add_argument("--model", action="append", default=None, type=Path)
    countpointer.add_argument("--model-dir", default=None, type=Path)
    countpointer.add_argument("--raw-stat-candidates", required=True, type=Path)
    countpointer.add_argument("--out", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit-raw":
        summary_path = audit_raw(args.model, args.raw, args.out)
        print(json.dumps({"summary": str(summary_path)}, indent=2))
        return
    if args.command == "build-counts":
        summary_path = build_counts(
            model_paths=args.model,
            model_dir=args.model_dir,
            raw_path=args.raw,
            out_path=args.out,
        )
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
    if args.command == "countpointer-export":
        manifest_path = export_countpointer(
            model_paths=args.model,
            model_dir=args.model_dir,
            raw_stat_candidates=args.raw_stat_candidates,
            out_dir=args.out,
        )
        print(json.dumps({"manifest": str(manifest_path)}, indent=2))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
