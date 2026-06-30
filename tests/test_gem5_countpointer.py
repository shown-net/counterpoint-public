from pathlib import Path
import shutil

import pandas as pd
import pytest

from counterpoint.gem5 import (
    build_count_relations,
    build_countpointer_artifacts,
    build_q_atoms,
    load_fdl_model,
)
from counterpoint.gem5.cli import export_countpointer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/gem5/arm_o3"
BACKEND_SIDECAR = MODEL_DIR / "backend_core.json"
BRANCH_SIDECAR = MODEL_DIR / "branch_frontend.json"
L1D_SIDECAR = MODEL_DIR / "memory_l1d.json"


def _candidate_rows(model) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for counter, binding in sorted(model.sidecar.counter_bindings.items()):
        stat = str(binding.gem5_stat)
        rows.append(
            {
                "gem5_stat": stat,
                "anchor_module_id": str(model.sidecar.module_id),
                "anchor_counter": str(counter),
                "anchor_gem5_stat": stat,
            }
        )
    return rows


def _raw_candidates(*models) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model in models:
        rows.extend(_candidate_rows(model))
    return pd.DataFrame(rows)


def test_countpointer_builds_q_atoms_from_fdl_bindings() -> None:
    model = load_fdl_model(L1D_SIDECAR)
    raw = _raw_candidates(model)
    raw.loc[len(raw)] = {
        "gem5_stat": "system.cpu.dcache.ReadReq.hits::total",
        "anchor_module_id": "memory_l1d",
        "anchor_counter": "unused_counter",
        "anchor_gem5_stat": "system.cpu.dcache.ReadReq.hits::total",
    }

    q_atoms = build_q_atoms((model,), raw).q_atoms

    assert set(q_atoms["module_id"].astype(str)) == {"memory_l1d"}
    assert set(q_atoms["counter"].astype(str)) == set(model.sidecar.counter_bindings)
    assert q_atoms["q_atom_id"].astype(str).str.startswith("q_atom:memory_l1d:").all()
    assert list(q_atoms.columns) == ["q_atom_id", "module_id", "counter", "gem5_stat"]
    assert {"pmu_event", "event_group_id", "head_type"}.isdisjoint(q_atoms.columns)


def test_countpointer_rejects_invalid_q_atom_binding_candidate() -> None:
    model = load_fdl_model(L1D_SIDECAR)
    raw = _raw_candidates(model)
    raw.loc[0] = {
        "gem5_stat": "system.cpu.dcache.WriteReq.hits::total",
        "anchor_module_id": "memory_l1d",
        "anchor_counter": "l1d_read_accesses",
        "anchor_gem5_stat": "system.cpu.dcache.ReadReq.accesses::total",
    }

    with pytest.raises(ValueError, match="q_atom binding candidate gem5_stat must match"):
        build_q_atoms((model,), raw)


def test_countpointer_exports_count_relations_and_terms() -> None:
    model = load_fdl_model(L1D_SIDECAR)
    q_atoms = build_q_atoms((model,), _raw_candidates(model)).q_atoms
    relations = build_count_relations((model,), q_atoms)

    relation_rows = relations.count_relation.to_dict(orient="records")
    assert relation_rows
    assert list(relations.count_relation.columns) == ["relation_id", "sense", "rhs"]
    assert set(relations.count_relation["sense"].astype(str)) <= {"eq", "le", "ge"}

    terms = relations.count_relation_terms
    assert not terms.empty
    assert list(terms.columns) == ["relation_id", "q_atom_id", "coefficient"]
    assert not (terms["coefficient"].astype(float) == 0.0).any()
    assert {"feature_id", "term_role", "counter", "gem5_stat", "module_id"}.isdisjoint(terms.columns)
    assert set(terms["q_atom_id"].astype(str)) <= set(q_atoms["q_atom_id"].astype(str))
    assert set(terms["relation_id"].astype(str)) <= set(
        relations.count_relation["relation_id"].astype(str)
    )


def test_countpointer_handles_backend_core_role_aware_relations() -> None:
    model = load_fdl_model(BACKEND_SIDECAR)
    q_atoms = build_q_atoms((model,), _raw_candidates(model)).q_atoms
    relations = build_count_relations((model,), q_atoms)

    assert not relations.count_relation.empty
    assert set(relations.count_relation["sense"].astype(str)) <= {"eq", "le", "ge"}
    assert "term_role" not in relations.count_relation_terms.columns
    assert "relation_kind" not in relations.count_relation.columns
    assert set(relations.count_relation_terms["q_atom_id"].astype(str)) <= set(
        q_atoms["q_atom_id"].astype(str)
    )


def test_countpointer_cli_export_writes_minimal_artifacts() -> None:
    workspace = ROOT / "tests/.countpointer_export"
    raw_path = ROOT / "tests/.countpointer_raw_candidates.parquet"
    if workspace.exists():
        shutil.rmtree(workspace)
    models = (load_fdl_model(BACKEND_SIDECAR), load_fdl_model(L1D_SIDECAR))
    _raw_candidates(*models).to_parquet(raw_path)
    manifest = export_countpointer(
        model_paths=[BACKEND_SIDECAR, L1D_SIDECAR],
        model_dir=None,
        raw_stat_candidates=raw_path,
        out_dir=workspace,
    )
    try:
        root = manifest.parent
        expected = {
            "q_atoms.parquet",
            "count_relation.parquet",
            "count_relation_terms.parquet",
            "h1_countpointer_manifest.json",
        }
        assert {path.name for path in root.iterdir()} == expected

        q_atoms = pd.read_parquet(root / "q_atoms.parquet")
        relations = pd.read_parquet(root / "count_relation.parquet")
        terms = pd.read_parquet(root / "count_relation_terms.parquet")
        assert list(q_atoms.columns) == ["q_atom_id", "module_id", "counter", "gem5_stat"]
        assert list(relations.columns) == ["relation_id", "sense", "rhs"]
        assert list(terms.columns) == ["relation_id", "q_atom_id", "coefficient"]
        assert "pmu_event" not in q_atoms.columns
        assert "event_group_id" not in q_atoms.columns
        assert "head_type" not in q_atoms.columns
        assert "constraint_inequalities.parquet" not in {path.name for path in root.iterdir()}
        assert "constraint_feature_map.parquet" not in {path.name for path in root.iterdir()}
    finally:
        shutil.rmtree(workspace)
        raw_path.unlink()
