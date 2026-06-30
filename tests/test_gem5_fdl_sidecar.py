import json
from pathlib import Path

import pandas as pd
import pytest
from upath.solver import ModelFeasibilityProblem
from upath.solver.cvconstraints import ExactValues

from counterpoint.gem5 import (
    audit_raw_stat_long,
    audit_raw_stat_long_path,
    build_counter_binding_table,
    build_counter_observations,
    load_fdl_model,
    load_sidecar,
    numeric_gem5_value,
    read_raw_stat_long,
    gem5_stat_coverage,
    validate_raw_stat_long,
    validate_source_references,
)
from counterpoint.gem5.cli import audit_raw, build_counts, validate_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/gem5/arm_o3"
BACKEND_SIDECAR = MODEL_DIR / "backend_core.json"
FRONTEND_STALL_SIDECAR = MODEL_DIR / "frontend_stall.json"
COMMIT_SIDECAR = MODEL_DIR / "commit_core.json"
BRANCH_SIDECAR = MODEL_DIR / "branch_frontend.json"
L1D_SIDECAR = MODEL_DIR / "memory_l1d.json"
L2_SIDECAR = MODEL_DIR / "memory_l2.json"
ICACHE_SIDECAR = MODEL_DIR / "icache_frontend.json"
TLB_SIDECAR = MODEL_DIR / "tlb_mmu.json"


def _binding_map(model) -> dict[str, str]:
    return {
        counter: binding.gem5_stat
        for counter, binding in model.sidecar.counter_bindings.items()
    }


def _assert_binding_contract(model) -> None:
    assert set(model.sidecar.counter_bindings) == set(model.mfd.Counters())
    for counter, binding in model.sidecar.counter_bindings.items():
        assert counter in model.mfd.Counters()
        assert binding.declaration_ref in model.sidecar.source_refs
        declaration = model.sidecar.source_refs[binding.declaration_ref]
        declaration_text = "\n".join(declaration.snippets)
        assert (
            "statistics::units::Count::get()" in declaration_text
            or "statistics::units::Cycle::get()" in declaration_text
        )


def _raw_row(gem5_stat: str, gem5_value: float) -> dict[str, object]:
    return {
        "workload_id": "w0",
        "parent_window_id": "win0",
        "window_id": "w000",
        "gem5_stat": gem5_stat,
        "gem5_value": gem5_value,
    }


def _binding_rows(
    sidecar_path: Path,
    values_by_counter: dict[str, float],
) -> list[dict[str, object]]:
    model = load_fdl_model(sidecar_path)
    missing = set(model.sidecar.counter_bindings) - set(values_by_counter)
    if missing:
        raise AssertionError(f"missing test values for counters: {sorted(missing)}")
    return [
        _raw_row(binding.gem5_stat, float(values_by_counter[counter]))
        for counter, binding in model.sidecar.counter_bindings.items()
    ]


def test_sidecar_rejects_flow_and_g_group(tmp_path: Path) -> None:
    fdl = tmp_path / "bad.fdl"
    fdl.write_text("FLOW { done; }\n", encoding="utf-8")
    sidecar = tmp_path / "bad.json"
    sidecar.write_text(
        """{
  "model_id": "bad",
  "module_id": "bad",
  "fdl_path": "bad.fdl",
  "gem5_revision": "unused",
  "source_refs": {},
  "counter_bindings": {},
  "flow": [],
  "g_group": "G0_progress"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="banned keys"):
        load_sidecar(sidecar)


def test_backend_core_fdl_loads() -> None:
    model = load_fdl_model(BACKEND_SIDECAR)

    assert model.sidecar.module_id == "backend_core"
    _assert_binding_contract(model)
    assert _binding_map(model) == {
        "commit_running_cycles": "system.cpu.commit.status::running",
        "core_cycles": "system.cpu.numCycles",
        "decode_blocked_cycles": "system.cpu.decode.status::Blocked",
        "decode_running_cycles": "system.cpu.decode.status::Running",
        "fetch_icache_wait_cycles": "system.cpu.fetch.status::icacheWaitResponse",
        "fetch_running_cycles": "system.cpu.fetch.status::running",
    }
    assert len(model.mfd.Paths()) == 18


def test_frontend_stall_fdl_loads() -> None:
    model = load_fdl_model(FRONTEND_STALL_SIDECAR)

    assert model.sidecar.module_id == "frontend_stall"
    _assert_binding_contract(model)
    assert _binding_map(model) == {
        "core_cycles": "system.cpu.numCycles",
        "fetch_icache_wait_response_cycles": "system.cpu.fetch.status::icacheWaitResponse",
        "fetch_icache_wait_retry_cycles": "system.cpu.fetch.status::icacheWaitRetry",
        "fetch_running_cycles": "system.cpu.fetch.status::running",
        "fetch_squashing_cycles": "system.cpu.fetch.status::squashing",
        "frontend_icache_stall_cycles": "system.cpu.fetchStats0.icacheStallCycles",
    }
    assert len(model.mfd.Paths()) == 10


def test_commit_core_fdl_loads() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)

    assert model.sidecar.module_id == "commit_core"
    _assert_binding_contract(model)
    assert set(model.mfd.Counters()) == {
        "committed_control",
        "committed_float_add",
        "committed_float_mem_read",
        "committed_float_mem_write",
        "committed_instructions",
        "committed_int_alu",
        "committed_int_div",
        "committed_int_mult",
        "committed_mem_read",
        "committed_mem_write",
        "committed_simd_alu",
        "committed_uops",
    }
    assert len(model.mfd.Paths()) == 40


def test_commit_core_sidecar_bindings_match_fdl_counters() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)

    _assert_binding_contract(model)
    assert _binding_map(model) == {
        "committed_control": "system.cpu.commitStats0.committedControl::IsControl",
        "committed_float_add": "system.cpu.commitStats0.committedInstType::FloatAdd",
        "committed_float_mem_read": "system.cpu.commitStats0.committedInstType::FloatMemRead",
        "committed_float_mem_write": "system.cpu.commitStats0.committedInstType::FloatMemWrite",
        "committed_instructions": "system.cpu.thread_0.numInsts",
        "committed_int_alu": "system.cpu.commitStats0.committedInstType::IntAlu",
        "committed_int_div": "system.cpu.commitStats0.committedInstType::IntDiv",
        "committed_int_mult": "system.cpu.commitStats0.committedInstType::IntMult",
        "committed_mem_read": "system.cpu.commitStats0.committedInstType::MemRead",
        "committed_mem_write": "system.cpu.commitStats0.committedInstType::MemWrite",
        "committed_simd_alu": "system.cpu.commitStats0.committedInstType::SimdAlu",
        "committed_uops": "system.cpu.thread_0.numOps",
    }


def test_load_model_rejects_binding_outside_fdl(tmp_path: Path) -> None:
    fdl = tmp_path / "unit.fdl"
    fdl.write_text("FLOW { incr event; done; }\n", encoding="utf-8")
    sidecar = tmp_path / "unit.json"
    sidecar.write_text(
        """{
  "model_id": "unit",
  "module_id": "unit",
  "fdl_path": "unit.fdl",
  "gem5_revision": "unused",
  "source_refs": {
    "event_increment": {
      "path": "unit.cc",
      "symbol": "Unit::event",
      "snippets": ["stats.event++;"]
    }
  },
  "counter_bindings": {
    "not_in_fdl": {
      "gem5_stat": "system.cpu.event",
      "declaration_ref": "event_increment",
      "increment_refs": ["event_increment"]
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present in FDL"):
        load_fdl_model(sidecar)


def test_sidecar_rejects_old_stat_selector_field(tmp_path: Path) -> None:
    fdl = tmp_path / "unit.fdl"
    fdl.write_text("FLOW { incr event; done; }\n", encoding="utf-8")
    sidecar = tmp_path / "unit.json"
    sidecar.write_text(
        """{
  "model_id": "unit",
  "module_id": "unit",
  "fdl_path": "unit.fdl",
  "gem5_revision": "unused",
  "source_refs": {
    "event_increment": {
      "path": "unit.cc",
      "symbol": "Unit::event",
      "snippets": ["stats.event++;"]
    }
  },
  "counter_bindings": {
    "event": {
      "stat_selector": "system.cpu.event",
      "declaration_ref": "event_increment",
      "increment_refs": ["event_increment"]
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys mismatch"):
        load_sidecar(sidecar)


def test_commit_core_exact_feasibility() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "committed_uops": 10,
                "committed_instructions": 8,
                "committed_mem_read": 2,
                "committed_mem_write": 1,
                "committed_float_mem_read": 0,
                "committed_float_mem_write": 0,
                "committed_control": 3,
            }
        ),
    ).Solve()
    too_many_instructions = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "committed_uops": 10,
                "committed_instructions": 11,
                "committed_mem_read": 2,
                "committed_mem_write": 1,
                "committed_float_mem_read": 0,
                "committed_float_mem_write": 0,
                "committed_control": 3,
            }
        ),
    ).Solve()

    assert feasible.Status() == "Optimal"
    assert too_many_instructions.Status() == "Infeasible"


def test_backend_core_exact_feasibility() -> None:
    model = load_fdl_model(BACKEND_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues({"core_cycles": 20}),
    ).Solve()
    negative_cycles = ModelFeasibilityProblem(
        model.mfd,
        ExactValues({"core_cycles": -1}),
    ).Solve()

    assert feasible.Status() == "Optimal"
    assert negative_cycles.Status() == "Infeasible"


def test_frontend_stall_exact_feasibility() -> None:
    model = load_fdl_model(FRONTEND_STALL_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues({"core_cycles": 20, "frontend_icache_stall_cycles": 3}),
    ).Solve()
    too_many_frontend_stalls = ModelFeasibilityProblem(
        model.mfd,
        ExactValues({"core_cycles": 20, "frontend_icache_stall_cycles": 21}),
    ).Solve()

    assert feasible.Status() == "Optimal"
    assert too_many_frontend_stalls.Status() == "Infeasible"


def test_branch_frontend_exact_feasibility() -> None:
    model = load_fdl_model(BRANCH_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "committed_branches": 10,
                "branch_mispredicts": 2,
            }
        ),
    ).Solve()
    too_many_mispredicts = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "committed_branches": 10,
                "branch_mispredicts": 11,
            }
        ),
    ).Solve()

    assert set(model.mfd.Counters()) == {
        "branch_mispredicts",
        "btb_hits",
        "btb_lookups",
        "btb_mispredicted",
        "committed_branches",
        "cond_incorrect",
        "cond_predicted",
    }
    assert feasible.Status() == "Optimal"
    assert too_many_mispredicts.Status() == "Infeasible"


def test_memory_l1d_exact_feasibility() -> None:
    model = load_fdl_model(L1D_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l1d_overall_accesses": 10,
                "l1d_read_accesses": 6,
                "l1d_read_mshr_misses": 3,
            }
        ),
    ).Solve()
    too_many_read_misses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l1d_overall_accesses": 10,
                "l1d_read_accesses": 6,
                "l1d_read_mshr_misses": 7,
            }
        ),
    ).Solve()
    too_many_read_accesses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l1d_overall_accesses": 10,
                "l1d_read_accesses": 11,
                "l1d_read_mshr_misses": 1,
            }
        ),
    ).Solve()

    assert set(model.mfd.Counters()) == {
        "l1d_overall_accesses",
        "l1d_read_accesses",
        "l1d_read_hits",
        "l1d_read_misses",
        "l1d_read_mshr_misses",
        "l1d_write_accesses",
        "l1d_write_misses",
        "l1d_write_mshr_misses",
    }
    assert feasible.Status() == "Optimal"
    assert too_many_read_misses.Status() == "Infeasible"
    assert too_many_read_accesses.Status() == "Infeasible"


def test_memory_l2_exact_feasibility() -> None:
    model = load_fdl_model(L2_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l2_overall_accesses": 20,
                "l2_demand_accesses": 12,
                "l2_demand_misses": 4,
            }
        ),
    ).Solve()
    too_many_misses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l2_overall_accesses": 20,
                "l2_demand_accesses": 12,
                "l2_demand_misses": 13,
            }
        ),
    ).Solve()
    too_many_demand_accesses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "l2_overall_accesses": 20,
                "l2_demand_accesses": 21,
                "l2_demand_misses": 4,
            }
        ),
    ).Solve()

    assert set(model.mfd.Counters()) == {
        "l2_blocked_no_mshrs",
        "l2_demand_accesses",
        "l2_demand_hits",
        "l2_demand_misses",
        "l2_overall_accesses",
        "l2_overall_hits",
        "l2_overall_misses",
        "l2_replacements",
    }
    assert feasible.Status() == "Optimal"
    assert too_many_misses.Status() == "Infeasible"
    assert too_many_demand_accesses.Status() == "Infeasible"


def test_icache_frontend_exact_feasibility() -> None:
    model = load_fdl_model(ICACHE_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "icache_overall_accesses": 20,
                "icache_demand_accesses": 12,
                "icache_demand_misses": 4,
            }
        ),
    ).Solve()
    too_many_misses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "icache_overall_accesses": 20,
                "icache_demand_accesses": 12,
                "icache_demand_misses": 13,
            }
        ),
    ).Solve()
    too_many_demand_accesses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "icache_overall_accesses": 20,
                "icache_demand_accesses": 21,
                "icache_demand_misses": 4,
            }
        ),
    ).Solve()

    assert set(model.mfd.Counters()) == {
        "icache_demand_accesses",
        "icache_demand_hits",
        "icache_demand_misses",
        "icache_overall_accesses",
        "icache_overall_hits",
        "icache_overall_misses",
        "icache_read_accesses",
        "icache_read_misses",
    }
    assert feasible.Status() == "Optimal"
    assert too_many_misses.Status() == "Infeasible"
    assert too_many_demand_accesses.Status() == "Infeasible"


def test_tlb_mmu_exact_feasibility() -> None:
    model = load_fdl_model(TLB_SIDECAR)
    feasible = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "dtlb_accesses": 18,
                "dtlb_misses": 3,
                "dtlb_read_accesses": 10,
                "dtlb_read_misses": 2,
                "dtlb_write_accesses": 8,
                "dtlb_write_misses": 1,
                "itlb_accesses": 7,
                "itlb_misses": 0,
                "itlb_inst_accesses": 7,
                "itlb_inst_misses": 0,
            }
        ),
    ).Solve()
    too_many_read_misses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "dtlb_accesses": 18,
                "dtlb_misses": 12,
                "dtlb_read_accesses": 10,
                "dtlb_read_misses": 11,
                "dtlb_write_accesses": 8,
                "dtlb_write_misses": 1,
                "itlb_accesses": 7,
                "itlb_misses": 0,
                "itlb_inst_accesses": 7,
                "itlb_inst_misses": 0,
            }
        ),
    ).Solve()
    mismatched_dtlb_access_total = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "dtlb_accesses": 17,
                "dtlb_misses": 3,
                "dtlb_read_accesses": 10,
                "dtlb_read_misses": 2,
                "dtlb_write_accesses": 8,
                "dtlb_write_misses": 1,
                "itlb_accesses": 7,
                "itlb_misses": 0,
                "itlb_inst_accesses": 7,
                "itlb_inst_misses": 0,
            }
        ),
    ).Solve()
    too_many_inst_misses = ModelFeasibilityProblem(
        model.mfd,
        ExactValues(
            {
                "dtlb_accesses": 18,
                "dtlb_misses": 3,
                "dtlb_read_accesses": 10,
                "dtlb_read_misses": 2,
                "dtlb_write_accesses": 8,
                "dtlb_write_misses": 1,
                "itlb_accesses": 7,
                "itlb_misses": 8,
                "itlb_inst_accesses": 7,
                "itlb_inst_misses": 8,
            }
        ),
    ).Solve()

    assert set(model.mfd.Counters()) == {
        "dtlb_accesses",
        "dtlb_inserts",
        "dtlb_misses",
        "dtlb_read_accesses",
        "dtlb_read_hits",
        "dtlb_read_misses",
        "dtlb_write_accesses",
        "dtlb_write_hits",
        "dtlb_write_misses",
        "itlb_accesses",
        "itlb_inserts",
        "itlb_misses",
        "itlb_inst_accesses",
        "itlb_inst_hits",
        "itlb_inst_misses",
    }
    assert feasible.Status() == "Optimal"
    assert too_many_read_misses.Status() == "Infeasible"
    assert mismatched_dtlb_access_total.Status() == "Infeasible"
    assert too_many_inst_misses.Status() == "Infeasible"


def _raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        COMMIT_SIDECAR,
        {
            "committed_control": 3.0,
            "committed_float_add": 0.0,
            "committed_float_mem_read": 0.0,
            "committed_float_mem_write": 0.0,
            "committed_instructions": 8.0,
            "committed_int_alu": 0.0,
            "committed_int_div": 0.0,
            "committed_int_mult": 0.0,
            "committed_mem_read": 2.0,
            "committed_mem_write": 1.0,
            "committed_simd_alu": 0.0,
            "committed_uops": 10.0,
        },
    )


def _backend_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        BACKEND_SIDECAR,
        {
            "commit_running_cycles": 10.0,
            "core_cycles": 20.0,
            "decode_blocked_cycles": 2.0,
            "decode_running_cycles": 12.0,
            "fetch_icache_wait_cycles": 3.0,
            "fetch_running_cycles": 11.0,
        },
    )


def _frontend_stall_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        FRONTEND_STALL_SIDECAR,
        {
            "core_cycles": 20.0,
            "fetch_icache_wait_response_cycles": 3.0,
            "fetch_icache_wait_retry_cycles": 1.0,
            "fetch_running_cycles": 12.0,
            "fetch_squashing_cycles": 2.0,
            "frontend_icache_stall_cycles": 3.0,
        },
    )


def _branch_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        BRANCH_SIDECAR,
        {
            "branch_mispredicts": 2.0,
            "btb_hits": 6.0,
            "btb_lookups": 8.0,
            "btb_mispredicted": 1.0,
            "committed_branches": 10.0,
            "cond_incorrect": 2.0,
            "cond_predicted": 8.0,
        },
    )


def _l1d_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        L1D_SIDECAR,
        {
            "l1d_overall_accesses": 10.0,
            "l1d_read_accesses": 6.0,
            "l1d_read_hits": 3.0,
            "l1d_read_misses": 3.0,
            "l1d_read_mshr_misses": 1.0,
            "l1d_write_accesses": 4.0,
            "l1d_write_misses": 2.0,
            "l1d_write_mshr_misses": 1.0,
        },
    )


def _l2_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        L2_SIDECAR,
        {
            "l2_blocked_no_mshrs": 1.0,
            "l2_demand_accesses": 12.0,
            "l2_demand_hits": 8.0,
            "l2_demand_misses": 4.0,
            "l2_overall_accesses": 20.0,
            "l2_overall_hits": 8.0,
            "l2_overall_misses": 4.0,
            "l2_replacements": 2.0,
        },
    )


def _icache_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        ICACHE_SIDECAR,
        {
            "icache_demand_accesses": 12.0,
            "icache_demand_hits": 8.0,
            "icache_demand_misses": 4.0,
            "icache_overall_accesses": 20.0,
            "icache_overall_hits": 8.0,
            "icache_overall_misses": 4.0,
            "icache_read_accesses": 12.0,
            "icache_read_misses": 4.0,
        },
    )


def _tlb_raw_stat_rows() -> list[dict[str, object]]:
    return _binding_rows(
        TLB_SIDECAR,
        {
            "dtlb_accesses": 18.0,
            "dtlb_inserts": 3.0,
            "dtlb_misses": 3.0,
            "dtlb_read_accesses": 10.0,
            "dtlb_read_hits": 8.0,
            "dtlb_read_misses": 2.0,
            "dtlb_write_accesses": 8.0,
            "dtlb_write_hits": 7.0,
            "dtlb_write_misses": 1.0,
            "itlb_accesses": 7.0,
            "itlb_inserts": 0.0,
            "itlb_inst_accesses": 7.0,
            "itlb_inst_hits": 7.0,
            "itlb_inst_misses": 0.0,
            "itlb_misses": 0.0,
        },
    )


def test_raw_stat_long_contract(tmp_path: Path) -> None:
    rows = _raw_stat_rows()
    frame = pd.DataFrame(rows)
    validate_raw_stat_long(frame)

    path = tmp_path / "raw.parquet"
    frame.to_parquet(path)
    assert len(read_raw_stat_long(path)) == len(rows)
    with pytest.raises(ValueError, match="finite"):
        numeric_gem5_value(float("nan"), "bad")
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_stat_long(pd.DataFrame([rows[0], dict(rows[0])]))
    with pytest.raises(ValueError, match="missing columns"):
        validate_raw_stat_long(
            pd.DataFrame(
                [
                    {
                        "workload_id": "w0",
                        "parent_window_id": "win0",
                        "window_id": "w000",
                        "stat_selector": "system.cpu.thread_0.numOps",
                        "stat_value": 10.0,
                    }
                ]
            )
        )


def test_audit_raw_stat_long_reports_missing_stat() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)
    rows = [
        row
        for row in _raw_stat_rows()
        if row["gem5_stat"] != "system.cpu.commitStats0.committedControl::IsControl"
    ]
    report = audit_raw_stat_long(model, pd.DataFrame(rows))

    assert report.iloc[0]["audit_status"] == "MissingStat"
    assert report.iloc[0]["missing_counters"] == ("committed_control",)
    assert report.iloc[0]["missing_gem5_stats"] == (
        "system.cpu.commitStats0.committedControl::IsControl",
    )


def test_audit_raw_stat_long_path_and_coverage(tmp_path: Path) -> None:
    model = load_fdl_model(COMMIT_SIDECAR)
    path = tmp_path / "raw.parquet"
    pd.DataFrame(_raw_stat_rows()).to_parquet(path)

    report = audit_raw_stat_long_path(model, path)
    coverage = gem5_stat_coverage(model, read_raw_stat_long(path)).set_index(
        "counter"
    )

    assert report.iloc[0]["audit_status"] == "Optimal"
    assert coverage.loc["committed_uops", "present_in_table"]
    assert coverage.loc["committed_control", "missing_window_count"] == 0


def test_backend_core_audit_raw_stat_long() -> None:
    report = audit_raw_stat_long(
        load_fdl_model(BACKEND_SIDECAR),
        pd.DataFrame(_backend_raw_stat_rows()),
    )

    assert report.iloc[0]["audit_status"] == "Optimal"
    assert report.iloc[0]["counter_values"]["core_cycles"] == 20.0
    assert report.iloc[0]["counter_values"]["commit_running_cycles"] == 10.0


def test_frontend_stall_audit_raw_stat_long() -> None:
    report = audit_raw_stat_long(
        load_fdl_model(FRONTEND_STALL_SIDECAR),
        pd.DataFrame(_frontend_stall_raw_stat_rows()),
    )

    assert report.iloc[0]["audit_status"] == "Optimal"
    assert report.iloc[0]["counter_values"]["core_cycles"] == 20.0
    assert report.iloc[0]["counter_values"]["frontend_icache_stall_cycles"] == 3.0


def test_branch_frontend_and_l1d_audit_raw_stat_long() -> None:
    branch_report = audit_raw_stat_long(
        load_fdl_model(BRANCH_SIDECAR),
        pd.DataFrame(_branch_raw_stat_rows()),
    )
    l1d_report = audit_raw_stat_long(
        load_fdl_model(L1D_SIDECAR),
        pd.DataFrame(_l1d_raw_stat_rows()),
    )

    assert branch_report.iloc[0]["audit_status"] == "Optimal"
    assert l1d_report.iloc[0]["audit_status"] == "Optimal"


def test_l2_icache_and_tlb_audit_raw_stat_long() -> None:
    l2_report = audit_raw_stat_long(
        load_fdl_model(L2_SIDECAR),
        pd.DataFrame(_l2_raw_stat_rows()),
    )
    icache_report = audit_raw_stat_long(
        load_fdl_model(ICACHE_SIDECAR),
        pd.DataFrame(_icache_raw_stat_rows()),
    )
    tlb_report = audit_raw_stat_long(
        load_fdl_model(TLB_SIDECAR),
        pd.DataFrame(_tlb_raw_stat_rows()),
    )

    assert l2_report.iloc[0]["audit_status"] == "Optimal"
    assert icache_report.iloc[0]["audit_status"] == "Optimal"
    assert tlb_report.iloc[0]["audit_status"] == "Optimal"


def test_build_counter_observations_and_binding_table() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)
    observations = build_counter_observations((model,), pd.DataFrame(_raw_stat_rows()))
    bindings = build_counter_binding_table((model,))

    assert list(observations.columns) == [
        "workload_id",
        "parent_window_id",
        "window_id",
        "model_id",
        "module_id",
        "counter",
        "counter_value",
        "audit_status",
    ]
    assert len(observations) == 12
    committed_instructions = observations[
        observations["counter"] == "committed_instructions"
    ].iloc[0]
    assert committed_instructions["counter_value"] == 8.0
    assert committed_instructions["audit_status"] == "Optimal"
    assert list(bindings.columns) == [
        "model_id",
        "module_id",
        "counter",
        "gem5_stat",
    ]
    assert dict(zip(bindings["counter"], bindings["gem5_stat"], strict=False))[
        "committed_instructions"
    ] == "system.cpu.thread_0.numInsts"


def test_build_counter_observations_missing_stat_keeps_counter_rows() -> None:
    model = load_fdl_model(COMMIT_SIDECAR)
    rows = [
        row
        for row in _raw_stat_rows()
        if row["gem5_stat"] != "system.cpu.commitStats0.committedControl::IsControl"
    ]
    observations = build_counter_observations((model,), pd.DataFrame(rows))

    assert set(observations["audit_status"]) == {"MissingStat"}
    missing = observations[observations["counter"] == "committed_control"].iloc[0]
    present = observations[observations["counter"] == "committed_uops"].iloc[0]
    assert pd.isna(missing["counter_value"])
    assert present["counter_value"] == 10.0


def test_source_refs_validate_against_stub(tmp_path: Path) -> None:
    fdl = tmp_path / "unit.fdl"
    fdl.write_text("FLOW { incr event; done; }\n", encoding="utf-8")
    source = tmp_path / "unit.cc"
    source.write_text(
        "void Unit::event() {\n"
        "    stats.event++;\n"
        "}\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "unit.json"
    sidecar.write_text(
        """{
  "model_id": "unit",
  "module_id": "unit",
  "fdl_path": "unit.fdl",
  "gem5_revision": "unused",
  "source_refs": {
    "event_increment": {
      "path": "unit.cc",
      "symbol": "Unit::event",
      "snippets": ["stats.event++;"]
    }
  },
  "counter_bindings": {
    "event": {
      "gem5_stat": "system.cpu.event",
      "declaration_ref": "event_increment",
      "increment_refs": ["event_increment"]
    }
  }
}
""",
        encoding="utf-8",
    )
    evidence = validate_source_references(
        load_sidecar(sidecar),
        tmp_path,
        check_revision=False,
    )

    assert evidence[0].symbol_line == 1
    assert evidence[0].snippet_lines == (2,)


def test_audit_raw_cli_writes_report_and_summary(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "audit.parquet"
    pd.DataFrame(_raw_stat_rows()).to_parquet(raw_path)

    summary_path = audit_raw(COMMIT_SIDECAR, raw_path, out_path)

    assert out_path.exists()
    assert summary_path.exists()


def test_build_counts_cli_writes_counter_outputs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "counter_observations.parquet"
    pd.DataFrame(_raw_stat_rows()).to_parquet(raw_path)

    summary_path = build_counts(
        model_paths=[COMMIT_SIDECAR],
        model_dir=None,
        raw_path=raw_path,
        out_path=out_path,
    )

    bindings_path = tmp_path / "counter_observations.bindings.parquet"
    observations = pd.read_parquet(out_path)
    bindings = pd.read_parquet(bindings_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(observations) == 12
    assert len(bindings) == 12
    assert summary["row_count"] == 12
    assert summary["bindings_path"] == str(bindings_path)
