#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Nicholas Lindsay

"""run_model_scaling.py

This script collects data on runtime, feasibility testing results, and constraints for the
Haswell MMU models across various subsets of counters. Results are written into the case
study's local ``scaling`` directory, regardless of the current working directory.

- constraints_stats.csv records the model constraint properties
- results.csv records feasibility results associated with each model
- runtime.csv records the runtime required to analyze each model

"""


import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, NamedTuple, Tuple

from pexpr import base, dataset, response

import pandas as pd

import counterpoint.model_building as mb
from counterpoint import upath_expr


CASE_STUDY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = CASE_STUDY_ROOT / "models"
COUNTER_GROUPS_PATH = CASE_STUDY_ROOT / "misc" / "counter_groups.json"
DATASET_PATH = CASE_STUDY_ROOT / "data" / "experimental_data.parquet"
SCALING_DIR = CASE_STUDY_ROOT / "scaling"


def main():
    # Determine the model evaluation order
    model_order: list[Path] = sorted(MODEL_DIR.glob("m*.fdlm4"))
    model_order += sorted(MODEL_DIR.glob("a*.fdlm4"))
    model_order += sorted(MODEL_DIR.glob("r*.fdlm4"))
    model_order += sorted(MODEL_DIR.glob("t*.fdlm4"))

    print("Run models in this order:")
    print("\n".join(str(p) for p in model_order))
    print()

    if not COUNTER_GROUPS_PATH.exists():
        raise FileNotFoundError(
            f"counter_groups.json not found at {COUNTER_GROUPS_PATH.resolve()}"
        )
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"experimental_data.parquet not found at {DATASET_PATH.resolve()}"
        )

    with COUNTER_GROUPS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    counter_sets: dict[str, list[str]] = data.get("counter_sets", {})
    counter_set_order: list[str] = data.get("counter_set_order", [])

    # Compute the counters expected for each model set
    expected_counters: dict[str, list[str]] = {}
    current_counters: list[str] = []
    for name in counter_set_order:
        # Add counters to current set
        current_counters += counter_sets[name]

        # Prepend "+" to name as counter sets are additive

        # Update dictionary
        expected_counters[f"{name}"] = current_counters.copy()

    # Create a Pandas Series mapping the counter group to it's count
    counter_set_sizes = { name : len(expected_counters[name]) for name in expected_counters }
    counter_set_sizes_s = pd.Series(counter_set_sizes)

    # We load the dataset of performance counter observations so that we can profile the
    # analysis times.
    #
    # We perform some manipulation of the dataset to maximize the number of observations that
    # can be tested. This is because some observations are missing certain performance counters.
    #
    # In particular, we make the following transformations:
    # 1. If `counter X = counter Y1 + counter Y2 + ...` and all counters on the RHS are known, we compute the LHS.
    # 2. If `counter X = counter X4k + counter X2m + counter X1g` and only the counter on the LHS is known and the page size is 4KB, we assume that `counter X2m` and `counter X1g` are zero.

    # Load entire dataset
    ds_all_raw: dataset.Dataset = dataset.Dataset.from_parquet(
        str(DATASET_PATH),
        "experimental_data_raw",
    )

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for analysis"""

        # Drop machine name configuration level
        df = df.droplevel("machine", axis=1)

        # Drop THP_2mb page size
        df = df.loc[:,df.columns.get_level_values("pagesize") != "THP_2mb"]

        # Create pseudo-counter for all page walker loads
        df.loc[repr(base.Counter("page_walker_loads"))] = sum(
            df.loc[repr(base.Counter(f"page_walker_loads.{x}"))]
            for x in ["dtlb_l1", "dtlb_l2", "dtlb_l3", "dtlb_memory"]
        )

        # TODO: drop individual walker load counters

        return df

    ds_all: dataset.Dataset = ds_all_raw.Apply(preprocess, "experimental data")

    # For workloads with a 4KB page size that are missing counters for larger page sizes,
    # assume that those counts are zero.
    ds_all.DeriveNaNs(
        base.Counter("dtlb_load_misses.stlb_hit_2m"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_store_misses.stlb_hit_2m"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_load_misses.walk_completed_2m_4m"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_store_misses.walk_completed_2m_4m"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_load_misses.walk_completed_1g"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_store_misses.walk_completed_1g"),
        base.IfThenElse(base.ConfigurationLevel("pagesize") == "4KB", 0, float("NaN"))
    )

    # Infer missing values for STLB hits
    ds_all.DeriveNaNs(
        base.Counter("dtlb_load_misses.stlb_hit"),
        base.Counter("dtlb_load_misses.stlb_hit_4k") + base.Counter("dtlb_load_misses.stlb_hit_2m")
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_store_misses.stlb_hit"),
        base.Counter("dtlb_store_misses.stlb_hit_4k") + base.Counter("dtlb_store_misses.stlb_hit_2m")
    )

    # Infer missing values for walks completed
    ds_all.DeriveNaNs(
        base.Counter("dtlb_load_misses.walk_completed"),
        base.Counter("dtlb_load_misses.walk_completed_4k") +
            base.Counter("dtlb_load_misses.walk_completed_2m_4m") +
            base.Counter("dtlb_load_misses.walk_completed_1g")
    )
    ds_all.DeriveNaNs(
        base.Counter("dtlb_store_misses.walk_completed"),
        base.Counter("dtlb_store_misses.walk_completed_4k") +
            base.Counter("dtlb_store_misses.walk_completed_2m_4m") +
            base.Counter("dtlb_store_misses.walk_completed_1g")
    )

    # Select only those observations which contain a full set of hardware performance counters
    def select_observations_with_all_counters(
            df: pd.DataFrame, counters: list[str]
    ) -> pd.DataFrame:
        """Select only those observations which contain complete set of counters"""
        rows = [ f"Counter('{x}')" for x in counters]
        return df.loc[rows].dropna(axis=1)

    ds_all_counters = ds_all.Apply(
        lambda df : select_observations_with_all_counters(df, expected_counters["Refs"]),
        "FullCounterObservations"
    )

    ## Parameters for report generation

    REL_ERR_TOL: float = 0.000002
    """Relative error tolerance"""

    CONFIDENCE_LEVEL: float = 0.99
    """Confidence level"""

    NOISE_HANDLING_MODE: upath_expr.NoiseHandlingMode = upath_expr.NoiseHandlingMode.AlignedBoundingBox
    """Noise handling bounding box construction method"""

    class ModelResults(NamedTuple):
        """Statistics associated with model evaluation"""

        observations: int
        """Number of observations tested"""

        infeasible: int
        """Number of infeasible observations"""

        equality_violations: int
        """Number of violated equalities"""

        inequality_violations: int
        """Number of violated inequalities"""

        analysis_cores: int
        """Number of cores available for parallelized analysis"""

    class ConstraintStatistics(NamedTuple):
        """Statistics about constraints generated by an MFD"""

        equalities: int
        """Number of equality constraints"""

        inequalities: int
        """Number of inequality constraints"""

        trivial_inequalities: int
        """Number of trivial inequalities (containing only one non-zero term)"""

        equality_terms: int
        """Number of terms in equality constraints with non-zero coefficients"""

        inequality_terms: int
        """Number of terms in inequality constriants with non-zero coefficients"""

        trivial_inequality_terms: int
        """Number of terms in non-trivial inequality constraints with non-zero coefficients"""

    def flatten_nested_dict(
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Any]:
        """
        Flatten a nested dict[str, dict[str, Any]] into a dict[(str, str), Any].

        Example:
            {"a": {"x": 1, "y": 2}, "b": {"z": 3}}
        becomes:
            {("a", "x"): 1, ("a", "y"): 2, ("b", "z"): 3}
        """
        out: Dict[Tuple[str, str], Any] = {}

        for key1, inner in data.items():
            for key2, value in inner.items():
                out[(key1, key2)] = value

        return out

    def save_nested_dict_to_csv(
            d: dict[str, dict[str, Any]],
            path: Path
    ):
        """Save nested dictionary to disk as csv"""

        df = pd.DataFrame.from_dict(
            flatten_nested_dict(d),
            orient="index"
        )
        df.index = pd.MultiIndex.from_tuples(df.index, names=["model", "cset"])
        with path.open("w", encoding="utf-8") as file:
            df.to_csv(file)

    models: dict[str, dict[str, mb.ModelUnderTest]] = {}
    """Models by name and counter set"""

    timing: dict[str, dict[str, mb.ModelTimingStats]] = {}
    """Model timing statistics"""

    results: dict[str, dict[str, ModelResults]] = {}
    """Miscellaneous model statistics"""

    constraint_stats: dict[str, dict[str, ConstraintStatistics]] = {}
    """Constraint statistics by model and counter set"""

    print("Starting to run models")
    SCALING_DIR.mkdir(parents=True, exist_ok=True)

    for path in model_order:

        # Create models for each counter set
        for cset, counters in expected_counters.items():
            # Load the model
            try:
                model = mb.ModelUnderTest(
                    str(path),
                    counters,
                    drop_unexpected_counters = True,
                    error_on_missing_counters = True,
                    # enable the additional counters
                    use_m4 = True,
                    m4_args = ["-DPSC_COUNTERS=1"],
                )
            except:
                # Cannot load model, perhaps because the model is missing counters.
                # Skip this model.
                continue

            # Do not add model if it has no metadata
            if model.metadata is None:
                continue

            # Provide progress update
            print(f"Analyzing {model.metadata.name} with {cset}")

            # Update our set of models
            models.setdefault(model.metadata.name, {})
            models[model.metadata.name][cset] = model

            # Update constraint statistics
            constraint_stats.setdefault(model.metadata.name, {})
            constraint_stats[model.metadata.name][cset] = ConstraintStatistics(
                equalities=model.mfd.counter_cone.equalities.shape[0],
                inequalities=model.mfd.counter_cone.inequalities.shape[0],
                trivial_inequalities=((model.mfd.counter_cone.inequalities != 0).sum() == 1).sum(),
                equality_terms=(model.mfd.counter_cone.equalities != 0).sum().sum(),
                inequality_terms=(model.mfd.counter_cone.inequalities != 0).sum().sum(),
                trivial_inequality_terms=0
            )

            # Count the number of cores available for analysis
            analysis_cores: int = os.process_cpu_count() if os.process_cpu_count() is not None else 1

            # Do not perform feasibility analysis for the "MMU$" group, since these are
            # not real counters and so no data is recorded.
            if cset != "MMU$":
                # Trigger analysis
                _ = model.report(
                    ds_all_counters,
                    confidence_level=CONFIDENCE_LEVEL,
                    rel_err_tol=REL_ERR_TOL,
                    constraint_satisfaction=True,
                    depth=1,
                    start=CONFIDENCE_LEVEL,
                    noise_handling=NOISE_HANDLING_MODE
                )

                # Get feasibility testing results with same parameters as report generation
                feasibility_r: response.Response = model.model_feasibility(
                    ds_all_counters,
                    rel_err_tol=REL_ERR_TOL,
                    depth=1,
                    start=CONFIDENCE_LEVEL,
                    noise_handling=NOISE_HANDLING_MODE
                )

                # Count the total number of observations
                observations = len(feasibility_r.series)

                # Count the number of infeasible observations
                infeasible = (feasibility_r.series > CONFIDENCE_LEVEL).sum()

                # Get equality testing results with same parameters as report generation
                equalities_rs: list[tuple[str,response.Response]] = model.equality_satisfaction(
                    ds_all_counters,
                    confidence_level=CONFIDENCE_LEVEL,
                    rel_err_tol=REL_ERR_TOL,
                    noise_handling=NOISE_HANDLING_MODE
                )

                # Count the number of equality violations
                equality_violations: int = 0
                for _, r in equalities_rs:
                    equality_violations += (r.series == upath_expr.ConstraintSatisfactionOutcome.VIOLATED).sum()

                # Get inequality testing results with same parameters as report generation
                inequalities_rs: list[tuple[str,response.Response]] = model.inequality_satisfaction(
                    ds_all_counters,
                    confidence_level=CONFIDENCE_LEVEL,
                    rel_err_tol=REL_ERR_TOL,
                    noise_handling=NOISE_HANDLING_MODE
                )

                # Count the number of inequality violations
                inequality_violations: int = 0
                for _, r in inequalities_rs:
                    inequality_violations += (r.series == upath_expr.ConstraintSatisfactionOutcome.VIOLATED).sum()

                # Store statistics
                results.setdefault(model.metadata.name, {})
                results[model.metadata.name][cset] = ModelResults(
                    observations, infeasible, equality_violations, inequality_violations, analysis_cores
                )

            # Store timing results (including for MMU$)
            timing.setdefault(model.metadata.name, {})
            timing[model.metadata.name][cset] = model.profile

            # Checkpoint data to disk
            save_nested_dict_to_csv(timing, SCALING_DIR / "runtime.csv")
            save_nested_dict_to_csv(results, SCALING_DIR / "results.csv")
            save_nested_dict_to_csv(
                constraint_stats,
                SCALING_DIR / "constraints_stats.csv",
            )

            # Flush stoud and stderr
            sys.stdout.flush()
            sys.stderr.flush()

    print("Model scaling script complete.")

if __name__ == "__main__":
    main()
