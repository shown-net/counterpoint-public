# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025 Nicholas Lindsay

"""Helper functions and classes for tracking data in the model building process"""

from dataclasses import dataclass
import functools  # for automatic memoization
import json # for reading metadata
import math
from pathlib import Path
import textwrap
import time
from typing import NamedTuple, Optional

import pandas as pd

from pexpr.dataset import Dataset
from pexpr.response import Response
import upath
import upath.fdl.FdlToGraph
from upath.mfd.finaldiagram import FinalMicroFlowDiagram

import pexpr.base
import counterpoint.upath_expr as upath_expr
from counterpoint.upath_expr import NoiseHandlingMode

# Helpers for printing reports
LINE_LENGTH = 120
DOUBLE_LINE = "=" * LINE_LENGTH + "\n"
SINGLE_LINE = "-" * LINE_LENGTH + "\n"
STAR_LINE = "*" * LINE_LENGTH + "\n"

def read_event_list_file(path: str) -> list[str]:
    """Extract names of performance counter events from event list file"""
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.startswith("#")]

def stringify_constraint(a: pd.Series, relation: str) -> str:
    """Write an constraint in a human-readable form"""

    # Collect positive and negative terms
    a_pos = a[a > 0]
    a_neg = a[a < 0]

    # Helper function for formatting individual terms
    def format_term(name, coeff):
        if math.isclose(coeff, 1):
            return f"{name}"
        else:
            return f"{coeff} * {name}"

    # Form LHS and RHS
    lhs = " + ".join(format_term(name, coeff) for name, coeff in a_pos.items())
    rhs = " + ".join(format_term(name, -coeff) for name, coeff in a_neg.items())

    # Replace empty sides with zeros
    if lhs == "":
        lhs = "0"
    if rhs == "":
        rhs = "0"

    # Write as relation
    return f"{lhs} {relation} {rhs}"


def model_feasibility_report(
        results: Response,
        rel_err_tol: float,
        depth: int,
        start: float,
        noise_handling: NoiseHandlingMode
) -> str:
    """Generate a textual report from model feasibility results"""

    # Header
    report: str = STAR_LINE
    report += f"MODEL FEASIBILITY REPORT\n"
    report += STAR_LINE
    report += "\n"

    # Result summary
    report += DOUBLE_LINE
    report += f"(Relative error tolerance = {rel_err_tol})\n"
    report += f"Number passing by confidence level (depth={depth}, start={start:.2f}, noise_handling={noise_handling})\n"
    report += DOUBLE_LINE

    # Compute value counts
    value_counts = results.series.value_counts()
    value_counts.name = "#"
    value_counts.index.name = "Confidence Level"
    value_counts.sort_index(ascending=True, inplace=True)
    value_counts["Total"] = value_counts.sum()

    # Compute percentages and combine with value counts in a single DataFrame
    counts_and_pct = value_counts.to_frame()
    counts_and_pct["%"] = (value_counts / value_counts["Total"]) * 100.0
    report += counts_and_pct.reset_index().to_string(index=False) + "\n"

    report += DOUBLE_LINE
    report += "\n"

    # Per-configuration results
    report += DOUBLE_LINE
    report += "Per-configuration results\n"
    report += DOUBLE_LINE
    results.series.name = "Conf. Level"
    report += results.series.reset_index().to_string(index=False) + "\n"
    report += DOUBLE_LINE
    report += "\n"

    return report


def constraint_satisfaction_report(results: list[tuple[str, Response]]) -> str:
    """Generate a textual report from constraint satisfaction results"""

    report: str = ""

    for ineq, response in results:
        report += DOUBLE_LINE
        report += textwrap.fill(f"{ineq}", width=LINE_LENGTH) + "\n"
        report += SINGLE_LINE

        # Pre-compute total number of observations
        n_total = len(response.series)

        # Configurations which satisfy constraint
        passing = response.series[
            response.series == upath_expr.ConstraintSatisfactionOutcome.SATISFIED
        ]

        # Find the total number that pass
        n_passing = len(passing)
        if n_passing == 0:
            report += f"There were no passing observations.\n"
        else:
            pct_satisfied = 100 * (n_passing / n_total)
            report += f"There were {n_passing}/{n_total} passing observations ({pct_satisfied:.1f}%).\n"

        # Configurations which are inconclusive because not all counters were measured
        inconclusive = response.series[
            response.series == upath_expr.ConstraintSatisfactionOutcome.INCONCLUSIVE
        ]

        # Find the total number of inconclusive observations
        n_inconclusive = len(inconclusive)
        if n_inconclusive == 0:
            report += f"There were no inconclusive observations.\n"
        else:
            pct_inconclusive = 100 * (n_inconclusive / n_total)
            report += f"There were {n_inconclusive}/{n_total} inconclusive observations ({pct_inconclusive:.1f}%).\n"

        # Configurations which definitively fail
        failing = response.series[
            response.series == upath_expr.ConstraintSatisfactionOutcome.VIOLATED
        ]

        # Find the total number of failing observations
        n_failing = len(failing)
        if n_failing == 0:
            report += "There were no failing observations.\n"
        else:
            pct_failing = 100 * (n_failing / n_total)
            report += f"There were {n_failing}/{n_total} failing observations ({pct_failing:.1f}%):\n"
            report += "\n"
            failing.name = "Failing configurations"
            report += failing.index.to_frame(False).to_string(index=False) + "\n"

        # report += SINGLE_LINE
        # # Configurations which are inconclusive because not all counters were measured
        # inconclusive.name = "Inconclusive configurations"
        # report += "Inconclusive configurations:\n"
        # report += "\n"
        # if len(inconclusive) == 0:
        #     report += "No inconclusive configurations\n"
        # else:
        #     report += inconclusive.index.to_frame(False).to_string(index=False)
        #     report += "\n"

        report += DOUBLE_LINE
        report += "\n"

    return report

class ModelMetadata(NamedTuple):
    """Metedata about a model"""

    name: str
    """Model name"""

    features: list[str]
    """List of modelled features"""

@dataclass
class ModelTimingStats:
    """Stats about execution time"""

    load_time: Optional[float] = None
    """Time taken to load model (s)"""

    cone_time: Optional[float] = None
    """Time taken to compute model cone (s)"""

    statistic_time: Optional[float] = None
    """Time taken to compute data means and covariances (s)"""

    feasibility_test_time: Optional[float] = None
    """Time taken to establish model feasibility (s)"""

    equality_evaluation_time: Optional[float] = None
    """Time taken to evaluate equality constraints for infeasible observations (s)"""

    inequality_evaluation_time: Optional[float] = None
    """Time taken to evaluate inequality constraints for infeasible observations (s)"""

class ModelUnderTest:
    """A model under test in the model building procedure"""

    _filepath: str
    _metadata: Optional[ModelMetadata]
    _mfd: FinalMicroFlowDiagram
    _profile: ModelTimingStats

    def __init__(
            self,
            filepath: str,
            expected_counters: list[str],
            **kwargs
        ) -> None:
        """Load MFD from filepath.

        Arguments:
            filepath: str
                Path to FDL file
            expected_counters: list[str]
                Expected performance counters in MFD

        Optional arguments:
            drop_unexpected_counters: bool [default: False]
                Drop counters in MFD that are not in expected_counters
            error_on_missing_counters: bool [default: False]
                Raise a ValueError if MFD does not contain all expected_counters
            parse_cgroups : bool [default: True]
                Parse counter groups from FDL file.
            use_m4 : bool [default: False]
                Run m4 macro preprocessor before parsing FDL file.
            m4_args : list[str] (default: [])
                m4 macro preprocessor arguments.
        """
        self._filepath = filepath
        drop_unexpected_counters = kwargs.pop("drop_unexpected_counters", False)
        error_on_missing_counters = kwargs.pop("error_on_missing_counters", False)
        parse_cgroups = kwargs.pop("parse_cgroups", True)
        use_m4 = kwargs.pop("use_m4", False)
        m4_args = kwargs.pop("m4_args", [])

        # The json file, if it exists, provides metadata about the model.
        if use_m4:
            suffix = ".fdlm4"
        else:
            suffix = ".fdl"
        jsonpath = Path(filepath.removesuffix(suffix) + ".json")
        if jsonpath.exists():
            with jsonpath.open("r") as f:
                data = json.load(f)
                try:
                    self._metadata = ModelMetadata(**data)
                except TypeError as e:
                    raise ValueError(f"{jsonpath} is not valid format")
        else:
            self._metadata = None
            print(f"Warning: metadata file {jsonpath} does not exist")

        # Measure time required to load MFD
        start_time = time.time()

        # Load unfinalized MFD
        partial = upath.fdl.FdlToGraph.build_graph_from_fdl(
            filepath,
            parse_cgroups=parse_cgroups,
            use_m4=use_m4,
            m4_args=m4_args
        )

        # If required, drop unexpected counters from model
        if drop_unexpected_counters:
            for counter in partial.Counters():
                if counter not in expected_counters:
                        partial.RemoveCounter(counter)

        # Finalize the MFD
        self._mfd = FinalMicroFlowDiagram(partial)

        # End timing measurement
        end_time = time.time()
        load_time = end_time - start_time

        # Measure time taken to find model cone
        start_time = time.time()
        _ = self._mfd.counter_cone
        end_time = time.time()
        cone_time = end_time - start_time

        # Record timing information
        self._profile = ModelTimingStats(load_time=load_time, cone_time=cone_time)

        # Check that there are no missing counters
        if error_on_missing_counters:
            for counter in expected_counters:
                if counter not in partial.Counters():
                    raise ValueError(f"MFD missing {counter}!")

        # Verify model counters align with expectation
        if not self._counter_name_check(self._mfd, expected_counters):
            print("WARNING: mismatch in counters between model and expected")

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    @property
    def mfd(self) -> FinalMicroFlowDiagram:
        return self._mfd

    @property
    def profile(self) -> ModelTimingStats:
        return self._profile

    @staticmethod
    def _counter_name_check(mfd: FinalMicroFlowDiagram, expected: list[str]) -> bool:
        """Returns true if and only if counters in mfd match expected"""

        model_counters = mfd.Counters()

        okay = True

        for counter in model_counters:
            if counter not in expected:
                print(f"{counter} in model is not expected")
                okay = False

        for counter in expected:
            if counter not in model_counters:
                print(f"expected {counter} not in model")
                okay = False

        return okay

    @property
    def inequality_strings(self) -> list[str]:
        """Returns list of model inequalities as strings"""
        ineq_strings: list[str] = []
        for _, ineq in self.mfd.counter_cone.inequalities.iterrows():
            ineq_strings.append(stringify_constraint(ineq, "<="))
        return ineq_strings

    @property
    def equality_strings(self) -> list[str]:
        """Returns list of equalities as strings"""
        eq_strings: list[str] = []
        for _, eq in self.mfd.counter_cone.equalities.iterrows():
            eq_strings.append(stringify_constraint(eq, "=="))
        return eq_strings

    @functools.cache
    def model_feasibility(
        self,
        ds: Dataset,
        rel_err_tol: float,
        depth: int = 3,
        start: float = 0.5,
        noise_handling: NoiseHandlingMode = NoiseHandlingMode.AlignedBoundingBox
    ) -> Response:
        """Returns confidence level required for model feasibility.

        IMPORTANT: if ds is modified between calls to this function, then changes to ds
        will NOT be reflected by this function."""
        start_time = time.time()

        problem = upath_expr.MfdModelFeasibilityProblem(
            self.mfd,
            rel_err_tol,
            depth,
            start,
            noise_handling
        )
        response: Response =  ds.Evaluate(problem)

        end_time = time.time()
        self._profile.feasibility_test_time = (end_time - start_time)

        return response

    def model_feasibility_report(
            self,
            ds: Dataset,
            rel_err_tol: float,
            depth: int,
            start: float,
            noise_handling: NoiseHandlingMode
        ) -> str:
        """Returns a model feasibility report"""
        results = self.model_feasibility(ds, rel_err_tol, depth, start, noise_handling)
        return model_feasibility_report(
            results,
            rel_err_tol,
            depth,
            start,
            noise_handling
        )

    @functools.cache
    def inequality_satisfaction(
        self,
        ds: Dataset,
        confidence_level: float,
        rel_err_tol: float,
        noise_handling: NoiseHandlingMode
    ) -> list[tuple[str, Response]]:
        """Boolean responses for which model cone inequalities are violated at CL.

        IMPORTANT: if ds is modified between calls to this function, then changes to ds
        will NOT be reflected by this function."""

        # TODO: optimize further by testing constraints at 0% confidence level using
        #       trivial test (no linear programming required). use results to exclude
        #       configurations from later tests

        # TODO: don't test trivial constraints (X >= 0)

        # TODO: drop configurations that fail counter group equality tests

        # Test only those configurations which are known to fail at this confidence level.
        # We can find these by performing a binary model feasibility search with depth 1
        # starting at the confidence level.
        model_feasibility_results = self.model_feasibility(
            ds=ds,
            rel_err_tol=rel_err_tol,
            depth=1,
            start=confidence_level,
            noise_handling=noise_handling
        )
        configs_to_test = model_feasibility_results.series > confidence_level

        def select_config(config: dict) -> bool:
            """Select configurations to test based on configs_to_test"""
            as_tuple = tuple(config[k] for k in configs_to_test.index.names)
            return as_tuple in configs_to_test

        # List of inequalities
        results: list[tuple[str, Response]] = []

        start_time: float = time.time()

        for _, ineq in self.mfd.counter_cone.inequalities.iterrows():
            ineq_str = stringify_constraint(ineq, "<=")
            problem = upath_expr.ConstraintSatisfactionProblem(
                equalities=pd.DataFrame(),
                inequalities=ineq.to_frame().T,
                counter_groups=self.mfd.Graph().counter_groups,
                confidence=confidence_level,
                rel_err_tol=rel_err_tol,
                select=pexpr.base.TestConfigWithPredicate(select_config, drop=["time"]),
                noise_handling=noise_handling
            )
            response = ds.Evaluate(problem)
            results.append((ineq_str, response))

        end_time: float = time.time()
        self._profile.inequality_evaluation_time = (end_time - start_time)

        return results

    def inequality_satisfaction_report(
        self,
        ds: Dataset,
        confidence_level: float,
        rel_err_tol: float,
        evaluate: bool,
        noise_handling: NoiseHandlingMode
    ) -> str:
        """Returns an inequality satisfaction report"""
        if evaluate:
            results = self.inequality_satisfaction(
                ds,
                confidence_level,
                rel_err_tol,
                noise_handling
            )
            return constraint_satisfaction_report(results)
        else:
            report: str = ""

            report += DOUBLE_LINE
            report += "List of inequalities (not tested)\n"
            report += SINGLE_LINE

            for s in self.inequality_strings:
                report += s
                report += "\n"

            report += DOUBLE_LINE
            report += "\n"

            return report

    @functools.cache
    def equality_satisfaction(
        self,
        ds: Dataset,
        confidence_level: float,
        rel_err_tol: float,
        noise_handling: NoiseHandlingMode
    ) -> list[tuple[str, Response]]:
        """Boolean responses for which model equalities are violated at CL.

        IMPORTANT: if ds is modified between calls to this function, then changes to ds
        will NOT be reflected by this function."""

        # TODO: optimize further by testing constraints at 0% confidence level using
        #       trivial test (no linear programming required). use results to exclude
        #       configurations from later tests

        # TODO: consider adding a seperate test just for the counter group equalities

        # TODO: drop configurations that fail counter group equality tests

        # Test only those configurations which are known to fail at this confidence level.
        # We can find these by performing a binary model feasibility search with depth 1
        # starting at the confidence level.
        model_feasibility_results = self.model_feasibility(
            ds=ds,
            rel_err_tol=rel_err_tol,
            depth=1,
            start=confidence_level,
            noise_handling=noise_handling
        )
        configs_to_test = model_feasibility_results.series > confidence_level

        def select_config(config: dict) -> bool:
            """Select configurations to test based on configs_to_test"""
            as_tuple = tuple(config[k] for k in configs_to_test.index.names)
            return as_tuple in configs_to_test

        # List of equalities
        results: list[tuple[str, Response]] = []

        start_time: float = time.time()

        for _, eq in self.mfd.counter_cone.equalities.iterrows():
            eq_str = stringify_constraint(eq, "==")
            problem = upath_expr.ConstraintSatisfactionProblem(
                equalities=eq.to_frame().T,
                inequalities=pd.DataFrame(),
                counter_groups=self.mfd.Graph().counter_groups,
                confidence=confidence_level,
                rel_err_tol=rel_err_tol,
                select=pexpr.base.TestConfigWithPredicate(select_config, drop=["time"]),
                noise_handling=noise_handling
            )
            response = ds.Evaluate(problem)
            results.append((eq_str, response))

        end_time: float = time.time()
        self._profile.equality_evaluation_time = (end_time - start_time)

        return results

    def equality_satisfaction_report(
            self, ds: Dataset,
            confidence_level: float,
            rel_err_tol: float,
            evaluate: bool,
            noise_handling: NoiseHandlingMode
        ) -> str:
        """Returns an equality satisfaction report"""
        if evaluate:
            results = self.equality_satisfaction(
                ds, confidence_level, rel_err_tol, noise_handling
            )
            return constraint_satisfaction_report(results)
        else:
            report: str = ""

            report += DOUBLE_LINE
            report += "List of equalities (not tested)\n"
            report += SINGLE_LINE

            for s in self.equality_strings:
                report += s
                report += "\n"

            report += DOUBLE_LINE
            report += "\n"

            return report

    def constraint_satisfaction_report(
        self,
        ds: Dataset,
        confidence_level: float,
        rel_err_tol: float,
        evaluate: bool,
        noise_handling: NoiseHandlingMode
    ) -> str:
        """Return report for equality and inequality constraints"""

        report: str = STAR_LINE
        if evaluate:
            report += (
                f"CONSTRAINT SATISFACTION REPORT @ {confidence_level} CONFIDENCE LEVEL ({noise_handling}, RELATIVE ERROR TOLERANCE = {rel_err_tol})\n"
            )
        else:
            report += "CONSTRAINT ENUMERATION"
            report += "\n"

        report += STAR_LINE
        report += "\n"

        report += self.equality_satisfaction_report(
            ds, confidence_level, rel_err_tol, evaluate, noise_handling
        )
        report += self.inequality_satisfaction_report(
            ds, confidence_level, rel_err_tol, evaluate, noise_handling
        )

        return report

    def model_facts(self) -> str:
        """List facts about the model in a report-like form"""

        report: str = STAR_LINE
        report += "MODEL FACTS\n"
        report += STAR_LINE
        report += "\n"

        report += DOUBLE_LINE

        report += f"Total paths: {len(self.mfd.Paths())}\n"
        report += f"Total nodes: {len(self.mfd.Graph().Nodes())}\n"
        report += f"Total counters: {len(self.mfd.Counters())}\n"

        report += DOUBLE_LINE
        report += "\n"

        return report

    def report(self, ds: Dataset, **kwargs) -> str:
        """Generate a textual report of model results against ds.

        Arguments:
            ds: Dataset
                Dataset to evaluate against model

        Optional arguments:
            confidence_level: float [default: 0.5]
                Confidence level for constraint satisfaction tests
            rel_err_tol: float [default: 0.000002]
                Relative error tolerence (as fraction of max event count)
            constraint_satisfaction : bool [default: False]
                Identify configurations that fail constraints
            depth : int [default: 3]
                Observation feasibility confidence level binary search depth
            start : float [default: 0.5]
                Starting point for binary search
            noise_handling : NoiseHandlingMode
                Approach for handling noise
        """

        confidence_level: float = kwargs.get("confidence_level", 0.5)
        rel_err_tol: float = kwargs.get("rel_err_tol", 0.000002)
        constraint_satisfaction: bool = kwargs.get("constraint_satisfaction", False)
        depth: int = kwargs.get("depth", 3)
        start: float = kwargs.get("start", 0.5)
        noise_handling: NoiseHandlingMode = kwargs.get(
            "noise_handling", NoiseHandlingMode.AlignedBoundingBox
        )

        report: str = ""

        # HACKY: Invalidate cached means and covariances and recompute
        upath_expr._InvalidateUpathStats()

        start_time: float = time.time()

        upath_expr._GetUpathStats(ds.df)

        end_time: float = time.time()
        self._profile.statistic_time = (end_time - start_time)

        # Ensure that these are not recomputed
        upath_expr._LockUpathStats()

        report += self.model_facts()
        report += self.model_feasibility_report(
            ds, rel_err_tol, depth, start, noise_handling
        )
        report += self.constraint_satisfaction_report(
                    ds,
                    confidence_level,
                    rel_err_tol,
                    constraint_satisfaction,
                    noise_handling
                  )

        # Stats can now be modified
        upath_expr._UnlockUpathStats()

        return report
