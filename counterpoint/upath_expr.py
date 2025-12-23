# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025 Nicholas Lindsay

"""Pexpr wrapper for uFlow"""

from enum import Enum
from multiprocessing import Pool
from typing import Any, NamedTuple, Optional

import pandas as pd

from pexpr.base import ZeroaryExpr, Expr, Counter, TestConfigWithPredicate
from upath.mfd.finaldiagram import FinalMicroFlowDiagram
from upath.mfd.meta import CounterGroup
import upath.solver as solver
import upath.solver.cvconstraints as cvconstraints

MIN_SAMPLES = 30  # minimum number of samples

class NoiseHandlingMode(Enum):
    """Approach for handling noise"""
    AlignedBoundingBox = "AlignedBoundingBox"
    SimpleBox = "SimpleBox"

class _UpathStats:
    """Statistics from dataset dataframe used for uflow calculations"""

    _means: pd.DataFrame  # sample means
    _covar: pd.DataFrame  # estimated covariance of sample mean

    def __init__(self, df: pd.DataFrame):
        # Find all non-time levels
        non_time_levels = [x for x in df.columns.names if x != "time"]

        # Groups dataframe by all configuration levels other than "time"
        def group_by_config(df: pd.DataFrame, keys: bool = False):
            return df.groupby(level=non_time_levels, group_keys=keys)

        # Find all counters in dataset
        counters = [x for x in df.index if x.startswith("Counter(")]  # hacky

        # Extract the counters from the Dataset
        counter_df = df.loc[counters]

        # Replace index with counter name
        counter_df.index = counter_df.index.map(lambda x: eval(x).name)
        counter_df.index.name = "counter"

        # Transpose
        counter_df = counter_df.T

        # For each configuration, remove samples that do not have all performance counters
        # for that configuration.
        def remove_incomplete(g):
            all_nan_rows = g.isna().all(axis=1)
            cols_to_remove = g.loc[~all_nan_rows].isna().any(axis=0)
            return g.loc[:, ~cols_to_remove]

        counter_df: pd.DataFrame = group_by_config(counter_df).apply(remove_incomplete)  # type: ignore

        # Find the number of samples per configuratino
        n_samples = group_by_config(counter_df, False).count()

        # Drop configurations with fewer than MIN_SAMPLES samples
        counter_df = counter_df[n_samples >= MIN_SAMPLES]
        n_samples = n_samples[n_samples >= MIN_SAMPLES]

        # Find the means per configuration
        means = group_by_config(counter_df).mean()

        # Find the covariance per configuration
        covar = group_by_config(counter_df, True).cov(ddof=1)

        # Estimate sample mean variance-covariance matrix
        sample_mean_covar = (covar.T / n_samples.stack()).T

        # Update variables
        self._means = means
        self._covar = sample_mean_covar

    @staticmethod
    def _counter_repr_to_name(x: str) -> str:
        """Converts repr(Counter...) to counter name"""
        e = eval(x)
        assert isinstance(e, Counter)
        return e.name

    @property
    def means(self) -> pd.DataFrame:
        return self._means

    @property
    def covar(self) -> pd.DataFrame:
        return self._covar

    def stats_subset_clean(
        self, config, counters: list[str]
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Return means and covars for valid counters recorded for config.

        Counters with NaN statistics are dropped."""
        # Compute metric for single configuration
        means = self.means.loc[config]
        covar = self.covar.loc[config]

        # Select only the subset of counters present in the model
        counter_subset = [x for x in means.index if x in counters]
        means_subset = means[counter_subset]
        covar_subset = covar.loc[counter_subset, counter_subset]

        # From these, use only those that don't have NaN values
        means_subset_clean = means_subset.dropna()
        covar_subset_clean = covar_subset.dropna(how="all", axis=0).dropna(
            how="all", axis=1
        )

        # Return as tuple
        return means_subset_clean, covar_subset_clean

    def construct_cv_constraints(
        self,
        config,
        counters: list[str],
        confidence: float,
        rel_err_tol: float,
        mode: NoiseHandlingMode
    ) -> cvconstraints.CounterValueConstraints:
        """Construct cvconstraints for config, list of counters, and error tolerence"""

        # Check parameters
        assert confidence >= 0.0
        assert confidence <= 1.0

        # Get statistics
        means, covar = self.stats_subset_clean(config, counters)

        # Construct confidence region
        if confidence == 0:
            return cvconstraints.ExactValues(means.to_dict())
        else:
            if mode == NoiseHandlingMode.AlignedBoundingBox:
                return cvconstraints.NormalDistBoundingBox(
                    mean=means,
                    covar=covar,
                    confidence_level=confidence,
                    rel_err_tol=rel_err_tol
                )
            elif mode == NoiseHandlingMode.SimpleBox:
                return cvconstraints.NormalDistSimpleBox(
                    mean=means,
                    covar=covar,
                    confidence_level=confidence,
                    rel_err_tol=rel_err_tol
                )
            else:
                raise ValueError(f"Unknown NoiseHandlingMode {mode}")


# Cache stats for a DataFrame
_upath_stats_cache_data: Optional[_UpathStats] = None
_upath_stats_cache_tag: Optional[int] = None
_upath_stats_locked: bool = False

def _GetUpathStats(df: pd.DataFrame) -> _UpathStats:
    """Gets Upath statistics for a Dataset"""

    global _upath_stats_cache_tag
    global _upath_stats_cache_data

    if _upath_stats_cache_tag is None or _upath_stats_cache_tag != id(df):
        if _upath_stats_locked:
            raise RuntimeError("Cannot recompute upath stats: locked")

        print("Computing upath stats.")
        _upath_stats_cache_data = _UpathStats(df)
        _upath_stats_cache_tag = id(df)

    assert _upath_stats_cache_data is not None
    return _upath_stats_cache_data

def _InvalidateUpathStats():
    """Invalidate cached statistics"""

    global _upath_stats_cache_tag
    global _upath_stats_cache_data

    _upath_stats_cache_tag = None
    _upath_stats_cache_tag = None

def _LockUpathStats():
    """Prevent stats from changing"""
    global _upath_stats_locked

    _upath_stats_locked = True

def _UnlockUpathStats():
    """Prevent stats from changing"""
    global _upath_stats_locked

    _upath_stats_locked = False

class _ModelFeasibilityProblemInstanceArgs(NamedTuple):
    """Arguments for model feasibility problem instance"""

    stats: _UpathStats
    model: FinalMicroFlowDiagram
    configuration: Any
    rel_err_tol: float
    search_depth: int
    start: float
    noise_handling: NoiseHandlingMode

def _SolveModelFeasibilityProblemInstance(
    args: _ModelFeasibilityProblemInstanceArgs,
) -> tuple[str, float]:
    """Returns (configuration, confidence level) for model feasibility problem instance"""

    means_subset_clean, covar_subset_clean = args.stats.stats_subset_clean(
        args.configuration, args.model.Counters()
    )

    # Check at a 0% confidence level
    value_constraints = cvconstraints.ExactValues(means_subset_clean.to_dict())
    lp_problem = solver.ModelFeasibilityProblem(args.model, value_constraints)
    lp_solution = lp_problem.Solve()
    if lp_solution.Status() == "Optimal":
        return (args.configuration, 0.0)

    # Otherwise perform binary search on the confidence level until a confidence level
    # which results in feasibility is reached.
    d = 0
    lb = 0.0  # lower bound
    ub = 1.0  # upper bound
    while d < args.search_depth:
        if d == 0:
            # Biased starting point
            trial_cl = args.start
        else:
            # Compute the trial confidence level
            trial_cl = 0.5 * (lb + ub)

        # Construct the constraints at the current confidence level
        if args.noise_handling == NoiseHandlingMode.AlignedBoundingBox:
            value_constraints = cvconstraints.NormalDistBoundingBox(
                means_subset_clean, covar_subset_clean, trial_cl, args.rel_err_tol
            )
        elif args.noise_handling == NoiseHandlingMode.SimpleBox:
            value_constraints = cvconstraints.NormalDistSimpleBox(
                means_subset_clean, covar_subset_clean, trial_cl, args.rel_err_tol
            )
        else:
            raise ValueError(f"Unknown NoiseHandlingMode {args.noise_handling}")

        # Test for feasibility
        lp_problem = solver.ModelFeasibilityProblem(args.model, value_constraints)
        lp_solution = lp_problem.Solve()
        if lp_solution.Status() == "Optimal":
            # Reduce upper bound on result
            ub = trial_cl
        else:
            # Reduce lower bound on result
            lb = trial_cl

        # Increment depth
        d += 1

    # Return the upper bound confidence level, which the model is known to pass with. Note
    # that if binary search doesn't find any feasible solutions, this value will be 1.0,
    # which is mathematically garuanteed to be feasible (since it results in an infinitely
    # wide confidence interval).
    return (args.configuration, ub)


class MfdModelFeasibilityProblem(ZeroaryExpr):
    """Find minimum confidence level for feasibility of a model"""

    _model: FinalMicroFlowDiagram
    _rel_err_tol: float # relative error tolerance
    _search_depth: int  # binary search depth
    _start: float # starting confidence level for binary search
    _noise_handling: NoiseHandlingMode

    def __init__(
            self,
            model: FinalMicroFlowDiagram,
            rel_err_tol: float,
            search_depth: int,
            start: float = 0.5,
            noise_handling: NoiseHandlingMode = NoiseHandlingMode.AlignedBoundingBox
    ):
        super().__init__()
        self._model = model
        self._rel_err_tol = rel_err_tol
        self._search_depth = search_depth
        self._start = start
        self._noise_handling = noise_handling

    def __repr__(self):
        return f"{self.__class__.__name__}({self._model!r},{self._rel_err_tol!r},{self._search_depth!r},{self._start!r},{self._noise_handling!r})"

    def fork(self, children: list[Expr]) -> Expr:
        raise NotImplementedError

    def eimpl(self, ctx: pd.DataFrame) -> pd.Series:
        # Find means and variance-covariance matrix
        stats = _GetUpathStats(ctx)

        # Solve for all configurations in parallel
        with Pool() as p:
            problems = [
                _ModelFeasibilityProblemInstanceArgs(
                    stats,
                    self._model,
                    config,
                    self._rel_err_tol,
                    self._search_depth,
                    self._start,
                    self._noise_handling
                )
                for config in stats.means.index
            ]
            results = {
                c[0]: c[1]
                for c in p.imap_unordered(
                    _SolveModelFeasibilityProblemInstance,
                    problems,
                    chunksize=32
                )
            }

        # Convert to series
        series = pd.Series(results)

        # Ensure Index has correct level names
        series.index.names = stats.means.index.names

        # Sort by configuration
        series = series.sort_index()

        return series

    def _to_latex_impl(self, depth: int) -> str:
        raise NotImplementedError


class ConstraintSatisfactionOutcome(Enum):
    SATISFIED = "Satisfied"
    INCONCLUSIVE = "Inconclusive"
    VIOLATED = "Violated"


class _ConstraintSatisfactionProblemInstanceArgs(NamedTuple):
    """Arguments for inequality satisfaction problem instance"""

    stats: _UpathStats
    counters: list[str]
    equalities: pd.DataFrame
    inequalities: pd.DataFrame
    counter_group_equalities: pd.DataFrame
    configuration: Any
    confidence: float
    rel_err_tol: float
    noise_handling: NoiseHandlingMode


def _counter_derivable_directly_from_counter_groups(
    counter: str, measured: set[str], cg_equalities: pd.DataFrame
) -> bool:
    """Returns true if counter is derivable from measured and counter group equalities"""
    if counter not in cg_equalities:
        return False

    # Select all equalities where counter appears with a non-zero coefficient
    relevant_equalities = cg_equalities[cg_equalities[counter] != 0]

    # For each equality, see if counter can be derived from others
    for _, eq in relevant_equalities.iterrows():
        counter_determined = True
        # Check all the counters that appear in the equality
        for c, coeff in eq.items():
            if coeff == 0:
                # Ignore counters not in equality
                continue
            if c == counter:
                # Ignore the counter we are testing
                continue
            if c in measured:
                # Ignore defined counters
                continue
            else:
                # Another undefined counter in equation =>
                # counter value not determined by this equation
                counter_determined = False
                break

        if counter_determined:
            return True

    # Counter could not be derived directly from any counter group
    return False


def _counters_derivable_directly_from_counter_groups(
    counters: set[str], measured: set[str], cg_equalities: pd.DataFrame
) -> bool:
    """Returns true iff all counters are directly derivable from counter groups"""
    return all(
        _counter_derivable_directly_from_counter_groups(c, measured, cg_equalities)
        for c in counters
    )


def _counters_known_or_directly_derivable_from_counter_groups(
    counters: set[str], measured: set[str], cg_equalities: pd.DataFrame
) -> bool:
    """Returns true iff counters are known or derivable directly from counter_groups"""

    # Find counters that are not directly measured
    diff = counters - measured

    if len(diff) == 0:
        # All counters are measured
        return True
    else:
        # Return true only if the counters that are not directly measured can be derived
        # from other measured counters using the counter group equalities
        return all(
            _counter_derivable_directly_from_counter_groups(c, measured, cg_equalities)
            for c in counters
        )


def _SolveConstraintSatisfactionProblemInstance(
    args: _ConstraintSatisfactionProblemInstanceArgs,
) -> tuple[str, ConstraintSatisfactionOutcome]:
    """Returns (configuration, outcome) for constraint satisfaction problem"""

    # Construct counter value constraints
    value_constraints = args.stats.construct_cv_constraints(
        args.configuration,
        args.counters,
        args.confidence,
        args.rel_err_tol,
        args.noise_handling
    )

    # Combine equalities and counter group equalities
    lp_equalities = (
        pd.concat((args.equalities, args.counter_group_equalities), axis=0)
        .fillna(0)
        .reset_index(drop=True)
    )

    # Solve
    lp_problem = solver.ConstraintSatisfactionProblem(
        value_constraints, lp_equalities, args.inequalities
    )
    lp_solution = lp_problem.Solve()

    # Interpret feasibility
    if lp_solution.Status() == "Optimal":

        # The results are inconclusive if any of the following conditions hold:
        # 1. There is an equality which has a non-recorded counter with a non-zero
        #    coefficient, and the non-recorded counter cannot be inferred directly from
        #    counter groups and other counters.
        # 2. There is an inequality which has a non-recorded counter with a coefficent
        #    greater than zero, and the non-recorded counter cannot be inferred directly
        #    from counter groups and other counters.

        for _, eq in args.equalities.iterrows():
            # Find set of counters that appear in this equality
            critical_vars = set(eq[eq != 0].index)

            # NOTE: if eq is a counter group equality, then all arguments are required
            #       (this is the intended behavior)
            if not _counters_known_or_directly_derivable_from_counter_groups(
                critical_vars, value_constraints.counters, args.counter_group_equalities
            ):
                return (
                    args.configuration,
                    ConstraintSatisfactionOutcome.INCONCLUSIVE,
                )

        for _, ineq in args.inequalities.iterrows():
            # Find set of counters that appear in inequality with coefficient > 0
            critical_vars = set(ineq[ineq > 0].index)

            if not _counters_known_or_directly_derivable_from_counter_groups(
                critical_vars, value_constraints.counters, args.counter_group_equalities
            ):
                return (args.configuration, ConstraintSatisfactionOutcome.INCONCLUSIVE)

        # Otherwise, the equality is satisfied and the results are conclusive
        return (args.configuration, ConstraintSatisfactionOutcome.SATISFIED)
    else:
        assert lp_solution.Status() == "Infeasible"
        return (args.configuration, ConstraintSatisfactionOutcome.VIOLATED)


def index_entry_to_dict(index, entry) -> dict[str, Any]:
    """Convert MultiIndex entry to dict"""
    return {index.names[i]: entry[i] for i in range(len(index.names))}


class ConstraintSatisfactionProblem(ZeroaryExpr):
    """Determine if constraints satisfied by performance counter values"""

    _equalities: pd.DataFrame
    _inequalities: pd.DataFrame
    _counter_groups: list[CounterGroup]
    _confidence: float
    _rel_err_tol: float
    _select: Optional[TestConfigWithPredicate]
    _noise_handling: NoiseHandlingMode

    def __init__(
        self,
        equalities: pd.DataFrame,
        inequalities: pd.DataFrame,
        counter_groups: list[CounterGroup],
        confidence: float,
        rel_err_tol: float,
        select: Optional[TestConfigWithPredicate] = None,
        noise_handling: NoiseHandlingMode = NoiseHandlingMode.AlignedBoundingBox
    ):
        super().__init__()
        self._equalities = equalities
        self._inequalities = inequalities
        self._counter_groups = counter_groups
        self._confidence = confidence
        self._rel_err_tol = rel_err_tol
        self._select = select
        self._noise_handling = noise_handling

    @property
    def equalities(self) -> pd.DataFrame:
        return self._equalities

    @property
    def inequalities(self) -> pd.DataFrame:
        return self._inequalities

    @property
    def counter_groups(self) -> list[CounterGroup]:
        return self._counter_groups

    @property
    def confidence(self) -> float:
        return self._confidence

    def __repr__(self):
        return f"{self.__class__.__name__}({self._equalities!r},{self._inequalities!r},{self._confidence!r},{self._select!r})"

    def fork(self, children: list[Expr]) -> Expr:
        raise NotImplementedError

    def eimpl(self, ctx: pd.DataFrame) -> pd.Series:
        # Find means and variance-covariance matrix
        stats = _GetUpathStats(ctx)

        # Get selection mask
        mask = pd.Series(True, index=ctx.columns.droplevel("time").drop_duplicates())
        if self._select is not None:
            mask = self._select.eimpl(ctx)

        # Equalities implied by counter groups
        counter_group_equalities: pd.DataFrame
        if self._counter_groups:
            counter_group_equalities = pd.concat(
                [g.equality for g in self._counter_groups], axis=1
            ).T.fillna(0.0)
        else:
            counter_group_equalities = pd.DataFrame(columns=self._equalities.columns)

        # Use counters from equalities, inequalities, and counter groups
        counters: list[str] = (
            self._equalities.columns.union(self._inequalities.columns)
            .union(counter_group_equalities.columns)
            .to_list()
        )

        # Solve for all configurations in parallel
        with Pool() as p:
            problems = [
                _ConstraintSatisfactionProblemInstanceArgs(
                    stats,
                    counters,
                    self.equalities,
                    self.inequalities,
                    counter_group_equalities,
                    config,
                    self._confidence,
                    self._rel_err_tol,
                    self._noise_handling
                )
                for config in stats.means.index if mask[config]
            ]
            results = {
                c[0]: c[1]
                for c in p.imap_unordered(
                    _SolveConstraintSatisfactionProblemInstance,
                    problems,
                    chunksize=32
                )
            }

        # Convert to series
        series = pd.Series(results)

        # Ensure Index has correct level names
        series.index.names = stats.means.index.names

        # Sort by configuration
        series = series.sort_index()

        return series

    def _to_latex_impl(self, depth: int) -> str:
        raise NotImplementedError
