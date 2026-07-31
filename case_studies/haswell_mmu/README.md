# Haswell MMU Case Study

This directory adapts the public Haswell MMU artifact into the `counterpoint-gem5`
mainline repository. The mainline already vendors the CounterPoint tool, so this case
study keeps only the lightweight integration surface needed to preserve the published
model source and reproduce the scaling workflow against the in-tree package.

The case study corresponds to the paper:

> Nick Lindsay, Caroline Trippel, Anurag Khandelwal, and Abhishek Bhattacharjee. 2026.
> **CounterPoint: Using Hardware Event Counters to Refute and Refine Microarchitectural
> Assumptions.** In *Proceedings of the 31st ACM International Conference on Architectural
> Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26).*
> Association for Computing Machinery, New York, NY, USA, 459–475.
> https://doi.org/10.1145/3779212.3790145

The original public artifact was validated against `counterpoint-public` commit
`257be9e22c5e7be9dc3a170b4a9cfc5e50a76753`. This repository already contains that tool
surface in-tree, so no nested `counterpoint/` checkout is needed here.

## What Is Tracked In Mainline

### Directory structure

This case-study directory intentionally tracks only source inputs and minimal metadata:

- `misc/`: miscellaneous metadata required by analysis scripts
- `models/`: published Haswell MMU model source (`.fdlm4`) and per-model metadata (`.json`)
- `scripts/`: reproducibility helpers adapted to the mainline package layout

Heavy artifact outputs remain ignored by `.gitignore`:

- `data/`
- `generated_figures/`
- `analysis/`
- `scaling/`
- `notebooks/`
- report subdirectories under `models/`

## Mainline Usage

Install the repository root as your working environment, then run the case-study tooling
from the repository root or any other directory. The scripts resolve paths relative to
their own file location.

Example:

```bash
"${PYTHON:-python3}" -m pip install -e .
```

Optional notebook/plotting dependencies for this case study are listed in
`case_studies/haswell_mmu/requirements.txt`.

## Data And Models

### Dataset

The published dataset remains external to the tracked mainline tree and is expected at:

`case_studies/haswell_mmu/data/experimental_data.parquet`

That directory is intentionally ignored because it is large artifact data rather than
source code.

### $\mu$-path Decision Diagrams (uDDs)

The `models` directory contains the Haswell MMU uDD source explored in the paper. Report
directories such as `models/basic/` and `models/basic.simple_bbox/` are generated outputs
and remain ignored.

## Scaling Workflow

The script `run_model_scaling.py` analyzes uDDs with various subsets of hardware event
counters to understand how model constraints and tool runtime scales with increasing
numbers of events.

Run:

```bash
"${PYTHON:-python3}" case_studies/haswell_mmu/scripts/run_model_scaling.py
```

Outputs are written to `case_studies/haswell_mmu/scaling/`.

The script produces three `csv` files:
- `constraints_stats.csv` reports how the constraints vary with the model and counters
- `results.csv` reports statistics on observations tested and their feasibility
- `runtime.csv` reports the runtime of each analysis phase
