# CounterPoint

CounterPoint is a microarchitectural modelling framework for refuting and refining
microarchitectural assumptions using hardware event counters.

CounterPoint is built around novel $\mu$-path Decision Diagrams (uDDs), which describes
how a micro-op interacts with the microarchitecture, including hardware event counters.
uDDs are specified using the *flow description language* (FDL), a domain-specific
language for this purpose. CounterPoint then applying a series of transformation passes
that convert the FDL file into a set of rules that can determine if hardware event counter
observations are consistent with the model. If the observations are not consistent, then
CounterPoint returns concrete counter-examples, enabling model refinement.


### Citing CounterPoint

We would appreciate if you could cite our paper if you use it for your work:

> Nick Lindsay, Caroline Trippel, Anurag Khandelwal, and Abhishek Bhattacharjee. 2026.
> **CounterPoint: Using Hardware Event Counters to Refute and Refine Microarchitectural
> Assumptions.** In *Proceedings of the 31st ACM International Conference on Architectural
> Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26).*
> Association for Computing Machinery, New York, NY, USA, 459–475.
> https://doi.org/10.1145/3779212.3790145

### Haswell MMU Case Study

We have used CounterPoint to explore the memory management unit (MMU) implementation on
an Intel Haswell Server microprocessor. The results are presented in the above paper. You
can find the full public dataset and analysis outputs as a Zenodo artifact:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18017051.svg)](https://doi.org/10.5281/zenodo.18017051)

This repository also vendors a lightweight, mainline-adapted version of that case study
under `case_studies/haswell_mmu/`, keeping the model source, scaling script, and minimal
metadata in-tree while leaving heavyweight artifact outputs external/ignored.

### Contributing

We are excited to see how CounterPoint can be used to solve hardware event counter related
problems. If you are interested in working on CounterPoint or expanding it for your own
work, please get in touch with Nick Lindsay.

If you would like to leave comments, queries, or suggestions, please open a
[GitHub issue](https://github.com/NicholasLindsay/counterpoint-public/issues)
or start a [GitHub discussion](https://github.com/NicholasLindsay/counterpoint-public/discussions).

### Status

CounterPoint is the start of a journey to apply formal modelling techniques to increase
the interpretabiility and utility of hardware event counters. CounterPoint is a research
tool and, whilst we will try our best to retain a stable codebase, we cannot promise for
now that there will be not be API breaking changes.

## Getting started with CounterPoint

To get started quickly, we recommend downloading the Zenodo artifact for our ASPLOS paper.

### Zenodo artifact (ASPLOS'26 paper)

This artifact contains the dataset, models, and CounterPoint analysis scipts from our
ASPLOS'26 paper:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18017051.svg)](https://doi.org/10.5281/zenodo.18017051)

### Cloning this repository

This repository can be cloned with the following command:
```bash
git clone --recurse-submodules https://github.com/NicholasLindsay/counterpoint-public.git counterpoint
```

This repository uses `git` submodules. When cloning, use `--recurse-submodules` to
automatically initialize and update all submodules.

If you have already cloned the repository without this flag, you can initialize the
submodules afterwards:

```bash
git submodule update --init --recursive
```

### Using CounterPoint

CounterPoint is a Python package, making it perfect for use in Jupyer notebooks and other
Python environments.
To add CounterPoint to your project, you simply need to add this repository as a Python
dependency (e.g. via `requirements.txt` or `pyproject.toml` files) and install using
`pip`. This will automatically build CounterPoint and all of it's dependencies.


## CounterPoint internals

CounterPoint internally is build on top of two Python packages that we have developed:
`upath` and `Pexpr`. We provide a high level summary of these packages here; for more
details, please refer to their respective `README.md`.

### `upath`

`upath` implements the parsing, building and analysis of uDDs. `upath` consists of two
components:
- A front-end which loads `.fdl` files and constructs an internal uDD representation
- A solver which checks hardware event counter values for consistency with the uDD


### `Pexpr`

`Pexpr` is a custom Python package designed to work with large hardware event counter
datasets. `Pexpr` supports standard mathematical and statistical operations. At the core
of `Pexpr` are Pexpressions: functions which accept performance counter measurements and
return the result of some computation. `Pexpr` can be used for evaluating performance
counter metrics, computing summary statistics, plotting performance counter values, and
more.

CounterPoint provides a wrapper (`upath_expr.py`) that enables `upath` to be used within
`Pexpr` expressions. CounterPoint uses this wrapper to evaluate uDDs on large performance
counter datasets.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file
for details.
