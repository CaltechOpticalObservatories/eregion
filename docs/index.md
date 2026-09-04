# Eregion

A modular Python framework for processing and analysis of imaging detector data (CCD/CMOS).
Supports direct task usage in scripts and orchestration in config-driven DAG workflows.

## Key Features

- Modular tasks — preprocessing, calibration, analysis, and image generation.
- Dual usage — call tasks directly in scripts or chain them in a pipeline.
- Reusable image ops — shared combine/stack core functionality in `core`.
- Config-driven orchestration — Prefect-ready flows kept in `pipeline`.
- Optional lazy execution — support for generator-style tasks.

## Installation

```bash
git clone git@github.com:CaltechOpticalObservatories/eregion.git
cd eregion
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Install additional dependencies for testing and development:

```bash
pip install -e '.[dev,test]'
```

## Usage

**Scripted** — import task classes from `eregion.tasks` and call `run(...)` or `__call__(...)`.

**Orchestrated** — define a pipeline flow YAML describing a DAG of tasks, then run it from Python:

```python
from eregion.pipeline import PipelineEngine

engine = PipelineEngine("path/to/pipeline.yaml")
engine.run()
```

or from the command line:

```bash
eregion validate path/to/pipeline.yaml   # build the DAG and print the execution plan, no tasks run
eregion run path/to/pipeline.yaml        # build the DAG and execute it end-to-end

# Override ${...} placeholders in the config, and allow ${VAR} to fall back to env vars
eregion run path/to/pipeline.yaml --var data_dir=/data/raw --env
```

Run `eregion --help` for the full list of commands and options.

```{toctree}
:hidden:
:maxdepth: 2

design/index
api/index
```
