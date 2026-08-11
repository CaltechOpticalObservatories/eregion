# Detector Characterization Framework

A modular Python framework for processing and analysis of imaging detector data (CCD/CMOS).  
Supports direct task usage in scripts and orchestration in config‑driven DAG workflows.

---

## Key Features

- Modular tasks — preprocessing, calibration, analysis, and image generation.
- Dual usage — call tasks directly in scripts or chain them in a pipeline.
- Reusable image ops — shared combine/stack core functionalities in `core`.
- Config‑driven orchestration — Prefect‑ready flows kept in `pipeline`.
- Optional lazy execution — support for generator‑style tasks.

---

## Project Layout

```text
src/eregion/
├── cli/                       # `eregion` command-line interface
│   ├── commands/              # one module per subcommand (run, validate, ...)
│   └── main.py                # Typer app / entry point
├── configs/                  # YAML configuration files and code
│   ├── detectors/            # YAML configs for different detectors (e.g., DEIMOS, LRIS)
│   ├── pipeline_flows/       # YAML configs defining flows for different processing pipelines
│   └── config.py             # Config loading and validation classes/functions
├── core/                     # Reusable core algorithms
│   └── image_operations.py   # image combine/stack ops
├── datamodels/
│   ├── image.py              # Flexible DetImage data class to hold image data, outputs and metadata
├── pipeline/                 # Engine for YAML-defined DAG workflows, uses Prefect to wrap tasks and flows
│   └── engine.py             
├── tasks/                    # Modular processing/analysis tasks with defined inputs/outputs
│   ├── analysis.py           # analysis tasks (e.g., ptc, linearity)
│   ├── calibration.py        # calibration tasks (e.g., masterbias, masterflat)
│   ├── imagegen.py           # for generating DetImage instances from detector config and input image data
│   ├── preprocessing.py      # preprocessing tasks (e.g., overscan trim, bias subtract)
│   └── task.py               # Base Task and LazyTask abstract classes
├── utils/                    # Utility functions
│   ├── image_utils.py        # array manipulation, etc.
│   ├── io_utils.py           # file I/O utilities (e.g., FITS read/write)
│   └── misc_utils.py         # miscellaneous utilities (e.g., logging setup)
README.md
data/                          # example data (e.g., raw images)
playground/                    # example notebooks for testing
tests/                         # unit tests
```

All internal code imports the package as `eregion.<subpackage>` (e.g. `from eregion.utils import configure_logger`), and installed usage is `import eregion`, `from eregion.tasks import ...`, etc.

### Usage
Scripted: import task classes from tasks/* and call run(...) or __call__(...).

Orchestrated: define a pipeline flow YAML (see `configs/pipeline_flows/example.yaml`) describing a
DAG of tasks, then run it either from Python:

```python
from eregion.pipeline import PipelineEngine

engine = PipelineEngine("path/to/pipeline.yaml")
engine.run()
```

or from the command line, without writing a notebook/script for each run:

```bash
  eregion validate path/to/pipeline.yaml   # build the DAG and print the execution plan, no tasks run
  eregion run path/to/pipeline.yaml        # build the DAG and execute it end-to-end

  # Override ${...} placeholders in the config, and allow ${VAR} to fall back to env vars
  eregion run path/to/pipeline.yaml --var data_dir=/data/raw --env
```

Run `eregion --help` for the full list of commands and options. The CLI is intentionally minimal
for now (`run`/`validate`); new subcommands live as one module per command under
`src/eregion/cli/commands/` so the surface can grow without touching existing commands.

### Status
Early development. More tasks and flows to be added.

## Installation

### Clone the repository
```bash
  git clone git@github.com:CaltechOpticalObservatories/eregion.git
  cd eregion
```

### Create and activate a virtual environment
```bash
  python3 -m venv venv
  source venv/bin/activate
```

### Install the package
- Standard installation:
```bash
  pip install .
```
- Development installation:
```bash
  pip install -e .
```
### Install additional dependencies for testing and development
```bash
  pip install -e .[dev]
```