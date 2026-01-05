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
eregion/
├── core/                    # reusable core algorithms
│   └── image_operations.py   # image combine/stack ops
├── data/                     # sample/raw data for testing
│   └── deimos_raw/
│       ├── *.fits
├── datamodels/
│   ├── detector_config.py    # For loading and parsing detector configuration files
│   ├── image.py              # DetImage (and child) data classes to hold image data and metadata
│   └── image_utils.py        # utilities for manipulating image data from numpy arrays to xarray DataArrays
├── pipeline/                 # Prefect flows / runners (orchestration layer)
├── playground/
│   ├── basic_ccd.yaml        # example config(s)
│   ├── deimos.yaml           # DEIMOS example config
│   └── test.ipynb            # notebook for quick testing and exmaple usage
├── tasks/                    # modular processing/analysis tasks
│   ├── analysis.py           # analysis tasks (e.g., ptc, linearity)
│   ├── calibration.py        # calibration tasks (e.g., masterbias, masterflat)
│   ├── imagegen.py           # For generating DetImage instances from detector config and input image data
│   ├── preprocessing.py      # preprocessing tasks (e.g., overscan trim, bias subtract)
│   └── task.py               # base Task and LazyTask abstract classes
├── tests/                    # unit tests
└── README.md
```

### Usage
Scripted: import task classes from tasks/* and call run(...) or __call__(...).

Orchestrated: define Prefect flows in pipeline/ that load a YAML from playground/ and execute steps using a generic runner.

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