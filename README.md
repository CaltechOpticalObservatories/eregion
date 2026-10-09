# Detector Characterization Framework

A modular Python framework for processing and analysis of imaging detector data (CCD/CMOS).  
Eregion provides the building blocks — tasks, data models, and core algorithms — and you compose
them into whatever workflow you need, in a script, a notebook, or your own orchestrator.

---

## Key Features

- Modular tasks — preprocessing, calibration, analysis, and image generation.
- Bring your own orchestration — tasks are plain Python objects you call directly, so they drop
  into a script, a notebook, or any workflow engine you already run.
- Reusable image ops — shared combine/stack core functionalities in `core`.
- Traceable results — every task returns a typed `TaskResult` carrying its data and provenance.
- Optional lazy execution — support for generator‑style tasks.

---

## Project Layout

```text
src/eregion/
├── cli/                       # `eregion` command-line interface
│   ├── commands/              # one module per subcommand (validate, ...)
│   └── main.py                # Typer app / entry point
├── configs/                  # YAML configuration files and code
│   ├── detectors/            # YAML configs for different detectors (e.g., DEIMOS, LRIS)
│   └── config.py             # Config loading and validation classes/functions
├── core/                     # Reusable core algorithms
│   └── image_operations.py   # image combine/stack ops
├── datamodels/
│   ├── image.py              # Flexible DetImage data class to hold image data, outputs and metadata
│   └── results.py            # TaskResult classes wrapping task outputs with their provenance
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

Import task classes from `eregion.tasks`, call `run(...)` (or `__call__(...)` for the raw-array
convenience path), and feed each result into the next task. You own the control flow, so ordering,
branching, parallelism, and retries are whatever your script or orchestrator does:

```python
from eregion.tasks import ImageCreator
from eregion.tasks.calibration import MasterBias
from eregion.tasks.preprocessing import BiasSubtraction, ScanSubtraction

detector_config = "src/eregion/configs/detectors/deimos_singledet.yaml"

# Build a master bias out of the bias frames
bias_res = ImageCreator(detector_config=detector_config).run(
    input_source="/path/to/data/*_bias_*.fits",
)
oscan_sub = ScanSubtraction(which_scan="serial_overscan", method="median_by_axis")
bias_res = oscan_sub.run(images=bias_res.data)
master_bias = MasterBias(method="median").run(images=bias_res.data).master_bias

# Apply it to the science frames
flat_res = ImageCreator(detector_config=detector_config).run(
    input_source="/path/to/data/*_flat_*.fits",
)
flat_res = oscan_sub.run(images=flat_res.data)
corrected = BiasSubtraction(only_image_area=True).run(
    images=flat_res.data, master_bias=master_bias,
)
```

Each task returns a `TaskResult` (see `datamodels/results.py`) holding the output data alongside
the parameters and upstream references used to produce it, so provenance survives however you
wire the steps together.

For large or streaming datasets, tasks that subclass `LazyTask` also expose `lazy_run(...)`, which
yields results batch by batch instead of materializing everything at once:

```python
creator = ImageCreator(detector_config=detector_config, max_batch_size=10)
for batch in creator.lazy_run(input_source="/path/to/data/*_flat_*.fits"):
    batch = oscan_sub.run(images=batch.data)
    batch = BiasSubtraction(only_image_area=True).run(
        images=batch.data, master_bias=master_bias,
    )
```

See `playground/` for complete, runnable end-to-end examples.

### Command line

The `eregion` CLI covers the parts of the framework that do not need a workflow, so a detector
config can be sanity-checked without writing a script:

```bash
  eregion validate path/to/detector.yaml   # load the config and print the detector it describes

  # Override ${...} placeholders in the config, and allow ${VAR} to fall back to env vars
  eregion validate path/to/detector.yaml --var width=2048 --env

  eregion --version
```

Run `eregion --help` for the full list of commands and options. The CLI is intentionally minimal;
new subcommands live as one module per command under `src/eregion/cli/commands/` so the surface
can grow without touching existing commands. Note that it does not execute workflows — composing
and running tasks is done in Python, as above.

### Status
Early development. More tasks to be added.

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
- Editable installation:
```bash
  pip install -e .
```
### Install additional dependencies for testing and development
```bash
  pip install -e '.[dev,test]'
```