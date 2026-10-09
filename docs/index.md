# Eregion

A modular Python framework for processing and analysis of imaging detector data (CCD/CMOS).
Eregion provides the building blocks — tasks, data models, and core algorithms — and you compose
them into whatever workflow you need, in a script, a notebook, or your own orchestrator.

## Key Features

- Modular tasks — preprocessing, calibration, analysis, and image generation.
- Bring your own orchestration — tasks are plain Python objects you call directly, so they drop
  into a script, a notebook, or any workflow engine you already run.
- Reusable image ops — shared combine/stack core functionality in `core`.
- Traceable results — every task returns a typed `TaskResult` carrying its data and provenance.
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

See `playground/` in the repository for complete, runnable end-to-end examples.

## Command line

The `eregion` CLI covers the parts of the framework that do not need a workflow, so a detector
config can be sanity-checked without writing a script:

```bash
eregion validate path/to/detector.yaml   # load the config and print the detector it describes

# Override ${...} placeholders in the config, and allow ${VAR} to fall back to env vars
eregion validate path/to/detector.yaml --var width=2048 --env

eregion --version
```

Run `eregion --help` for the full list of commands and options. It does not execute workflows —
composing and running tasks is done in Python, as above.

```{toctree}
:hidden:
:maxdepth: 2

design/index
api/index
```
