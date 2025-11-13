# Detector Characterization Framework

A modular Python framework for processing and analysis of imaging detector data (CCD/CMOS).  
Supports direct task usage in scripts and orchestration in config‑driven DAG workflows.

---

## Key Features

- Modular tasks — preprocessing, calibration, analysis, and image generation.
- Dual usage — call tasks directly in scripts or chain them in a pipeline.
- Reusable image ops — shared combine/stack utilities in `utils`.
- Config‑driven orchestration — Prefect‑ready flows kept in `pipeline`.
- Optional lazy execution — support for generator‑style tasks.

---

## Project Layout

```text
eregion/
├── data/                     # sample/raw data for testing
│   └── deimos_raw/
│       ├── *.fits
├── datamodels/
│   ├── detector_config.py    # For loading and parsing detector configuration files
│   └── image.py              # Image data model (DetImage)
├── pipeline/                 # Prefect flows / runners (orchestration layer)
├── playground/
│   ├── basic_ccd.yaml        # example config(s)
│   ├── deimos.yaml           # DEIMOS example config
│   └── test.ipynb            # notebook for quick experiments
├── tasks/                    # modular processing/analysis tasks
│   ├── analysis.py           # analysis tasks (e.g., ptc, linearity)
│   ├── calibration.py        # calibration tasks (e.g., masterbias, masterflat)
│   ├── imagegen.py           # For generating DetImage instances from detector config and FITS files
│   ├── preprocessing.py      # preprocessing tasks (e.g., overscan trim, bias subtract)
│   └── task.py               # base Task and LazyTask abstract classes
├── tests/                    # unit tests
├── utils/                    # reusable utility functions
│   └── image_operations.py   # image combine/stack ops
└── README.md
