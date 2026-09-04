# Design

Eregion: A flexible and modular detector characterization framework

## 1. Overview

### 1.1 Purpose

Characterizing the performance of imaging detectors is a critical component in
astronomical instrumentation. There are diverse preprocessing and analysis tasks
involved, such as measuring bias and dark signals, gain, charge-transfer efficiency, and
linearity, to name a few, and each task can have multiple techniques or algorithms for
execution. And though some core algorithms can be reused, there are many unique
analysis tasks needed for different detector types. However, existing characterization
pipelines are often rigid, project-specific, or difficult to adapt across facilities and detector
technologies. Data handling, processing logic, and analysis are frequently mixed together,
which makes the system harder to maintain. Intermediate results are rarely reusable, and
running the same workflow on a different detector can require substantial rewriting.

Eregion addresses these issues by clearly separating data representation, processing
logic, and workflow orchestration. The framework provides a modular system for detector
characterization, a configuration-driven workflow engine, and reusable building blocks
(core algorithms) for processing and analysis. This makes it easier to develop new
procedures, reuse existing methods across detectors, run both automated pipelines and
interactive analyses, and generate standardizable reports. This framework is designed to
facilitate collaborative development of characterization procedures across teams and
facilities.

### 1.2 Scope

Eregion supports processing detector data from common formats such as FITS files,
NumPy arrays, or arrays from memory buffers. The framework is designed to support
multiple detector types, including CCD, CMOS-APS, HgCdTe hybrid arrays, and APD
arrays. Detector images are loaded through customizable config-driven data models (and
associated loading task) to support detector specific properties (for example, amplifier
outputs and overscan regions for CCD), focal plane position of the detector, etc.

Basic processing and common analysis tasks are included that inherit from a base `Task`
class. A prescription to write and add custom tasks is provided.

Algorithmic processing and analysis functions are kept in a 'core' library for easy import.

Typical use cases include laboratory detector testing, instrument development, and
automated calibration workflows. The design is intentionally detector-agnostic so that the
same framework can be reused across different systems.

## 2. High-Level Architecture

The framework is organized into four main components:

- **Data Layer** — Handles input data and detector-specific structure. Converts raw data
  into standardized objects.
- **Task Layer** — Modular processing and analysis units. Each task performs a single
  well-defined operation.
- **Core Library** — Standalone library of algorithmic functions and image operations.
- **Pipeline Engine** — Builds workflows from configuration files. Manages execution
  order with dependency tracking. Handles errors and supports parallel execution.

This separation allows each part of the system to evolve independently. Tasks do not need
to know how workflows are executed, and the pipeline engine does not need to know the
details of detector structure.

### Design Principles

- **Modular** — Tasks are defined with clear dependencies and I/O to function as building
  blocks in a pipeline. Plug-ins provided for users to add custom tasks.
- **Configurable** — Workflows and detector structures can be updated flexibly in YAML
  configs.
- **Reusable** — Tasks can run standalone or inside pipelines.
- **Extensible** — New tasks and detectors can be added without modifying core code.

## 3. Config-Driven Data Model

### 3.1 Inputs

Eregion supports multiple input formats, including FITS images, NumPy arrays, and xarray
DataArrays. Rather than hard-coding detector assumptions, the structure of the input
image is defined through a YAML configuration. It instructs how to initialize Eregion's image
data model (`DetImage`) for the given input format.

### 3.2 Intermediate Data: DetImage and Output

The framework uses two main abstractions to represent data during processing. A
`DetImage` represents the full detector image and contains the pixel data and associated
metadata. It also manages a collection of `Output` objects, each corresponding to a
sub-region of the detector, such as an amplifier.

An `Output` provides access to specific regions of the detector, including active pixels and
overscan regions. This allows operations that depend on detector layout, such as overscan
subtraction, to be performed correctly at the appropriate level.

- The `DetImage` instance is designed to contain the outputs of a **single** detector
  (not mosaic). So, the config structure should describe what is contained in each
  FITS file.
- The flexibility provided by YAML allows instantiating image(s):
  - From one FITS file with one extension that has all the detectors/outputs
  - OR from multiple FITS extensions, one per detector/output
  - OR from multiple FITS files, one per detector/output.
- Users should specify which type of `Output` class to use (based on their detector
  type and available options). Users can create new classes for new types of
  detectors that inherit from `Output`.

The example below (Fig. 1) describes a single CCD detector with two outputs (channels)
that are read out from the same FITS extension. The data for each output is defined by a
slice of the FITS extension (`ext_slice`), and the location of that data in the full
`DetImage` data array is also defined by a slice (`data_slice`).

This approach allows the same processing code to work across different detectors without
modification. Only the configuration needs to be changed.

```{code-block} yaml
:caption: "Fig. 1: Example of a detector configuration"

---
description: Basic single CCD detector, one channel
detector_type: CCD # (specify the type of detector, e.g., CCD, CMOS, H2RG, etc.)
detector_output_class: CCDOutput # (use the correct class here for the detector output type)

# List of Output class objects, each containing a list of outputs.
# An output is described by the FITS file it is in, the extension in that file, and the slices that define its data.
objects:
  - name: 'det_1'
    class: DetImage
    filename_format: '*.fits*'  # Use wildcard to pattern match filename if needed
    properties:
      x_size: 2048
      y_size: 4096
      saturation_level: 65535  # ADU
      pixel_size: 0.015  # mm
    outputs:
      - id: 'chan_1'
        ext_id: 1  # FITS extension ID
        ext_slice: [!slice [0, 4096], !slice [0, 1024]]   # Slice of the ext_id that has the data for this output (left half)
        data_slice: [!slice [0, 4096], !slice [0, 1024]]  # Slice of the full DetImage data array where this output's data will go (left half)
        serial_prescan: !slice [0, 20]        # 20 prescan columns w.r.t. the ext_slice
        serial_overscan: !slice [1004, 1024]  # 20 overscan columns w.r.t. the ext_slice
        parallel_prescan: !slice [0, 20]      # 20 prescan rows w.r.t. the ext_slice
        parallel_overscan: !slice [4076, 4096] # 20 overscan rows w.r.t. the ext_slice
        parallel_axis: 'y'      # First axis in the data array (rows) represent parallel readout direction
        readout_pixel: [0, 0]   # Readout of this amplifier (top left)
        gain: 1.0                # electrons/ADU
        read_noise: 5.0          # electrons
        bias_level: 1000         # ADU
      - id: 'chan_2'
        ext_id: 1
        ext_slice: [!slice [0, 4096], !slice [1024, 2048]]   # (right half)
        data_slice: [!slice [0, 4096], !slice [1024, 2048]]  # (right half)
        serial_prescan: !slice [1024, 1004]  # 20 prescan columns w.r.t. the ext_slice (on the right since readout is from the right)
        serial_overscan: !slice [20, 0]      # 20 overscan columns w.r.t. the ext_slice (on the right since readout is from the right)
        parallel_prescan: !slice [0, 20]       # 20 prescan rows w.r.t. the ext_slice
        parallel_overscan: !slice [4076, 4096] # 20 overscan rows w.r.t. the ext_slice
        parallel_axis: 'y'
        readout_pixel: [0, 2047]  # Readout of this amplifier (top right)
        gain: 1.0                 # electrons/ADU
        read_noise: 5.0           # electrons
        bias_level: 1000          # ADU
    focal_plane_position:
      x_cen: 0.0    # mm
      y_cen: 0.0    # mm
      angle: 0.0    # degrees
```

### 3.3 Output Products

Processing steps (Tasks, see {ref}`task-and-lazytask`) produce new data products that can be
passed downstream or analyzed independently. These can include updated `DetImage`
objects, enriched metadata, and derived quantities such as gain or bias level. The
processing step results are required to be packaged as a dictionary with keys representing
the result type/name. This standardization allows easy access across different parts of a
pipeline, even across different workflows.

## 4. Modular Processing Nodes with Defined I/O

(task-and-lazytask)=
### 4.1 Task and LazyTask

Processing in Eregion is built around tasks. A `Task` is the basic unit of computation and
provides a `run()` method for execution. Tasks can also be called directly like functions,
making them easy to use outside of pipelines.

For streaming scenarios, the framework provides a `LazyTask` abstraction. Instead of
processing all data at once, a lazy task yields results incrementally as images become
available. This allows pipelines to operate on data streams or datasets that do not fit
entirely in memory.

### 4.2 Task Design Pattern

Each task follows a simple and consistent structure. It takes initialization `args` (required)
and `kwargs` (optional), applies processing logic by calling core algorithm functions in the
`run()` method for `DetImage` objects, or in the `__call__()` method for NumPy arrays,
and returns a dictionary containing the results. This pattern keeps tasks small, predictable,
and easy to compose.

For example, a bias subtraction task takes a master bias as initial input, performs
subtraction per input image during a run, and returns a list of corrected images. For tasks
that apply the same processing to a list of inputs, parallelization is handled internally using
the `joblib` module.

Orchestration of tasks this way also allows supporting different methods/algorithms of
performing the task (e.g., bias or gain estimation) with the same input under the same task
call.

### 4.3 Core Algorithms as a Library

The actual processing logic is implemented as standalone functions that are independent
of the task framework. Examples include subtraction of arrays, image combination,
sigma-clipping, etc. By separating algorithms from task orchestration, the algorithms become
available as a standalone library for direct importation.

### 4.4 Example Tasks

Common tasks in the framework include image loading, master bias creation, bias
subtraction, overscan subtraction, bad pixel masking, etc. The image loading task
(`ImageCreator`) is a primary part of the framework and handles correct instantiation of
`DetImage` objects given a list of inputs and a detector configuration.

Each task has clearly defined inputs and outputs and can be executed either
independently or as part of a pipeline. This flexibility allows users to mix interactive
analysis with automated workflows.

Example flow:

- `ImageCreator` → produces `DetImage`s from input
- `MasterBias` → consumes multiple 'bias' `DetImage`s, produces master bias
- `BiasSubtraction` → consumes master bias and non-bias `DetImage`s and produces
  corrected `DetImage`s
- ... (further tasks on results of bias-subtracted `DetImage`s)

Such flows can automatically be orchestrated from pipeline YAML configuration as
described in the next section.

## 5. Config-Driven Pipeline Engine

### 5.1 YAML-Based Workflows

Pipelines in Eregion are also defined using YAML configuration files (see Fig. 2). Each task
entry specifies the task class, initialization parameters, runtime inputs, and dependencies.
This declarative approach makes workflows easy to read, modify, and reproduce.

### 5.2 DAG Creation and Execution

From the YAML configuration, the pipeline engine constructs a Directed Acyclic Graph
(DAG) per pipeline where nodes correspond to tasks and edges represent dependencies.
Tasks are loaded dynamically, and execution proceeds in dependency order. Another DAG
is constructed at the pipeline level to run all defined pipelines in the dependency order.

The engine is responsible for resolving inputs between tasks, executing them in the correct
sequence, and storing results. We use the [Prefect](https://www.prefect.io/) package to
orchestrate the workflow dynamically. The nodes/tasks are wrapped in Prefect "tasks" and
the pipelines in "flows" for execution. Prefect handles retries, and concurrent execution of
the same generation of tasks. This allows users to focus on defining workflows rather than
managing execution details.

### 5.3 Execution Modes

Eregion supports both eager and lazy execution at the pipeline level. For eager pipelines, all
data is processed at once, and complete results are returned. This is useful for small
datasets and debugging. For lazy pipelines, data is processed incrementally, and results
are yielded as they are produced, which is more suitable for large datasets or streaming
applications. A mix of lazy and eager pipelines can be defined in the YAML.

### 5.4 TaskResult Class

The dictionary results from tasks are wrapped in a `TaskResult` object that encapsulates
both the output data and its provenance. It includes the data itself, the parameters used
for execution, references to upstream dependencies, and a timestamp.

This structure ensures that every output can be traced back to its origin, including the
exact inputs, the sequence of tasks applied, and the parameters used at each step. This
level of traceability is critical for scientific workflows.

```{code-block} yaml
:caption: "Fig. 2: Example pipeline YAML structure"

debug: false  # Optional: set to true to enable debug mode (more verbose logging, etc.)

pipelines:
  - name: PIPE_1  # Name of the pipeline flow, required
    description: Pipeline flow 1
    lazy: false  # Set true if this sub-pipeline should be run lazily (i.e. as images arrive)

    nodes:  # List of tasks (nodes) in the pipeline flow
      - name: TASK_1  # Name of the task node, required
        task: package.module.class  # Path to the Class of the task to run, must be a subclass of `Task` defined in tasks.task
        init:  # Initialization parameters (for Task.__init__)
          inputs:  # Specify any args needed from outputs of other tasks in this config
            arg_1: pipe_name.node_name.data.key  # Output of tasks are wrapped in TaskResult objects by the engine, and the data
                                                  # produced by the task is in the TaskResult.data dict; specify the path to the
                                                  # data you want to use as input for this task
            # etc.
          params:  # Specify any additional kwargs (which are not task outputs) needed; refer to the task documentation for required and optional params and kwargs
            param_1: value
            param_2: value
            # etc.
        run:  # Run-time (Task.run() or Task.lazy_run()) inputs and parameters, as above; use `inputs` for data coming from
              # outputs of other tasks, and `params` for any additional parameters
          inputs:
            arg_1: pipe_name.node_name.data.key
            # etc.
          params:
            param_1: value
            param_2: value
            # etc.

      - name: TASK_2
        task: package.module.class
        init:
          inputs:
            arg_1: PIPE_1.TASK_1.data.key  # Example of using output from TASK_1 as input for TASK_2
          params:
            param_1: value
            # etc.
        run:
          inputs:
            arg_1: PIPE_1.TASK_1.data.key  # Example of using output from TASK_1 as input for TASK_2
          params:
            param_1: value
            # etc.
        depends_on: [TASK_1]  # should be specified if this task depends on the output of another task; ensures correct execution order in the pipeline flow

  - name: PIPE_2
    description: Pipeline flow 2
    lazy: true  # This sub-pipeline will be run lazily (i.e. as images arrive)
    nodes:
      - name: TASK_3
        task: package.module.class
        init:
          inputs:
            arg_1: PIPE_1.TASK_1.data.key  # Example of using output from a task in another pipeline flow as input
          params:
            param_1: value
            # etc.
        run:
          inputs:
            arg_1: PIPE_1.TASK_2.data.key  # Example of using output from a task in another pipeline flow as input
          params:
            param_1: value
            # etc.
        depends_on: [PIPE_1.TASK_1, PIPE_1.TASK_2]  # specify dependencies across pipeline flows as well
```

### 5.5 Pipeline Example

A simple pipeline (Fig. 3) might consist of loading images, creating a master bias,
subtracting the bias from the images, and computing summary statistics. Each step
consumes the outputs of previous steps and produces structured results that can be
further processed or analyzed.

```{code-block} yaml
:caption: "Fig. 3: Example of a simple pipeline for basic calibration of images"

debug: false
pipelines:
  - name: calib_flow
    description: Pipeline flow to create a master bias frame from bias images
    lazy: false
    nodes:
      - name: image_creator
        task: tasks.imagegen.ImageCreator
        init:
          params:
            detector_config: "/path/to/eregion/configs/detectors/deimos_singledet.yaml"
        run:
          params:
            input_source: "/path/to/data/PTC/SCI/20250812-101359/*_bias_*.fits"
            identifier_func: tasks.custom.guess_image_type_from_filename_DEIMOS

      - name: master_bias
        task: tasks.calibration.MasterBias
        init:
          params:
            method: "median"
        run:
          inputs:
            bias_images: calib_flow.image_creator.data.bias
        depends_on: [calib_flow.image_creator]

  - name: preproc_flow
    description: Example pre-processing pipeline flow
    lazy: false
    nodes:
      - name: image_creator
        task: tasks.imagegen.ImageCreator
        init:
          params:
            detector_config: "/path/to/eregion/configs/detectors/deimos_singledet.yaml"
        run:
          params:
            input_source: "/path/to/data/PTC/SCI/20250812-101359/*flat_0.000*.fits"
            identifier_func: tasks.custom.guess_image_type_from_filename_DEIMOS

      - name: bias_subtraction
        task: tasks.preprocessing.BiasSubtraction
        init:
          inputs:
            master_biases: calib_flow.master_bias.data.master_biases
        run:
          inputs:
            images: preproc_flow.image_creator.data.flat
        depends_on: [preproc_flow.image_creator, calib_flow.master_bias]
```

## 6. Standardized Characterization Reports

Eregion tasks will support the generation of standardized outputs such as summary
statistics, calibration products, and diagnostic plots like photon transfer curves.
Functionality will be created to collate the task results and automatically populate a
standard report format, which can be used standalone or integrated with the pipeline
engine.

## 7. Provenance and Quality Assurance

Provenance is a core feature of the engine framework. Every result includes information
about the task that produced it, the inputs it depended on, and the parameters used. This
makes it possible to fully reconstruct how any output was generated.

Quality assurance is supported through metadata flags, validation checks within tasks,
and reproducible pipeline definitions. Together, these features ensure that results are both
reliable and auditable.

## 8. Error Handling

Error handling is managed by the orchestration layer, which detects task-level failures, and
provides logging for debugging. The pipeline engine supports retry mechanisms. Errors
propagate clearly through the pipeline and are associated with specific tasks, making it
easier to identify and fix issues.

## 9. Performance

Performance is addressed at both the pipeline and task levels. Independent tasks can be
executed in parallel, while individual tasks can use optimized numerical operations or
parallel processing libraries (`joblib`).

## 10. Design Tradeoffs

The framework makes several deliberate tradeoffs. Using YAML for pipeline definition
improves reproducibility and clarity, but a Python API would offer more flexibility. Similarly,
keeping tasks small improves reusability, but excessively fine granularity can introduce
overhead. Lazy execution reduces memory usage but can make debugging more complex.

These tradeoffs are chosen to favor clarity, reproducibility, and modularity, which are
critical for scientific workflows.

## 11. Summary

Eregion is a modular and extensible framework for detector characterization that
emphasizes clean separation of concerns, configuration-driven workflows, and strong
provenance tracking. By standardizing how data is represented, processed, and passed
through pipelines, it simplifies the development of characterization procedures and
enables reuse across detectors and projects.

The framework supports both interactive and automated use cases and is designed to
scale from small experiments to large calibration workflows while maintaining
reproducibility and clarity.
