EIP-1 redesign the Task
-----------------------

## Goals

	- decouple Task from any kind of concurrency, parallelism, or workflow engine hooks
	- allow for flexible Task execution without horrific ergonomics
	- allow for more optimisations in terms of memory & IO
	- allow for more easy collection of metadata and configuration
	- Task must be runnable standalone (plain script / CLI / notebook / test) with zero required setup. This means no
	  ExecutionAdapter, no engine, no config beyond the Task's own constructor args
	- a user must be able to drop a Task into their own Prefect (or other) pipeline without the Task needing any awareness that it's happening

## Non-goals / open questions for this EIP

	- This document only covers the `Task` interface itself. Need to expand on replacement of
	  `PipelineEngine.execute_pipeline`'s Prefect-based generation-by-generation scheduling
	  (`eregion/pipeline/engine.py`, `make_prefect_task`/`make_prefect_flow`) This is "remove Prefect"
	  work and needs its own design pass once Task's shape here is settled. `TaskConcurrencyType` and
	  `TaskDataDependencyType` are meant to be the input to that future scheduler
	- Not yet addressed: how `ExecutionAdapter` subclasses would actually get wired up by an external engine (e.g.
	  would Prefect's `@task`/`@flow` decorators wrap `Task.run`/`incremental_generate` directly, or go through the
	  adapter?).
	- Not yet addressed: migration path for existing concrete tasks (`BiasSubtraction`, `ScanSubtraction`,
	  `SigmaClipMasking`, the PTC tasks) off of today's `Task`/`LazyTask`/`run`/`lazy_run`/`TaskResult.combine`
	  interface and onto `incremental_generate`/`TaskCompletion`. 
	


