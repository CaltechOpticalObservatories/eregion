from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Type, TypeVar, Generator, Iterable, Optional, Sequence


class TaskConcurrencyType(Enum):
    TRIVIAL = auto()
    """The task is intrinsically trivially parallelisable. Any number of this Task may be spawned
    and a whole dataset split between them, the result combined afterwards, without any penalty

    Examples include: anything which produces single lines of a table based on single or small sets of images e.g. PTC,
    trap pumping, EPER trail analysis, Fe-55 cluster analysis, actually most things we need
    """

    COMBINABLE_SHARED_STATE = auto()
    """The Task contains internal shared state which depends on previous data. This data can however
    be combined during execution, i.e. multiple Task instances could run and in the middle of the processing,
    the internal state could be combined between instances with only modest penalty.

    Examples include: producing a master bias image where the shared state can be combined without needing strict ordering,
    for example a Welford combination on master flat or bias which is done by mean-stacking

    """

    NO_SHARED_STATE = auto()
    """The Task cannot combine shared state sensibly. Only one instance of this Task may exist for a particular
    data run, hence drastically reducing concurrency the workflow engine can add

    Examples include: basically anything that makes a non-linear statistical combination through an image stack. For example: a (hypothetical) Welford     combination by median stacking
    """


class TaskDataDependencyType(Enum):
    SINGLE_INCREMENTAL_PROCESS = auto()
    """The Task can produce a sensible incremental output  when its internal coroutine is given a single instance of the input types
       Examples include: overscan subtraction, cosmic ray stripping, finding traps in an image, getting EPER trails out of a binned image
    """

    MULTI_INCREMENTAL_PROCESS = auto()
    """The Task requires multiple instances of the input types per meaningful incremental output update. Examples include: PTC (multiple images needed to make the diff pairs)"""

    NO_INCREMENTAL = auto()
    """The Task requires an entire input dataset (potentially many instances of the input type) before output can be produced
    Examples include: Full Astier fit of a PTC curve, non-linearity model fitting, CTI model fitting,  etc
    """


class ExecutionAdapter:
    """This class contains abstraction functions for anything the task needs to talk to the execution environment about.
    These should be relatively rare, for most tasks, the knowledge of the data dependency and concurrency allows the pipeline/ workflow engine
    to intelligently schedule and run incremental tasks. Keep everything else in here.
    """

@dataclass
class TaskCompletion:
    """This class (probably a dataclass actually), contains information about the task's execution, for example how many iterations were done, any warnings raised etc"""


_OutputType = TypeVar("output")
_InputType = TypeVar("input")
_InputCollectionType = Sequence[_InputType]
_OutputCollectionType = Sequence[_OutputType]

class Task:
    inputs: dict[str, Type]  = dict()
    """The input names and types that the task takes on each iteration. For example, a DetImage to build master bias"""

    output_types: dict[str, Type]  = dict()
    """The output names and types of an (incremental) run of the task. For things that generate tables, this would be the
    minimal unit of possible meaningful information, i.e. one row of that table. The final table is produced by combining these rows"""

    concurrency: TaskConcurrencyType = TaskConcurrencyType.NO_SHARED_STATE
    """The type of concurrency the Task can be run from the workflow engine / "pipeline". Assumed the most conservative value,
    even though this will mostly be overridden"""

    data_dependency: TaskDataDependencyType = TaskDataDependencyType.NO_INCREMENTAL
    """The data dependency type of the task. The DAG build can use this information to work out how to dynamically assign input data to potentially mmany instances of the Task as orchestrated byu the workflow engine / pipeline.  Assumed the most conservative value, even though it will mostly be overridden"""

    def __init__(self, exec_adapt: Optional[ExecutionAdapter] = None, **kwargs):
        """Initialize the Task.

        :param exec_adapt: Optional[ExecutionAdapter]
            Defaults to a plain no-op ExecutionAdapter() when omitted, so a Task can be constructed and run standalone
            (script/CLI/notebook/test) with no orchestrator wiring at all. Only pass a real adapter when something
            about the execution environment (a workflow engine, a cluster scheduler, ...) actually needs to be talked to.

        Set up logging, diagnostics, etc. Go through all the kwargs, which should be checked against the `inputs` attribute for both names and types
        All the info in kwargs gets captured in some metadata for traceability. The subclass _may_ setup any internal data structures needed for a run, but this may also be done in the main run coroutine and that may be much more readable and convenient
        """
        self.exec_adapt = exec_adapt or ExecutionAdapter()

    def incremental_generate(self) -> Generator[_InputType, _OutputType, TaskCompletion]:
        """This is where the main work happens. Should be written as a coroutine, which accepts the .send() method for sending in incremental inputs. It yields incremental outputs. In cases where multiple inputs are needed for a single output, the generator should NOT yield an output on every input. It will throw an error if a downstream consumer tries to yield from it in that case. Other machinery (below) will exist to check whether it's currently safe to yield from the incremental generator

        This allows to model maps, folds, filters, and simple input output tasks with one paradigm (I think). It's possible (and likely) we want subclasses of this to add relevant nice ergonomics to common cases (like maps).
        """

    @property
    def incremental_result_ready(self) -> bool:
        """Whether an incremental result is ready to be yielded from the current generator."""

    def run(self, inputs: _InputCollectionType) -> _OutputCollectionType:
        """a convenience function that lets one run the task on an entire collection of inputs and produce a collection of outputs.
        Default implementation would just repeatedly pass stuff into the internal coroutine and produce the entire output collection.
        For some cases, may be written such that the ExecutionAdapter can talk to the environment to enable some level of concurrency / parallelism
        """

    def collect_outputs(self, outputs: Iterable[_OutputType]) -> _OutputCollectionType:
        """any specific extra actions which need to be done by the Task itself to collect a bunch of incremental outputs into an output collection"""


# Worked example
# Sketch of how eregion.tasks.preprocessing.BiasSubtraction would look rewritten against this interface, as a sanity
# check that the design covers a real, already-shipped task. 
#
# BiasSubtraction is a good test case because it's the simplest kind of task: exactly one input produces exactly one output

class BiasSubtractionSketch(Task):
    inputs = {"image": "DetImage"}
    output_types = {"image": "DetImage"}

    concurrency = TaskConcurrencyType.TRIVIAL
    """Each image's bias subtraction is completely independent of every other image in the run. Any number of
    instances of this Task could be spawned and the input dataset split arbitrarily between them."""

    data_dependency = TaskDataDependencyType.SINGLE_INCREMENTAL_PROCESS
    """One image in, one bias-subtracted image out. No state needs to be carried between increments other than the
    master_bias set up at construction time, so a meaningful output can be produced from a single input."""

    def __init__(self, master_bias, exec_adapt: Optional[ExecutionAdapter] = None, **kwargs):
        super().__init__(exec_adapt=exec_adapt, **kwargs)
        self.master_bias = master_bias

    def incremental_generate(self):
        n_iterations = 0
        img = yield
        while img is not None:
            matching_bias = self.master_bias.filter(f'det_id == "{img.id}"')
            if not matching_bias:
                raise ValueError(f"No matching master bias found for DetImage with name '{img.id}'.")
            img.set_data(img.data - matching_bias[0].data)
            n_iterations += 1
            img = yield img
        return TaskCompletion(n_iterations=n_iterations)

    @property
    def incremental_result_ready(self) -> bool:
        # SINGLE_INCREMENTAL_PROCESS tasks always have a result ready after every .send(). There's never a partial
        # state waiting on more input, unlike a MULTI_INCREMENTAL_PROCESS task such as PTC.
        return True

# Standalone use needs nothing beyond the task's own constructor:
#   task = BiasSubtractionSketch(master_bias=master_bias_bundle)
#   corrected_images = task.run(science_images)
#
# A user who does want this inside their own Prefect flow supplies an ExecutionAdapter subclass and/or wraps the
# call site in @task/@flow themselves.
 
