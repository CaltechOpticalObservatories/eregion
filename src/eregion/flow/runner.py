"""
Execution of the flows described in a yaml configuration.

`FlowRunner` builds the graph (see `nodes.py`), checks it can run (`dry_run`), and then
runs every flow in dependency order, collecting each task's `TaskResult`.
"""
from collections.abc import Iterator
from copy import deepcopy
import inspect

from eregion.configs import FlowConfig
from eregion.utils import configure_logger

from .nodes import Flow, FlowNode, build_flows
from .refs import resolve_inputs

logger = configure_logger(__name__)


class FlowRunner:
    """
    Runner to construct and manage execution of flows described in a yaml configuration.

    The runner reads a flow configuration (path or dict), builds the task nodes and the
    flows that group them, works out the order both must run in, and executes them,
    collecting the results.

    Parameters
    ----------
    config_input : str | dict
        Path to a configuration file (e.g. YAML/JSON) or a dictionary containing flow
        definitions that will be parsed by `FlowConfig`.
    runtime_variables : dict, optional
        Dictionary of runtime variables that can be used to resolve placeholders in the configuration.
    enable_env_vars : bool, optional
        If True, environment variables can be used to resolve placeholders in the configuration.
    dry_run : bool, optional
        If True (the default), `dry_run()` validates the wiring between tasks while the runner is
        built, so configuration mistakes are raised before any task is executed.

    Attributes
    ----------
    flow_config : FlowConfig
        Parsed and validated flow configuration object.
    flows : dict
        Mapping of flow name -> `Flow`, each holding its own nodes and their order.
    flow_order : tuple
        Generations of flow names, in the order the flows must run.
    nodes : dict
        Mapping of fully qualified node name -> `FlowNode`, across every flow. Each node
        holds an already-initialised task, built once here and reused for every execution. In a
        lazy flow that means one instance serves every batch, so a task must derive its
        per-run state from its run arguments; anything cached on `self` persists across batches.
    results : dict
        Stores TaskResult objects keyed by node/task names produced during runs.
    dummy_results : dict
        Dummy TaskResult objects produced by the dry run, keyed by node/task names; empty when the
        dry run is disabled. Useful to inspect which payload fields each node will produce.
    """

    def __init__(self, config_input: str | dict, runtime_variables: dict | None = None,
                 enable_env_vars: bool = False, dry_run: bool = True):

        self.runtime_variables = dict(runtime_variables or {})
        self.enable_env_vars = enable_env_vars
        self.flow_config = FlowConfig(
            config_input,
            runtime_variables=self.runtime_variables,
            enable_env_vars=self.enable_env_vars,
        )
        config = self.flow_config.config
        logger.info("Number of flows defined: {}".format(len(config["flows"])))

        self.flows, self.flow_order = build_flows(config)
        self.nodes = {name: node for flow in self.flows.values()
                      for name, node in flow.nodes.items()}
        self.results = {}

        # Validate the wiring between tasks up front, using dummy results in place of real ones
        self.dry_run_enabled = dry_run
        self.dummy_results = self.dry_run() if dry_run else {}

    def ordered_flows(self):
        """
        Iterate the flows in the order they must run.
        :return: Iterator[Flow]
        """
        for generation in self.flow_order:
            for flow_name in generation:
                yield self.flows[flow_name]

    ## Execute the flows
    def run(self):
        """
        Execute every flow in dependency order, one at a time.

        Flows within a generation are independent and could in principle run together, but
        they all write to the single `self.results` mapping, so they are executed sequentially.
        :return: None, self.results holds the results of every executed task.
        """
        for flow in self.ordered_flows():
            try:
                if flow.lazy:
                    # Drive the lazy flow to completion by consuming its generator
                    for _ in self.execute_lazy_flow(flow):
                        pass
                else:
                    self.execute_eager_flow(flow)
            except Exception:
                logger.exception(f"Flow '{flow.name}' failed")
                raise
            logger.info(f"Flow '{flow.name}' complete")

    def execute_eager_flow(self, flow: Flow) -> dict:
        """
        Execute every node of an eager flow once, in dependency order.
        :param flow: Flow
            The flow to execute.
        :return: dict, self.results, updated in place with this flow's results.
        """
        for node in flow.ordered_nodes():
            self.results[node.name] = execute_task(node, self.results)
        return self.results

    def execute_lazy_flow(self, flow: Flow) -> Iterator[dict]:
        """
        Execute a lazy flow, one batch at a time.

        The flow's source node yields batches from its `lazy_run()`. Each batch is fed to the
        rest of the flow in a temporary results mapping, and what comes out is combined into
        `self.results`, so that each node's entry accumulates across batches.

        Every node runs its single task instance, built when the runner was, so that instance is
        shared by every batch. A task must therefore derive its per-run state from its run
        arguments; anything it caches on `self` is carried over into the batches that follow.
        :param flow: Flow
            The lazy flow to execute.
        :return: Generator that yields self.results after each batch.
        """
        source = flow.nodes[flow.source]

        # A temporary mapping keeps each batch's results apart from the accumulated ones
        batch_results = deepcopy(self.results)
        batches = execute_task(source, batch_results, lazy=True)
        if not isinstance(batches, Iterator):
            raise TypeError(f"Source node '{source.name}' for lazy flow '{flow.name}' did not "
                            f"return an iterator")

        for batch in batches:
            batch_results[source.name] = batch.model_copy(
                update={"upstream": source.upstream, "params": source.params})
            for node in flow.ordered_nodes(skip_source=True):
                batch_results[node.name] = execute_task(node, batch_results)

            self._accumulate(flow, batch_results)
            yield self.results

    def _accumulate(self, flow: Flow, batch_results: dict):
        """
        Fold one batch's results into the accumulated results of a lazy flow.
        :param flow: Flow
            The flow being executed, whose nodes are the ones to fold in.
        :param batch_results: dict
            The results produced for the batch just finished.
        :return: None, self.results is updated in place.
        """
        for node_name in flow.nodes:
            batch_result = batch_results[node_name]
            accumulated = self.results.get(node_name)
            self.results[node_name] = (batch_result if accumulated is None
                                       else accumulated.combine(batch_result))

    ## Validate the flows without executing them
    def dry_run(self) -> dict:
        """
        Walk the whole DAG in execution order without executing any task.

        Each node is checked with `dry_run_task` and contributes a dummy TaskResult, which the
        nodes downstream of it resolve their inputs against. Wiring mistakes (an unknown node or
        payload field, an argument a task does not take, a missing required argument) therefore
        surface when the runner is built rather than part-way through a long run. Payload values
        are placeholders, so their types are not checked.
        :return: dict
            Mapping of fully qualified node name -> dummy TaskResult that the node would produce.
        """
        dummy_results = {}
        for flow in self.ordered_flows():
            for node in flow.ordered_nodes():
                dummy_results[node.name] = dry_run_task(node, dummy_results,
                                                        lazy=node.name == flow.source)
        logger.info(f"Dry run passed for {len(dummy_results)} nodes")
        return dummy_results


def execute_task(node: FlowNode, results: dict, lazy: bool = False):
    """
    Execute a single node's task.
     - Resolve the node's inputs against the results produced so far; an input can be a single
       reference or a list/dict of references, in which case the task receives a list/dict of
       resolved values
     - Call the task's run or lazy_run method with the resolved inputs and parameters
     - Return the task's own TaskResult (subclass) with upstream dependencies and params attached
     - If lazy is True, call the task's lazy_run method and return the raw generator of TaskResult
       batches, unmodified. The caller is responsible for iterating through the results, attaching
       upstream/params metadata to each yielded batch, and feeding them downstream.
    :param node: FlowNode
        The node to execute, holding its initialised task, inputs, params and dependencies.
    :param results: dict
        Results produced so far, keyed by fully qualified node name, used to resolve the inputs.
    :param lazy: bool
        Whether to execute the task in lazy mode. If True, the task must have a lazy_run method
        that returns an iterator.
    :return: TaskResult | Generator[TaskResult, None, None]
        The task's own TaskResult with 'upstream'/'params' set, or (if lazy) a generator of
        TaskResults still missing 'upstream'/'params', which the caller must attach.
    """
    run_inputs = resolve_inputs(node.run_inputs, results, node.name)
    task = node.task

    if lazy:
        if not hasattr(task, "lazy_run"):
            raise ValueError(f"Task '{task.name}' does not have a 'lazy_run' method for lazy execution")
        return task.lazy_run(**run_inputs, **node.run_params)

    result = task.run(**run_inputs, **node.run_params)
    return result.model_copy(update={"upstream": node.upstream, "params": node.params})


def dry_run_task(node: FlowNode, dummy_results: dict, lazy: bool = False):
    """
    Check that a single node can be called, without executing it.
     - Resolve its inputs against the dummy results of its upstream nodes, which checks that every
       referenced node and payload field exists
     - Bind the resolved inputs and the run params to the signature of run()/lazy_run(), which
       checks for missing, unexpected or duplicated arguments
    The task itself is built when the node is, so a bad 'init.params' block is already reported
    by `build_node` and does not need checking here.
    :param node: FlowNode
        The node to check, holding its initialised task, inputs, params and dependencies.
    :param dummy_results: dict
        Dummy results of the nodes checked so far, keyed by fully qualified node name.
    :param lazy: bool
        Whether the node is the source of a lazy flow, and is therefore called via lazy_run().
    :return: TaskResult
        A dummy instance of the task's own result class, standing in for what it would produce.
    """
    # Dummy payloads carry no data to filter on, so filter calls are skipped rather than applied
    run_inputs = resolve_inputs(node.run_inputs, dummy_results, node.name, apply_queries=False)

    task = node.task
    task_name = type(task).__name__
    run_method = "lazy_run" if lazy else "run"
    if lazy and not hasattr(task, run_method):
        raise ValueError(f"Task '{node.name}' does not have a 'lazy_run' method for lazy execution")
    try:
        inspect.signature(getattr(task, run_method)).bind(**run_inputs, **node.run_params)
    except TypeError as e:
        raise TypeError(f"Node '{node.name}': the configured inputs {sorted(run_inputs)} and params "
                        f"{sorted(node.run_params)} do not match "
                        f"{task_name}.{run_method}(): {e}") from e

    return task.dummy_run()
