from typing import Iterator
from copy import deepcopy
import graphlib
import inspect
import re

from eregion.configs import PipelineConfig
from eregion.utils import configure_logger, load_class

import concurrent.futures as cf

logger = configure_logger(__name__)

class PipelineEngine:
    """
    Engine to construct and manage execution of pipelines described in a yaml configuration.

    The engine reads a pipeline configuration (path or dict), builds task node objects for each
    pipeline, computes dependency-based execution orders (both pipeline-level and per-pipeline tasks),
    and saves execution results.

    Parameters
    ----------
    pipeline_config_input : str | dict
        Path to a configuration file (e.g. YAML/JSON) or a dictionary containing pipeline
        definitions that will be parsed by `PipelineConfig`.
    runtime_variables : dict, optional
        Dictionary of runtime variables that can be used to resolve placeholders in the configuration.
    enable_env_vars : bool, optional
        If True, environment variables can be used to resolve placeholders in the configuration.
    dry_run : bool, optional
        If True (the default), `dry_run()` validates the wiring between tasks while the engine is
        built, so configuration mistakes are raised before any task is executed.

    Attributes
    ----------
    pipeline_config : PipelineConfig
        Parsed and validated pipeline configuration object.
    debug : bool
        Flag controlling verbose logging; taken from configuration if present.
    pipelines : dict
        Mapping of pipeline name -> pipeline configuration augmented with `nodes_dict` that
        maps task nodes to their runtime objects and metadata.
    execution_orders : tuple
        Tuple of (pipeline_order, node_orders) describing topological execution order:
        - pipeline_order: ordered list/generations of pipeline names based on inter-pipeline deps
        - node_orders: mapping pipeline_name -> ordered generations of node names
    results : dict
        Stores TaskResult objects keyed by node/task names produced during runs.
    dummy_results : dict
        Dummy TaskResult objects produced by the dry run, keyed by node/task names; empty when the
        dry run is disabled. Useful to inspect which payload fields each node will produce.
    """

    def __init__(self, pipeline_config_input: str | dict, runtime_variables: dict | None = None,
                 enable_env_vars: bool = False, dry_run: bool = True):

        self.runtime_variables = dict(runtime_variables or {})
        self.enable_env_vars = enable_env_vars
        self.pipeline_config = PipelineConfig(
            pipeline_config_input,
            runtime_variables=self.runtime_variables,
            enable_env_vars=self.enable_env_vars,
        )
        self.debug = self.pipeline_config.config.get("debug", False)
        logger.info("Number of pipelines defined: {}".format(len(self.pipeline_config.config["pipelines"])))

        # Build the pipeline nodes and note their dependencies
        all_node_dependencies = {}
        self.pipelines = {}
        for pipeline_cfg in self.pipeline_config.config["pipelines"]:
            nodes, node_dependencies = self.build_pipeline_nodes(pipeline_cfg)
            pipeline_cfg["nodes_dict"] = nodes
            self.pipelines[pipeline_cfg["name"]] = pipeline_cfg
            all_node_dependencies.update(node_dependencies)

        self.execution_orders = self.build_execution_order(all_node_dependencies)
        self.results = {}

        # Validate the wiring between tasks up front, using dummy results in place of real ones
        self.dry_run_enabled = dry_run
        self.dummy_results = self.dry_run() if dry_run else {}

    def update(self, new_config: str | dict, runtime_variables: dict | None = None, enable_env_vars: bool | None = None):
        """
            Update the pipeline configuration and rebuild the pipelines and execution orders.
        :param new_config: str or dict
            New pipeline configuration as a path to a YAML/JSON file or a dictionary.
        :param runtime_variables: dict, optional
            New runtime variables to use for resolving placeholders in the configuration.
        :param enable_env_vars: bool, optional
            Whether to enable environment variable interpolation in the new configuration.
        """
        self.__init__(
            pipeline_config_input=new_config,
            runtime_variables=self.runtime_variables if runtime_variables is None else runtime_variables,
            enable_env_vars=self.enable_env_vars if enable_env_vars is None else enable_env_vars,
            dry_run=self.dry_run_enabled,
        )

    @staticmethod
    def build_pipeline_nodes(pipeline_cfg: dict):
        """
        Build the task nodes for a single pipeline based on its configuration.
         - For each node, load the corresponding task class
         - Extract the run inputs (that are outputs of other nodes) and the init/run params from the config
         - Track the dependencies of each node based on the inputs and explicit depends_on field
        :param pipeline_cfg: dict
            Configuration dictionary for a single pipeline, containing at least the keys "name", "lazy", and "nodes".
            Each node should have at least "name" and "task", and can optionally have "init", "run", and "depends_on"
            fields. Only the "run" block may declare "inputs"; outputs of other tasks are never passed to
            `Task.__init__`, so an "init" block only takes "params".
        :return: tuple
            A tuple containing:
            - nodes: dict mapping node names to their corresponding task instances, inputs, parameters
                and upstream dependencies.
            - node_dependencies: dict mapping node names to a set of their upstream dependencies
        """
        node_dependencies = {}
        nodes = {}
        pipe_name = pipeline_cfg["name"]
        for node in pipeline_cfg["nodes"]:
            # get task instance
            eregion_task = load_class(node["task"])
            # get init params
            init_block = node.get("init", {})
            if "inputs" in init_block:
                raise ValueError(f"Node '{node['name']}' declares 'init.inputs'; outputs of other tasks cannot be "
                                 f"passed to Task.__init__. Move them to the node's 'run.inputs' block, 'init' only "
                                 f"takes 'params'.")
            init_params = init_block.get("params", {})
            # get run params and inputs, qualify the inputs and record upstream dependencies
            run_block = node.get("run", {})
            run_params = run_block.get("params", {})
            run_inputs = run_block.get("inputs", {})
            upstream = node.get("depends_on", [])
            ## Ensure that node names in upstream list are named fully with pipeline name
            upstream = [f"{pipe_name}.{dep}" for dep in upstream if '.' not in dep]
            ## sanity check: if run_inputs is not empty, the corresponding dependencies must be listed in depends_on
            ## If not, log a warning and add them to the dependencies list
            node_dependencies[f"{pipeline_cfg['name']}.{node['name']}"] = set(upstream)
            node_dict = {"task": eregion_task, "init_params": init_params,
                         "run_inputs": run_inputs, "run_params": run_params, "upstream": upstream,
                         "params": {'init': init_params, 'run': run_params | run_inputs}}
            nodes[f"{pipeline_cfg['name']}.{node['name']}"] = node_dict

        return nodes, node_dependencies

    @staticmethod
    def qualify_inputs(node: dict, pipeline_name: str) -> dict:
        """
        Qualify the references of a node's inputs and record the nodes they point to as dependencies.

        References written without a pipeline name are assumed to belong to `pipeline_name`, and any
        referenced node missing from `depends_on` is appended to `upstream`; both cases are logged as
        warnings. An input entry can be a single reference or a list/dict of them, so each entry is
        walked structurally.
        :param node: dict
            Node entry holding the node's 'run' block with 'inputs' and 'params', and 'depends_on' list.
        :param pipeline_name: str
            Name of the pipeline the node belongs to, used to qualify same-pipeline references that omit the pipeline name.
        :return: dict
            The inputs with every reference fully qualified, in the same structure as `run_inputs`.
        """
        def qualify_ref(arg_ref: str, arg_path: str) -> str:
            upnode_name, _, _ = parse_upstream_ref(arg_ref, pipeline_name)
            # Ensure that task_path (which should be a node name) is named fully with pipeline name
            ref_base, ref_paren, ref_call = arg_ref.partition("(")
            if '.' not in ref_base:
                fixed_ref = f"{upnode_name}({ref_call}" if ref_paren else upnode_name
                logger.warning(f"Input '{arg_path}' for task '{node_name}' references '{arg_ref}' "
                               f"without pipeline name. Assuming it is from the same pipeline and converting it "
                               f"to '{fixed_ref}'.")
                arg_ref = fixed_ref
            if upnode_name not in upstream:
                logger.warning(f"Input '{arg_path}' for task '{node_name}' references '{upnode_name}' "
                               f"which is not listed in 'depends_on'. Adding it to the dependencies.")
                upstream.append(upnode_name)
            return arg_ref


        return {arg_name: map_input_refs(arg_spec, qualify_ref, arg_name)
                for arg_name, arg_spec in run_inputs.items()}

    @staticmethod
    def resolve_inputs(run_inputs: dict, upstream_results: dict, node_name: str, dry: bool = False) -> dict:
        """
        Resolve a node's input references into the values produced by its upstream tasks.

        An input entry can be a single reference or a list/dict of them, in which case the resolved
        value keeps that structure.
        :param run_inputs: dict
            Fully qualified node inputs, mapping argument name -> reference(s).
        :param upstream_results: dict
            Results of the node's dependencies, keyed by fully qualified node name.
        :param node_name: str
            Fully qualified name of the node being resolved, used in error messages.
        :param dry: bool
            If True the upstream results are dummies (see `dry_run`), so a filter call on a
            placeholder value is skipped instead of raising.
        :return: dict
            Mapping of argument name -> resolved value(s), ready to be passed to the task.
        """
        pipeline_name = node_name.split('.')[0]

        def resolve_ref(ref: str, arg_path: str):
            task_path, field_name, call_arg = parse_upstream_ref(ref, pipeline_name)
            if task_path not in upstream_results:
                raise ValueError(f"Upstream task '{task_path}' not found in results")
            result = upstream_results[task_path]
            try:
                value = result[field_name] if field_name is not None else result
            except KeyError as e:
                raise KeyError(f"Input '{arg_path}' for task '{node_name}' references field '{field_name}', which "
                               f"'{task_path}' ({type(result).__name__}) does not produce; it produces "
                               f"{list(result.keys())}") from e
            # Dummy payloads carry no data to filter on, so the call is left out of a dry run
            if call_arg is not None and not dry:
                if not callable(value):
                    raise TypeError(f"Input '{arg_path}' for task '{node_name}' references "
                                    f"'{task_path}.{field_name}' with a filter call, but its value "
                                    f"(type {type(value).__name__}) is not callable.")
                value = value(call_arg)
            return value

        return {arg_name: map_input_refs(arg_spec, resolve_ref, arg_name)
                for arg_name, arg_spec in run_inputs.items()}

    @staticmethod
    def collect_upstream_results(node_name: str, node_dict: dict, results: dict) -> dict:
        """
        Pick the results of a node's dependencies out of the results collected so far.
        :param node_name: str
            Fully qualified name of the node whose dependencies are needed, used in error messages.
        :param node_dict: dict
            Node entry holding the node's 'upstream' dependencies.
        :param results: dict
            Results collected so far, keyed by fully qualified node name.
        :return: dict
            Results of the node's dependencies, keyed by fully qualified node name.
        """
        upstream_results = {}
        for upstream_name in node_dict["upstream"]:
            if upstream_name not in results:
                raise ValueError(f"Upstream task '{upstream_name}' of node '{node_name}' not found in results")
            upstream_results[upstream_name] = results[upstream_name]
        return upstream_results

    @staticmethod
    def build_execution_order(node_dependencies: dict):
        # First determine the execution order of the pipelines based on their inter-pipeline dependencies
        pipe_names = set([name.split('.')[0] for name in node_dependencies.keys()]) # Get unique pipeline names
        pipe_dependencies = {pipe: set() for pipe in pipe_names}

        # For each node, check which pipelines its dependencies belong to and add those to the node's pipeline dependencies set
        for node, node_dep in node_dependencies.items():
            pipe_name = node.split('.')[0]
            dep_pipe_names = set([dep.split('.')[0] for dep in node_dep])
            pipe_dependencies[pipe_name].update(dep_pipe_names - {pipe_name})
        pipe_order = get_dag_order(pipe_dependencies) # Get DAG order of pipelines
        logger.info(f"Pipeline order: {pipe_order}")

        # Then determine the execution order of the tasks within each pipeline
        node_orders = {}
        for pnames in pipe_order:
            for pipe_name in pnames:
                # Get the subset of nodes that belong to this pipeline and their dependencies
                # All dependencies are kept as is to get the full ordering, the correct handling of execution is done later
                pipe_nodes = {node: deps for node, deps in node_dependencies.items() if node.startswith(pipe_name + '.')}
                node_order = get_dag_order(pipe_nodes) # Get DAG order of nodes within this pipeline
                node_orders[pipe_name] = node_order
                logger.info(f"Pipeline {pipe_name} order: {node_order}")

        return pipe_order, node_orders

    @staticmethod
    def execute_task(node_name, node_dict, upstream_results, lazy=False):
        """
        Execute a single task given its node_dict containing the task instance, inputs, and parameters.
         - Resolve the run inputs against the upstream results; an input can be a single reference or a
           list/dict of references, in which case the task receives a list/dict of resolved values
         - Call the task's run or lazy_run method with the resolved inputs and parameters
         - Return the task's own TaskResult (subclass) with upstream dependencies and params attached
         - If lazy is True, call the task's lazy_run method and return the raw generator of TaskResult
           batches, unmodified. The caller is responsible for iterating through the results, attaching
           upstream/params metadata to each yielded batch, and feeding them downstream.
         :param node_name: The name of the node to execute.
         :param node_dict: dict
            A dictionary containing the task instance, inputs, parameters, and upstream dependencies.
         :param upstream_results: dict
            A dictionary containing the results of previously executed tasks that are dependencies, used for resolving inputs.
         :param lazy: bool
            Whether to execute the task in lazy mode. If True, the task must have a lazy_run method that returns an iterator.
         :return: TaskResult | Generator[TaskResult, None, None]
             The task's own TaskResult with 'upstream'/'params' set, or (if lazy) a generator of TaskResults
             still missing 'upstream'/'params', which the caller must attach.
        """

        init_params = node_dict["init_params"]
        run_inputs = PipelineEngine.resolve_inputs(node_dict["run_inputs"], upstream_results, node_name)
        run_params = node_dict["run_params"]
        eregion_task = node_dict["task"](name=node_name, **init_params)

        if lazy:
            if not hasattr(eregion_task, "lazy_run"):
                raise ValueError(f"Task '{eregion_task.name}' does not have a 'lazy_run' method for lazy execution")
            return eregion_task.lazy_run(**run_inputs, **run_params)

        res = eregion_task.run(**run_inputs, **run_params)
        return res.model_copy(update={"upstream": node_dict["upstream"], "params": node_dict["params"]})

    def execute_pipeline(self, pipe_name, node_order, nodes_dict, results):
        """
        Execute a single pipeline given its name, execution order of its nodes, the nodes_dict containing task instances
        , and the results dict containing results of previously executed tasks (both from this pipeline and other pipelines).
         - Only nodes belonging to the current pipeline are executed, nodes from other pipelines in the order that are
           dependencies are checked for their results in the results dict and passed as upstream_results to the current nodes
        :param pipe_name: str
            Name of the pipeline being executed, used for logging and checking node ownership.
        :param node_order: list of sets
            Execution order of the nodes in this pipeline, as a list of generations (sets) of node names that can be executed in parallel.
        :param nodes_dict: dict
            Dictionary mapping node names to their corresponding task instances, inputs, parameters and upstream dependencies.
        :param results: dict
            Dictionary containing results, keys are node/task names, values are TaskResult objects.
            In eager pipelines, same as self.results
            In lazy pipelines, a temp results dict that is updated iteratively and merged into self.results after each iteration.
         :return: dict
             Updated results dictionary after adding results of all executed tasks from this pipeline.
        """
        for node_names in node_order:
            for node_name in node_names:
                # For nodes belonging to other pipelines, check that their results are available
                if node_name.split('.')[0] != pipe_name:
                    if node_name not in results:
                        raise ValueError(f"Upstream task '{node_name}' not found in results")
                    continue

                upstream_results = self.collect_upstream_results(node_name, nodes_dict[node_name], results)
                results[node_name] = self.execute_task(node_name, nodes_dict[node_name], upstream_results, lazy=False)

        return results

    def get_source_node(self, pipe_name: str) -> str:
        """
        Return the fully qualified source node of a lazy pipeline.
        :param pipe_name: str
            Name of the lazy pipeline whose 'source' node is wanted.
        :return: str
            Fully qualified name of the source node, checked to start the pipeline's execution order.
        """
        source_node = self.pipelines[pipe_name]["source"]
        source_node = f"{pipe_name}.{source_node}" if '.' not in source_node else source_node
        # Sanity check: source_node must be in the first generation of the node order
        if source_node not in self.execution_orders[1][pipe_name][0]:
            raise ValueError(f"Source node '{source_node}' for lazy pipeline '{pipe_name}' must be in the "
                             f"first generation of the node order")
        return source_node

    def execute_eager_pipeline(self, pipe_name):
        """
        Execute a single eager pipeline (non-lazy) given its name.
         - This is a wrapper around self.execute_pipeline that retrieves the node order and nodes_dict for the given
           pipeline name
         - self.results is updated directly
        :param pipe_name: str
            Name of the pipeline being executed, used for retrieving node order and nodes_dict, and for logging.
        :return: None, self.results is updated in place with the results of executing this pipeline.
        """
        node_order = self.execution_orders[1][pipe_name]
        nodes_dict = self.pipelines[pipe_name]["nodes_dict"]
        self.results = self.execute_pipeline(pipe_name, node_order, nodes_dict, self.results)
        return self.results

    def execute_lazy_pipeline(self, pipe_name):
        """
        Execute a single lazy pipeline given its name.
        - Lazy pipelines must have a single source node in the first generation of the node order that returns an
          iterator when executed with self.execute_task with lazy=True
        - Execute the rest of the pipeline for each item in the source node's iterator,
          feeding the result of each iteration as input, storing iteration results in a temp results dict.
        - After each iteration, merge the temp results into self.results by combining results of the same nodes across iterations
        :param pipe_name: str
            Name of the pipeline being executed, used for retrieving node order and nodes_dict, and for logging.
        :return: Generator that yields self.results after each iteration of the source node.
        """
        node_order = self.execution_orders[1][pipe_name]
        nodes_dict = self.pipelines[pipe_name]["nodes_dict"]
        source_node = self.get_source_node(pipe_name)

        # Check that rest of the nodes in first generation have been executed and their results are available
        for node_name in node_order[0]:
            if node_name.split('.')[0] != pipe_name:
                if node_name not in self.results:
                    raise ValueError(f"Upstream task '{node_name}' not found in results")

        # For lazy pipelines, we need to execute the source node using self.execute_task
        # that returns a generator of TaskResult batches, and then feed it downstream.
        temp_results = deepcopy(self.results)  # Use a temporary results dict to store the iterating results
        source_node_dict = nodes_dict[source_node]
        source_upstream_results = self.collect_upstream_results(source_node, source_node_dict, temp_results)
        source_res_iterator = self.execute_task(source_node, source_node_dict, source_upstream_results, lazy=True)
        if not isinstance(source_res_iterator, Iterator):
            raise ValueError(f"Source node '{source_node}' for lazy pipeline '{pipe_name}' did not return an iterator")

        # Iterate through the source result and feed it downstream to the rest of the nodes in the pipeline according
        # to the execution order
        for item in source_res_iterator:
            temp_results[source_node] = item.model_copy(update={"upstream": source_node_dict["upstream"],
                                                                  "params": source_node_dict["params"]})
            temp_results = self.execute_pipeline(pipe_name, node_order[1:], nodes_dict, temp_results)

            # Merge the temp_results into self.results (a bit convoluted as we need to add this iterations result
            # for all nodes into the main results dict)
            for node_name in nodes_dict.keys():
                if node_name not in self.results:
                    self.results[node_name] = temp_results[node_name]
                else:
                    self.results[node_name] = self.results[node_name].combine(temp_results[node_name])
            yield self.results

    ## Validate the pipelines without executing them
    @staticmethod
    def dry_run_task(node_name: str, node_dict: dict, dummy_results: dict, lazy: bool = False):
        """
        Check that a single node can be built and called, without executing it.
         - Instantiate the task with its configured init params
         - Resolve its inputs against the dummy results of its upstream nodes, which checks that every
           referenced node and payload field exists
         - Bind the resolved inputs and the run params to the signature of run()/lazy_run(), which
           checks for missing, unexpected or duplicated arguments
        :param node_name: str
            Fully qualified name of the node to check.
        :param node_dict: dict
            Node entry holding the task class, inputs, params and upstream dependencies.
        :param dummy_results: dict
            Dummy results of the nodes checked so far, keyed by fully qualified node name.
        :param lazy: bool
            Whether the node is the source of a lazy pipeline, and is therefore called via lazy_run().
        :return: TaskResult
            A dummy instance of the task's own result class, standing in for what it would produce.
        """
        upstream_results = PipelineEngine.collect_upstream_results(node_name, node_dict, dummy_results)
        # Upstream payloads are placeholders here, so filter calls on them are skipped rather than applied
        run_inputs = PipelineEngine.resolve_inputs(node_dict["run_inputs"], upstream_results, node_name, dry=True)

        task_class = node_dict["task"]
        try:
            eregion_task = task_class(name=node_name, **node_dict["init_params"])
        except (TypeError, ValueError) as e:
            raise type(e)(f"Node '{node_name}': cannot initialise {task_class.__name__} with the configured "
                          f"'init.params' {sorted(node_dict['init_params'])}: {e}") from e

        run_method = "lazy_run" if lazy else "run"
        if lazy and not hasattr(eregion_task, run_method):
            raise ValueError(f"Task '{node_name}' does not have a 'lazy_run' method for lazy execution")
        try:
            inspect.signature(getattr(eregion_task, run_method)).bind(**run_inputs, **node_dict["run_params"])
        except TypeError as e:
            raise TypeError(f"Node '{node_name}': the configured inputs {sorted(run_inputs)} and params "
                            f"{sorted(node_dict['run_params'])} do not match "
                            f"{task_class.__name__}.{run_method}(): {e}") from e

        return eregion_task.dummy_run()

    def dry_run(self):
        """
        Walk the whole DAG in execution order without executing any task.

        Each node is checked with `dry_run_task` and contributes a dummy TaskResult, which the nodes
        downstream of it resolve their inputs against. Wiring mistakes (an unknown node or payload
        field, an argument a task does not take, a missing required argument) therefore surface when
        the engine is built rather than part-way through a long run. Payload values are placeholders,
        so their types are not checked.
        :return: dict
            Mapping of fully qualified node name -> dummy TaskResult that the node would produce.
        """
        dummy_results = {}
        for pipe_names in self.execution_orders[0]:
            for pipe_name in pipe_names:
                nodes_dict = self.pipelines[pipe_name]["nodes_dict"]
                source_node = self.get_source_node(pipe_name) if self.pipelines[pipe_name]["lazy"] else None
                for node_names in self.execution_orders[1][pipe_name]:
                    for node_name in node_names:
                        # Nodes from other pipelines have been checked when that pipeline was walked
                        if node_name.split('.')[0] != pipe_name:
                            if node_name not in dummy_results:
                                raise ValueError(f"Upstream task '{node_name}' of pipeline '{pipe_name}' is not "
                                                 f"produced by any earlier pipeline")
                            continue
                        if node_name not in nodes_dict:
                            raise ValueError(f"Node '{node_name}' is referenced by pipeline '{pipe_name}' but is "
                                             f"not defined in it")
                        dummy_results[node_name] = self.dry_run_task(node_name, nodes_dict[node_name],
                                                                     dummy_results, lazy=node_name == source_node)
        logger.info(f"Dry run passed for {len(dummy_results)} nodes")
        return dummy_results

    ## Execute the pipelines
    def run(self, max_workers=4):
        """
        Execute all pipelines, generation by generation, in a thread pool.
        :param max_workers: int
            Maximum number of pipelines to execute concurrently within a generation.
        :return: None, self.results holds the results of every executed task.
        """

        pipe_order = self.execution_orders[0]
        for pipe_names in pipe_order:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                # Submit each pipeline in this generation to the executor
                for pipe_name in pipe_names:
                    if self.pipelines[pipe_name]["lazy"]:
                        # Run the lazy pipeline to completion inside the worker thread by consuming its generator
                        def _run_lazy(pn):
                            last_result = None
                            for iteration_result in self.execute_lazy_pipeline(pn):
                                last_result = iteration_result
                            return last_result

                        fut = executor.submit(_run_lazy, pipe_name)
                    else:
                        # Eager pipeline updates self.results in place
                        fut = executor.submit(self.execute_eager_pipeline, pipe_name)

                    future_map[fut] = pipe_name

                # Wait for all pipelines in this generation to finish
                done, _ = cf.wait(future_map.keys(), return_when=cf.ALL_COMPLETED)

                # Check for exceptions and re-raise after logging
                for fut in done:
                    pipe_name = future_map[fut]
                    try:
                        fut.result()
                        logger.info(f"Pipeline '{pipe_name}' complete")
                    except Exception as exc:
                        logger.exception(f"Pipeline '{pipe_name}' failed: {exc}")
                        raise

################################# Helper functions #################################
_UPSTREAM_REF_CALL_PATTERN = re.compile(r"^(?P<base>[^()]+)\((?P<quote>['\"])(?P<arg>.*)(?P=quote)\)$")

def parse_upstream_ref(ref: str, pipeline_name: str) -> tuple[str, str | None, str | None]:
    """
    Parse an upstream reference string into (task_path, field_name, call_arg).

    Pipeline and node names must not contain '.'. Accepted forms:
      - 'node'                          -> (f'{pipeline_name}.node', None, None)  # same-pipeline, whole result
      - 'pipeline.node'                 -> ('pipeline.node', None, None)          # whole result
      - 'pipeline.node.field'           -> ('pipeline.node', 'field', None)       # single payload field
      - "pipeline.node.field('query')"  -> ('pipeline.node', 'field', 'query')    # field's value is called
                                            with 'query' (e.g. ImageBundle.__call__ for filtering by a
                                            pandas query string); the field's value must be callable.
    :param ref: str
        Reference string as written in the pipeline config.
    :param pipeline_name: str
        Name of the pipeline the reference appears in, used to qualify same-pipeline
        references that omit the pipeline name.
    :return: tuple[str, str | None, str | None]
        (task_path, field_name, call_arg). field_name is None when the whole upstream TaskResult is
        requested; call_arg is None unless a "field('query')" call was used.
    """
    call_match = _UPSTREAM_REF_CALL_PATTERN.match(ref)
    base_ref = call_match.group("base") if call_match else ref
    call_arg = call_match.group("arg") if call_match else None

    parts = base_ref.split(".")
    if len(parts) == 1:
        task_path, field_name = f"{pipeline_name}.{parts[0]}", None
    elif len(parts) == 2:
        task_path, field_name = base_ref, None
    else:
        task_path, field_name = ".".join(parts[:2]), ".".join(parts[2:])

    if call_arg is not None and field_name is None:
        raise ValueError(f"Invalid upstream reference '{ref}': a filter call requires a field name "
                          f"(use 'pipeline.node.field(...)').")

    return task_path, field_name, call_arg

def map_input_refs(spec, func, arg_path: str):
    """
    Apply `func` to every upstream reference in an input entry, keeping the entry's structure.

    A task argument can be wired to a single upstream output ('images: pipe.node.field'), or to
    several at once by giving a list or dict of references (nested arbitrarily), e.g.
    'task_results: [pipe.node_a, pipe.node_b]'. The task then receives a list/dict of the resolved
    values, in the order/under the keys written in the config.
    :param spec: str | list | dict
        Input entry as written in the config: a reference string, or a list/dict of them.
    :param func: callable
        Called as func(reference, arg_path) for each reference string; its return value takes the
        reference's place in the returned structure.
    :param arg_path: str
        Location of `spec` within the node's inputs, used in log and error messages
        (e.g. "task_results[0]").
    :return: str | list | dict
        `spec` with every reference string replaced by the corresponding `func` return value.
    """
    match spec:
        case str():
            return func(spec, arg_path)
        case list():
            return [map_input_refs(item, func, f"{arg_path}[{i}]") for i, item in enumerate(spec)]
        case dict():
            return {key: map_input_refs(value, func, f"{arg_path}.{key}") for key, value in spec.items()}
        case _:
            raise TypeError(f"Input '{arg_path}' must be an upstream reference string, or a list/dict of them, "
                            f"got {type(spec).__name__}")

def get_dag_order(deps):
    """
    Get the execution order of tasks based on their dependencies using topological sorting.
    :param deps: dict
        A dictionary where keys are task names and values are sets of task names that the key task depends on
    :return: list of sets
        A list of generations of task names that can be executed in parallel, ordered by their dependencies
    """
    ts = graphlib.TopologicalSorter(deps)
    try:
        # check for cycles
        ts.prepare()
    except graphlib.CycleError as e:
        raise ValueError(f"Cyclic dependency detected in pipeline: {e}")
    # Get generations of tasks that can be executed in parallel
    execution_order = []
    while ts.is_active():
        generation = ts.get_ready()
        execution_order.append(generation)
        ts.done(*generation)
    return execution_order