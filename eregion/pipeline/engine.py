from typing import Iterator
from copy import deepcopy
import graphlib

from tasks import TaskResult
from configs import PipelineConfig
from utils import configure_logger, load_class

from prefect import task, flow
from prefect.futures import wait
from prefect.task_runners import ThreadPoolTaskRunner

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
    """

    def __init__(self, pipeline_config_input: str | dict):

        self.pipeline_config = PipelineConfig(pipeline_config_input)
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

    def update(self, new_config: str | dict):
        """
            Update the pipeline configuration and rebuild the pipelines and execution orders.
        :param new_config: str or dict
            New pipeline configuration as a path to a YAML/JSON file or a dictionary.
        """
        self.__init__(pipeline_config_input=new_config)

    def build_pipeline_nodes(self, pipeline_cfg: dict):
        """
        Build the task nodes for a single pipeline based on its configuration.
         - For each node, load the corresponding task class
         - Extract the inputs (that are outputs of other nodes) for init and run arguments from the config
         - Track the dependencies of each node based on the inputs and explicit depends_on field
        :param pipeline_cfg: dict
            Configuration dictionary for a single pipeline, containing at least the keys "name", "lazy", and "nodes".
            Each node should have at least "name" and "task", and can optionally have "init", "run", and "depends_on"
            fields.
        :return: tuple
            A tuple containing:
            - nodes: dict mapping node names to their corresponding task instances, inputs, parameters
                and upstream dependencies.
            - node_dependencies: dict mapping node names to a set of their upstream dependencies
        """
        node_dependencies = {}
        nodes = {}
        for node in pipeline_cfg["nodes"]:
            eregion_task = load_class(node["task"])
            init_block = node.get("init", {})
            init_inputs = init_block.get("inputs", {})
            init_params = init_block.get("params", {})
            run_block = node.get("run", {})
            run_inputs = run_block.get("inputs", {})
            run_params = run_block.get("params", {})
            upstream = node.get("depends_on", [])

            # Ensure that node names in upstream list are named fully with pipeline name
            for i, dep in enumerate(upstream):
                if '.' not in dep:
                    upstream[i] = f"{pipeline_cfg['name']}.{dep}"

            # sanity check: if init_inputs and run_inputs are not empty, the corresponding dependencies must be listed in depends_on
            # If not, log a warning and add them to the dependencies list
            def check_inputs(inputs):
                for arg_name, arg_ref in inputs.items():
                    upnode_name, _, key = arg_ref.partition(".data.")
                    # Ensure that task_path (which should be a node name) is named fully with pipeline name
                    if '.' not in upnode_name:
                        logger.warning(f"Input '{arg_name}' for task '{node['name']}' references '{upnode_name}' "
                                     f"without pipeline name. Assuming it is from the same pipeline and converting it "
                                     f"to '{pipeline_cfg['name']}.{upnode_name}'.")
                        upnode_name = f"{pipeline_cfg['name']}.{upnode_name}"
                        inputs[arg_name] = f"{upnode_name}.data.{key}"
                    if upnode_name not in upstream:
                        logger.warning(f"Input '{arg_name}' for task '{node['name']}' references '{upnode_name}' "
                                       f"which is not listed in 'depends_on'. Adding it to the dependencies.")
                        upstream.append(upnode_name)

            check_inputs(init_inputs)
            check_inputs(run_inputs)

            node_dependencies[f"{pipeline_cfg['name']}.{node['name']}"] = set(upstream)
            node_dict = {"task": eregion_task, "init_inputs": init_inputs, "init_params": init_params,
                         "run_inputs": run_inputs, "run_params": run_params, "upstream": upstream,
                         "params": {'init': init_params | init_inputs, 'run': run_params | run_inputs}}
            nodes[f"{pipeline_cfg['name']}.{node['name']}"] = node_dict

        return nodes, node_dependencies

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
    def execute_task(node_name, node_dict, upstream_results, lazy=False) -> TaskResult:
        """
        Execute a single task given its node_dict containing the task instance, inputs, and parameters.
         - Resolve the inputs using self.resolve_inputs
         - Call the task's run or lazy_run method with the resolved inputs and parameters
         - Return a TaskResult containing the task name, output data, upstream dependencies, and parameters
         - If lazy is True, call the task's lazy_run method and return an iterator wrapped in TaskResult.
           The caller is responsible for iterating through the results and feeding them downstream.
         :param node_name: The name of the node to execute.
         :param node_dict: dict
            A dictionary containing the task instance, inputs, parameters, and upstream dependencies.
         :param upstream_results: dict
            A dictionary containing the results of previously executed tasks that are dependencies, used for resolving inputs.
         :param lazy: bool
            Whether to execute the task in lazy mode. If True, the task must have a lazy_run method that returns an iterator.
         :return: TaskResult
             A TaskResult containing the task name, output data (or iterator if lazy), upstream dependencies, and parameters.
        """

        def resolve_inputs(input_spec: dict) -> dict:
            resolved = {}
            for arg_name, ref in input_spec.items():
                task_path, _, key = ref.partition(".data.")
                if task_path not in upstream_results:
                    raise ValueError(f"Upstream task '{task_path}' not found in results")
                resolved[arg_name] = upstream_results[task_path].data[key]
            return resolved

        init_inputs = resolve_inputs(node_dict["init_inputs"])
        init_params = node_dict["init_params"]
        run_inputs = resolve_inputs(node_dict["run_inputs"])
        run_params = node_dict["run_params"]
        eregion_task = node_dict["task"](name=node_name, **init_params, **init_inputs)

        if lazy:
            if not hasattr(eregion_task, "lazy_run"):
                raise ValueError(f"Task '{eregion_task.name}' does not have a 'lazy_run' method for lazy execution")
            res = eregion_task.lazy_run(**run_inputs, **run_params)
        else:
            res = eregion_task.run(**run_inputs, **run_params)
        return TaskResult(task_name=eregion_task.name, data=res, upstream=node_dict["upstream"],
                          params=node_dict["params"])

    def execute_pipeline(self, pipe_name, node_order, nodes_dict, results):
        """
        Execute a single pipeline given its name, execution order of its nodes, the nodes_dict containing task instances
        , and the results dict containing results of previously executed tasks (both from this pipeline and other pipelines).
         - Each node task is wrapped in a Prefect task instance that calls self.execute_task
         - Nodes belonging to same generation in the execution order are executed in parallel using Prefect's task runner
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
            submitted, futures = [], []
            for node_name in node_names:
                # For nodes belonging to other pipelines, check that their results are available
                if node_name.split('.')[0] != pipe_name:
                    if node_name not in results:
                        raise ValueError(f"Upstream task '{node_name}' not found in results")
                    continue

                prefect_task = make_prefect_task(node_name, self.execute_task) # Wrap in Prefect task
                upstream_results = {up: results[up] for up in nodes_dict[node_name]["upstream"]}
                futures.append(prefect_task.submit(node_name=node_name, node_dict=nodes_dict[node_name],
                                                   upstream_results=upstream_results, lazy=False)) # Submit for parallel execution
                submitted.append(node_name)

            wait(futures) # Wait for all tasks in this generation to complete before moving to the next generation
            for node_name, future in zip(submitted, futures):
                results[node_name] = future.result()

        return results

    def execute_eager_pipeline(self, pipe_name):
        """
        Execute a single eager pipeline (non-lazy) given its name.
         - This is a wrapper around self.execute_pipeline that retrieves the node order and nodes_dict for the given
           pipeline name
         - Pipeline is executed as a Prefect flow that calls self.execute_pipeline
         - self.results is updated directly
        :param pipe_name: str
            Name of the pipeline being executed, used for retrieving node order and nodes_dict, and for logging.
        :return: None, self.results is updated in place with the results of executing this pipeline.
        """
        node_order = self.execution_orders[1][pipe_name]
        nodes_dict = self.pipelines[pipe_name]["nodes_dict"]
        pipe_prefect_flow = make_prefect_flow(pipe_name, self.execute_pipeline)
        self.results = pipe_prefect_flow(pipe_name, node_order, nodes_dict, self.results)
        return self.results

    def execute_lazy_pipeline(self, pipe_name):
        """
        Execute a single lazy pipeline given its name.
        - Lazy pipelines must have a single source node in the first generation of the node order that returns an
          iterator when executed with self.execute_task with lazy=True
        - Execute the rest of the pipeline (as a Prefect flow) for each item in the source node's iterator,
          feeding the result of each iteration as input, storing iteration results in a temp results dict.
        - After each iteration, merge the temp results into self.results by combining results of the same nodes across iterations
        :param pipe_name: str
            Name of the pipeline being executed, used for retrieving node order and nodes_dict, and for logging.
        :return: Generator that yields self.results after each iteration of the source node.
        """
        node_order = self.execution_orders[1][pipe_name]
        nodes_dict = self.pipelines[pipe_name]["nodes_dict"]
        source_node = self.pipelines[pipe_name]["source"]
        source_node = f"{pipe_name}.{source_node}" if '.' not in source_node else source_node

        # Sanity check: source_node must be in the first generation of the node order
        if source_node not in node_order[0]:
            raise ValueError(f"Source node '{source_node}' for lazy pipeline '{pipe_name}' must be in the "
                             f"first generation of the node order")
        # Check that rest of the nodes in first generation have been executed and their results are available
        for node_name in node_order[0]:
            if node_name.split('.')[0] != pipe_name:
                if node_name not in self.results:
                    raise ValueError(f"Upstream task '{node_name}' not found in results")

        # For lazy pipelines, we need to execute the source node using self.execute_task
        # that returns an iterator wrapped in TaskResult, and then feed it downstream.
        temp_results = deepcopy(self.results)  # Use a temporary results dict to store the iterating results
        source_res = self.execute_task(nodes_dict[source_node], temp_results, lazy=True)
        source_res_iterator = source_res.data
        if not isinstance(source_res_iterator, Iterator):
            raise ValueError(f"Source node '{source_node}' for lazy pipeline '{pipe_name}' did not return an iterator")

        # Iterate through the source result and feed it downstream to the rest of the nodes in the pipeline according
        # to the execution order
        for item in source_res_iterator:
            temp_results[source_node] = TaskResult(task_name=source_res.task_name, upstream=source_res.upstream,
                                                   params=source_res.params, data=item)
            pipe_prefect_flow = make_prefect_flow(pipe_name, self.execute_pipeline)
            temp_results = pipe_prefect_flow(pipe_name=pipe_name, node_order=node_order[1:], nodes_dict=nodes_dict,
                                                temp_results=temp_results)

            # Merge the temp_results into self.results (a bit convoluted as we need to add this iterations result
            # for all nodes into the main results dict)
            for node_name in nodes_dict.keys():
                if node_name not in self.results:
                    self.results[node_name] = temp_results[node_name]
                else:
                    self.results[node_name] = self.results[node_name].combine(temp_results[node_name])
            yield self.results

    ## Execute the pipelines
    def run(self, max_workers=4):
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

def make_prefect_task(task_name, task_func, retries=3, retry_delay_seconds=3):
    """
    Wrap a task function in a Prefect task with the given name and retry logic.
    :param task_name: str
        Name of the Prefect task, used for logging and tracking in Prefect.
    :param task_func: function
        The function that implements the task logic
    :param retries: int
        Number of times to retry the task in case of failure before giving up.
    :param retry_delay_seconds: int
        Number of seconds to wait between retries.
    :return: Prefect task
        A Prefect task that wraps the given task function with the specified name and retry logic.
    """
    @task(name=task_name, retries=retries, retry_delay_seconds=retry_delay_seconds)
    def prefect_task(*args, **kwargs):
        return task_func(*args, **kwargs)
    return prefect_task

def make_prefect_flow(flow_name, flow_func, task_runner=ThreadPoolTaskRunner, max_workers=4):
    """
    Wrap a flow function in a Prefect flow with the given name and task runner for parallel execution.
    :param flow_name: str
        Name of the Prefect flow, used for logging and tracking in Prefect.
    :param flow_func: function
        The function that implements the flow logic.
    :param task_runner: Prefect task runner class
        The Prefect task runner class to use for parallel execution of tasks within the flow
        (e.g., ThreadPoolTaskRunner or ProcessPoolTaskRunner).
    :param max_workers: int
        The maximum number of worker threads/processes to use for parallel execution of tasks within the flow
    :return: Prefect flow
        A Prefect flow that wraps the given flow function with the specified name and task runner for parallel execution.
    """
    @flow(name=flow_name, task_runner=task_runner(max_workers=max_workers))
    def prefect_flow(*args, **kwargs):
        return flow_func(*args, **kwargs)
    return prefect_flow