"""
The flow graph: nodes, flows, and how both are built from a parsed configuration.

A node pairs an initialised `Task` with the wiring it was configured with; a flow is an
ordered group of nodes. Everything here is built once, before anything is executed, so that a
mistake in the configuration is reported up front rather than part-way through a long run.
"""
from dataclasses import dataclass, field
import graphlib

from eregion.tasks.task import Task
from eregion.utils import configure_logger, load_class

from .refs import parse_inputs, referenced_nodes, render_inputs

logger = configure_logger(__name__)


@dataclass(frozen=True, slots=True)
class FlowNode:
    """
    A single node of a flow, built once from its yaml entry.

    The task is already initialised: `init.params` are static by contract (outputs of other
    tasks may only be wired through `run.inputs`), so nothing about the task depends on what
    the flow produces and it can be constructed up front rather than per execution.

    :param name: str
        Fully qualified node name, 'flow.node'.
    :param task: Task
        The initialised task instance this node runs.
    :param flow: str
        Name of the flow the node belongs to.
    :param run_inputs: dict
        Arguments wired to other nodes' results, as `ResultRef` objects (a single reference, or
        a list/dict of them).
    :param run_params: dict
        Static arguments passed to run()/lazy_run() alongside the resolved inputs.
    :param upstream: list[str]
        Fully qualified names of the nodes this one depends on, ordered and de-duplicated.
    :param init_params: dict
        The static params the task was initialised with, kept for provenance.
    """
    name: str
    task: Task
    flow: str = ""
    run_inputs: dict = field(default_factory=dict)
    run_params: dict = field(default_factory=dict)
    upstream: list[str] = field(default_factory=list)
    init_params: dict = field(default_factory=dict)

    @property
    def params(self) -> dict:
        """
        The node's configuration, recorded on every TaskResult it produces.

        References are rendered back to their canonical strings, because results are serialised
        to JSON when they are saved.
        :return: dict
        """
        return {"init": self.init_params, "run": self.run_params | render_inputs(self.run_inputs)}


@dataclass(frozen=True, slots=True)
class Flow:
    """
    One flow: its own nodes, in the order they must run.

    :param name: str
        Name of the flow, as given in the config.
    :param lazy: bool
        Whether the flow is driven in batches by a source node's `lazy_run()`.
    :param nodes: dict
        This flow's nodes only, keyed by fully qualified name.
    :param node_order: tuple
        Generations of this flow's own node names, in dependency order. Nodes in other
        flows are left out: a flow only runs once everything it depends on has, so its
        own nodes are all that it has to order.
    :param source: str | None
        For a lazy flow, the fully qualified name of the node whose `lazy_run()` yields the
        batches that drive it; None for an eager flow.
    """
    name: str
    lazy: bool
    nodes: dict
    node_order: tuple = ()
    source: str | None = None

    @property
    def depends_on(self) -> set:
        """
        The other flows this one needs results from.
        :return: set[str]
            Names of the flows this one depends on, which therefore run before it.
        """
        return {dep.split('.')[0] for node in self.nodes.values()
                for dep in node.upstream} - {self.name}

    def ordered_nodes(self, skip_source: bool = False):
        """
        Iterate this flow's nodes in dependency order.
        :param skip_source: bool
            Whether to leave out the source node, which a lazy run executes separately.
        :return: Iterator[FlowNode]
        """
        for generation in self.node_order:
            for node_name in generation:
                if skip_source and node_name == self.source:
                    continue
                yield self.nodes[node_name]


def build_node(node_cfg: dict, flow_name: str) -> tuple[FlowNode, set[str]]:
    """
    Build one node from its config entry.

    The task is initialised here, from `init.params` alone, and the node's inputs are parsed into
    `ResultRef` objects. A node referenced by an input is a dependency whether or not
    `depends_on` names it, so the two sources are merged.
    :param node_cfg: dict
        The node's config entry: at least 'name' and 'task', optionally 'init', 'run',
        'depends_on'.
    :param flow_name: str
        Name of the flow the node belongs to.
    :return: tuple[FlowNode, set[str]]
        The node, and the set of node names it depends on.
    """
    node_name = f"{flow_name}.{node_cfg['name']}"

    # Outputs of other tasks are never passed to Task.__init__
    init_block = node_cfg.get("init", {})
    if "inputs" in init_block:
        raise ValueError(f"Node '{node_cfg['name']}' declares 'init.inputs'; outputs of other tasks cannot be "
                         f"passed to Task.__init__. Move them to the node's 'run.inputs' block, 'init' only "
                         f"takes 'params'.")
    init_params = init_block.get("params", {})

    # Build the task now: its params are static, so nothing about it depends on the run
    task_class = load_class(node_cfg["task"])
    try:
        task = task_class(name=node_name, **init_params)
    except (TypeError, ValueError, NotImplementedError) as e:
        raise type(e)(f"Node '{node_name}': cannot initialise {task_class.__name__} with the configured "
                      f"'init.params' {sorted(init_params)}: {e}") from e

    run_block = node_cfg.get("run", {})
    run_inputs = parse_inputs(run_block.get("inputs", {}), node_name)

    # Dependencies: those declared, plus those implied by the inputs
    upstream = [dep if '.' in dep else f"{flow_name}.{dep}" for dep in node_cfg.get("depends_on", [])]
    for referenced in referenced_nodes(run_inputs):
        if referenced not in upstream:
            logger.warning(f"Task '{node_name}' references '{referenced}' in its inputs, which is not "
                           f"listed in 'depends_on'. Adding it to the dependencies.")
            upstream.append(referenced)
    upstream = list(dict.fromkeys(upstream))  # keep declaration order, drop repeats

    node = FlowNode(name=node_name, task=task, flow=flow_name, run_inputs=run_inputs,
                        run_params=run_block.get("params", {}), upstream=upstream,
                        init_params=init_params)
    return node, set(upstream)


def build_flow_nodes(flow_cfg: dict) -> tuple[dict, dict]:
    """
    Build all the nodes of a single flow.
    :param flow_cfg: dict
        Config of one flow, with at least the keys 'name' and 'nodes'.
    :return: tuple[dict, dict]
        The flow's nodes keyed by fully qualified name, and each node's dependency set.
    """
    flow_name = flow_cfg["name"]
    nodes, dependencies = {}, {}
    for node_cfg in flow_cfg["nodes"]:
        node, upstream = build_node(node_cfg, flow_name)
        if node.name in nodes:
            raise ValueError(f"Duplicate node name '{node_cfg['name']}' in flow '{flow_name}'")
        nodes[node.name] = node
        dependencies[node.name] = upstream
    return nodes, dependencies


def validate_dependencies(all_nodes: dict, dependencies: dict, flow_names: set):
    """
    Check that every dependency in the whole config points at a node that exists.

    Done once all flows are built, because a node may depend on one in another flow.
    `graphlib` cannot catch this: it treats an unknown dependency as a root node, not an error.
    :param all_nodes: dict
        Every node defined in the config, keyed by fully qualified name.
    :param dependencies: dict
        Mapping of fully qualified node name -> the set of names it depends on.
    :param flow_names: set
        Names of the flows defined, to tell a mistyped node from a mistyped flow.
    :return: None, raises ValueError on the first dependency that cannot be resolved.
    """
    for node_name, deps in dependencies.items():
        flow_name = node_name.split('.')[0]
        for dep in sorted(deps):
            if dep in all_nodes:
                continue
            dep_flow = dep.split('.')[0]
            if dep_flow == flow_name:
                raise ValueError(f"Node '{dep}' is referenced by flow '{flow_name}' but is "
                                 f"not defined in it")
            if dep_flow in flow_names:
                raise ValueError(f"Node '{dep}' is referenced by '{node_name}' but is not defined "
                                 f"in flow '{dep_flow}'")
            raise ValueError(f"Node '{dep}' is referenced by '{node_name}' but flow "
                             f"'{dep_flow}' is not defined in the config")


def dag_order(dependencies: dict) -> tuple:
    """
    Sort names into generations that may be run in order, by topological sort.
    :param dependencies: dict
        Mapping of name -> the set of names it depends on.
    :return: tuple
        Generations of names, each depending only on names in earlier generations.
    """
    sorter = graphlib.TopologicalSorter(dependencies)
    try:
        sorter.prepare()
    except graphlib.CycleError as e:
        raise ValueError(f"Cyclic dependency detected in flow: {e}")

    order = []
    while sorter.is_active():
        generation = sorter.get_ready()
        order.append(generation)
        sorter.done(*generation)
    return tuple(order)


def resolve_source(flow_cfg: dict, nodes: dict, node_order: tuple) -> str:
    """
    Work out which node drives a lazy flow, and check that it can.
    :param flow_cfg: dict
        Config of the lazy flow, which must carry a 'source' key.
    :param nodes: dict
        The flow's own nodes, keyed by fully qualified name.
    :param node_order: tuple
        The flow's node order, used to check the source starts it.
    :return: str
        Fully qualified name of the source node.
    """
    flow_name = flow_cfg["name"]
    source = flow_cfg["source"]
    source = source if '.' in source else f"{flow_name}.{source}"
    if source not in nodes:
        raise ValueError(f"Source node '{source}' for lazy flow '{flow_name}' is not defined")
    # The source drives every other node, so nothing in this flow may come before it
    if source not in node_order[0]:
        raise ValueError(f"Source node '{source}' for lazy flow '{flow_name}' must be in the "
                         f"first generation of the node order")
    return source


def build_flows(config: dict) -> tuple[dict, tuple]:
    """
    Build every flow described by a configuration, ready to execute.

    Nodes are built and their dependencies checked, then both the flows and the nodes within
    each flow are sorted into dependency order.
    :param config: dict
        The parsed flow configuration, with a 'flows' key.
    :return: tuple[dict, tuple]
        The flows keyed by name, and the generations of flow names in execution order.
    """
    all_nodes, all_dependencies, configs = {}, {}, {}
    for flow_cfg in config["flows"]:
        nodes, dependencies = build_flow_nodes(flow_cfg)
        duplicates = set(nodes) & set(all_nodes)
        if duplicates:
            raise ValueError(f"Duplicate flow name '{flow_cfg['name']}': nodes "
                             f"{sorted(duplicates)} are already defined")
        configs[flow_cfg["name"]] = flow_cfg
        all_nodes.update(nodes)
        all_dependencies.update(dependencies)

    validate_dependencies(all_nodes, all_dependencies, set(configs))

    # Order the flows by the dependencies that cross between them
    flow_dependencies = {name: set() for name in configs}
    for node_name, deps in all_dependencies.items():
        flow_name = node_name.split('.')[0]
        flow_dependencies[flow_name].update({dep.split('.')[0] for dep in deps} - {flow_name})
    flow_order = dag_order(flow_dependencies)
    logger.info(f"Flow order: {flow_order}")

    # Order each flow's own nodes. Dependencies on other flows are left out: those
    # flows run first, so their results are already in hand before this one starts.
    flows = {}
    for flow_name, flow_cfg in configs.items():
        own = {name: {dep for dep in all_dependencies[name] if dep.split('.')[0] == flow_name}
               for name in all_nodes if name.split('.')[0] == flow_name}
        node_order = dag_order(own)
        logger.info(f"Flow {flow_name} order: {node_order}")
        nodes = {name: all_nodes[name] for name in own}
        lazy = flow_cfg["lazy"]
        flows[flow_name] = Flow(
            name=flow_name,
            lazy=lazy,
            nodes=nodes,
            node_order=node_order,
            source=resolve_source(flow_cfg, nodes, node_order) if lazy else None,
        )

    return flows, flow_order
