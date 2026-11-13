"""
Tests for eregion.flow.runner.FlowRunner.

Covers:
  - ResultRef.parse grammar (bare node, flow.node, flow.node.field, filter-call, errors)
  - build_flow_nodes: bare-reference auto-qualification and missing-dependency auto-add,
    references nested in list/dict inputs, rejection of 'init.inputs'
  - execute_task: whole-result references, list/dict inputs, unknown upstream task, unknown field,
    non-callable filter, non-reference input
  - Eager flow end-to-end: field-reference wiring between real TaskResult subclasses
  - Lazy flow end-to-end: source-node iteration and combine() accumulation across batches
  - Filter-call reference end-to-end (a callable payload field, e.g. ImageBundle-style filtering)
  - List input end-to-end: several upstream results collected into one task argument
  - Dry run from __init__: dummy results per node, unknown field/node, signature mismatches,
    filter calls left unapplied, and opting out with dry_run=False
"""

import pytest

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task, LazyTask
from eregion.flow.runner import FlowRunner, execute_task
from eregion.flow.nodes import FlowNode, build_flow_nodes
from eregion.flow.refs import ResultRef, parse_inputs


# ---------------------------------------------------------------------------
# Dummy tasks / results used as fixtures throughout this module.
# Referenced by dotted path (e.g. "tests.test_flow.EchoTask") when building
# flow configs, matching how real flow YAML configs reference tasks.
# ---------------------------------------------------------------------------

class EchoResult(TaskResult):
    value: int = 0


class EchoTask(Task):
    task_result = EchoResult

    def run(self, x):
        return EchoResult(value=x + 1)


class SourceResult(TaskResult):
    value: int = 0


class SourceLazyTask(LazyTask):
    task_result = SourceResult

    def lazy_run(self, n=3):
        for i in range(n):
            yield SourceResult(value=i)


class CountingResult(TaskResult):
    value: int = 0
    instance_id: int = 0


class CountingTask(Task):
    """Records the identity of the instance that ran, to show instance reuse across batches."""
    task_result = CountingResult

    def run(self, x):
        return CountingResult(value=x, instance_id=id(self))


class FilterableBox:
    """Minimal stand-in for ImageBundle's callable pandas-query filtering."""

    def __init__(self, tag):
        self.tag = tag

    def __call__(self, query):
        return FilterableBox(f"{self.tag}:{query}")


class BoxResult(TaskResult):
    data: object = None


class MakeBox(Task):
    task_result = BoxResult

    def run(self):
        return BoxResult(data=FilterableBox("all"))


class ConsumeBox(Task):
    task_result = BoxResult

    def run(self, images):
        return BoxResult(data=images)


# ---------------------------------------------------------------------------
# ResultRef.parse
# ---------------------------------------------------------------------------

def test_result_ref_bare_node_is_qualified_with_flow_name():
    assert ResultRef.parse("node", "PIPE") == ResultRef("PIPE.node")


def test_result_ref_flow_node_has_no_field():
    assert ResultRef.parse("PIPE.node", "PIPE") == ResultRef("PIPE.node")


def test_result_ref_flow_node_field():
    assert ResultRef.parse("PIPE.node.field", "PIPE") == ResultRef("PIPE.node", "field")


def test_result_ref_filter_call():
    ref = 'PIPE.node.field(\'type == "bias"\')'
    assert ResultRef.parse(ref, "PIPE") == ResultRef("PIPE.node", "field", 'type == "bias"')


def test_result_ref_filter_call_without_field_raises():
    with pytest.raises(ValueError, match="filter call requires a field name"):
        ResultRef.parse("node('q')", "PIPE")


def test_result_ref_non_string_raises():
    with pytest.raises(TypeError, match="must be a string"):
        ResultRef.parse(5, "PIPE")


@pytest.mark.parametrize("text", [
    "PIPE.node",
    "PIPE.node.field",
    'PIPE.node.field(\'type == "bias"\')',
])
def test_result_ref_round_trips_through_its_canonical_text(text):
    # str(ref) is what gets recorded in a result's params and serialised to JSON
    assert str(ResultRef.parse(text, "PIPE")) == text


# ---------------------------------------------------------------------------
# build_flow_nodes
# ---------------------------------------------------------------------------

def test_build_flow_nodes_auto_qualifies_bare_node_reference():
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_flow.EchoTask", "run": {"inputs": {"x": "A"}}},
        ],
    }
    nodes, node_dependencies = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.B"].run_inputs["x"] == ResultRef("PIPE.A")
    assert nodes["PIPE.B"].upstream == ["PIPE.A"]
    assert node_dependencies["PIPE.B"] == {"PIPE.A"}


def test_build_flow_nodes_auto_adds_missing_dependency():
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {
                "name": "B",
                "task": "tests.test_flow.EchoTask",
                "run": {"inputs": {"x": "PIPE.A.value"}},
                # depends_on intentionally omitted; should be inferred from the input reference.
            },
        ],
    }
    nodes, node_dependencies = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.B"].upstream == ["PIPE.A"]
    assert node_dependencies["PIPE.B"] == {"PIPE.A"}


def test_build_flow_nodes_walks_list_and_dict_inputs():
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 2}}},
            {
                "name": "C",
                "task": "tests.test_flow.ConsumeBox",
                # Bare references nested in a list/dict must be qualified and picked up as dependencies too.
                "run": {"inputs": {"images": ["A", "PIPE.B.value"], "extras": {"a": "A"}}},
            },
        ],
    }
    nodes, node_dependencies = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.C"].run_inputs == {"images": [ResultRef("PIPE.A"), ResultRef("PIPE.B", "value")],
                                          "extras": {"a": ResultRef("PIPE.A")}}
    assert sorted(nodes["PIPE.C"].upstream) == ["PIPE.A", "PIPE.B"]
    assert node_dependencies["PIPE.C"] == {"PIPE.A", "PIPE.B"}


def test_build_flow_nodes_rejects_init_inputs():
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_flow.ConsumeBox", "init": {"inputs": {"images": "PIPE.A.value"}}},
        ],
    }
    with pytest.raises(ValueError, match="init.inputs"):
        build_flow_nodes(flow_cfg)


def test_build_flow_nodes_keeps_a_fully_qualified_dependency():
    # Writing 'depends_on' out in full is what every real config does, so it must survive verbatim.
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {
                "name": "B",
                "task": "tests.test_flow.EchoTask",
                "run": {"inputs": {"x": "PIPE.A.value"}},
                "depends_on": ["PIPE.A"],
            },
        ],
    }
    nodes, node_dependencies = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.B"].upstream == ["PIPE.A"]
    assert node_dependencies["PIPE.B"] == {"PIPE.A"}


def test_build_flow_nodes_keeps_a_cross_flow_dependency():
    # A dependency on another flow can only ever be written qualified.
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {
                "name": "B",
                "task": "tests.test_flow.EchoTask",
                "run": {"inputs": {"x": "OTHER.A.value"}},
                "depends_on": ["OTHER.A"],
            },
        ],
    }
    nodes, node_dependencies = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.B"].upstream == ["OTHER.A"]
    assert node_dependencies["PIPE.B"] == {"OTHER.A"}


def test_build_flow_nodes_dedupes_a_dependency_named_twice():
    # 'A' in depends_on and 'PIPE.A' in an input are the same node, and must be recorded once.
    flow_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
            {
                "name": "B",
                "task": "tests.test_flow.EchoTask",
                "run": {"inputs": {"x": "PIPE.A.value"}},
                "depends_on": ["A"],
            },
        ],
    }
    nodes, _ = build_flow_nodes(flow_cfg)

    assert nodes["PIPE.B"].upstream == ["PIPE.A"]


def test_build_flow_nodes_holds_an_initialised_task_instance():
    flow_cfg = {
        "name": "PIPE",
        "nodes": [{"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}}],
    }
    nodes, _ = build_flow_nodes(flow_cfg)

    assert isinstance(nodes["PIPE.A"].task, EchoTask)
    assert nodes["PIPE.A"].task.name == "PIPE.A"


def test_build_flow_nodes_reports_a_bad_init_param():
    flow_cfg = {
        "name": "PIPE",
        # DeleteResult.__init__ takes 'name' only, so an extra init param cannot be applied
        "nodes": [{"name": "A", "task": "eregion.tasks.bookkeeping.DeleteResult",
                   "init": {"params": {"nonexistent": 1}}}],
    }
    with pytest.raises(TypeError, match="cannot initialise DeleteResult"):
        build_flow_nodes(flow_cfg)


# ---------------------------------------------------------------------------
# validate_dependencies
# ---------------------------------------------------------------------------

def _single_node_config(nodes, name="PIPE_V"):
    return {"flows": [{"name": name, "lazy": False, "nodes": nodes}]}


def test_validate_dependencies_rejects_an_unknown_node_in_the_same_flow():
    config = _single_node_config([
        {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
        {"name": "B", "task": "tests.test_flow.EchoTask", "run": {"inputs": {"x": "PIPE_V.typo.value"}}},
    ])
    with pytest.raises(ValueError, match="not defined in it"):
        FlowRunner(config)


def test_validate_dependencies_rejects_an_unknown_node_in_a_known_flow():
    config = {
        "flows": [
            {"name": "P1", "lazy": False,
             "nodes": [{"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}}]},
            {"name": "P2", "lazy": False,
             "nodes": [{"name": "B", "task": "tests.test_flow.EchoTask",
                        "run": {"inputs": {"x": "P1.typo.value"}}}]},
        ]
    }
    with pytest.raises(ValueError, match="not defined in flow 'P1'"):
        FlowRunner(config)


def test_validate_dependencies_rejects_an_unknown_flow():
    config = _single_node_config([
        {"name": "B", "task": "tests.test_flow.EchoTask", "run": {"inputs": {"x": "NOPE.A.value"}}},
    ])
    with pytest.raises(ValueError, match="flow 'NOPE' is not defined"):
        FlowRunner(config)


def test_a_bad_lazy_source_is_rejected_at_build_time_even_without_a_dry_run():
    config = {
        "flows": [
            {"name": "PIPE_S", "lazy": True, "source": "typo",
             "nodes": [{"name": "src_node", "task": "tests.test_flow.SourceLazyTask"}]},
        ]
    }
    with pytest.raises(ValueError, match="Source node 'PIPE_S.typo'"):
        FlowRunner(config, dry_run=False)


# ---------------------------------------------------------------------------
# execute_task (unit-level)
# ---------------------------------------------------------------------------

def _node(task, run_inputs, name="PIPE.B", upstream=(), run_params=None):
    """Build a FlowNode as the runner does: initialised task, inputs parsed into ResultRefs."""
    return FlowNode(
        name=name,
        task=task(name=name),
        flow=name.split(".")[0],
        run_inputs=parse_inputs(run_inputs, name),
        run_params=run_params or {},
        upstream=list(upstream),
    )


def test_execute_task_returns_typed_result_with_metadata_attached():
    node = _node(EchoTask, {}, name="PIPE.A", upstream=["PIPE.A"], run_params={"x": 5})

    result = execute_task(node, {})

    assert isinstance(result, EchoResult)
    assert result.value == 6
    assert result.upstream == ["PIPE.A"]
    # 'params' is derived from the node's init/run configuration
    assert result.params == {"init": {}, "run": {"x": 5}}


def test_execute_task_resolves_whole_result_reference():
    a_result = EchoResult(value=41)
    node = _node(ConsumeBox, {"images": "PIPE.A"}, upstream=["PIPE.A"])

    result = execute_task(node, {"PIPE.A": a_result})

    assert result.data is a_result


def test_execute_task_unknown_upstream_task_raises_value_error():
    node = _node(EchoTask, {"x": "PIPE.missing.value"})

    with pytest.raises(ValueError, match="has not produced a result yet"):
        execute_task(node, {})


def test_execute_task_unknown_field_raises_key_error():
    node = _node(EchoTask, {"x": "PIPE.A.nonexistent"}, upstream=["PIPE.A"])

    with pytest.raises(KeyError):
        execute_task(node, {"PIPE.A": EchoResult(value=1)})


def test_execute_task_filter_call_on_noncallable_field_raises_type_error():
    node = _node(EchoTask, {"x": "PIPE.A.value('q')"}, upstream=["PIPE.A"])

    with pytest.raises(TypeError, match="not callable"):
        execute_task(node, {"PIPE.A": EchoResult(value=1)})


def test_execute_task_resolves_a_list_of_references_into_a_list():
    a_result, b_result = EchoResult(value=1), EchoResult(value=2)
    node = _node(ConsumeBox, {"images": ["PIPE.A", "PIPE.B.value"]}, upstream=["PIPE.A", "PIPE.B"])

    result = execute_task(node, {"PIPE.A": a_result, "PIPE.B": b_result})

    assert result.data == [a_result, 2]


def test_execute_task_resolves_a_dict_of_references_into_a_dict():
    node = _node(ConsumeBox, {"images": {"first": "PIPE.A.value", "second": ["PIPE.B.value"]}},
                           upstream=["PIPE.A", "PIPE.B"])

    result = execute_task(node,
                                         {"PIPE.A": EchoResult(value=1), "PIPE.B": EchoResult(value=2)})

    assert result.data == {"first": 1, "second": [2]}


def test_non_reference_input_is_rejected_when_the_node_is_built():
    # Inputs are parsed up front, so a bad input shape fails at build time rather than at run time
    flow_cfg = {
        "name": "PIPE",
        "nodes": [{"name": "B", "task": "tests.test_flow.EchoTask", "run": {"inputs": {"x": 5}}}],
    }
    with pytest.raises(TypeError, match="must be a string"):
        build_flow_nodes(flow_cfg)


# ---------------------------------------------------------------------------
# Eager flow, end-to-end
# ---------------------------------------------------------------------------

def test_eager_flow_resolves_field_references_and_returns_typed_results():
    config = {
        "debug": False,
        "flows": [
            {
                "name": "PIPE_E",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 5}}},
                    {
                        "name": "B",
                        "task": "tests.test_flow.EchoTask",
                        "run": {"inputs": {"x": "PIPE_E.A.value"}},
                        "depends_on": ["A"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    runner.run()

    a = runner.results["PIPE_E.A"]
    b = runner.results["PIPE_E.B"]

    assert isinstance(a, EchoResult)
    assert isinstance(b, EchoResult)
    assert a.value == 6
    assert b.value == 7
    assert a.upstream == []
    assert b.upstream == ["PIPE_E.A"]


# ---------------------------------------------------------------------------
# Lazy flow, end-to-end
# ---------------------------------------------------------------------------

def test_lazy_flow_combines_batches_across_iterations():
    config = {
        "debug": False,
        "flows": [
            {
                "name": "PIPE_L",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    {"name": "src_node", "task": "tests.test_flow.SourceLazyTask", "run": {"params": {"n": 3}}},
                    {
                        "name": "D",
                        "task": "tests.test_flow.EchoTask",
                        "run": {"inputs": {"x": "PIPE_L.src_node.value"}},
                        "depends_on": ["src_node"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    runner.run()

    src = runner.results["PIPE_L.src_node"]
    d = runner.results["PIPE_L.D"]

    assert isinstance(src, SourceResult)
    assert isinstance(d, EchoResult)
    assert src.value == [0, 1, 2]
    assert d.value == [1, 2, 3]


def test_lazy_flow_reuses_one_task_instance_across_batches():
    # The task is built once with the node, so every batch is handled by the same instance.
    config = {
        "flows": [
            {
                "name": "PIPE_LI",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    {"name": "src_node", "task": "tests.test_flow.SourceLazyTask", "run": {"params": {"n": 3}}},
                    {
                        "name": "counter",
                        "task": "tests.test_flow.CountingTask",
                        "run": {"inputs": {"x": "PIPE_LI.src_node.value"}},
                        "depends_on": ["src_node"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    expected_id = id(runner.nodes["PIPE_LI.counter"].task)
    runner.run()

    result = runner.results["PIPE_LI.counter"]
    assert result.value == [0, 1, 2]                      # three batches were processed
    assert result.instance_id == [expected_id] * 3        # all by the node's single task instance


def test_lazy_flow_runs_an_independent_node_alongside_the_source():
    # A lazy flow may hold a node with no dependencies besides the source. Only the source is
    # driven by the batch iterator; every other node, including this one, runs per batch.
    config = {
        "flows": [
            {
                "name": "PIPE_LN",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    {"name": "src_node", "task": "tests.test_flow.SourceLazyTask", "run": {"params": {"n": 2}}},
                    {"name": "independent", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 10}}},
                    {
                        "name": "down",
                        "task": "tests.test_flow.EchoTask",
                        "run": {"inputs": {"x": "PIPE_LN.src_node.value"}},
                        "depends_on": ["src_node"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    runner.run()

    assert runner.results["PIPE_LN.src_node"].value == [0, 1]
    assert runner.results["PIPE_LN.down"].value == [1, 2]
    assert runner.results["PIPE_LN.independent"].value == [11, 11]


# ---------------------------------------------------------------------------
# Filter-call reference, end-to-end
# ---------------------------------------------------------------------------

def test_filter_call_reference_invokes_callable_field_end_to_end():
    config = {
        "debug": False,
        "flows": [
            {
                "name": "PIPE",
                "lazy": False,
                "nodes": [
                    {"name": "box", "task": "tests.test_flow.MakeBox"},
                    {
                        "name": "consumer",
                        "task": "tests.test_flow.ConsumeBox",
                        "run": {"inputs": {"images": "PIPE.box.data('type == \"bias\"')"}},
                        "depends_on": ["box"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    runner.run()

    result = runner.results["PIPE.consumer"].data
    assert result.tag == 'all:type == "bias"'

# ---------------------------------------------------------------------------
# List/dict inputs, end-to-end
# ---------------------------------------------------------------------------

def test_flow_feeds_a_list_of_upstream_results_to_a_single_argument():
    config = {
        "debug": False,
        "flows": [
            {
                "name": "PIPE_M",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
                    {"name": "B", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 10}}},
                    {
                        "name": "collect",
                        "task": "tests.test_flow.ConsumeBox",
                        # e.g. bookkeeping.SaveResult, which takes a list of whole TaskResults
                        "run": {"inputs": {"images": ["PIPE_M.A", "PIPE_M.B"]}},
                        "depends_on": ["A", "B"],
                    },
                ],
            }
        ],
    }

    runner = FlowRunner(config)
    runner.run()

    collected = runner.results["PIPE_M.collect"].data
    assert [result.value for result in collected] == [2, 11]
    assert sorted(runner.results["PIPE_M.collect"].upstream) == ["PIPE_M.A", "PIPE_M.B"]


# ---------------------------------------------------------------------------
# Dry run (called from __init__)
# ---------------------------------------------------------------------------

def _two_node_config(consumer_run):
    return {
        "flows": [
            {
                "name": "PIPE_D",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_flow.EchoTask", "run": {"params": {"x": 1}}},
                    {"name": "consumer", "task": "tests.test_flow.ConsumeBox", "run": consumer_run,
                     "depends_on": ["A"]},
                ],
            }
        ],
    }


def test_dry_run_produces_a_dummy_result_for_every_node():
    runner = FlowRunner(_two_node_config({"inputs": {"images": "PIPE_D.A.value"}}))

    assert set(runner.dummy_results) == {"PIPE_D.A", "PIPE_D.consumer"}
    assert isinstance(runner.dummy_results["PIPE_D.A"], EchoResult)
    assert list(runner.dummy_results["PIPE_D.consumer"].keys()) == ["data"]
    assert runner.results == {}  # nothing was actually executed


def test_dry_run_rejects_a_reference_to_a_field_the_upstream_task_does_not_produce():
    with pytest.raises(KeyError, match="does not produce"):
        FlowRunner(_two_node_config({"inputs": {"images": "PIPE_D.A.nonexistent"}}))


def test_dry_run_rejects_inputs_that_do_not_match_the_run_signature():
    with pytest.raises(TypeError, match=r"do not match ConsumeBox.run\(\)"):
        FlowRunner(_two_node_config({"inputs": {"imagez": "PIPE_D.A.value"}}))


# A reference to an undefined node is now caught before the dry run; see the
# 'validate_dependencies' tests above, which cover all three kinds of unresolvable reference.


def test_dry_run_checks_the_lazy_run_signature_of_a_source_node():
    config = {
        "flows": [
            {
                "name": "PIPE_DL",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    # SourceLazyTask.lazy_run() takes 'n', not 'm'
                    {"name": "src_node", "task": "tests.test_flow.SourceLazyTask", "run": {"params": {"m": 3}}},
                ],
            }
        ],
    }
    with pytest.raises(TypeError, match=r"do not match SourceLazyTask.lazy_run\(\)"):
        FlowRunner(config)


def test_dry_run_leaves_filter_calls_unapplied_on_dummy_payloads():
    config = {
        "flows": [
            {
                "name": "PIPE_DF",
                "lazy": False,
                "nodes": [
                    {"name": "box", "task": "tests.test_flow.MakeBox"},
                    {
                        "name": "consumer",
                        "task": "tests.test_flow.ConsumeBox",
                        "run": {"inputs": {"images": "PIPE_DF.box.data('type == \"bias\"')"}},
                        "depends_on": ["box"],
                    },
                ],
            }
        ],
    }
    runner = FlowRunner(config)  # dummy BoxResult.data is None, the filter call must not be attempted

    assert runner.dummy_results["PIPE_DF.consumer"].data is None


def test_dry_run_can_be_disabled():
    config = _two_node_config({"inputs": {"images": "PIPE_D.A.nonexistent"}})
    runner = FlowRunner(config, dry_run=False)  # builds despite the broken wiring

    assert runner.dummy_results == {}
    with pytest.raises(KeyError, match="does not produce"):
        runner.run()
