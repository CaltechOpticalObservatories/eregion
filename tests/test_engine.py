"""
Tests for eregion.pipeline.engine.PipelineEngine.

Covers:
  - parse_upstream_ref grammar (bare node, pipeline.node, pipeline.node.field, filter-call, errors)
  - build_pipeline_nodes: bare-reference auto-qualification and missing-dependency auto-add,
    references nested in list/dict inputs, rejection of 'init.inputs'
  - execute_task: whole-result references, list/dict inputs, unknown upstream task, unknown field,
    non-callable filter, non-reference input
  - Eager pipeline end-to-end: field-reference wiring between real TaskResult subclasses
  - Lazy pipeline end-to-end: source-node iteration and combine() accumulation across batches
  - Filter-call reference end-to-end (a callable payload field, e.g. ImageBundle-style filtering)
  - List input end-to-end: several upstream results collected into one task argument
  - Dry run from __init__: dummy results per node, unknown field/node, signature mismatches,
    filter calls left unapplied, and opting out with dry_run=False
"""

import pytest

from eregion.datamodels import TaskResult
from eregion.tasks.task import Task, LazyTask
from eregion.pipeline.engine import PipelineEngine, parse_upstream_ref


# ---------------------------------------------------------------------------
# Dummy tasks / results used as fixtures throughout this module.
# Referenced by dotted path (e.g. "tests.test_engine.EchoTask") when building
# pipeline configs, matching how real pipeline YAML configs reference tasks.
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
# parse_upstream_ref
# ---------------------------------------------------------------------------

def test_parse_upstream_ref_bare_node_is_qualified_with_pipeline_name():
    assert parse_upstream_ref("node", "PIPE") == ("PIPE.node", None, None)


def test_parse_upstream_ref_pipeline_node_has_no_field():
    assert parse_upstream_ref("PIPE.node", "PIPE") == ("PIPE.node", None, None)


def test_parse_upstream_ref_pipeline_node_field():
    assert parse_upstream_ref("PIPE.node.field", "PIPE") == ("PIPE.node", "field", None)


def test_parse_upstream_ref_filter_call():
    ref = 'PIPE.node.field(\'type == "bias"\')'
    assert parse_upstream_ref(ref, "PIPE") == ("PIPE.node", "field", 'type == "bias"')


def test_parse_upstream_ref_filter_call_without_field_raises():
    with pytest.raises(ValueError, match="filter call requires a field name"):
        parse_upstream_ref("node('q')", "PIPE")


# ---------------------------------------------------------------------------
# build_pipeline_nodes
# ---------------------------------------------------------------------------

def test_build_pipeline_nodes_auto_qualifies_bare_node_reference():
    pipeline_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_engine.EchoTask", "run": {"inputs": {"x": "A"}}},
        ],
    }
    nodes, node_dependencies = PipelineEngine.build_pipeline_nodes(pipeline_cfg)

    assert nodes["PIPE.B"]["run_inputs"]["x"] == "PIPE.A"
    assert nodes["PIPE.B"]["upstream"] == ["PIPE.A"]
    assert node_dependencies["PIPE.B"] == {"PIPE.A"}


def test_build_pipeline_nodes_auto_adds_missing_dependency():
    pipeline_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
            {
                "name": "B",
                "task": "tests.test_engine.EchoTask",
                "run": {"inputs": {"x": "PIPE.A.value"}},
                # depends_on intentionally omitted; should be inferred from the input reference.
            },
        ],
    }
    nodes, node_dependencies = PipelineEngine.build_pipeline_nodes(pipeline_cfg)

    assert nodes["PIPE.B"]["upstream"] == ["PIPE.A"]
    assert node_dependencies["PIPE.B"] == {"PIPE.A"}


def test_build_pipeline_nodes_walks_list_and_dict_inputs():
    pipeline_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 2}}},
            {
                "name": "C",
                "task": "tests.test_engine.ConsumeBox",
                # Bare references nested in a list/dict must be qualified and picked up as dependencies too.
                "run": {"inputs": {"images": ["A", "PIPE.B.value"], "extras": {"a": "A"}}},
            },
        ],
    }
    nodes, node_dependencies = PipelineEngine.build_pipeline_nodes(pipeline_cfg)

    assert nodes["PIPE.C"]["run_inputs"] == {"images": ["PIPE.A", "PIPE.B.value"], "extras": {"a": "PIPE.A"}}
    assert sorted(nodes["PIPE.C"]["upstream"]) == ["PIPE.A", "PIPE.B"]
    assert node_dependencies["PIPE.C"] == {"PIPE.A", "PIPE.B"}


def test_build_pipeline_nodes_rejects_init_inputs():
    pipeline_cfg = {
        "name": "PIPE",
        "nodes": [
            {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
            {"name": "B", "task": "tests.test_engine.ConsumeBox", "init": {"inputs": {"images": "PIPE.A.value"}}},
        ],
    }
    with pytest.raises(ValueError, match="init.inputs"):
        PipelineEngine.build_pipeline_nodes(pipeline_cfg)


# ---------------------------------------------------------------------------
# execute_task (unit-level, no Prefect involved)
# ---------------------------------------------------------------------------

def _node_dict(task, run_inputs, upstream=(), run_params=None):
    return {
        "task": task,
        "init_params": {},
        "run_inputs": run_inputs,
        "run_params": run_params or {},
        "upstream": list(upstream),
        "params": {},
    }


def test_execute_task_returns_typed_result_with_metadata_attached():
    node_dict = _node_dict(EchoTask, {}, run_params={"x": 5})
    node_dict["upstream"] = ["PIPE.A"]
    node_dict["params"] = {"run": {"x": 5}}

    result = PipelineEngine.execute_task("PIPE.A", node_dict, {})

    assert isinstance(result, EchoResult)
    assert result.value == 6
    assert result.upstream == ["PIPE.A"]
    assert result.params == {"run": {"x": 5}}


def test_execute_task_resolves_whole_result_reference():
    a_result = EchoResult(value=41)
    node_dict = _node_dict(ConsumeBox, {"images": "PIPE.A"}, upstream=["PIPE.A"])

    result = PipelineEngine.execute_task("PIPE.B", node_dict, {"PIPE.A": a_result})

    assert result.data is a_result


def test_execute_task_unknown_upstream_task_raises_value_error():
    node_dict = _node_dict(EchoTask, {"x": "PIPE.missing.value"})

    with pytest.raises(ValueError, match="not found in results"):
        PipelineEngine.execute_task("PIPE.B", node_dict, {})


def test_execute_task_unknown_field_raises_key_error():
    node_dict = _node_dict(EchoTask, {"x": "PIPE.A.nonexistent"}, upstream=["PIPE.A"])

    with pytest.raises(KeyError):
        PipelineEngine.execute_task("PIPE.B", node_dict, {"PIPE.A": EchoResult(value=1)})


def test_execute_task_filter_call_on_noncallable_field_raises_type_error():
    node_dict = _node_dict(EchoTask, {"x": "PIPE.A.value('q')"}, upstream=["PIPE.A"])

    with pytest.raises(TypeError, match="not callable"):
        PipelineEngine.execute_task("PIPE.B", node_dict, {"PIPE.A": EchoResult(value=1)})


def test_execute_task_resolves_a_list_of_references_into_a_list():
    a_result, b_result = EchoResult(value=1), EchoResult(value=2)
    node_dict = _node_dict(ConsumeBox, {"images": ["PIPE.A", "PIPE.B.value"]}, upstream=["PIPE.A", "PIPE.B"])

    result = PipelineEngine.execute_task("PIPE.C", node_dict, {"PIPE.A": a_result, "PIPE.B": b_result})

    assert result.data == [a_result, 2]


def test_execute_task_resolves_a_dict_of_references_into_a_dict():
    node_dict = _node_dict(ConsumeBox, {"images": {"first": "PIPE.A.value", "second": ["PIPE.B.value"]}},
                           upstream=["PIPE.A", "PIPE.B"])

    result = PipelineEngine.execute_task("PIPE.C", node_dict,
                                         {"PIPE.A": EchoResult(value=1), "PIPE.B": EchoResult(value=2)})

    assert result.data == {"first": 1, "second": [2]}


def test_execute_task_non_reference_input_raises_type_error():
    node_dict = _node_dict(EchoTask, {"x": 5})

    with pytest.raises(TypeError, match="must be an upstream reference string"):
        PipelineEngine.execute_task("PIPE.B", node_dict, {})


# ---------------------------------------------------------------------------
# Eager pipeline, end-to-end
# ---------------------------------------------------------------------------

def test_eager_pipeline_resolves_field_references_and_returns_typed_results():
    config = {
        "debug": False,
        "pipelines": [
            {
                "name": "PIPE_E",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 5}}},
                    {
                        "name": "B",
                        "task": "tests.test_engine.EchoTask",
                        "run": {"inputs": {"x": "PIPE_E.A.value"}},
                        "depends_on": ["A"],
                    },
                ],
            }
        ],
    }

    engine = PipelineEngine(config)
    engine.run()

    a = engine.results["PIPE_E.A"]
    b = engine.results["PIPE_E.B"]

    assert isinstance(a, EchoResult)
    assert isinstance(b, EchoResult)
    assert a.value == 6
    assert b.value == 7
    assert a.upstream == []
    assert b.upstream == ["PIPE_E.A"]


# ---------------------------------------------------------------------------
# Lazy pipeline, end-to-end
# ---------------------------------------------------------------------------

def test_lazy_pipeline_combines_batches_across_iterations():
    config = {
        "debug": False,
        "pipelines": [
            {
                "name": "PIPE_L",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    {"name": "src_node", "task": "tests.test_engine.SourceLazyTask", "run": {"params": {"n": 3}}},
                    {
                        "name": "D",
                        "task": "tests.test_engine.EchoTask",
                        "run": {"inputs": {"x": "PIPE_L.src_node.value"}},
                        "depends_on": ["src_node"],
                    },
                ],
            }
        ],
    }

    engine = PipelineEngine(config)
    engine.run()

    src = engine.results["PIPE_L.src_node"]
    d = engine.results["PIPE_L.D"]

    assert isinstance(src, SourceResult)
    assert isinstance(d, EchoResult)
    assert src.value == [0, 1, 2]
    assert d.value == [1, 2, 3]


# ---------------------------------------------------------------------------
# Filter-call reference, end-to-end
# ---------------------------------------------------------------------------

def test_filter_call_reference_invokes_callable_field_end_to_end():
    config = {
        "debug": False,
        "pipelines": [
            {
                "name": "PIPE",
                "lazy": False,
                "nodes": [
                    {"name": "box", "task": "tests.test_engine.MakeBox"},
                    {
                        "name": "consumer",
                        "task": "tests.test_engine.ConsumeBox",
                        "run": {"inputs": {"images": "PIPE.box.data('type == \"bias\"')"}},
                        "depends_on": ["box"],
                    },
                ],
            }
        ],
    }

    engine = PipelineEngine(config)
    engine.run()

    result = engine.results["PIPE.consumer"].data
    assert result.tag == 'all:type == "bias"'

# ---------------------------------------------------------------------------
# List/dict inputs, end-to-end
# ---------------------------------------------------------------------------

def test_pipeline_feeds_a_list_of_upstream_results_to_a_single_argument():
    config = {
        "debug": False,
        "pipelines": [
            {
                "name": "PIPE_M",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
                    {"name": "B", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 10}}},
                    {
                        "name": "collect",
                        "task": "tests.test_engine.ConsumeBox",
                        # e.g. bookkeeping.SaveResult, which takes a list of whole TaskResults
                        "run": {"inputs": {"images": ["PIPE_M.A", "PIPE_M.B"]}},
                        "depends_on": ["A", "B"],
                    },
                ],
            }
        ],
    }

    engine = PipelineEngine(config)
    engine.run()

    collected = engine.results["PIPE_M.collect"].data
    assert [result.value for result in collected] == [2, 11]
    assert sorted(engine.results["PIPE_M.collect"].upstream) == ["PIPE_M.A", "PIPE_M.B"]


# ---------------------------------------------------------------------------
# Dry run (called from __init__)
# ---------------------------------------------------------------------------

def _two_node_config(consumer_run):
    return {
        "pipelines": [
            {
                "name": "PIPE_D",
                "lazy": False,
                "nodes": [
                    {"name": "A", "task": "tests.test_engine.EchoTask", "run": {"params": {"x": 1}}},
                    {"name": "consumer", "task": "tests.test_engine.ConsumeBox", "run": consumer_run,
                     "depends_on": ["A"]},
                ],
            }
        ],
    }


def test_dry_run_produces_a_dummy_result_for_every_node():
    engine = PipelineEngine(_two_node_config({"inputs": {"images": "PIPE_D.A.value"}}))

    assert set(engine.dummy_results) == {"PIPE_D.A", "PIPE_D.consumer"}
    assert isinstance(engine.dummy_results["PIPE_D.A"], EchoResult)
    assert list(engine.dummy_results["PIPE_D.consumer"].keys()) == ["data"]
    assert engine.results == {}  # nothing was actually executed


def test_dry_run_rejects_a_reference_to_a_field_the_upstream_task_does_not_produce():
    with pytest.raises(KeyError, match="does not produce"):
        PipelineEngine(_two_node_config({"inputs": {"images": "PIPE_D.A.nonexistent"}}))


def test_dry_run_rejects_inputs_that_do_not_match_the_run_signature():
    with pytest.raises(TypeError, match=r"do not match ConsumeBox.run\(\)"):
        PipelineEngine(_two_node_config({"inputs": {"imagez": "PIPE_D.A.value"}}))


def test_dry_run_rejects_a_reference_to_an_undefined_node():
    with pytest.raises(ValueError, match="not defined in it"):
        PipelineEngine(_two_node_config({"inputs": {"images": "PIPE_D.missing.value"}}))


def test_dry_run_checks_the_lazy_run_signature_of_a_source_node():
    config = {
        "pipelines": [
            {
                "name": "PIPE_DL",
                "lazy": True,
                "source": "src_node",
                "nodes": [
                    # SourceLazyTask.lazy_run() takes 'n', not 'm'
                    {"name": "src_node", "task": "tests.test_engine.SourceLazyTask", "run": {"params": {"m": 3}}},
                ],
            }
        ],
    }
    with pytest.raises(TypeError, match=r"do not match SourceLazyTask.lazy_run\(\)"):
        PipelineEngine(config)


def test_dry_run_leaves_filter_calls_unapplied_on_dummy_payloads():
    config = {
        "pipelines": [
            {
                "name": "PIPE_DF",
                "lazy": False,
                "nodes": [
                    {"name": "box", "task": "tests.test_engine.MakeBox"},
                    {
                        "name": "consumer",
                        "task": "tests.test_engine.ConsumeBox",
                        "run": {"inputs": {"images": "PIPE_DF.box.data('type == \"bias\"')"}},
                        "depends_on": ["box"],
                    },
                ],
            }
        ],
    }
    engine = PipelineEngine(config)  # dummy BoxResult.data is None, the filter call must not be attempted

    assert engine.dummy_results["PIPE_DF.consumer"].data is None


def test_dry_run_can_be_disabled():
    config = _two_node_config({"inputs": {"images": "PIPE_D.A.nonexistent"}})
    engine = PipelineEngine(config, dry_run=False)  # builds despite the broken wiring

    assert engine.dummy_results == {}
    with pytest.raises(KeyError, match="does not produce"):
        engine.run()
