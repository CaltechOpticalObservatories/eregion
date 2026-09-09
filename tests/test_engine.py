"""
Tests for eregion.pipeline.engine.PipelineEngine.

Covers:
  - parse_upstream_ref grammar (bare node, pipeline.node, pipeline.node.field, filter-call, errors)
  - build_pipeline_nodes: bare-reference auto-qualification and missing-dependency auto-add
  - execute_task: whole-result references, unknown upstream task, unknown field, non-callable filter
  - Eager pipeline end-to-end: field-reference wiring between real TaskResult subclasses
  - Lazy pipeline end-to-end: source-node iteration and combine() accumulation across batches
  - Filter-call reference end-to-end (a callable payload field, e.g. ImageBundle-style filtering)
"""

from copy import deepcopy

import pytest
from prefect.settings import PREFECT_API_URL, get_current_settings

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


# ---------------------------------------------------------------------------
# execute_task (unit-level, no Prefect involved)
# ---------------------------------------------------------------------------

def _node_dict(task, run_inputs, upstream=(), run_params=None):
    return {
        "task": task,
        "init_inputs": {},
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