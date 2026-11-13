"""
References from one flow node's input to another node's result.

A node wires an argument of its `run()` to the output of an upstream node by writing a
reference string in the flow yaml, e.g. ``images: masterbias_flow.bias_creator.data``.
`ResultRef` is that reference, parsed once when the node is built; from then on the runner
works with the parsed form and never looks at the text again.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
import re

from eregion.utils import configure_logger

logger = configure_logger(__name__)

# 'base(<quote>arg<quote>)' - the optional filter call at the end of a reference
_CALL_PATTERN = re.compile(r"^(?P<base>[^()]+)\((?P<quote>['\"])(?P<arg>.*)(?P=quote)\)$")


@dataclass(frozen=True, slots=True)
class ResultRef:
    """
    One reference to an upstream node's result, as written in a node's 'run.inputs' block.

    Accepted forms, all parsed by `parse`:
      - ``node``                          the whole TaskResult of a node in the same flow
      - ``flow.node``                 the whole TaskResult of a node in any flow
      - ``flow.node.field``           one named payload field of that result
      - ``flow.node.field('query')``  that field's value called with 'query', e.g. to filter
                                          an ImageBundle by a pandas query string

    Flow and node names must not contain '.'.

    :param node: str
        Fully qualified name of the node referred to, 'flow.node'.
    :param field: str | None
        Payload field of that node's result, or None for the whole TaskResult.
    :param query: str | None
        Argument the field's value is called with, or None when there is no filter call.
    """
    node: str
    field: str | None = None
    query: str | None = None

    @classmethod
    def parse(cls, text: str, default_flow: str) -> "ResultRef":
        """
        Parse a reference string, filling in the flow name when it is omitted.

        A reference written without a flow name is taken to point at the flow it appears
        in, which is logged so that an accidental omission is visible.
        :param text: str
            The reference exactly as written in the config.
        :param default_flow: str
            Flow the reference appears in, used when the reference names no flow.
        :return: ResultRef
            The parsed reference, with `node` always fully qualified.
        """
        if not isinstance(text, str):
            raise TypeError(f"An upstream reference must be a string, got {type(text).__name__}")

        call = _CALL_PATTERN.match(text)
        base = call.group("base") if call else text
        query = call.group("arg") if call else None

        parts = base.split(".")
        match len(parts):
            case 1:
                node, field = f"{default_flow}.{parts[0]}", None
                logger.warning(f"Reference '{text}' does not name a flow. Assuming it is from the same "
                               f"flow and reading it as '{default_flow}.{text}'.")
            case 2:
                node, field = base, None
            case _:
                node, field = ".".join(parts[:2]), ".".join(parts[2:])

        if query is not None and field is None:
            raise ValueError(f"Invalid upstream reference '{text}': a filter call requires a field name "
                             f"(use 'flow.node.field(...)').")

        return cls(node=node, field=field, query=query)

    def __str__(self) -> str:
        """:return: str, the reference in canonical, fully qualified form."""
        text = self.node if self.field is None else f"{self.node}.{self.field}"
        return text if self.query is None else f"{text}('{self.query}')"

    def resolve(self, results: Mapping[str, Any], apply_query: bool = True) -> Any:
        """
        Look up the value this reference points at among the results produced so far.
        :param results: Mapping[str, TaskResult]
            Results available to the node being resolved, keyed by fully qualified node name.
        :param apply_query: bool
            Whether to apply the filter call, if the reference has one. The dry run passes False
            because its payloads are placeholders with nothing to filter on.
        :return: Any
            The whole upstream TaskResult, or one of its payload fields, optionally filtered.
        """
        if self.node not in results:
            raise ValueError(f"references '{self.node}', which has not produced a result yet")

        result = results[self.node]
        if self.field is None:
            return result

        try:
            value = result[self.field]
        except KeyError as e:
            raise KeyError(f"references field '{self.field}', which '{self.node}' "
                           f"({type(result).__name__}) does not produce; it produces "
                           f"{list(result.keys())}") from e

        if self.query is not None and apply_query:
            if not callable(value):
                raise TypeError(f"references '{self.node}.{self.field}' with a filter call, but its "
                                f"value (type {type(value).__name__}) is not callable.")
            value = value(self.query)
        return value


def map_refs(spec: Any, func: Callable[[Any, str], Any], arg_path: str) -> Any:
    """
    Apply `func` to every reference in one input entry, keeping the entry's structure.

    An argument can be wired to a single upstream output (``images: pipe.node.field``) or to
    several at once by giving a list or dict of references, nested arbitrarily, e.g.
    ``task_results: [pipe.node_a, pipe.node_b]``. The task then receives a list/dict of the
    corresponding values, in the order or under the keys written in the config.
    :param spec: Any
        An input entry: a single reference, or a list/dict of them.
    :param func: Callable
        Called as ``func(reference, arg_path)`` for each reference; its return value takes that
        reference's place in the returned structure.
    :param arg_path: str
        Where `spec` sits within the node's inputs, used in messages (e.g. "task_results[0]").
    :return: Any
        `spec` with every reference replaced by the corresponding `func` return value.
    """
    match spec:
        case list():
            return [map_refs(item, func, f"{arg_path}[{i}]") for i, item in enumerate(spec)]
        case dict():
            return {key: map_refs(value, func, f"{arg_path}.{key}") for key, value in spec.items()}
        case _:
            return func(spec, arg_path)


def parse_inputs(run_inputs: dict, node_name: str) -> dict:
    """
    Parse a node's whole 'run.inputs' block into `ResultRef` objects.
    :param run_inputs: dict
        The block as written in the config, mapping argument name -> reference(s).
    :param node_name: str
        Fully qualified name of the node the inputs belong to; its flow fills in references
        that name none, and it names the node in error messages.
    :return: dict
        The same structure with every reference string replaced by a `ResultRef`.
    """
    flow_name = node_name.split('.')[0]

    def parse_one(text: Any, arg_path: str) -> ResultRef:
        try:
            return ResultRef.parse(text, flow_name)
        except (TypeError, ValueError) as e:
            raise type(e)(f"Input '{arg_path}' of node '{node_name}': {e}") from e

    return {arg_name: map_refs(spec, parse_one, arg_name) for arg_name, spec in run_inputs.items()}


def resolve_inputs(run_inputs: dict, results: Mapping[str, Any], node_name: str,
                   apply_queries: bool = True) -> dict:
    """
    Resolve a node's parsed inputs into the values its task should be called with.
    :param run_inputs: dict
        The node's inputs, holding `ResultRef` objects (see `parse_inputs`).
    :param results: Mapping[str, TaskResult]
        Results produced so far, keyed by fully qualified node name.
    :param node_name: str
        Fully qualified name of the node being resolved, used in error messages.
    :param apply_queries: bool
        Whether to apply filter calls; the dry run passes False.
    :return: dict
        Mapping of argument name -> resolved value(s), ready to pass to the task.
    """
    def resolve_one(ref: ResultRef, arg_path: str) -> Any:
        try:
            return ref.resolve(results, apply_query=apply_queries)
        except (ValueError, KeyError, TypeError) as e:
            message = e.args[0] if e.args else str(e)
            raise type(e)(f"Input '{arg_path}' for task '{node_name}' {message}") from e

    return {arg_name: map_refs(spec, resolve_one, arg_name) for arg_name, spec in run_inputs.items()}


def referenced_nodes(run_inputs: dict) -> list[str]:
    """
    List the nodes a node's inputs refer to, in the order they first appear.
    :param run_inputs: dict
        The node's inputs, holding `ResultRef` objects (see `parse_inputs`).
    :return: list[str]
        Fully qualified node names, de-duplicated, in first-seen order.
    """
    found = []
    for arg_name, spec in run_inputs.items():
        map_refs(spec, lambda ref, _: found.append(ref.node), arg_name)
    return list(dict.fromkeys(found))


def render_inputs(run_inputs: dict) -> dict:
    """
    Render parsed inputs back to their canonical reference strings.

    Used to record a node's wiring on the results it produces, which are serialised to JSON.
    :param run_inputs: dict
        The node's inputs, holding `ResultRef` objects (see `parse_inputs`).
    :return: dict
        The same structure with every `ResultRef` replaced by its canonical string.
    """
    return {arg_name: map_refs(spec, lambda ref, _: str(ref), arg_name)
            for arg_name, spec in run_inputs.items()}
