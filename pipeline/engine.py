from typing import Iterator
from tasks.task import TaskResult
from configs.config import PipelineConfig

from prefect import task, flow
import importlib
import logging


class PipelineEngine:
    def __init__(self, pipeline_config: str | dict):
        self.pipeline_config = PipelineConfig(pipeline_config)
        self.pipeline = self.build_flow()

    def update(self, new_config: str | dict):
        self.pipeline_config = PipelineConfig(new_config)
        self.pipeline = self.build_flow()

    @staticmethod
    def load_task_class(path: str):
        module, cls = path.rsplit(".", 1)
        return getattr(importlib.import_module(module), cls)

    @staticmethod
    def resolve_inputs(input_spec: dict, results: dict) -> dict:
        resolved = {}
        for arg_name, ref in input_spec.items():
            task_name, _, key = ref.partition(".data.")
            if task_name not in results:
                raise ValueError(f"Upstream task '{task_name}' not found")
            resolved[arg_name] = results[task_name].data[key]
        return resolved

    @staticmethod
    def execute_task(
        task_cls,
        name: str,
        init_params: dict,
        run_inputs: dict,
        run_params: dict,
        upstream: list[str],
        lazy: bool = False,
        debug: bool = False,
    ) -> TaskResult | Iterator[TaskResult]:

        eregion_task = task_cls(name=name, **init_params)
        eregion_task.set_logging_level(logging.DEBUG if debug else logging.INFO)

        merged_params = {
            "init": init_params,
            "run": run_params,
            "lazy": lazy
        }

        if lazy:
            if not hasattr(eregion_task, "lazy_run"):
                raise TypeError(f"Task '{name}' does not support lazy execution")

            def generator():
                for item in eregion_task.lazy_run(**run_inputs, **run_params):
                    if not isinstance(item, dict):
                        raise TypeError("lazy_run must yield dicts")
                    yield TaskResult(
                        task_name=name,
                        data=item,
                        params=merged_params,
                        upstream=upstream,
                    )
            return generator()

        # eager execution
        result = eregion_task.run(**run_inputs, **run_params)
        if not isinstance(result, dict):
            raise TypeError("run() must return a dict")
        return TaskResult(
            task_name=name,
            data=result,
            params=merged_params,
            upstream=upstream,
        )

    def build_flow(self):
        cfg = self.pipeline_config.config

        @flow
        def pipeline_flow():
            results = {}

            for node in cfg["pipeline"]:
                name = node["name"]
                task_cls = self.load_task_class(node["task"])

                lazy = node.get("lazy", False)
                init_params = node.get("init", {})

                run_block = node.get("run", {})
                input_spec = run_block.get("inputs", {})
                run_params = run_block.get("params", {})

                # upstream_results = {
                #     dep: results[dep]
                #     for dep in node.get("depends_on", [])
                # }
                upstream = node.get("depends_on", [])

                run_inputs = self.resolve_inputs(input_spec, results)

                @task(name=name)
                def prefect_task():
                    return self.execute_task(
                        task_cls=task_cls,
                        name=name,
                        init_params=init_params,
                        run_inputs=run_inputs,
                        run_params=run_params,
                        upstream=upstream,
                        lazy=lazy,
                        debug=cfg.get("debug", False),
                    )

                results[name] = prefect_task()

            return results

        return pipeline_flow


    ## Execute the pipeline
    def run(self):
        return self.pipeline()

