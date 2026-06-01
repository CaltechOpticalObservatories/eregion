from collections.abc import Iterator
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing import Any
import os
import multiprocessing
from astropy.time import Time
import inspect

from utils import configure_logger, load_class

# Base abstract class for tasks, should have a call method for direct execution and a run method for pipeline workflows
class Task(ABC):
    required_keys = []

    def __init__(self, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.n_jobs = kwargs.get('n_jobs') or self.get_default_n_jobs()
        self.logger = configure_logger(self.name)

        for key in self.required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required keyword argument: {key}")
        self.meta = {}
        self.meta.update(kwargs)

        self._method = None
        if 'method' in self.meta:
            self.set_method(self.meta['method'])

    def set_method(self, method):
        self._method = method
        self.verify_method()

    def verify_method(self):
        if self._method not in self.methods:
            self.logger.warning(f"Supplied method '{self._method}' is not implemented for task {self.name}.")
            self.print_methods()
            raise NotImplementedError

    @property
    def method(self):
        """Return the currently selected method for this task."""
        return self.methods[self._method] if self._method else None

    @property
    def methods(self):
        """
        Return a dictionary of available methods for this task and their function signatures.
        :return: dict
            Dictionary with method names as keys and function signatures as values.
        """
        return {}

    def print_methods(self):
        """
        Print a list available methods for this task and their function signatures.
        """
        self.logger.info(f"Available methods for {self.name}:")
        for method, func_path in self.methods.items():
            sig = inspect.signature(load_class(func_path))
            self.logger.info(f"- {method}: {func_path}, {sig}")

    @abstractmethod
    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Run the task. Output should be a dict with string keys and any type of resulting data."""
        pass

    def __call__(self, *args, **kwargs):
        """Directly execute the task."""
        return self.run(*args, **kwargs)


    def set_logging_level(self, level: int):
        """
        Set the logging level.
        :param level: int
            Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.logger.setLevel(level)

    ####### Some parallelization utility functions #######
    @staticmethod
    def get_default_n_jobs(max_fraction: float = 0.75, min_jobs: int = 1) -> int:
        """
        Determine a safe default number of parallel jobs.

        - Respects scheduler-provided limits
        - Avoids using all cores by default
        """
        # Common scheduler environment variables
        for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
            if var in os.environ:
                try:
                    return max(min_jobs, int(os.environ[var]))
                except ValueError:
                    pass

        try:
            cpu_count = multiprocessing.cpu_count()
        except ModuleNotFoundError:
            cpu_count = 1
        return max(min_jobs, int(cpu_count * max_fraction))



class LazyTask(Task):
    """
    Abstract base class for tasks that support lazy (generator-based) execution.
    """
    def __init__(self, name=None, watch_mode=False, poll_interval=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.watch_mode = watch_mode
        self.poll_interval = poll_interval

    @abstractmethod
    def lazy_run(self, *args, **kwargs) -> Iterator[dict[str, Any]]:
        """Run the task lazily, yielding results (which should be in dict format)."""
        pass

    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Default run executes the lazy_run and collects all results."""
        res_dict = {}
        for batch in self.lazy_run(*args, **kwargs):
            for key, value in batch.items():
                if key in res_dict:
                    # If the key already exists, we merge the values. Extend lists, convert everything else into lists and extend
                    res_dict[key] = [res_dict[key]] if not isinstance(res_dict[key], list) else res_dict[key]
                    value_list = [value] if not isinstance(value, list) else value
                    res_dict[key].extend(value_list)
                else:
                    res_dict[key] = value
        return res_dict




class TaskResult(BaseModel):
    task_name: str
    data: dict[str, Any]

    params: dict[str, Any] = Field(default_factory=dict)
    upstream: list[str] = Field(default_factory=list)
    timestamp: Time | list[Time] = Field(default_factory=lambda: Time.now())

    model_config = {"arbitrary_types_allowed": True}

    def combine(self, other: 'TaskResult') -> 'TaskResult':
        """
        Combine this TaskResult with another with the same task_name, merging their data and metadata. (useful for lazy iterations)
        :param other: TaskResult
            Another TaskResult to combine with.
        :return: TaskResult
            A new TaskResult containing the combined data and metadata.
        """
        if self.task_name != other.task_name:
            raise ValueError("Can only combine TaskResults with the same task_name.")

        combined_data = self.data.copy()
        for key, value in other.data.items():
            if key in combined_data:
                # If the key already exists, we merge the values. Extend lists, convert everything else into lists and extend
                combined_data[key] = [combined_data[key]] if not isinstance(combined_data[key], list) else combined_data[key]
                value_list = [value] if not isinstance(value, list) else value
                combined_data[key].extend(value_list)
            else:
                combined_data[key] = value

        combined_params = {**self.params, **other.params}
        combined_upstream = list(set(self.upstream + other.upstream))
        combined_timestamp = self.timestamp if isinstance(self.timestamp, list) else [self.timestamp]
        other_timestamp = other.timestamp if isinstance(other.timestamp, list) else [other.timestamp]
        combined_timestamp.extend(other_timestamp)

        return TaskResult(
            task_name=self.task_name,
            data=combined_data,
            params=combined_params,
            upstream=combined_upstream,
            timestamp=combined_timestamp
        )

TaskResult.model_rebuild()
