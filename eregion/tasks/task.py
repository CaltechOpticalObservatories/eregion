from collections.abc import Iterator
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing import Any

import os
import logging
import multiprocessing
from astropy.time import Time

# Base abstract class for tasks, should have a call method for direct execution and a run method for pipeline workflows
class Task(ABC):
    required_keys = []

    def __init__(self, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.n_jobs = kwargs.get('n_jobs') or self.get_default_n_jobs()
        self.logger = self.configure_logger()

        self.meta = {}
        self.meta.update(kwargs)
        for key in self.required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required keyword argument: {key}")

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the task in a pipeline workflow."""
        pass

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
        print(f"Available methods for {self.name}:")
        for method, signature in self.methods.items():
            print(f"- {method}: {signature}")

    def __call__(self, *args, **kwargs):
        """Directly execute the task."""
        return self.run(*args, **kwargs)

    def configure_logger(self):
        """
        Configure the logger for the task.
        """
        logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

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
    def lazy_run(self, *args, **kwargs) -> Iterator:
        """Run the task lazily, yielding results."""
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """Default run executes the lazy_run and collects all results."""
        return list(self.lazy_run(*args, **kwargs))


class TaskResult(BaseModel):
    task_name: str
    data: dict[str, Any]

    params: dict[str, Any] = Field(default_factory=dict)
    upstream: list[str] = Field(default_factory=list)
    timestamp: Time = Field(default_factory=lambda: Time.now())

    model_config = {"arbitrary_types_allowed": True}

TaskResult.model_rebuild()
