from collections.abc import Iterator
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict
import os
import multiprocessing

# Base abstract class for tasks, should have a call method for direct execution and a run method for pipeline workflows
class Task(ABC):
    required_keys = []

    def __init__(self, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.n_jobs = kwargs.get('n_jobs') or self.get_default_n_jobs()
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

        cpu_count = multiprocessing.cpu_count()
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
    data: Dict[str, Any]

    params: Dict[str, Any] = Field(default_factory=dict)
    upstream: Dict[str, "TaskResult"] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}

TaskResult.model_rebuild()
