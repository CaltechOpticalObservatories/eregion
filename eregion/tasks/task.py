from abc import ABC, abstractmethod
from typing import Callable, Generator
import os
import multiprocessing
import inspect

from utils import configure_logger, load_class
from datamodels import TaskResult

# Base abstract class for tasks, should have a call method for direct execution and a run method for pipeline workflows
class Task(ABC):
    required_keys = []
    task_result = TaskResult

    def __init__(self, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.n_jobs = kwargs.get('n_jobs') or self.get_default_n_jobs()
        self.logger = configure_logger(self.name)

        for key in self.required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required keyword argument: {key}")
        self.meta = {}
        self.meta.update(kwargs)

        self.method_name = None
        if 'method' in self.meta:
            self.set_method(self.meta['method'])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__name__ in {"Task", "LazyTask"}:
            return

        if "task_result" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a 'task_result' attribute and set it to a TaskResult subclass.")

        tr = cls.__dict__['task_result']
        if not isinstance(tr, type) or not issubclass(tr, TaskResult):
            raise TypeError(f"{cls.__name__}'s 'task_result' attribute must be a subclass of TaskResult, "
                            f"got {tr} instead.")

    @abstractmethod
    def run(self, *args, **kwargs) -> TaskResult:
        pass

    def dummy_run(self):
        return self.task_result.get_empty_instance()

    def __call__(self, *args, **kwargs):
        """Directly execute the task."""
        raise NotImplementedError

    def set_method(self, method_name):
        self.logger.info(f"Setting method '{method_name}' for task {self.name}.")
        self.method_name = method_name
        self.verify_method()

    def verify_method(self):
        if self.method_name not in self.methods.keys():
            self.logger.warning(f"Supplied method '{self.method_name}' is not implemented for task {self.name}.")
            self.print_methods()
            raise NotImplementedError

    @property
    def method(self):
        """Return the currently selected method's Callable for this task."""
        if self.method_name:
            func = self.methods[self.method_name]
            if isinstance(func, Callable):
                return func
            else:
                try:
                    return load_class(func)
                except Exception as e:
                    raise ValueError(f"Failed to load method '{self.method_name}' for task {self.name}: {e}")
        else:
            raise ValueError(f"Supplied method '{self.method_name}' is not implemented for task {self.name}.")

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
            if isinstance(func_path, str):
                func_path = load_class(func_path)
            sig = inspect.signature(func_path)
            self.logger.info(f"- {method}: {func_path}, {sig}")

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
    def __init__(self, name=None, watch_mode=False, poll_interval=0, max_batch_size=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.watch_mode = watch_mode
        self.poll_interval = poll_interval
        self.max_batch_size = max_batch_size

    @abstractmethod
    def lazy_run(self, *args, **kwargs) -> Generator[TaskResult]:
        """Run the task lazily, yielding results (which should be in dict format)."""
        pass

    def run(self, *args, **kwargs) -> TaskResult:
        for i, batch_result in enumerate(self.lazy_run(*args, **kwargs)):
            if i==0:
                results = batch_result
            else:
                results = results.combine(batch_result)
        return results
