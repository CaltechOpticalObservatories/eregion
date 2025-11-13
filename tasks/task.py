from collections.abc import Iterator
from abc import ABC, abstractmethod

# Base abstract class for tasks, should have a call method for direct execution and a run method for pipeline workflows
class Task(ABC):
    required_keys = []
    depends_on = []

    def __init__(self, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.meta = {}
        self.meta.update(kwargs)
        for key in self.required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required keyword argument: {key}")

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the task in a pipeline workflow."""
        pass

    def __call__(self, *args, **kwargs):
        """Directly execute the task."""
        return self.run(*args, **kwargs)

class LazyTask(Task):
    """
    Abstract base class for tasks that support lazy (generator-based) execution.
    """
    @abstractmethod
    def lazy_run(self, *args, **kwargs) -> Iterator:
        """Run the task lazily, yielding results."""
        pass

    def run(self, *args, **kwargs):
        """Default run executes the lazy_run and collects all results."""
        return list(self.lazy_run(*args, **kwargs))

class IOTask(Task):
    """
    A child class of Task specifically for handling FITS I/O operations.
    To be used as a base class for tasks that read from or write to FITS files, like ImageCreator.
    """
    required_keys = ['input_path']

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
