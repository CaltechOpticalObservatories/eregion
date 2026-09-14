from eregion.tasks import Task
from eregion.datamodels import TaskResult

class DeleteResult(Task):
    task_result = TaskResult
    """
    Task to delete a result from the pipeline engine's self.results dictionary to free up memory.
    """
    def __init__(self, name: str = "delete"):
        super().__init__(name=name)

    def run(self, result) -> TaskResult:
        del result
        return self.task_result()


class SaveResult(Task):
    task_result = TaskResult
    """
    Task to save a result from the pipeline engine's self.results dictionary to disk.
    """
    def __init__(self, name: str = "save"):
        super().__init__(name=name)

    def run(self, task_results: list, output_dir: str) -> TaskResult:
        for result in task_results:
            result.save(output_dir)
        return self.task_result()
