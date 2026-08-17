"""
Used only to exercise the CLI's commands
end-to-end without depending on real detector data.
"""
from eregion.tasks import Task
from eregion.datamodels import TaskResult


class EchoResult(TaskResult):
    value: int = 0


class EchoTask(Task):
    task_result = EchoResult

    def run(self, value: int = 0, **kwargs) -> EchoResult:
        return self.task_result(value=value)
