import pytest
import os

os.environ.setdefault("PREFECT_LOGGING_TO_API_ENABLED", "false")

from prefect.server.api.server import SubprocessASGIServer

@pytest.fixture(scope="session", autouse=True)
def stop_prefect_ephemeral_server():
    yield
    for server in list(SubprocessASGIServer._instances.values()):
        server.stop()