import pytest

from support import FakeOpenAI, write_credentials


@pytest.fixture
def fake():
    return FakeOpenAI()


@pytest.fixture
def directory(tmp_path):
    folder = tmp_path / "data"
    write_credentials(folder)
    return folder
