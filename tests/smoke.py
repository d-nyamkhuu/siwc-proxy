"""Check a built wheel or sdist works when installed on its own (run by the release workflow, not pytest).

    uv run --isolated --no-project --with dist/*.whl tests/smoke.py
"""

import tempfile
from importlib.metadata import entry_points, version

from click.testing import CliRunner
from fastapi.testclient import TestClient

import siwc_proxy
from siwc_proxy.cli import main
from siwc_proxy.config import Settings
from siwc_proxy.server import create_app

assert siwc_proxy.__version__ == version("siwc-proxy")
assert {e.name for e in entry_points(group="console_scripts") if e.value == "siwc_proxy.cli:main"} == {"siwc-proxy"}

with tempfile.TemporaryDirectory() as folder:
    result = CliRunner().invoke(main, ["--data-dir", folder, "status", "--json"])
    assert result.exit_code == 0 and '"signed_out"' in result.output, result.output
    with TestClient(create_app(Settings(data_dir=folder, api_key="smoke")), base_url="http://127.0.0.1:8787") as client:
        assert client.get("/healthz").json()["service"] == "siwc-proxy"

print(f"siwc-proxy {siwc_proxy.__version__} OK")
