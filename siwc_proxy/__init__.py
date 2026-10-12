"""Local OpenAI-compatible proxy for Sign in with ChatGPT token sharing."""

from importlib.metadata import version

# pyproject.toml is the single source, so `uv version --bump` is all a release needs.
__version__ = version("siwc-proxy")
