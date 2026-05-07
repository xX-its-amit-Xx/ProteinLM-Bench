"""ProteinLM-Bench: benchmarking protein language models on mutation effect prediction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("proteinlm-bench")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.1.0"

__all__ = ["__version__"]
