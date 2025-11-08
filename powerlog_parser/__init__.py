"""Power log parsing utilities."""

from importlib import metadata

from .config import PowerLogParserConfig
from .parser import PowerLogParser

try:
	__version__ = metadata.version("powerlog-parser")
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
	__version__ = "0.0.0"

__all__ = ["PowerLogParser", "PowerLogParserConfig", "__version__"]
