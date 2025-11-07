"""Configuration helpers for the powerlog parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - module may not be installed
    yaml = None  # type: ignore[assignment]


@dataclass
class PowerLogParserConfig:
    """Configuration options loaded from a YAML file."""

    power_log_dir: Optional[Path] = None
    nodes: Optional[List[str]] = None
    max_interval_seconds: int = 300
    timezone: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[Path]) -> "PowerLogParserConfig":
        """Load configuration values from *path* if it exists."""
        if path is None:
            return cls()

        expanded = Path(path).expanduser()
        if not expanded.exists():
            return cls()

        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load configuration files. "
                "Install it or remove the --config option."
            )

        with expanded.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            raise ValueError("Configuration file must contain a mapping at the top level.")

        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "PowerLogParserConfig":
        """Create a configuration object from a plain mapping."""
        power_log_dir = data.get("power_log_dir")
        nodes = data.get("nodes")
        max_interval_seconds = data.get("max_interval_seconds")
        timezone = data.get("timezone")

        return cls(
            power_log_dir=Path(power_log_dir).expanduser() if power_log_dir else None,
            nodes=_normalise_nodes(nodes),
            max_interval_seconds=int(max_interval_seconds)
            if max_interval_seconds is not None
            else 300,
            timezone=str(timezone) if timezone is not None else None,
        )

    def with_overrides(
        self,
        *,
        power_log_dir: Optional[Path] = None,
        nodes: Optional[Iterable[str]] = None,
        max_interval_seconds: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> "PowerLogParserConfig":
        """Return a new config with explicit overrides applied."""
        return PowerLogParserConfig(
            power_log_dir=_coalesce_path(power_log_dir, self.power_log_dir),
            nodes=list(nodes) if nodes is not None else self.nodes,
            max_interval_seconds=max_interval_seconds
            if max_interval_seconds is not None
            else self.max_interval_seconds,
            timezone=timezone if timezone is not None else self.timezone,
        )


def _normalise_nodes(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [node.strip() for node in value.split(",") if node.strip()]
    if isinstance(value, Iterable):
        normalised: List[str] = []
        for item in value:
            if not item:
                continue
            normalised.append(str(item).strip())
        return normalised or None
    raise TypeError("nodes must be a sequence or a comma separated string")


def _coalesce_path(*candidates: Optional[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None:
            return Path(candidate).expanduser()
    return None
