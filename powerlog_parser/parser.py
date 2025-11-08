"""Power log parsing logic."""

from __future__ import annotations

import io
import logging
import math
import os
import subprocess
from datetime import datetime, tzinfo
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PowerLogParserConfig

_LOGGER = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("~/.powerlog.yaml").expanduser()
_SUPPORTED_EXTENSIONS = {".log", ".csv", ".txt", ".zst"}


@dataclass
class NodeResult:
    """Container for per-node parsing results."""

    node: str
    time_series: pd.DataFrame
    summary: Dict[str, float]


class PowerLogParser:
    """Parse power log files and compute consumption metrics."""

    def __init__(
        self,
        *,
        power_log_dir: Optional[str | Path] = None,
        nodes: Optional[Sequence[str]] = None,
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        max_interval_seconds: Optional[int] = None,
        config_path: Optional[str | Path] = None,
        timezone: Optional[str] = None,
    ) -> None:
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time are required")

        config = PowerLogParserConfig.load(
            Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH
        )
        config = config.with_overrides(
            power_log_dir=Path(power_log_dir).expanduser() if power_log_dir else None,
            nodes=nodes,
            max_interval_seconds=max_interval_seconds,
            timezone=timezone,
        )

        if config.power_log_dir is None:
            raise ValueError(
                "power_log_dir must be provided either via CLI argument or configuration file"
            )

        self._power_log_dir = config.power_log_dir
        self._nodes = list(config.nodes or _default_nodes())
        self._max_interval = pd.Timedelta(seconds=config.max_interval_seconds)
        self._config_timezone = config.timezone

        self._start_time_raw = _coerce_timestamp(start_time)
        self._end_time_raw = _coerce_timestamp(end_time)

        self._timezone_info, self._timezone_source = _resolve_timezone(
            explicit=timezone,
            config=config.timezone,
            start=self._start_time_raw,
        )

        self._start_local = _localize_timestamp(
            self._start_time_raw, self._timezone_info, source=self._timezone_source
        )
        self._end_local = _localize_timestamp(
            self._end_time_raw, self._timezone_info, source=self._timezone_source
        )

        if (
            self._timezone_source == "system"
            and self._start_local.utcoffset() is not None
            and self._end_local.utcoffset() is not None
            and self._start_local.utcoffset() != self._end_local.utcoffset()
        ):
            raise ValueError(
                "Local timestamps cross a daylight saving time change. "
                "Specify --timezone or include explicit offsets."
            )

        self._start_time = _convert_to_utc(self._start_local)
        self._end_time = _convert_to_utc(self._end_local)
        if self._end_time <= self._start_time:
            raise ValueError("end_time must be greater than start_time")

        self._start_date_hint = self._derive_local_date(self._start_local)
        self._end_date_hint = self._derive_local_date(self._end_local)

    def parse_logs(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """Parse logs and return time series, per-node summary, and overall summary."""
        node_results: List[NodeResult] = []

        for node in self._nodes:
            result = self._parse_node(node)
            node_results.append(result)

        if not node_results:
            raise ValueError("No nodes to parse")

        time_series_df = pd.concat(
            [result.time_series for result in node_results], ignore_index=True
        ).sort_values(["timestamp", "nodename"])

        summary_by_nodes_rows = [result.summary for result in node_results]
        combined_summary = self._build_combined_summary(node_results)
        summary_by_nodes_df = pd.DataFrame(summary_by_nodes_rows)

        return time_series_df.reset_index(drop=True), summary_by_nodes_df, combined_summary

    def _parse_node(self, node: str) -> NodeResult:
        paths = list(self._discover_candidate_files(node))
        if not paths:
            raise FileNotFoundError(f"No log files found for node '{node}'")

        frames: List[pd.DataFrame] = []
        
        for path in paths:
            frame = _read_log_file(path)
            if frame is None:
                continue
            frames.append(frame)

        if not frames:
            raise ValueError(f"Node '{node}' has no readable records")

        raw_df = pd.concat(frames, ignore_index=True)
        raw_df = _normalise_raw_frame(raw_df)

        # Preserve local timestamps for output but convert everything to UTC for arithmetic.
        raw_df["timestamp_utc"] = raw_df["timestamp"].dt.tz_convert("UTC")

        # Filter to the interval plus a padding record on each side for interpolation.
        relevant_df = raw_df[
            (raw_df["timestamp_utc"] >= self._start_time - pd.Timedelta(hours=1))
            & (raw_df["timestamp_utc"] <= self._end_time + pd.Timedelta(hours=1))
        ].copy()

        if relevant_df.empty:
            raise ValueError(f"Node '{node}' has no data around requested time window")

        node_series = self._build_time_series(node, relevant_df)
        summary = self._summarise_node(node, node_series, relevant_df)

        return NodeResult(node=node, time_series=node_series, summary=summary)

    def _build_time_series(self, node: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp_utc")

        start_point = _extract_boundary_point(
            target=self._start_time,
            records=df,
            direction="start",
        )
        end_point = _extract_boundary_point(
            target=self._end_time,
            records=df,
            direction="end",
        )

        if start_point is None:
            raise ValueError(
                f"Node '{node}' does not have sufficient data before start_time for interpolation"
            )

        if end_point is None:
            raise ValueError(
                f"Node '{node}' does not have sufficient data after end_time for interpolation"
            )

        inside_mask = (df["timestamp_utc"] >= self._start_time) & (
            df["timestamp_utc"] <= self._end_time
        )
        inside_records = df[inside_mask].copy()

        if inside_records.empty:
            raise ValueError(
                f"Node '{node}' does not contain any measurements within the requested interval"
            )

        records: List[Dict[str, object]] = []

        records.append(
            {
                "nodename": node,
                "timestamp": start_point.timestamp_local,
                "timestamp_utc": start_point.timestamp_utc,
                "instant_watts": start_point.value,
                "point_type": start_point.point_type,
            }
        )

        for _, row in inside_records.iterrows():
            records.append(
                {
                    "nodename": node,
                    "timestamp": row["timestamp"],
                    "timestamp_utc": row["timestamp_utc"],
                    "instant_watts": _to_float(row["instant_watts"]),
                    "point_type": "recorded",
                }
            )

        records.append(
            {
                "nodename": node,
                "timestamp": end_point.timestamp_local,
                "timestamp_utc": end_point.timestamp_utc,
                "instant_watts": end_point.value,
                "point_type": end_point.point_type,
            }
        )

        node_df = pd.DataFrame(records).drop_duplicates(
            subset=["nodename", "timestamp_utc"], keep="last"
        )
        node_df = node_df.sort_values(["timestamp_utc", "nodename"]).reset_index(drop=True)
        return node_df

    def _summarise_node(
        self, node: str, node_series: pd.DataFrame, raw_df: pd.DataFrame
    ) -> Dict[str, float]:
        inside_raw = raw_df[
            (raw_df["timestamp_utc"] >= self._start_time)
            & (raw_df["timestamp_utc"] <= self._end_time)
        ].copy()
        inside_raw = inside_raw.sort_values("timestamp_utc")

        if inside_raw.empty:
            raise ValueError(
                f"Node '{node}' has no recorded samples within the requested window"
            )

        _validate_sampling(node, inside_raw["timestamp_utc"], self._max_interval)

        energy_kwh = _integrate_energy(node_series["timestamp_utc"], node_series["instant_watts"])

        weighted_average = _compute_weighted_average(
            node_series["timestamp_utc"], node_series["instant_watts"]
        )

        stats_series = inside_raw["instant_watts"].astype(float)

        summary: Dict[str, float] = {
            "nodename": node,
            "total_energy_kwh": float(energy_kwh),
            "maximum_watts": float(stats_series.max()),
            "average_watts": float(weighted_average),
            "median_watts": float(stats_series.median()),
            "minimum_watts": float(stats_series.min()),
            "data_points": float(len(inside_raw)),
        }

        return summary

    def _build_combined_summary(self, results: Iterable[NodeResult]) -> Dict[str, float]:
        node_summaries = [res.summary for res in results]
        combined_series = self._combine_power_profiles([res.time_series for res in results])

        if combined_series.empty:
            raise ValueError("Combined power profile is empty")

        energy_kwh = _integrate_energy(
            combined_series.index,
            combined_series["instant_watts"],
        )

        overall_stats = combined_series["instant_watts"]
        weighted_average = _compute_weighted_average(
            combined_series.index, combined_series["instant_watts"]
        )

        per_node_max = [summary["maximum_watts"] for summary in node_summaries]
        per_node_avg = [summary["average_watts"] for summary in node_summaries]
        per_node_median = [summary["median_watts"] for summary in node_summaries]
        per_node_min = [summary["minimum_watts"] for summary in node_summaries]

        summary = {
            "nodename": "ALL",
            "total_energy_kwh": float(
                np.sum([summary["total_energy_kwh"] for summary in node_summaries])
            ),
            "maximum_watts": float(overall_stats.max()),
            "average_watts": float(weighted_average),
            "median_watts": float(overall_stats.median()),
            "minimum_watts": float(overall_stats.min()),
            "maximum_watts_per_node": float(max(per_node_max)) if per_node_max else math.nan,
            "average_watts_per_node": float(np.mean(per_node_avg)) if per_node_avg else math.nan,
            "median_watts_per_node": float(np.median(per_node_median))
            if per_node_median
            else math.nan,
            "minimum_watts_per_node": float(min(per_node_min)) if per_node_min else math.nan,
            "data_points": float(len(combined_series)),
        }

        summary["total_energy_kwh"] = float(energy_kwh)

        return summary

    def _combine_power_profiles(self, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
        frames = [frame.copy() for frame in frames]
        if not frames:
            return pd.DataFrame(columns=["instant_watts"])

        timestamp_sets: List[pd.Series] = []

        base = self._start_time

        for frame in frames:
            ts = frame["timestamp_utc"].sort_values().drop_duplicates()
            timestamp_sets.append(ts)

        if not timestamp_sets:
            return pd.DataFrame(columns=["instant_watts"])

        union_index = pd.DatetimeIndex(sorted(pd.concat(timestamp_sets).unique()))
        base_ns = base.value
        union_seconds = (union_index.astype("int64") - base_ns) / 1e9

        power_matrix: List[np.ndarray] = []

        for frame in frames:
            frame_sorted = frame.sort_values("timestamp_utc")
            time_values = frame_sorted["timestamp_utc"].astype("int64")
            seconds = (time_values - base_ns) / 1e9
            watts = frame_sorted["instant_watts"].to_numpy(dtype=float)
            interpolated_values = np.interp(union_seconds, seconds, watts)
            power_matrix.append(interpolated_values)

        combined_watts = np.sum(np.vstack(power_matrix), axis=0)
        combined_series = pd.Series(combined_watts, index=union_index, name="instant_watts")
        return combined_series.to_frame()

    def _discover_candidate_files(self, node: str) -> Iterator[Path]:
        base = self._power_log_dir
        if base.is_file():
            yield base
            return

        node_dir = base / node
        search_roots = [node_dir, base]
        expanded_start = self._start_date_hint - pd.Timedelta(days=1)
        expanded_end = self._end_date_hint + pd.Timedelta(days=1)
        dates = _enumerate_dates(expanded_start, expanded_end)

        for root in search_roots:
            if not root.exists():
                continue

            if root.is_file():
                yield root
                continue

            for date in dates:
                year = f"{date.year:04d}"
                month = f"{date.month:02d}"
                filename_base = f"{date:%Y-%m-%d}.log"
                candidates = [
                    root / filename_base,
                    root / f"{filename_base}.zst",
                    root / year / month / filename_base,
                    root / year / month / f"{filename_base}.zst",
                    root / year / filename_base,
                    root / year / f"{filename_base}.zst",
                ]

                for candidate in candidates:
                    if candidate.exists() and candidate.suffix in _SUPPORTED_EXTENSIONS:
                        yield candidate

        # As a fallback, traverse nexted directories but keep it contained.
        #for globbed in base.glob("**/*"):
        #    if globbed.is_file() and any(globbed.name.endswith(ext) for ext in _SUPPORTED_EXTENSIONS):
        #        if node in globbed.parts or node_dir in globbed.parents:
        #            yield globbed

    def _derive_local_date(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp


def _coerce_timezone_info(identifier: str) -> tzinfo:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover - Python < 3.9
        try:
            import pytz
        except ImportError as exc:  # pragma: no cover - minimal environments
            raise ValueError(
                "Timezone support requires the standard zoneinfo module or pytz."
            ) from exc
        try:
            return pytz.timezone(identifier)
        except Exception as exc:
            raise ValueError(f"Unknown timezone identifier '{identifier}'.") from exc
    else:
        try:
            return ZoneInfo(identifier)
        except Exception as exc:
            raise ValueError(f"Unknown timezone identifier '{identifier}'.") from exc


def _detect_system_timezone() -> Optional[tzinfo]:
    tz_env = os.environ.get("TZ")
    if tz_env:
        try:
            return _coerce_timezone_info(tz_env)
        except ValueError:
            pass

    try:
        tzinfo_obj = datetime.now().astimezone().tzinfo
    except Exception:  # pragma: no cover - defensive guard
        return None

    key = getattr(tzinfo_obj, "key", None) or getattr(tzinfo_obj, "zone", None)
    if key:
        try:
            return _coerce_timezone_info(str(key))
        except ValueError:
            return tzinfo_obj

    return tzinfo_obj


def _default_nodes() -> List[str]:
    import socket

    return [socket.gethostname()]


def _coerce_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.to_datetime(value)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to parse timestamp '{value}': {exc}") from exc


def _resolve_timezone(
    *, explicit: Optional[str], config: Optional[str], start: pd.Timestamp
) -> Tuple[Optional[tzinfo], str]:
    if explicit:
        return _coerce_timezone_info(explicit), "explicit"
    if config:
        return _coerce_timezone_info(config), "config"
    if start.tzinfo is not None:
        return start.tzinfo, "provided"

    system_tz = _detect_system_timezone()
    if system_tz is None:
        raise ValueError(
            "Unable to determine the local timezone. Specify --timezone or configure one."
        )
    return system_tz, "system"


def _localize_timestamp(
    timestamp: pd.Timestamp, tzinfo_obj: Optional[tzinfo], *, source: str
) -> pd.Timestamp:
    if timestamp.tzinfo is not None:
        return timestamp

    if tzinfo_obj is None:
        return timestamp.tz_localize("UTC")

    try:
        if isinstance(tzinfo_obj, str):
            tz = _coerce_timezone_info(tzinfo_obj)
        else:
            tz = tzinfo_obj
        return timestamp.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    except Exception as exc:
        raise ValueError(
            "Unable to interpret naive timestamps with the inferred timezone "
            f"from {source!r}. Provide an explicit --timezone."
        ) from exc


def _convert_to_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@dataclass
class BoundaryPoint:
    timestamp_local: pd.Timestamp
    timestamp_utc: pd.Timestamp
    value: float
    point_type: str


def _extract_boundary_point(
    *, target: pd.Timestamp, records: pd.DataFrame, direction: str
) -> Optional[BoundaryPoint]:
    before = records[records["timestamp_utc"] <= target].tail(1)
    after = records[records["timestamp_utc"] >= target].head(1)

    if direction == "start":
        if not before.empty and before["timestamp_utc"].iloc[0] == target:
            return BoundaryPoint(
                timestamp_local=before["timestamp"].iloc[0],
                timestamp_utc=before["timestamp_utc"].iloc[0],
                value=float(before["instant_watts"].iloc[0]),
                point_type="recorded",
            )
    else:
        if not after.empty and after["timestamp_utc"].iloc[0] == target:
            return BoundaryPoint(
                timestamp_local=after["timestamp"].iloc[0],
                timestamp_utc=after["timestamp_utc"].iloc[0],
                value=float(after["instant_watts"].iloc[0]),
                point_type="recorded",
            )

    prev_row = before.iloc[-1] if not before.empty else None
    next_row = after.iloc[0] if not after.empty else None

    if prev_row is None or next_row is None:
        return None

    value = _linear_interpolate(
        target,
        prev_row["timestamp_utc"],
        next_row["timestamp_utc"],
        float(prev_row["instant_watts"]),
        float(next_row["instant_watts"]),
    )
    prev_ts = prev_row["timestamp"]
    tzinfo = prev_ts.tzinfo

    if target.tzinfo is None:
        target_utc = target.tz_localize("UTC")
    else:
        target_utc = target.tz_convert("UTC")

    if tzinfo is None:
        timestamp_local = target_utc.tz_localize(None)
    else:
        timestamp_local = target_utc.tz_convert(tzinfo)

    return BoundaryPoint(
        timestamp_local=timestamp_local,
        timestamp_utc=target_utc,
        value=value,
        point_type="interpolated",
    )


def _linear_interpolate(
    target: pd.Timestamp,
    left_time: pd.Timestamp,
    right_time: pd.Timestamp,
    left_value: float,
    right_value: float,
) -> float:
    total = (right_time - left_time) / np.timedelta64(1, "s")
    if total == 0:
        return float(left_value)

    elapsed = (target - left_time) / np.timedelta64(1, "s")
    ratio = float(elapsed) / float(total)
    return float(left_value + (right_value - left_value) * ratio)


def _normalise_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "collected_at",
        "timestamp",
    ]
    timestamp_col = None
    for candidate in candidates:
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        raise ValueError("Input data frame does not include a timestamp column")

    if "instant_watts" not in df.columns:
        raise ValueError("Input data frame must include 'instant_watts'")

    df = df.copy()
    df.rename(columns={timestamp_col: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["instant_watts"] = pd.to_numeric(df["instant_watts"], errors="coerce")
    df = df.dropna(subset=["timestamp", "instant_watts"])

    if "ipmi_timestamp" in df.columns:
        ipmi_series = pd.to_datetime(df["ipmi_timestamp"], errors="coerce", utc=True)
        if ipmi_series.isna().any():
            raise ValueError("Unable to parse one or more ipmi_timestamp values")

        timestamp_series = df["timestamp"]
        if timestamp_series.dt.tz is None:
            timestamp_utc = timestamp_series.dt.tz_localize("UTC")
        else:
            timestamp_utc = timestamp_series.dt.tz_convert("UTC")

        if ipmi_series.dt.tz is None:
            ipmi_utc = ipmi_series.dt.tz_localize("UTC")
        else:
            ipmi_utc = ipmi_series.dt.tz_convert("UTC")

        drift = (timestamp_utc - ipmi_utc).abs()
        if (drift > pd.Timedelta(minutes=1)).any():
            raise ValueError(
                "ipmi_timestamp deviates from collected_at by more than one minute"
            )

        df["collected_at"] = timestamp_utc
        df["timestamp"] = ipmi_utc

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    return df


def _read_log_file(path: Path) -> Optional[pd.DataFrame]:
    if path.suffix == ".zst":
        text_stream = _decompress_zst(path)
        if text_stream is None:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return pd.read_csv(io.StringIO(handle.read()))
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.warning("Failed to read %s: %s", path, exc)
                return None
        return pd.read_csv(text_stream)

    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.warning("Failed to read %s: %s", path, exc)
        return None


def _decompress_zst(path: Path) -> Optional[io.StringIO]:
    try:
        import zstandard as zstd
    except ImportError:
        try:
            output = subprocess.check_output(["zstdcat", str(path)], text=True)
            return io.StringIO(output)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug(
                "Unable to decompress %s without zstandard package: %s", path, exc
            )
            return None

    with path.open("rb") as handle:
        dctx = zstd.ZstdDecompressor()
        try:
            text = dctx.stream_reader(handle).read().decode("utf-8")
            return io.StringIO(text)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Failed to decompress %s: %s", path, exc)
            return None


def _validate_sampling(node: str, timestamps: pd.Series, max_interval: pd.Timedelta) -> None:
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        return
    largest = diffs.max()
    if largest > max_interval:
        raise ValueError(
            f"Node '{node}' has a sampling gap of {largest} which exceeds the allowed {max_interval}."
        )


def _integrate_energy(timestamps: pd.Series, watts: pd.Series | Sequence[float]) -> float:
    if len(timestamps) != len(watts):
        raise ValueError("Timestamp and watt arrays must have the same length")

    if len(timestamps) < 2:
        return 0.0

    ns_values = _timestamps_to_ns_array(timestamps)

    base_ns = ns_values[0]
    seconds = (ns_values - base_ns) / 1e9
    values = np.asarray(watts, dtype=float)
    energy_ws = np.trapezoid(values, seconds)
    energy_kwh = energy_ws / (3600.0 * 1000.0)
    return float(energy_kwh)


def _enumerate_dates(start_hint: pd.Timestamp, end_hint: pd.Timestamp) -> List[pd.Timestamp]:
    start_date = start_hint.normalize()
    end_date = end_hint.normalize()
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return list(dates)


def _to_float(value: object) -> float:
    return float(value) if value is not None else math.nan


def _timestamps_to_ns_array(timestamps: pd.Series | pd.Index) -> np.ndarray:
    if isinstance(timestamps, pd.Index):
        return timestamps.astype("int64").to_numpy()

    ns_series = timestamps.astype("int64")
    if hasattr(ns_series, "to_numpy"):
        return ns_series.to_numpy(dtype=np.int64)
    return np.asarray(ns_series, dtype=np.int64)


def _compute_weighted_average(
    timestamps: pd.Series | pd.Index, watts: pd.Series | Sequence[float]
) -> float:
    if len(timestamps) == 0:
        return math.nan

    values = np.asarray(watts, dtype=float)
    if len(values) < 2:
        return float(values[0]) if len(values) else math.nan

    ns_values = _timestamps_to_ns_array(timestamps)
    duration_ns = ns_values[-1] - ns_values[0]
    if duration_ns <= 0:
        return float(values[-1])

    seconds = (ns_values - ns_values[0]) / 1e9
    area = np.trapezoid(values, seconds)
    return float(area / (seconds[-1] - seconds[0]))
