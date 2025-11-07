from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from powerlog_parser import PowerLogParser


POWERLOG_TEST_ROOT = PROJECT_ROOT / "tests" / "powerlogs"
DEFAULT_POWER_LOG_DIR = POWERLOG_TEST_ROOT / "example1"
DEFAULT_NODES = ["fj-aox01", "fj-aox02"]
DEFAULT_START = "2025-11-01T00:02:30-04:00"
DEFAULT_END = "2025-11-01T00:17:30-04:00"
MULTI_DAY_DIR = POWERLOG_TEST_ROOT / "multi_day_zstd"
TIMEZONE_DIR = POWERLOG_TEST_ROOT / "timezone_test"
IPMI_DIR = POWERLOG_TEST_ROOT / "ipmi_timestamp_test"


def test_parse_multiple_nodes() -> None:
    parser = PowerLogParser(
        power_log_dir=DEFAULT_POWER_LOG_DIR,
        nodes=DEFAULT_NODES,
        start_time=DEFAULT_START,
        end_time=DEFAULT_END,
    )

    time_series_df, summary_df = parser.parse_logs()

    assert set(time_series_df["nodename"].unique()) == set(DEFAULT_NODES)

    node1_series = time_series_df[time_series_df["nodename"] == "fj-aox01"].reset_index(drop=True)
    node2_series = time_series_df[time_series_df["nodename"] == "fj-aox02"].reset_index(drop=True)

    assert node1_series.iloc[0]["point_type"] == "interpolated"
    assert node1_series.iloc[0]["instant_watts"] == pytest.approx(305.0)
    assert node2_series.iloc[0]["point_type"] == "recorded"

    node1_summary = summary_df[summary_df["nodename"] == "fj-aox01"].iloc[0]
    node2_summary = summary_df[summary_df["nodename"] == "fj-aox02"].iloc[0]
    combined_summary = summary_df[summary_df["nodename"] == "ALL"].iloc[0]

    def _expected_avg(series_df: pd.DataFrame, total_energy_kwh: float) -> float:
        duration_seconds = (
            series_df["timestamp_utc"].iloc[-1] - series_df["timestamp_utc"].iloc[0]
        ).total_seconds()
        if duration_seconds <= 0:
            return float(series_df["instant_watts"].iloc[-1])
        return (total_energy_kwh * 3_600_000.0) / duration_seconds

    def _energy_from_series(series_df: pd.DataFrame) -> float:
        if len(series_df) < 2:
            return 0.0
        ns = series_df["timestamp_utc"].astype("int64").to_numpy()
        seconds = (ns - ns[0]) / 1e9
        watts = series_df["instant_watts"].to_numpy(dtype=float)
        area = np.trapezoid(watts, seconds)
        return area / (3600.0 * 1000.0)

    expected_node1_energy = _energy_from_series(node1_series)
    expected_node2_energy = _energy_from_series(node2_series)

    assert node1_summary["total_energy_kwh"] == pytest.approx(expected_node1_energy, rel=1e-6)
    assert node2_summary["total_energy_kwh"] == pytest.approx(expected_node2_energy, rel=1e-6)

    expected_node1_avg = _expected_avg(node1_series, node1_summary["total_energy_kwh"])
    expected_node2_avg = _expected_avg(node2_series, node2_summary["total_energy_kwh"])

    assert node1_summary["maximum_watts"] == pytest.approx(310.0)
    assert node2_summary["maximum_watts"] == pytest.approx(215.0)
    assert combined_summary["maximum_watts"] == pytest.approx(517.5)
    assert node1_summary["average_watts"] == pytest.approx(expected_node1_avg, rel=1e-6)
    assert node2_summary["average_watts"] == pytest.approx(expected_node2_avg, rel=1e-6)

    overall_duration_seconds = (
        time_series_df["timestamp_utc"].max() - time_series_df["timestamp_utc"].min()
    ).total_seconds()
    expected_combined_avg = (
        combined_summary["total_energy_kwh"] * 3_600_000.0
    ) / overall_duration_seconds
    assert combined_summary["average_watts"] == pytest.approx(expected_combined_avg, rel=1e-6)
    assert combined_summary["total_energy_kwh"] == pytest.approx(
        node1_summary["total_energy_kwh"] + node2_summary["total_energy_kwh"], rel=1e-6
    )


def test_sampling_gap_validation(tmp_path: Path) -> None:
    delayed_log_dir = tmp_path
    node_dir = delayed_log_dir / "nodeA" / "2025" / "11"
    node_dir.mkdir(parents=True, exist_ok=True)
    log_path = node_dir / "2025-11-01.log"
    log_path.write_text(
        "collected_at,instant_watts\n"
        "2025-11-01T00:00:00-04:00,100\n"
        "2025-11-01T01:00:00-04:00,150\n",
        encoding="utf-8",
    )

    parser = PowerLogParser(
        power_log_dir=delayed_log_dir,
        nodes=["nodeA"],
        start_time="2025-11-01T00:10:00-04:00",
        end_time="2025-11-01T00:50:00-04:00",
    )

    with pytest.raises(ValueError):
        parser.parse_logs()


def test_multi_day_zstd_sources() -> None:
    parser = PowerLogParser(
        power_log_dir=MULTI_DAY_DIR,
        nodes=DEFAULT_NODES,
        start_time="2025-10-31T23:56:00-04:00",
        end_time="2025-11-01T00:06:00-04:00",
    )

    time_series_df, summary_df = parser.parse_logs()

    earliest = time_series_df["timestamp_utc"].min()
    assert earliest < pd.Timestamp("2025-11-01T04:00:00Z")
    assert set(time_series_df["nodename"].unique()) == set(DEFAULT_NODES)
    assert "ALL" in summary_df["nodename"].values


def test_timezone_spanning_interval() -> None:
    parser = PowerLogParser(
        power_log_dir=TIMEZONE_DIR,
        nodes=DEFAULT_NODES,
        start_time="2025-11-02T00:00:00-04:00",
        end_time="2025-11-02T01:15:00-05:00",
        max_interval_seconds=7200,
    )

    time_series_df, summary_df = parser.parse_logs()

    assert set(time_series_df["nodename"].unique()) == set(DEFAULT_NODES)
    assert set(time_series_df["point_type"].unique()) <= {"recorded", "interpolated"}
    assert summary_df.loc[summary_df["nodename"] == "ALL", "total_energy_kwh"].iloc[0] > 0


def test_no_data_in_interval_raises() -> None:
    parser = PowerLogParser(
        power_log_dir=DEFAULT_POWER_LOG_DIR,
        nodes=DEFAULT_NODES,
        start_time="2025-11-01T02:00:00-04:00",
        end_time="2025-11-01T03:00:00-04:00",
    )

    with pytest.raises(ValueError):
        parser.parse_logs()


def test_ipmi_timestamp_mismatch_raises(tmp_path: Path) -> None:
    log_dir = tmp_path / "nodeX" / "2025" / "11"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "2025-11-01.log"
    log_path.write_text(
        "collected_at,instant_watts,ipmi_timestamp\n"
        "2025-11-01T00:00:00-0400,100,\"Sat Nov  1 04:00:05 2025\"\n"
        "2025-11-01T00:05:00-0400,110,\"Sat Nov  1 04:05:05 2025\"\n",
        encoding="utf-8",
    )

    parser = PowerLogParser(
        power_log_dir=tmp_path,
        nodes=["nodeX"],
        start_time="2025-11-01T00:00:00-04:00",
        end_time="2025-11-01T00:10:00-04:00",
    )

    with pytest.raises(ValueError, match="ipmi_timestamp"):
        parser.parse_logs()


def test_ipmi_timestamp_dataset_mismatch() -> None:
    parser = PowerLogParser(
        power_log_dir=IPMI_DIR,
        nodes=["fj-aox01"],
        start_time="2025-11-01T00:00:00-04:00",
        end_time="2025-11-01T00:10:00-04:00",
    )

    with pytest.raises(ValueError, match="ipmi_timestamp"):
        parser.parse_logs()


def test_naive_times_crossing_dst_require_timezone() -> None:
    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/New_York"
        if hasattr(time, "tzset"):
            time.tzset()

        with pytest.raises(ValueError):
            PowerLogParser(
                power_log_dir=DEFAULT_POWER_LOG_DIR,
                nodes=DEFAULT_NODES[:1],
                start_time="2025-11-02T00:30:00",
                end_time="2025-11-02T02:30:00",
            )
    finally:
        if original_tz is not None:
            os.environ["TZ"] = original_tz
        else:
            os.environ.pop("TZ", None)
        if hasattr(time, "tzset"):
            time.tzset()
