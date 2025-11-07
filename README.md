# powerlog-parser

Python toolkit and CLI for parsing power log files, validating time-series integrity, and generating consolidated metrics across nodes.

## Features

- Time-zone aware parsing with automated DST checks.
- Supports plain CSV, directory structures, and `.zst` archives (falls back to plain text when decompression is unavailable).
- Validates `ipmi_timestamp` alignment with `collected_at`.
- Produces per-node time-series and combined summary reports, including properly weighted averages.
- Exposes both CLI (`powerlog-parser`) and importable Python API.

## Quick Start

```bash
mamba create -n bench-parser-py313 python=3.13 -y
mamba activate bench-parser-py313
mamba install -y pandas zstandard pytest

python3 -m pytest tests

python3 powerlog-parser --help
```

Parse power log files and report power consumption metrics for specified nodes over specified time intervals.

Usage:

```bash
powerlog-parser \
    [--config /nsimakov/powerlog/config.yaml] \
    [--power-log-dir /nsimakov/powerlog/fj-aox01/2025/11/2025-11-01.log.zst] \
    [--nodes node01,node02] \
    --start-time 2025-11-02T00:45:00.12-0400 \
    --end-time 2025-11-02T02:01:59.32-0500 \
    [-ots|--output-ts power_timeseries_report.csv] \
    [-sum|--output-summary power_summary_report.csv]
```

Where:
- `--config`: Path to the configuration YAML file specifying parsing options. [default: ~/.powerlog.yaml]
- `--power-log-dir`: Path to the directory containing power log files. [default: specified in config file]
- `--nodes`: Comma-separated list of node names to extract power consumption data for. [default: `hostname`]
- `--start-time`: Start time for the interval to extract data (ISO 8601 format).
- `--end-time`: End time for the interval to extract data (ISO 8601 format).
- `-ots` or `--output-ts`: Path to the output CSV file for time series power consumption data. [default: power_timeseries_report.csv]
- `-sum` or `--output-summary`: Path to the output CSV file for summary power consumption metrics. [default: power_summary_report.csv]


Power log directory structure example:

<node_name>/<year>/<month>/<year>-<month>-<day>.log[.zst]

Where:
- `<node_name>`: Name of the node (e.g., fj-aox01).
- `<year>`: Year of the log file (e.g., 2025).
- `<month>`: Month of the log file (e.g., 11).
- `<day>`: Day of the log file (e.g., 01).
- `.zst`: Optional compression extension if the log file is compressed using Zstandard. Typically currently logged file is uncompressed and previous days are compressed.

`./examples/` directory contains examples for the power-parser utility.
`./examples/fj-aox01/` contains sample power log files for testing purposes.

power log files format example:

```
collected_at,instant_watts,minimum_watts,maximum_watts,average_watts,ipmi_timestamp,sampling_seconds,state
2025-10-24T00:00:01-0400,301,300,315,302,"Fri Oct 24 04:00:09 2025",00000300,activated
2025-10-24T00:00:16-0400,301,300,315,302,"Fri Oct 24 04:00:24 2025",00000300,activated
2025-10-24T00:00:32-0400,301,300,315,302,"Fri Oct 24 04:00:40 2025",00000300,activated
2025-10-24T00:00:47-0400,301,300,315,302,"Fri Oct 24 04:00:55 2025",00000300,activated
```

Check that ipmi_timestamp is consistent with collected_at timestamp and not off by more than 1 second.

# Algorithm

Read power log files for specified nodes and time intervals.

Importantly each node record has different timestamps.

Make dataframe for each node with `instant_watts` power consumption metrics, it seems other metrics are often not accurate.

Make the first and last record interpolated between the two nearest records if start_time or end_time do not match exactly.

Ensure that the time intervals between records are less than or equal to specified value (default to 5 minutes).

Ensure that there is at least one record between start_time and end_time.

Calculate total energy consumption in kWh for each node over the specified time interval.

Calculate total energy consumption in kWh for all nodes combined over the specified time interval.

Record maximum, average, median and minimum power consumption in watts for each node over the specified time interval.

Record maximum, average, median and minimum power consumption in watts per node for all nodes over the specified time interval.

Output two CSV reports:
1. Time series report with power consumption metrics for each node at each timestamp.
2. Summary report with total energy consumption and power consumption statistics for each node and for all nodes combined.

output time series report example:

```
nodename,timestamp,point_type,instant_watts
fj-aox01,2025-11-01T00:00:00.120000-04:00,interpolated,305
fj-aox01,2025-11-01T00:05:00-04:00,recorded,310
fj-aox01,2025-11-01T00:10:00-04:00,recorded,300
...
```

output summary report example:

```
total_energy_kwh,maximum_watts,average_watts,median_watts,minimum_watts,maximum_watts_per_node,average_watts_per_node,median_watts_per_node,minimum_watts_per_node
0.256,320,305,303,290,320,305,303,290
```

The reported `average_watts` is the weighted average based on time intervals between records. i.e. (instant_watts[i] + instant_watts[i+1]) / 2 should be weighted by (time[i+1] - time[i]) in averaging.

This utility will also be used programmatically via its Python API.

Example CLI usage:

```bash
powerlog-parser \
    --power-log-dir ./examples \
    --nodes fj-aox01 \
    --start-time 2025-11-01T00:04:00-0400 \
    --end-time 2025-11-01T00:11:00-0400 \
    --output-ts power_timeseries_report.csv \
    --output-summary power_summary_report.csv


```

Example Python API usage:

```python
from powerlog_parser import PowerLogParser
parser = PowerLogParser(
    power_log_dir="./test/fj-aox01/2025/11/2025-11-01.log.zst",
    nodes=["fj-aox01"],
    start_time="2025-11-01T00:00:00.12",
    end_time="2025-11-01T01:59:59.32"
)
time_series_df, summary_df = parser.parse_logs()
time_series_df.to_csv("power_timeseries_report.csv", index=False)
summary_df.to_csv("power_summary_report.csv", index=False)
```

# Project Structure

follow the standard Python package structure.

# Python development setup

```bash
mamba create -n bench-parser-py313 python=3.13 -y
mamba activate bench-parser-py313
mamba install -y pandas zstandard pytest
```


# More system and unit tests

`pytest` is used for testing

```bash
mamba install pytest
pytest ./tests/
```

The powerlogs examples for tests are located in the `tests/powerlogs` directory. You can add your own test cases there as needed.

* `tests/powerlogs/example1` - contains sample power log files for node `fj-aox01` and `fj-aox02` for testing purposes.
  * raise error if no data in the specified time interval


* `tests/powerlogs/multi_day_zstd` - contains sample power log files for node `fj-aox01` and `fj-aox02` spanning multiple days with Zstandard compression for all but last day.

* `tests/powerlogs/timezone_test` - contains sample power log files for node `fj-aox01` and `fj-aox02` for testing time interval spanning across time zones.

* `tests/powerlogs/ipmi_timestamp_test` - contains sample power log files for node `fj-aox01` and `fj-aox02` for testing IPMI timestamp consistency with collected_at timestamp (one of them is inconsistent).

# License

This project is licensed under the MIT License - see the LICENSE file for details.
