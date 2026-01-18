"""
Data Collection Progress Monitor
==================================

At-a-glance monitoring of real-world data collection progress during
dissertation validation.  The script reports:

  1. Total record count relative to the 14,784-record target.
  2. Continuity — whether collection has proceeded without gaps or errors.
  3. Data quality (missing values, duplicate timestamps, schema conformance).
  4. Estimated time to reach a statistically meaningful dataset.

The target threshold is 2 weeks of 15-minute observations across 11 lines:
11 × 4 × 24 × 14 = 14,784 records.

Usage::

    # Full progress report:
    python scripts/check_collection_progress.py

    # Single-line summary:
    python scripts/check_collection_progress.py --summary
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

# Project root is added to sys.path so pandas/numpy can be imported when
# the script is invoked from the scripts/ subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_CSV: Path = PROJECT_ROOT / "data" / "tfl_merged.csv"
LOG_FILE: Path = PROJECT_ROOT / "data" / "collection.log"

#: 2 weeks of 15-minute observations across all 11 modelled lines.
TARGET_RECORDS: int = 11 * 4 * 24 * 14  # 14,784

TUBE_LINES = [
    "Bakerloo", "Central", "Circle", "District",
    "Hammersmith & City", "Jubilee", "Metropolitan",
    "Northern", "Piccadilly", "Victoria", "Waterloo & City",
]

REQUIRED_COLUMNS = [
    "timestamp", "line", "status", "delay_minutes",
    "temp_c", "precipitation_mm", "humidity", "crowding_index",
    "is_weekend", "hour", "day_of_week", "month", "peak_time", "is_holiday",
]

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    """Apply an ANSI colour escape when stdout is a terminal."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t: str) -> str:  return _c(t, "32")
def red(t: str)   -> str:  return _c(t, "31")
def yellow(t: str) -> str: return _c(t, "33")
def cyan(t: str)  -> str:  return _c(t, "36")
def bold(t: str)  -> str:  return _c(t, "1")
def dim(t: str)   -> str:  return _c(t, "2")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def _progress_bar(value: float, width: int = 40) -> str:
    """
    Render a simple ASCII progress bar.

    Args:
        value: Fraction complete in [0.0, 1.0].
        width: Total character width of the bar.

    Returns:
        String like ``[████████░░░░░░░░░░░░] 42%``.
    """
    clamped = max(0.0, min(1.0, value))
    filled = int(width * clamped)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    percent = int(clamped * 100)
    colour = green if clamped >= 0.75 else (yellow if clamped >= 0.3 else red)
    return f"[{colour(bar)}] {bold(str(percent) + '%')}"


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------


def check_data_quality(df: pd.DataFrame) -> Dict[str, object]:
    """
    Run a suite of data quality checks on the collected CSV.

    Checks target the failure modes most likely to break the ML pipeline:
    - Missing values (which propagate NaNs through feature engineering)
    - Duplicate (timestamp, line) pairs (which distort lag features)
    - Schema column presence
    - Value range violations

    Args:
        df: The loaded DataFrame from ``tfl_merged.csv``.

    Returns:
        Dictionary mapping quality metric names to their values.
    """
    issues: Dict[str, object] = {}

    null_counts = df.isnull().sum()
    issues["total_missing_values"] = int(null_counts.sum())
    issues["columns_with_nulls"] = null_counts[null_counts > 0].to_dict()

    dupe_mask = df.duplicated(subset=["timestamp", "line"], keep=False)
    issues["duplicate_rows"] = int(dupe_mask.sum())

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    issues["missing_columns"] = missing_cols

    range_issues = []
    if "delay_minutes" in df.columns and (df["delay_minutes"] < 0).any():
        range_issues.append("delay_minutes has negative values")
    if "crowding_index" in df.columns:
        if (df["crowding_index"] < 0).any() or (df["crowding_index"] > 1).any():
            range_issues.append("crowding_index outside [0, 1]")
    if "humidity" in df.columns:
        if (df["humidity"] < 0).any() or (df["humidity"] > 100).any():
            range_issues.append("humidity outside [0, 100]")
    issues["range_violations"] = range_issues

    return issues


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------


def find_collection_gaps(df: pd.DataFrame, expected_interval_minutes: int = 15) -> pd.DataFrame:
    """
    Identify gaps in the collection timeline larger than the expected interval.

    A gap indicates that the collection script was not running — e.g., the
    host machine was asleep, or an API failure silently dropped observations.

    Args:
        df: DataFrame with a ``timestamp`` column.
        expected_interval_minutes: Expected time between consecutive records
            for the same line.

    Returns:
        DataFrame of gaps with columns ``line``, ``gap_start``, ``gap_end``,
        ``gap_minutes``.
    """
    gaps = []
    # One missed cycle (2× the expected interval) is the tolerance threshold.
    tolerance_minutes = expected_interval_minutes * 2

    for line in df["line"].unique():
        line_df = df[df["line"] == line].sort_values("timestamp")
        timestamps = line_df["timestamp"]
        deltas = timestamps.diff()

        large_gaps = deltas[deltas > timedelta(minutes=tolerance_minutes)]
        for idx in large_gaps.index:
            gap_start = timestamps.loc[idx - 1] if idx > 0 else timestamps.iloc[0]
            gap_end = timestamps.loc[idx]
            gap_minutes = (gap_end - gap_start).total_seconds() / 60
            gaps.append(
                {
                    "line": line,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "gap_minutes": round(gap_minutes, 1),
                }
            )

    return pd.DataFrame(gaps) if gaps else pd.DataFrame(
        columns=["line", "gap_start", "gap_end", "gap_minutes"]
    )


# ---------------------------------------------------------------------------
# ETA computation
# ---------------------------------------------------------------------------


def estimate_completion(
    current_records: int,
    target_records: int,
    collection_start: datetime,
) -> Tuple[Optional[timedelta], Optional[datetime]]:
    """
    Estimate when collection will reach the target record count.

    The collection rate is derived from elapsed time and current record count,
    then extrapolated linearly to the target.

    Args:
        current_records: Number of records collected so far.
        target_records: Target record count.
        collection_start: When collection first began.

    Returns:
        Tuple of (time_remaining, estimated_completion_datetime).
        Both are ``None`` if the estimate cannot be computed.
    """
    if current_records <= 0:
        return None, None

    elapsed = datetime.now() - collection_start
    elapsed_seconds = elapsed.total_seconds()
    if elapsed_seconds <= 0:
        return None, None

    rate_per_second = current_records / elapsed_seconds
    remaining_records = max(0, target_records - current_records)
    seconds_remaining = remaining_records / rate_per_second if rate_per_second > 0 else None

    if seconds_remaining is None:
        return None, None

    time_remaining = timedelta(seconds=int(seconds_remaining))
    eta = datetime.now() + time_remaining
    return time_remaining, eta


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_full_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive progress report to the terminal.

    Sections:
        1. Overview (record count, date range, progress bar)
        2. Per-line breakdown
        3. Status distribution
        4. Weather summary
        5. Delay statistics
        6. Data quality report
        7. Gap analysis (if gaps found)
        8. ETA to dissertation-quality dataset

    Args:
        df: Loaded DataFrame from ``tfl_merged.csv``.
    """
    now = datetime.now()
    total_records = len(df)
    progress = total_records / TARGET_RECORDS

    print()
    print(cyan("=" * 65))
    print(bold(f"  London Underground Data Collection — Progress Report"))
    print(dim(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(cyan("=" * 65))

    # ── 1. Overview ─────────────────────────────────────────────────────
    print()
    print(bold("OVERVIEW"))
    print(f"  Records collected : {bold(str(total_records)):>15}  (target: {TARGET_RECORDS:,})")
    print(f"  Date range        : {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → "
          f"{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    span = df['timestamp'].max() - df['timestamp'].min()
    print(f"  Collection span   : {span.days} days, {span.seconds // 3600} hours")
    print()
    print(f"  Progress to target:")
    print(f"  {_progress_bar(progress)}")
    print(f"  {total_records:,} / {TARGET_RECORDS:,} records "
          f"({'complete' if progress >= 1.0 else 'in progress'})")

    # ── 2. Per-line breakdown ────────────────────────────────────────────
    print()
    print(bold("RECORDS PER LINE"))
    line_counts = df["line"].value_counts().reindex(TUBE_LINES, fill_value=0)
    max_line_records = line_counts.max() if not line_counts.empty else 1

    for line_name, count in line_counts.items():
        bar_width = 25
        bar_filled = int(bar_width * count / max_line_records)
        bar = "▪" * bar_filled + " " * (bar_width - bar_filled)
        status_icon = green("✓") if count > 0 else red("✗")
        print(f"  {status_icon} {line_name:<22} [{bar}] {count:>6,}")

    # ── 3. Status distribution ───────────────────────────────────────────
    if "status" in df.columns:
        print()
        print(bold("SERVICE STATUS DISTRIBUTION"))
        status_counts = df["status"].value_counts()
        for status, count in status_counts.items():
            pct = 100 * count / total_records
            icon = {"Good Service": green("●"), "Minor Delays": yellow("●"),
                    "Severe Delays": red("●")}.get(status, "●")
            print(f"  {icon} {status:<20} {count:>7,} records  ({pct:.1f}%)")

    # ── 4. Weather summary ───────────────────────────────────────────────
    if all(c in df.columns for c in ["temp_c", "precipitation_mm", "humidity"]):
        print()
        print(bold("WEATHER OBSERVATIONS"))
        print(f"  Temperature   : {df['temp_c'].min():.1f}°C — {df['temp_c'].max():.1f}°C"
              f"  (mean: {df['temp_c'].mean():.1f}°C)")
        print(f"  Precipitation : {df['precipitation_mm'].min():.1f} — "
              f"{df['precipitation_mm'].max():.1f} mm/h"
              f"  (mean: {df['precipitation_mm'].mean():.2f} mm/h)")
        print(f"  Humidity      : {df['humidity'].min():.0f}% — {df['humidity'].max():.0f}%"
              f"  (mean: {df['humidity'].mean():.1f}%)")
        rainy_pct = 100 * (df["precipitation_mm"] > 0).mean()
        print(f"  Rainy periods : {rainy_pct:.1f}% of observations")

    # ── 5. Delay summary ─────────────────────────────────────────────────
    if "delay_minutes" in df.columns:
        print()
        print(bold("DELAY STATISTICS"))
        print(f"  Mean delay    : {df['delay_minutes'].mean():.2f} minutes")
        print(f"  Median delay  : {df['delay_minutes'].median():.2f} minutes")
        print(f"  Max delay     : {df['delay_minutes'].max():.2f} minutes")
        print(f"  Std deviation : {df['delay_minutes'].std():.2f} minutes")

    # ── 6. Data quality ──────────────────────────────────────────────────
    print()
    print(bold("DATA QUALITY"))
    quality = check_data_quality(df)

    null_total = quality["total_missing_values"]
    null_icon = green("✓") if null_total == 0 else red("✗")
    print(f"  {null_icon} Missing values     : {null_total}")
    if quality["columns_with_nulls"]:
        for col, count in quality["columns_with_nulls"].items():
            print(f"       {col}: {count} nulls")

    dupe_count = quality["duplicate_rows"]
    dupe_icon = green("✓") if dupe_count == 0 else yellow("!")
    print(f"  {dupe_icon} Duplicate rows     : {dupe_count}")

    missing_cols = quality["missing_columns"]
    cols_icon = green("✓") if not missing_cols else red("✗")
    print(f"  {cols_icon} Schema columns     : "
          f"{'all present' if not missing_cols else ', '.join(missing_cols) + ' MISSING'}")

    range_issues = quality["range_violations"]
    range_icon = green("✓") if not range_issues else red("✗")
    print(f"  {range_icon} Value ranges       : "
          f"{'all valid' if not range_issues else '; '.join(range_issues)}")

    # ── 7. Gap analysis ──────────────────────────────────────────────────
    print()
    print(bold("COLLECTION GAPS  (> 30 minutes without data)"))
    gaps_df = find_collection_gaps(df)
    if gaps_df.empty:
        print(f"  {green('✓')} No significant gaps detected")
    else:
        significant = gaps_df[gaps_df["gap_minutes"] > 30].sort_values(
            "gap_minutes", ascending=False
        )
        print(f"  {yellow('!')} {len(significant)} gap(s) found:")
        for _, row in significant.head(5).iterrows():
            print(f"    {row['line']:<22} {row['gap_start'].strftime('%m-%d %H:%M')} → "
                  f"{row['gap_end'].strftime('%m-%d %H:%M')}  "
                  f"({row['gap_minutes']:.0f} min)")
        if len(significant) > 5:
            print(f"    … and {len(significant) - 5} more")

    # ── 8. ETA ───────────────────────────────────────────────────────────
    print()
    print(bold("ESTIMATED COMPLETION"))
    collection_start = df["timestamp"].min()
    time_remaining, eta = estimate_completion(
        current_records=total_records,
        target_records=TARGET_RECORDS,
        collection_start=collection_start,
    )

    if progress >= 1.0:
        print(f"  {green('✓')} Target reached!  You have enough data to retrain the pipeline.")
        print(f"     Run:  python train.py")
    elif time_remaining is not None and eta is not None:
        days = time_remaining.days
        hours = time_remaining.seconds // 3600
        print(f"  Remaining    : approx. {days}d {hours}h")
        print(f"  Target date  : {eta.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Target       : {TARGET_RECORDS:,} records (2 weeks of 15-min observations)")
    else:
        print(f"  {yellow('?')} Not enough data to estimate completion rate yet.")

    print()
    print(cyan("=" * 65))
    print()


def print_summary_line(df: pd.DataFrame) -> None:
    """
    Print a single summary line for quick status checks.

    Args:
        df: Loaded DataFrame from ``tfl_merged.csv``.
    """
    total = len(df)
    pct = min(100, int(100 * total / TARGET_RECORDS))
    span = df["timestamp"].max() - df["timestamp"].min()
    print(
        f"[Data Collection] {total:,} records ({pct}% of target) | "
        f"span: {span.days}d {span.seconds // 3600}h | "
        f"last: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the monitoring script."""
    parser = argparse.ArgumentParser(
        description="Monitor the progress of real-world tube data collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a single summary line instead of the full report.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_CSV,
        metavar="PATH",
        help=f"Path to the collected data CSV (default: {DATA_CSV}).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the progress monitoring script.

    Loads the CSV, parses timestamps, and displays either the full report
    or a one-line summary depending on the ``--summary`` flag.
    """
    args = _parse_args()

    if not args.csv.exists():
        print()
        print(red(f"  No data file found at: {args.csv}"))
        print()
        print("  Collection has not started yet, or the file is in a different location.")
        print("  To start collecting data, run:")
        print("    python data_collection.py --once   # test one snapshot")
        print("    python data_collection.py           # start continuous collection")
        print()
        sys.exit(0)

    try:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    except Exception as exc:
        print(red(f"  Failed to load {args.csv}: {exc}"))
        sys.exit(1)

    if df.empty:
        print(yellow("  The data file exists but contains no records yet."))
        sys.exit(0)

    if args.summary:
        print_summary_line(df)
    else:
        print_full_report(df)


if __name__ == "__main__":
    main()
