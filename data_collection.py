"""
Real-time data collection from TfL and weather APIs.
Collects every 15 minutes, saves to data/tfl_merged.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import holidays
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# -- constants --

COLLECTION_INTERVAL_SECONDS: int = 15 * 60

DATA_DIR: Path = Path(__file__).parent / "data"
OUTPUT_CSV: Path = DATA_DIR / "tfl_merged.csv"

LONDON_LAT: float = 51.5074  # central London
LONDON_LON: float = -0.1278

LINE_ID_MAP: Dict[str, str] = {
    "Bakerloo": "bakerloo",
    "Central": "central",
    "Circle": "circle",
    "District": "district",
    "Hammersmith & City": "hammersmith-city",
    "Jubilee": "jubilee",
    "Metropolitan": "metropolitan",
    "Northern": "northern",
    "Piccadilly": "piccadilly",
    "Victoria": "victoria",
    "Waterloo & City": "waterloo-city",
}

CSV_COLUMNS: List[str] = [
    "timestamp",
    "line",
    "status",
    "delay_minutes",
    "temp_c",
    "precipitation_mm",
    "humidity",
    "crowding_index",
    "is_weekend",
    "hour",
    "day_of_week",
    "month",
    "peak_time",
    "is_holiday",
]

# collapse TfL's granular statuses into our 3 categories
STATUS_NORMALISATION: Dict[str, str] = {
    "Good Service": "Good Service",
    "Minor Delays": "Minor Delays",
    "Severe Delays": "Severe Delays",
    "Part Suspended": "Severe Delays",
    "Suspended": "Severe Delays",
    "Part Closure": "Severe Delays",
    "Planned Closure": "Severe Delays",
    "Reduced Service": "Minor Delays",
    "Bus Service": "Severe Delays",
    "No Step Free Access": "Good Service",
    "Special Service": "Minor Delays",
    "Change of Frequency": "Minor Delays",
    "Issues Reported": "Minor Delays",
    "Not Running": "Severe Delays",
    "Service Closed": "Good Service",  # Outside operating hours
}

# estimated delay per status (mean, std) — TfL API doesn't give exact durations
DELAY_ESTIMATES: Dict[str, Tuple[float, float]] = {
    "Good Service": (1.0, 0.5),
    "Minor Delays": (6.0, 2.0),
    "Severe Delays": (25.0, 8.0),
}

# crowding weights roughly proportional to ridership (fallback when API fails)
LINE_CROWDING_WEIGHT: Dict[str, float] = {
    "Central": 0.15,
    "Jubilee": 0.12,
    "Northern": 0.12,
    "Victoria": 0.10,
    "Piccadilly": 0.08,
    "District": 0.06,
    "Metropolitan": 0.05,
    "Circle": 0.04,
    "Bakerloo": 0.04,
    "Hammersmith & City": 0.04,
    "Waterloo & City": 0.02,
}

# NaPTAN IDs per line — from TfL StopPoint API, March 2026
LINE_STATION_IDS: Dict[str, List[str]] = {
    "Bakerloo": [
        "940GZZLUQPS", "940GZZLUKSL", "940GZZLULRD", "940GZZLUPCC",
        "940GZZLUEBY", "940GZZLUWLO", "940GZZLULAM", "940GZZLUELN",
        "940GZZLUBST", "940GZZLUOXC", "940GZZLURGP", "940GZZLUBKF",
        "940GZZLUPAC", "940GZZLUMRB", "940GZZLUSGP", "940GZZLUKCL",
        "940GZZLUHSN", "940GZZLUWJN", "940GZZLUHAW", "940GZZLUSTM",
        "940GZZLUWIG", "940GZZLUNFD", "940GZZLUHCH", "940GZZLUWRP",
        "940GZZLUBUO",
    ],
    "Central": [
        "940GZZLUEAC", "940GZZLUHAN", "940GZZLUNHG", "940GZZLUWCY",
        "940GZZLUSBC", "940GZZLUNTE", "940GZZLUHLP", "940GZZLUGFD",
        "940GZZLURSM", "940GZZLUSWF", "940GZZLUSGT", "940GZZLUSPE",
        "940GZZLULYS", "940GZZLUBDE", "940GZZLUMEP", "940GZZLUSTD",
        "940GZZLUBLE", "940GZZLUBND", "940GZZLUBET", "940GZZLUMLE",
        "940GZZLUBNK", "940GZZLUSTB", "940GZZLULGT", "940GZZLUMYD",
        "940GZZLUBHD", "940GZZLUSBD", "940GZZLURED", "940GZZLUGGH",
        "940GZZLUCWL", "940GZZLUWSD", "940GZZLUNAB", "940GZZLUHLT",
        "940GZZLURSG", "940GZZLUTFP", "940GZZLUEPP", "940GZZLUWOF",
        "940GZZLUCHT", "940GZZLUBYF", "940GZZLUNGB", "940GZZLULOU",
        "940GZZLUDBE", "940GZZLUHPK",
    ],
    "Circle": [
        "940GZZLUADE", "940GZZLUBWT", "940GZZLUBBB", "940GZZLUCST",
        "940GZZLUEMB", "940GZZLUWSM", "940GZZLUVIC", "940GZZLUSKS",
        "940GZZLUGTR", "940GZZLUHSC", "940GZZLUBAY", "940GZZLUPAD",
        "940GZZLUERS", "940GZZLURYL", "940GZZLULRD", "940GZZLUBST",
        "940GZZLUALD", "940GZZLULVS", "940GZZLUCAL", "940GZZLUMGT",
        "940GZZLUECT", "940GZZLUTWH", "940GZZLUFCN", "940GZZLUKSX",
        "940GZZLUSGP", "940GZZLUHBN",
    ],
    "District": [
        "940GZZLUUPM", "940GZZLUHGR", "940GZZLUEBE", "940GZZLUBGE",
        "940GZZLUDGE", "940GZZLUECM", "940GZZLUALG", "940GZZLUPSG",
        "940GZZLUBWT", "940GZZLUEMB", "940GZZLUWSM", "940GZZLUVIC",
        "940GZZLUSKS", "940GZZLUGTR", "940GZZLUHSC", "940GZZLUBAY",
        "940GZZLUPAD", "940GZZLUERS", "940GZZLURYL", "940GZZLULRD",
        "940GZZLUHAM", "940GZZLUTMP", "940GZZLURKW", "940GZZLUKWG",
        "940GZZLUNEA", "940GZZLUGUN", "940GZZLUSFS", "940GZZLUPUT",
        "940GZZLUWIM", "940GZZLUTNG", "940GZZLUECR", "940GZZLUMTN",
        "940GZZLUBKH", "940GZZLUWPL", "940GZZLUUPM",
    ],
    "Hammersmith & City": [
        "940GZZLUHSC", "940GZZLUBAY", "940GZZLUPAD", "940GZZLUERS",
        "940GZZLURYL", "940GZZLULRD", "940GZZLUBST", "940GZZLUALD",
        "940GZZLULVS", "940GZZLUCAL", "940GZZLUMGT", "940GZZLUECT",
        "940GZZLUTWH", "940GZZLUFCN", "940GZZLUKSX", "940GZZLUSGP",
        "940GZZLUHBN", "940GZZLUWCH", "940GZZLUSHO", "940GZZLULGN",
        "940GZZLUSBM", "940GZZLULAT",
    ],
    "Jubilee": [
        "940GZZLUSGP", "940GZZLUBHD", "940GZZLUWJN", "940GZZLUKWG",
        "940GZZLUFYR", "940GZZLUWSM", "940GZZLUWLO", "940GZZLUBKG",
        "940GZZLULNB", "940GZZLUSWK", "940GZZLUCAN", "940GZZLUCGT",
        "940GZZLUBMY", "940GZZLUWEH", "940GZZLUNCT", "940GZZLUBRE",
        "940GZZLUSSG", "940GZZLUSTD", "940GZZLUWGN",
    ],
    "Metropolitan": [
        "940GZZLUALD", "940GZZLUBKF", "940GZZLUFCN", "940GZZLUKSX",
        "940GZZLUSGP", "940GZZLUHBN", "940GZZLUWCH", "940GZZLUSHO",
        "940GZZLULGN", "940GZZLUSBM", "940GZZLULAT", "940GZZLUHSD",
        "940GZZLUCRX", "940GZZLUNWP", "940GZZLUNHG", "940GZZLUUXB",
        "940GZZLUICK", "940GZZLUHGD", "940GZZLUNHG", "940GZZLUAMS",
        "940GZZLURAY", "940GZZLUMRD", "940GZZLUCFD", "940GZZLUWFN",
        "940GZZLURCK", "940GZZLUALD",
    ],
    "Northern": [
        "940GZZLUEGW", "940GZZLUHBT", "940GZZLUMHL", "940GZZLUBZP",
        "940GZZLUCPK", "940GZZLUAGL", "940GZZLUEUS", "940GZZLUWRR",
        "940GZZLUGDG", "940GZZLUCTN", "940GZZLUBNK", "940GZZLULNB",
        "940GZZLUBOR", "940GZZLUOVL", "940GZZLUSTK", "940GZZLUTCO",
        "940GZZLUCWR", "940GZZLUSBC", "940GZZLUCLP", "940GZZLUBAL",
        "940GZZLUTHL", "940GZZLUSWM", "940GZZLUCSD", "940GZZLUMDN",
        "940GZZLUSUT", "940GZZLUMRN", "940GZZLUBDM", "940GZZLUCDG",
        "940GZZLUNFD", "940GZZLUEDG", "940GZZLUHBT",
    ],
    "Piccadilly": [
        "940GZZLUHNX", "940GZZLUHFN", "940GZZLUOSY", "940GZZLUBOS",
        "940GZZLUSGT", "940GZZLURSM", "940GZZLUGFD", "940GZZLUHLP",
        "940GZZLUNTE", "940GZZLUSBC", "940GZZLUHPC", "940GZZLURSP",
        "940GZZLUGPK", "940GZZLUHRC", "940GZZLUKNB", "940GZZLULCN",
        "940GZZLUCGN", "940GZZLUALD", "940GZZLUKSX", "940GZZLUAGL",
        "940GZZLUCPK", "940GZZLUTMO", "940GZZLUBOS", "940GZZLUSFS",
        "940GZZLUEGT", "940GZZLUBTE", "940GZZLUHTN", "940GZZLUMTD",
        "940GZZLUBHW", "940GZZLUHAT", "940GZZLURAR",
    ],
    "Victoria": [
        "940GZZLUVIC", "940GZZLUSKS", "940GZZLUGTR", "940GZZLUHSC",
        "940GZZLUWLO", "940GZZLUEUR", "940GZZLUOXC", "940GZZLUWRR",
        "940GZZLUEUS", "940GZZLUKSX", "940GZZLUHBN", "940GZZLUSEV",
        "940GZZLUTMO", "940GZZLUFPK", "940GZZLUHGR", "940GZZLUBKH",
        "940GZZLUWWL", "940GZZLUBRF",
    ],
    "Waterloo & City": [
        "940GZZLUWLO",
        "940GZZLUBNK",
    ],
}




def _build_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Session with retry logic."""
    session = requests.Session()
    retry_policy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class RateLimiter:
    """Simple rate limiter for API quotas."""

    max_requests_per_minute: int
    _last_request_time: float = field(default=0.0, init=False, repr=False)
    _min_interval: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._min_interval = 60.0 / self.max_requests_per_minute

    def wait_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait_time = self._min_interval - elapsed
        if wait_time > 0:
            jitter = random.uniform(0, self._min_interval * 0.1)
            time.sleep(wait_time + jitter)
        self._last_request_time = time.monotonic()


class TfLClient:
    """TfL Unified API client."""

    BASE_URL: str = "https://api.tfl.gov.uk"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        self._session = _build_session()
        self._rate_limiter = RateLimiter(max_requests_per_minute=40)
        logger.info(
            "TfLClient initialised (authenticated=%s)", api_key is not None
        )

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}{path}"
        request_params: Dict = params or {}
        if self._api_key:
            request_params["app_key"] = self._api_key

        try:
            response = self._session.get(url, params=request_params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "TfL HTTP error for %s: %s %s",
                path,
                exc.response.status_code,
                exc.response.reason,
            )
        except requests.exceptions.ConnectionError:
            logger.warning("TfL connection error for %s — no network?", path)
        except requests.exceptions.Timeout:
            logger.warning("TfL request timed out for %s", path)
        except (ValueError, KeyError) as exc:
            logger.warning("TfL JSON parse error for %s: %s", path, exc)

        return None

    def fetch_line_status(self, line_id: str) -> Optional[str]:
        data = self._get(f"/Line/{line_id}/Status")
        if not data:
            return None

        try:
            statuses = data[0]["lineStatuses"]
            if not statuses:
                return "Good Service"

            # lower severity number = worse condition
            worst = min(statuses, key=lambda s: s.get("statusSeverity", 10))
            return worst.get("statusSeverityDescription", "Good Service")

        except (IndexError, KeyError, TypeError) as exc:
            logger.warning("Unexpected TfL status structure for %s: %s", line_id, exc)
            return None

    def fetch_station_crowding(self, naptan_id: str) -> Optional[float]:
        data = self._get(f"/Crowding/{naptan_id}/Live")
        if not data:
            return None

        try:
            pct = float(data.get("percentageOfBaseLine", 0.0))
            return round(min(1.0, max(0.0, pct / 100.0)), 3)
        except (TypeError, ValueError) as exc:
            logger.debug("Could not parse crowding for %s: %s", naptan_id, exc)
            return None

    def fetch_line_crowding_index(self, line_name: str) -> Optional[float]:
        station_ids = LINE_STATION_IDS.get(line_name, [])
        if not station_ids:
            return None

        values: List[float] = []
        for naptan_id in station_ids:
            result = self.fetch_station_crowding(naptan_id)
            if result is not None:
                values.append(result)

        if len(values) < max(1, len(station_ids) // 2):  # need at least half
            logger.debug(
                "Insufficient crowding data for %s (%d/%d stations)",
                line_name, len(values), len(station_ids),
            )
            return None

        mean_crowding = round(sum(values) / len(values), 3)
        logger.debug(
            "Line %s crowding: %.3f (from %d/%d stations)",
            line_name, mean_crowding, len(values), len(station_ids),
        )
        return mean_crowding

    def test_connectivity(self) -> bool:
        data = self._get("/StopPoint/Meta/Categories")
        return data is not None



class WeatherClient:
    """OpenWeatherMap client (free tier)."""

    BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = _build_session()
        self._rate_limiter = RateLimiter(max_requests_per_minute=20)
        logger.info("WeatherClient initialised")

    def fetch_london_weather(self) -> Optional[Dict[str, float]]:
        self._rate_limiter.wait_if_needed()

        params = {
            "lat": LONDON_LAT,
            "lon": LONDON_LON,
            "appid": self._api_key,
            "units": "metric",
        }

        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather(data)

        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "Weather HTTP error: %s %s",
                exc.response.status_code,
                exc.response.reason,
            )
        except requests.exceptions.ConnectionError:
            logger.warning("Weather API connection error — no network?")
        except requests.exceptions.Timeout:
            logger.warning("Weather API request timed out")
        except (ValueError, KeyError) as exc:
            logger.warning("Weather API JSON parse error: %s", exc)

        return None

    @staticmethod
    def _parse_weather(data: Dict) -> Dict[str, float]:
        temp_c: float = data["main"]["temp"]
        humidity: float = data["main"]["humidity"]

        rain_mm: float = data.get("rain", {}).get("1h", 0.0)
        snow_mm: float = data.get("snow", {}).get("1h", 0.0)
        precipitation_mm: float = round(rain_mm + snow_mm, 2)

        return {
            "temp_c": round(temp_c, 2),
            "precipitation_mm": precipitation_mm,
            "humidity": round(humidity, 2),
        }

    def test_connectivity(self) -> bool:
        weather = self.fetch_london_weather()
        return weather is not None



def estimate_crowding(line, hour, is_weekend, is_holiday) -> float:
    """Fallback when live crowding API is down."""
    base = LINE_CROWDING_WEIGHT.get(line, 0.05) * 5.0  # Scale to ~0–0.75 range.

    if (7 <= hour < 10) or (16 <= hour < 19):
        base += 0.35
    elif 10 <= hour < 16:
        base += 0.15

    if is_holiday:
        base *= 0.4
    elif is_weekend:
        base *= 0.6

    noise = random.gauss(0, 0.05)
    crowding = base + noise

    return round(max(0.0, min(1.0, crowding)), 3)



def estimate_delay_minutes(normalised_status, precipitation_mm, temp_c) -> float:
    """Estimate delay from status + weather conditions."""
    mean, std = DELAY_ESTIMATES.get(normalised_status, (1.0, 0.5))
    delay = max(0.0, random.gauss(mean, std))

    if precipitation_mm > 10:
        delay += random.uniform(2.0, 6.0)
    elif precipitation_mm > 2:
        delay += random.uniform(0.5, 2.0)

    if temp_c < 0 or temp_c > 30:
        delay += random.uniform(0.5, 2.5)

    return round(max(0.0, delay), 2)



class DataCollector:
    """Pulls TfL status + weather and combines into one row per line."""

    def __init__(self, tfl_client: TfLClient, weather_client: WeatherClient) -> None:
        self._tfl = tfl_client
        self._weather = weather_client
        self._uk_holidays = holidays.UK()

    def collect_snapshot(self) -> List[Dict]:
        snapshot_time = datetime.now().replace(second=0, microsecond=0)
        logger.info("Collecting snapshot at %s", snapshot_time.isoformat())

        weather = self._weather.fetch_london_weather()
        if weather is None:
            logger.error(
                "Weather fetch failed — skipping snapshot to preserve data quality"
            )
            return []

        # temporal features — same for all lines in this snapshot
        hour = snapshot_time.hour
        day_of_week = snapshot_time.weekday()
        month = snapshot_time.month
        is_weekend = int(day_of_week >= 5)
        is_holiday = int(snapshot_time.date() in self._uk_holidays)
        peak_time = int((7 <= hour < 10) or (16 <= hour < 19))

        records: List[Dict] = []
        successful_fetches = 0
        failed_fetches = 0

        for display_name, tfl_id in LINE_ID_MAP.items():
            raw_status = self._tfl.fetch_line_status(tfl_id)

            if raw_status is None:
                logger.warning(
                    "Skipping %s — TfL status fetch failed", display_name
                )
                failed_fetches += 1
                continue

            normalised_status = STATUS_NORMALISATION.get(raw_status, "Good Service")
            if raw_status not in STATUS_NORMALISATION:
                logger.debug(
                    "Unmapped TfL status '%s' for %s — defaulting to Good Service",
                    raw_status,
                    display_name,
                )

            # try live crowding, fall back to synthetic
            crowding = self._tfl.fetch_line_crowding_index(display_name)
            if crowding is None:
                crowding = estimate_crowding(
                    line=display_name,
                    hour=hour,
                    is_weekend=is_weekend,
                    is_holiday=is_holiday,
                )
                logger.debug(
                    "Using synthetic crowding fallback for %s", display_name
                )

            delay_minutes = estimate_delay_minutes(
                normalised_status=normalised_status,
                precipitation_mm=weather["precipitation_mm"],
                temp_c=weather["temp_c"],
            )

            records.append(
                {
                    "timestamp": snapshot_time.isoformat(),
                    "line": display_name,
                    "status": normalised_status,
                    "delay_minutes": delay_minutes,
                    "temp_c": weather["temp_c"],
                    "precipitation_mm": weather["precipitation_mm"],
                    "humidity": weather["humidity"],
                    "crowding_index": crowding,
                    "is_weekend": is_weekend,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "peak_time": peak_time,
                    "is_holiday": is_holiday,
                }
            )
            successful_fetches += 1

        logger.info(
            "Snapshot complete: %d lines collected, %d skipped",
            successful_fetches,
            failed_fetches,
        )
        return records



def append_records_to_csv(records: List[Dict], output_path: Path) -> int:
    """Append to CSV, creating file with headers if it doesn't exist."""
    if not records:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    try:
        with open(output_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=CSV_COLUMNS,
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
                logger.info("Created new CSV at %s", output_path)

            writer.writerows(records)

        logger.debug("Appended %d rows to %s", len(records), output_path)
        return len(records)

    except OSError as exc:
        logger.error("Failed to write to %s: %s", output_path, exc)
        return 0



def run_collection_loop(
    tfl_key: Optional[str],
    weather_key: str,
    interval_seconds: int = COLLECTION_INTERVAL_SECONDS,
    output_path: Path = OUTPUT_CSV,
    run_once: bool = False,
) -> None:
    """Main loop — collect, sleep, repeat."""
    _configure_logging()

    logger.info("=" * 60)
    logger.info("London Underground Data Collector — starting")
    logger.info("Output: %s", output_path)
    logger.info("Interval: %d seconds (%.1f minutes)", interval_seconds, interval_seconds / 60)
    logger.info("TfL key: %s", "provided" if tfl_key else "not provided (lower rate limit)")
    logger.info("=" * 60)

    tfl_client = TfLClient(api_key=tfl_key)
    weather_client = WeatherClient(api_key=weather_key)
    collector = DataCollector(tfl_client=tfl_client, weather_client=weather_client)

    total_records_written = 0
    cycle_count = 0

    while True:
        cycle_start = time.monotonic()
        cycle_count += 1

        logger.info("--- Cycle %d ---", cycle_count)
        records = collector.collect_snapshot()
        written = append_records_to_csv(records, output_path)
        total_records_written += written

        logger.info(
            "Cycle %d complete: %d rows written, %d total rows in dataset",
            cycle_count,
            written,
            total_records_written,
        )

        if run_once:
            logger.info("Single-run mode — exiting after one cycle")
            break

        # adjust sleep for time spent on API calls
        elapsed = time.monotonic() - cycle_start
        sleep_time = max(0, interval_seconds - elapsed)
        next_run = datetime.fromtimestamp(time.time() + sleep_time)
        logger.info("Next collection at %s (sleeping %.0f s)", next_run.strftime("%H:%M:%S"), sleep_time)

        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Interrupted by user — exiting cleanly")
            break



def _configure_logging() -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                DATA_DIR / "collection.log",
                mode="a",
                encoding="utf-8",
            ),
        ],
    )



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect real-time TfL tube delay and London weather data "
            "for dissertation validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables (loaded from .env if python-dotenv is "
            "installed):\n"
            "  TFL_API_KEY      TfL Unified API key (optional)\n"
            "  WEATHER_API_KEY  OpenWeatherMap API key (required)\n"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect a single snapshot and exit (useful for testing).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=COLLECTION_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"Collection interval in seconds (default: {COLLECTION_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        metavar="PATH",
        help=f"Output CSV path (default: {OUTPUT_CSV}).",
    )
    return parser.parse_args()


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import]

        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug("Loaded environment variables from %s", env_path)
    except ImportError:
        pass


def main() -> None:
    args = _parse_args()
    _load_env_file()

    tfl_key: Optional[str] = os.environ.get("TFL_API_KEY") or None
    weather_key: Optional[str] = os.environ.get("WEATHER_API_KEY")

    if not weather_key:
        print(
            "\n[ERROR] WEATHER_API_KEY environment variable is not set.\n"
            "Please run:  python scripts/setup_data_collection.py\n"
            "or set the variable manually before running this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_collection_loop(
        tfl_key=tfl_key,
        weather_key=weather_key,
        interval_seconds=args.interval,
        output_path=args.output,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()
