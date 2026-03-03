"""
API credentials and collection configuration for the real-data pipeline.

I support three configuration sources in priority order:

    1. Direct instantiation — highest priority, most useful in tests.
    2. Environment variables — preferred in production and CI environments.
    3. A ``.env`` file in the project root — convenient for local development.

I use dataclasses rather than ``pydantic`` or ``configparser`` to stay
consistent with the rest of the project (see ``config.py``) and to avoid
introducing additional dependencies.

Usage::

    from credentials.api_config import load_api_keys
    keys = load_api_keys()
    if not keys.is_valid():
        raise RuntimeError("WEATHER_API_KEY must be set")

Author: Marwan
Date: March 2026
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key container
# ---------------------------------------------------------------------------


@dataclass
class APIKeys:
    """
    API credentials for the data collection pipeline.

    I keep this class as a pure data container with no side effects on
    construction.  Validation is performed lazily via ``is_valid()``, which
    I call immediately before starting the collection loop.
    """

    tfl_key: Optional[str] = None
    """TfL Unified API key.  Optional — unauthenticated requests are allowed
    but subject to a lower rate limit (≈50 req/min vs 500 req/min)."""

    weather_key: Optional[str] = None
    """OpenWeatherMap API key.  Required for weather data collection."""

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "APIKeys":
        """
        I construct an ``APIKeys`` instance by reading environment variables.

        Missing variables are stored as ``None`` rather than raised immediately,
        allowing the caller to decide how to handle partial configuration.

        Returns:
            ``APIKeys`` populated from the current environment.
        """
        tfl_key = os.environ.get("TFL_API_KEY") or None
        weather_key = os.environ.get("WEATHER_API_KEY") or None

        logger.debug(
            "Loaded API keys from environment: TfL=%s, Weather=%s",
            "set" if tfl_key else "not set",
            "set" if weather_key else "not set",
        )

        return cls(tfl_key=tfl_key, weather_key=weather_key)

    @classmethod
    def from_env_file(cls, env_path: Optional[Path] = None) -> "APIKeys":
        """
        I load a ``.env`` file and construct an ``APIKeys`` instance.

        I prefer ``python-dotenv`` for robust parsing and fall back to a
        manual parser when it is unavailable.  Existing environment variables
        are not overwritten so that shell-level exports take precedence.

        Args:
            env_path: Path to the ``.env`` file.  Defaults to
                ``<project_root>/.env``.

        Returns:
            ``APIKeys`` populated from the ``.env`` file and/or environment.
        """
        resolved_path = env_path or (Path(__file__).parent.parent / ".env")

        if not resolved_path.exists():
            logger.debug(".env file not found at %s — using environment only", resolved_path)
            return cls.from_env()

        try:
            from dotenv import load_dotenv  # type: ignore[import]

            load_dotenv(resolved_path, override=False)
            logger.debug("Loaded .env from %s using python-dotenv", resolved_path)
        except ImportError:
            logger.debug("python-dotenv not installed — parsing .env manually")
            _parse_env_file_manually(resolved_path)

        return cls.from_env()

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def is_valid(self) -> bool:
        """
        I verify that all mandatory credentials are present.

        The TfL key is optional; the weather key is required.

        Returns:
            ``True`` if the minimum viable configuration is present.
        """
        return bool(self.weather_key)

    def validate(self) -> None:
        """
        I raise ``ValueError`` if required credentials are absent.

        Calling this at startup surfaces configuration problems immediately
        rather than allowing the first API request to fail silently.

        Raises:
            ValueError: If ``weather_key`` is not set.
        """
        if not self.weather_key:
            raise ValueError(
                "WEATHER_API_KEY is required but not set.  "
                "Run 'python scripts/setup_data_collection.py' to configure it."
            )

    # ------------------------------------------------------------------ #
    # Display helpers                                                      #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """I mask secrets when printing so they never appear in logs."""
        tfl_display = _mask(self.tfl_key) if self.tfl_key else "not set (optional)"
        weather_display = _mask(self.weather_key) if self.weather_key else "NOT SET (required)"
        return f"APIKeys(tfl={tfl_display}, weather={weather_display})"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Collection configuration
# ---------------------------------------------------------------------------


@dataclass
class CollectionConfig:
    """
    Tunable parameters for the data collection pipeline, centralised here
    so collection behaviour can be adjusted without touching business logic.
    """

    #: Polling interval; 900 s matches the synthetic data's 15-minute resolution.
    interval_seconds: int = 900

    #: Maximum retry attempts per API request before the call is abandoned.
    max_retries: int = 3

    #: Back-off factor passed to the urllib3 ``Retry`` adapter.
    backoff_factor: float = 0.5

    #: TfL request cap in requests per minute, kept below the authenticated limit.
    tfl_rate_limit_rpm: int = 40

    #: OpenWeatherMap request cap in requests per minute.
    weather_rate_limit_rpm: int = 20

    #: Destination CSV for collected records.
    output_csv: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "tfl_merged.csv"
    )

    #: File path for the collection activity log (mirrors console output).
    log_file: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "collection.log"
    )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_api_keys(env_path: Optional[Path] = None) -> APIKeys:
    """
    Load API keys using the preferred strategy for the current environment,
    delegating to ``APIKeys.from_env_file`` which tries ``python-dotenv``
    before falling back to raw environment variables.

    Args:
        env_path: Optional explicit path to a ``.env`` file.

    Returns:
        Populated ``APIKeys`` instance.
    """
    return APIKeys.from_env_file(env_path=env_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mask(secret: str, visible_chars: int = 4) -> str:
    """
    I mask a secret string for safe display in logs and terminal output.

    Args:
        secret: The string to mask.
        visible_chars: Number of trailing characters to reveal.

    Returns:
        Masked string, e.g. ``"****abcd"``.
    """
    if len(secret) <= visible_chars:
        return "*" * len(secret)
    return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]


def _parse_env_file_manually(path: Path) -> None:
    """
    I parse a ``.env`` file manually when ``python-dotenv`` is unavailable.

    I handle the most common formats:
    - ``KEY=value``
    - ``KEY="value"``
    - ``# comment`` lines (ignored)
    - Blank lines (ignored)

    I do not override variables that are already set in the environment.

    Args:
        path: Path to the ``.env`` file.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Existing environment variables are not overwritten (override=False semantics).
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("Could not read .env file at %s: %s", path, exc)
