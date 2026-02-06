"""
Operational scripts for data collection setup and monitoring.

Scripts:
    setup_data_collection.py      -- Interactive wizard: validates API keys, tests
                                     connectivity, writes ``.env``.
    check_collection_progress.py  -- Reports row count, completeness, and estimated
                                     time remaining for the two-week collection target.
"""

__all__ = [
    "setup_data_collection",
    "check_collection_progress",
]
