"""
Static tube line metadata (TfL published data).
"""

from typing import Dict


# Route length in km (from TfL infrastructure data)
LINE_LENGTH_KM: Dict[str, float] = {
    'Bakerloo':            23.2,
    'Central':             74.0,
    'Circle':              27.0,
    'District':            64.0,
    'Hammersmith & City':  25.0,
    'Jubilee':             36.2,
    'Metropolitan':        67.0,
    'Northern':            58.0,
    'Piccadilly':          71.0,
    'Victoria':            21.0,
    'Waterloo & City':      2.4,
}

# Number of stations per line
LINE_N_STATIONS: Dict[str, int] = {
    'Bakerloo':            25,
    'Central':             49,
    'Circle':              36,
    'District':            60,
    'Hammersmith & City':  29,
    'Jubilee':             27,
    'Metropolitan':        34,
    'Northern':            50,
    'Piccadilly':          53,
    'Victoria':            16,
    'Waterloo & City':      2,
}

# Interchange stations per line
LINE_N_INTERCHANGES: Dict[str, int] = {
    'Bakerloo':             7,
    'Central':             14,
    'Circle':              20,
    'District':            16,
    'Hammersmith & City':  10,
    'Jubilee':             10,
    'Metropolitan':        10,
    'Northern':            11,
    'Piccadilly':          13,
    'Victoria':             8,
    'Waterloo & City':      2,
}

# 1 = deep-bore tube, 0 = sub-surface (cut-and-cover)
LINE_IS_DEEP_TUBE: Dict[str, int] = {
    'Bakerloo':            1,
    'Central':             1,
    'Circle':              0,
    'District':            0,
    'Hammersmith & City':  0,
    'Jubilee':             1,
    'Metropolitan':        0,
    'Northern':            1,
    'Piccadilly':          1,
    'Victoria':            1,
    'Waterloo & City':     1,
}

# Number of TfL fare zones each line passes through
LINE_ZONE_COVERAGE: Dict[str, int] = {
    'Bakerloo':            4,
    'Central':             6,
    'Circle':              2,
    'District':            6,
    'Hammersmith & City':  4,
    'Jubilee':             5,
    'Metropolitan':        9,
    'Northern':            6,
    'Piccadilly':          6,   # includes Heathrow branches
    'Victoria':            3,
    'Waterloo & City':     1,
}

# Trains per hour during peak (approx 07:00-10:00, 16:00-19:00)
LINE_PEAK_TPH: Dict[str, int] = {
    'Bakerloo':            24,
    'Central':             32,
    'Circle':              12,
    'District':            24,
    'Hammersmith & City':  12,
    'Jubilee':             36,
    'Metropolitan':        16,
    'Northern':            26,
    'Piccadilly':          24,
    'Victoria':            36,
    'Waterloo & City':      6,  # peak-only service; runs infrequently
}

# Off-peak TPH
LINE_OFFPEAK_TPH: Dict[str, int] = {
    'Bakerloo':            16,
    'Central':             20,
    'Circle':               8,
    'District':            16,
    'Hammersmith & City':   8,
    'Jubilee':             24,
    'Metropolitan':         8,
    'Northern':            20,
    'Piccadilly':          16,
    'Victoria':            24,
    'Waterloo & City':      4,
}

# Baseline delay per line in minutes — roughly matches TfL published averages.
# Moved here from data.py so all static per-line constants live in one place.
LINE_BASE_DELAYS: Dict[str, float] = {
    'Bakerloo':            3.2,
    'Central':             3.0,
    'Circle':              4.0,
    'District':            3.8,
    'Hammersmith & City':  3.5,
    'Jubilee':             2.5,
    'Metropolitan':        2.5,
    'Northern':            3.5,
    'Piccadilly':          2.8,
    'Victoria':            2.0,
    'Waterloo & City':     1.5,
}


def _validate_metadata_completeness():
    canonical = set(LINE_LENGTH_KM.keys())
    all_dicts: Dict[str, Dict] = {
        'LINE_N_STATIONS':    LINE_N_STATIONS,
        'LINE_N_INTERCHANGES': LINE_N_INTERCHANGES,
        'LINE_IS_DEEP_TUBE':  LINE_IS_DEEP_TUBE,
        'LINE_ZONE_COVERAGE': LINE_ZONE_COVERAGE,
        'LINE_PEAK_TPH':      LINE_PEAK_TPH,
        'LINE_OFFPEAK_TPH':   LINE_OFFPEAK_TPH,
        'LINE_BASE_DELAYS':   LINE_BASE_DELAYS,
    }
    for dict_name, d in all_dicts.items():
        missing = canonical - set(d.keys())
        if missing:
            raise ValueError(
                f"line_metadata.{dict_name} is missing entries for: {sorted(missing)}. "
                "Add the missing lines to keep all metadata dictionaries in sync."
            )


# run check at import time so missing entries fail fast
_validate_metadata_completeness()


def get_topology_series(line_name: str) -> Dict[str, float]:
    return {
        'line_length_km':         LINE_LENGTH_KM.get(line_name, 0.0),
        'n_stations':             float(LINE_N_STATIONS.get(line_name, 0)),
        'n_interchange_stations': float(LINE_N_INTERCHANGES.get(line_name, 0)),
        'is_deep_tube':           float(LINE_IS_DEEP_TUBE.get(line_name, 0)),
        'zone_coverage':          float(LINE_ZONE_COVERAGE.get(line_name, 0)),
    }


def get_tph(line_name: str, is_peak: bool) -> int:
    if is_peak:
        return LINE_PEAK_TPH.get(line_name, 12)
    return LINE_OFFPEAK_TPH.get(line_name, 8)
