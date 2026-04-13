"""
Brand constants for the TfL dashboard.
"""

from typing import Dict, List


# Official TfL line colours
LINE_COLOURS: Dict[str, str] = {
    "Bakerloo":          "#B36305",
    "Central":           "#E32017",
    "Circle":            "#FFD300",
    "District":          "#00782A",
    "Hammersmith & City":"#F3A9BB",
    "Jubilee":           "#A0A5A9",
    "Metropolitan":      "#9B0056",
    "Northern":          "#000000",
    "Piccadilly":        "#003688",
    "Victoria":          "#0098D4",
    "Waterloo & City":   "#95CDBA",
}

STATUS_COLOURS: Dict[str, str] = {
    "Good Service":   "#00B140",
    "Minor Delays":   "#FFD300",
    "Moderate Delays":"#FF6600",
    "Severe Delays":  "#DC241F",
}

ALL_LINES: List[str] = list(LINE_COLOURS.keys())
DATA_COLLECTION_TARGET: int = 14_784
