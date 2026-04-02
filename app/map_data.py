"""
Schematic tube map coordinates for the London Underground network.

Each line is a list of (x, y) waypoints that approximate the
Harry Beck-style map layout.  Coordinates are unitless — they
only need to look right on a Plotly scatter chart.

Key interchange stations are stored separately so the map can
label them without cluttering every stop.
"""

from typing import Dict, List, Tuple

# (x, y) polyline waypoints per tube line
# Coordinates are designed to produce a clean schematic layout
LINE_PATHS: Dict[str, List[Tuple[float, float]]] = {
    "Bakerloo": [
        (3.0, 12.0), (3.5, 11.0), (4.0, 10.0), (4.5, 9.0),
        (5.0, 8.0), (5.5, 7.0), (6.0, 6.0), (6.5, 5.0),
        (7.0, 4.0), (7.5, 3.0),
    ],
    "Central": [
        (0.0, 7.0), (1.5, 7.0), (3.0, 7.0), (4.5, 7.0),
        (6.0, 7.0), (7.5, 7.0), (9.0, 7.0), (10.5, 7.0),
        (12.0, 7.0), (13.5, 7.0), (15.0, 7.0),
    ],
    "Circle": [
        (3.0, 9.0), (4.5, 9.5), (6.0, 9.5), (7.5, 9.5),
        (9.0, 9.5), (10.0, 9.0), (10.0, 8.0), (10.0, 7.0),
        (10.0, 6.0), (9.0, 5.5), (7.5, 5.5), (6.0, 5.5),
        (4.5, 5.5), (3.0, 6.0), (3.0, 7.0), (3.0, 8.0),
        (3.0, 9.0),
    ],
    "District": [
        (1.0, 5.0), (2.0, 5.0), (3.0, 5.5), (4.5, 5.5),
        (6.0, 5.5), (7.5, 5.5), (9.0, 5.5), (10.0, 6.0),
        (11.0, 6.0), (12.0, 5.5), (13.0, 5.0), (14.0, 4.5),
    ],
    "Hammersmith & City": [
        (1.0, 9.0), (2.0, 9.0), (3.0, 9.0), (4.5, 9.5),
        (6.0, 9.5), (7.5, 9.5), (9.0, 9.5), (10.0, 9.0),
        (10.5, 8.5),
    ],
    "Jubilee": [
        (2.0, 11.0), (3.0, 10.0), (4.0, 9.5), (5.0, 9.0),
        (6.0, 8.5), (7.0, 8.0), (8.0, 7.5), (9.0, 7.0),
        (10.0, 6.5), (11.0, 5.5), (12.0, 4.5), (13.0, 3.5),
    ],
    "Metropolitan": [
        (0.5, 12.0), (1.5, 11.5), (2.5, 11.0), (3.0, 10.0),
        (3.0, 9.0), (4.5, 9.5), (6.0, 9.5), (7.5, 9.5),
        (9.0, 9.5), (10.0, 9.0), (10.5, 8.5),
    ],
    "Northern": [
        (5.5, 13.0), (5.5, 12.0), (5.5, 11.0), (5.5, 10.0),
        (5.5, 9.0), (6.0, 8.0), (6.5, 7.0), (7.0, 6.0),
        (7.5, 5.0), (8.0, 4.0), (8.5, 3.0), (9.0, 2.0),
    ],
    "Piccadilly": [
        (0.5, 8.5), (1.5, 8.0), (2.5, 7.5), (3.5, 7.0),
        (5.0, 7.0), (6.0, 7.0), (7.0, 7.5), (8.0, 8.0),
        (9.0, 8.5), (10.0, 9.0), (11.0, 9.5), (12.0, 10.0),
        (13.0, 10.5),
    ],
    "Victoria": [
        (5.0, 12.0), (5.5, 11.0), (6.0, 10.0), (6.5, 9.0),
        (7.0, 8.0), (7.5, 7.0), (8.0, 6.0), (8.5, 5.0),
        (9.0, 4.0),
    ],
    "Waterloo & City": [
        (8.0, 6.5), (9.0, 6.5),
    ],
}


# Major interchange stations — (x, y, label)
INTERCHANGE_STATIONS: List[Tuple[float, float, str]] = [
    (3.0, 9.0,  "Baker Street"),
    (5.5, 9.0,  "King's Cross"),
    (7.0, 8.0,  "Euston"),
    (7.5, 7.0,  "Oxford Circus"),
    (8.0, 6.0,  "Green Park"),
    (9.0, 5.5,  "Westminster"),
    (10.0, 6.0, "Bank"),
    (6.0, 7.0,  "Tottenham Ct Rd"),
    (4.5, 9.5,  "Great Portland St"),
    (9.0, 7.0,  "Liverpool Street"),
    (3.0, 7.0,  "Notting Hill Gate"),
    (5.0, 8.0,  "Warren Street"),
    (6.0, 5.5,  "Victoria"),
    (10.0, 9.0, "Aldgate"),
    (7.5, 5.5,  "Waterloo"),
]
