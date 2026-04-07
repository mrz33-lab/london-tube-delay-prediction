"""
Tab renderers for the Streamlit dashboard.

Five tabs:
  1. Live Overview  — map + KPIs + 24-hour forecast for the selected line
  2. Simulator      — scenario / what-if analysis (unchanged)
  3. Model Diagnostics — consolidated: model comparison, error analysis,
                         per-line MAE, SHAP, line comparison, trends
  4. Data Pipeline  — collection status
  5. About          — project overview
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

from app.constants import LINE_COLOURS, STATUS_COLOURS, ALL_LINES, DATA_COLLECTION_TARGET
from app.charts import (
    create_gauge_chart,
    create_forecast_chart,
    create_line_heatmap,
    create_model_comparison_bar,
    create_feature_importance_chart,
    create_error_distribution,
    create_scatter_actual_vs_pred,
    create_line_perf_bar,
    create_collection_progress_chart,
    create_confusion_matrix_chart,
)
from app.data_loading import load_collection_status


# ─────────────────────────────────────────────────────────────────────────────
# Tube map
# ─────────────────────────────────────────────────────────────────────────────

_TM_COLORS = {
    "Bakerloo":           "#B36305",
    "Central":            "#E32017",
    "Circle":             "#FFD300",
    "District":           "#00782A",
    "Hammersmith & City": "#F3A9BB",
    "Jubilee":            "#A0A5A9",
    "Metropolitan":       "#9B0056",
    "Northern":           "#FFFFFF",
    "Piccadilly":         "#003688",
    "Victoria":           "#0098D4",
    "Waterloo & City":    "#95CDBA",
    "DLR":                "#00A4A7",
    "Elizabeth line":     "#60216E",
    "Lioness line":       "#FFBE18",
    "Mildmay line":       "#006DFF",
    "Suffragette line":   "#18FF74",
    "Weaver line":        "#800080",
    "Windrush line":      "#FF69B4",
    "Liberty line":       "#632E0E",
}

# (x, y, zone)
_TM_STATIONS = {
    # ── Zone 1 central ────────────────────────────────────────────────────────
    "King's Cross":          (520, 220, 1),
    "Euston":                (490, 230, 1),
    "Warren Street":         (470, 250, 1),
    "Oxford Circus":         (460, 280, 1),
    "Bond Street":           (440, 275, 1),
    "Paddington":            (400, 270, 1),
    "Baker Street":          (440, 250, 1),
    "Marylebone":            (430, 255, 1),
    "Regent's Park":         (460, 260, 1),
    "Great Portland Street": (465, 255, 1),
    "Tottenham Court Road":  (480, 285, 1),
    "Holborn":               (500, 290, 1),
    "Chancery Lane":         (510, 290, 1),
    "St Paul's":             (525, 295, 1),
    "Bank":                  (535, 295, 1),
    "Liverpool Street":      (555, 280, 1),
    "Moorgate":              (545, 270, 1),
    "Farringdon":            (520, 275, 1),
    "Barbican":              (530, 272, 1),
    "Angel":                 (530, 245, 1),
    "Old Street":            (550, 260, 1),
    "Aldgate":               (565, 290, 1),
    "Aldgate East":          (568, 293, 1),
    "Tower Hill":            (560, 305, 1),
    "London Bridge":         (545, 315, 1),
    "Borough":               (540, 320, 1),
    "Elephant & Castle":     (530, 335, 1),
    "Waterloo":              (520, 310, 1),
    "Westminster":           (505, 310, 1),
    "St James's Park":       (498, 308, 1),
    "Victoria":              (490, 315, 1),
    "Sloane Square":         (478, 320, 1),
    "South Kensington":      (465, 325, 1),
    "Knightsbridge":         (460, 310, 1),
    "Hyde Park Corner":      (458, 305, 1),
    "Green Park":            (475, 295, 1),
    "Piccadilly Circus":     (475, 290, 1),
    "Leicester Square":      (483, 288, 1),
    "Covent Garden":         (487, 287, 1),
    "Charing Cross":         (490, 295, 1),
    "Embankment":            (492, 305, 1),
    "Temple":                (500, 302, 1),
    "Blackfriars":           (510, 302, 1),
    "Cannon Street":         (530, 302, 1),
    "Mansion House":         (525, 300, 1),
    "Monument":              (537, 300, 1),
    # extra Z1 needed for routing
    "Vauxhall":              (507, 332, 1),
    "Pimlico":               (497, 322, 1),
    "Goodge Street":         (472, 282, 1),
    "Southwark":             (535, 312, 1),
    "Bermondsey":            (555, 325, 2),
    "Lambeth North":         (518, 325, 1),
    "Kennington":            (522, 345, 2),
    "Oval":                  (520, 357, 2),
    "Stepney Green":         (580, 285, 2),
    "Whitechapel":           (573, 290, 2),
    "Edgware Road":          (430, 268, 1),
    "Bayswater":             (425, 285, 2),
    "Notting Hill Gate":     (435, 295, 2),
    "Queensway":             (430, 290, 2),
    "Lancaster Gate":        (443, 287, 1),
    "Marble Arch":           (452, 283, 1),
    "Holland Park":          (427, 292, 2),
    "Shepherd's Bush":       (420, 292, 2),
    "Earl's Court":          (445, 330, 2),
    "Gloucester Road":       (455, 328, 1),
    "Barons Court":          (420, 330, 2),
    "West Kensington":       (432, 335, 2),
    # ── Outer / Zone 2-6 ─────────────────────────────────────────────────────
    "Wembley Park":          (380, 160, 4),
    "Harrow-on-the-Hill":    (350, 140, 5),
    "Stratford":             (620, 260, 2),
    "Canary Wharf":          (600, 320, 2),
    "North Greenwich":       (610, 335, 3),
    "Brixton":               (505, 380, 2),
    "Stockwell":             (508, 365, 2),
    "Clapham Common":        (495, 375, 2),
    "Clapham South":         (490, 385, 2),
    "Balham":                (488, 395, 3),
    "Tooting Broadway":      (485, 408, 3),
    "Wimbledon":             (460, 430, 3),
    "Richmond":              (350, 330, 4),
    "Hammersmith":           (400, 320, 2),
    "Ealing Broadway":       (310, 280, 3),
    "Uxbridge":              (270, 260, 6),
    "Heathrow Terminals 2&3":(280, 360, 6),
    "Upminster":             (720, 295, 6),
    "Barking":               (660, 290, 4),
    "Abbey Wood":            (650, 340, 4),
    "Clapham Junction":      (450, 355, 2),
    "Gospel Oak":            (490, 195, 2),
    "Highbury & Islington":  (535, 225, 2),
    "Seven Sisters":         (555, 195, 3),
    "Tottenham Hale":        (565, 190, 3),
    "Walthamstow Central":   (590, 185, 3),
    "Chingford":             (600, 155, 4),
    "Cockfosters":           (600, 140, 5),
    "Oakwood":               (590, 145, 5),
    "Southgate":             (580, 150, 4),
    "Arnos Grove":           (570, 155, 4),
    "Bounds Green":          (565, 165, 3),
    "Wood Green":            (560, 170, 3),
    "Turnpike Lane":         (558, 175, 3),
    "Manor House":           (555, 185, 3),
    "Finsbury Park":         (540, 205, 2),
    "Highgate":              (510, 185, 3),
    "Archway":               (515, 195, 3),
    "Tufnell Park":          (510, 200, 2),
    "Kentish Town":          (505, 210, 2),
    "Camden Town":           (495, 215, 2),
    "Mornington Crescent":   (492, 222, 2),
    "Edgware":               (430, 130, 5),
    "Burnt Oak":             (435, 145, 4),
    "Colindale":             (438, 155, 4),
    "Hendon Central":        (445, 165, 3),
    "Brent Cross":           (450, 175, 3),
    "Golders Green":         (455, 185, 2),
    "Hampstead":             (462, 195, 2),
    "Belsize Park":          (468, 205, 2),
    "Chalk Farm":            (475, 210, 2),
    "West Hampstead":        (440, 205, 2),
    "Kilburn":               (435, 220, 2),
    "Queen's Park":          (425, 230, 2),
    "Kensal Green":          (415, 235, 2),
    "Willesden Junction":    (405, 240, 3),
    "Harlesden":             (398, 245, 3),
    "Stonebridge Park":      (390, 248, 4),
    "Wembley Central":       (382, 242, 4),
    "Barking Riverside":     (670, 300, 4),
    "West Croydon":          (490, 460, 5),
    "Canada Water":          (565, 325, 2),
    "Rotherhithe":           (568, 322, 2),
    "Surrey Quays":          (570, 330, 2),
    "New Cross Gate":        (575, 345, 2),
    "Brockley":              (572, 352, 3),
    "Honor Oak Park":        (568, 358, 3),
    "Forest Hill":           (565, 365, 3),
    "Sydenham":              (562, 372, 3),
    "Crystal Palace":        (555, 378, 3),
    "Norwood Junction":      (550, 385, 3),
    # Northern line south
    "Clapham North":         (500, 370, 2),
    "Tooting Bec":           (487, 403, 3),
    "Colliers Wood":         (484, 415, 3),
    "South Wimbledon":       (482, 422, 3),
    "Morden":                (480, 430, 4),
    # District east
    "Mile End":              (590, 275, 2),
    "Bow Road":              (600, 275, 3),
    "Bromley-by-Bow":        (608, 278, 3),
    "West Ham":              (615, 270, 3),
    "Plaistow":              (625, 275, 3),
    "Upton Park":            (635, 278, 4),
    "East Ham":              (645, 282, 4),
    "Upney":                 (668, 290, 4),
    "Becontree":             (675, 290, 4),
    "Dagenham Heathway":     (685, 292, 5),
    "Dagenham East":         (695, 292, 5),
    "Elm Park":              (705, 293, 5),
    "Hornchurch":            (713, 294, 6),
    "Upminster Bridge":      (717, 295, 6),
    # Bethnal Green (Central/Weaver)
    "Bethnal Green":         (575, 270, 2),
    # Northern high barnet
    "High Barnet":           (540, 125, 5),
    "Totteridge & Whetstone":(537, 135, 5),
    "Woodside Park":         (535, 145, 4),
    "West Finchley":         (535, 155, 4),
    "Finchley Central":      (533, 165, 4),
    "East Finchley":         (520, 175, 3),
    "Mill Hill East":        (505, 155, 4),
    # Jubilee NW
    "Stanmore":              (350, 155, 5),
    "Canons Park":           (355, 165, 5),
    "Queensbury":            (360, 175, 4),
    "Kingsbury":             (365, 185, 4),
    "Neasden":               (385, 185, 3),
    "Dollis Hill":           (390, 195, 3),
    "Finchley Road":         (455, 200, 2),
    "Swiss Cottage":         (455, 210, 2),
    "St John's Wood":        (450, 225, 1),
    # Piccadilly west
    "Acton Town":            (370, 290, 3),
    "Hammersmith (Picc)":    (400, 320, 2),
    "Turnham Green":         (355, 305, 3),
    "Chiswick Park":         (365, 300, 3),
    "South Ealing":          (340, 295, 3),
    "Northfields":           (333, 292, 3),
    "Boston Manor":          (325, 290, 4),
    "Osterley":              (315, 290, 4),
    "Hounslow East":         (302, 290, 4),
    "Hounslow Central":      (295, 292, 4),
    "Hounslow West":         (288, 287, 5),
    "Hatton Cross":          (282, 368, 5),
    "Heathrow T5":           (270, 380, 6),
    "Heathrow T4":           (280, 385, 6),
    # Central west
    "West Ruislip":          (270, 275, 6),
    "Ruislip Gardens":       (278, 270, 6),
    "South Ruislip":         (285, 268, 5),
    "Northolt":              (295, 266, 5),
    "Greenford":             (303, 266, 4),
    "Perivale":              (312, 268, 4),
    "Hanger Lane":           (322, 272, 3),
    "North Acton":           (335, 280, 3),
    "East Acton":            (348, 283, 2),
    "White City":            (358, 285, 2),
    # H&C west
    "Goldhawk Road":         (415, 300, 2),
    "Shepherd's Bush Market":(418, 298, 2),
    "Wood Lane":             (420, 295, 2),
    "Latimer Road":          (422, 293, 2),
    "Ladbroke Grove":        (425, 290, 2),
    "Westbourne Park":       (415, 280, 2),
    "Royal Oak":             (410, 275, 2),
    # Bakerloo NW
    "Harrow & Wealdstone":   (350, 150, 5),
    "Kenton":                (360, 163, 5),
    "North Wembley":         (368, 178, 4),
    "Kilburn Park":          (432, 225, 2),
    "Maida Vale":            (428, 248, 2),
    "Warwick Avenue":        (418, 260, 2),
    # District west branches
    "Ravenscourt Park":      (375, 315, 2),
    "Stamford Brook":        (362, 318, 3),
    "Gunnersbury":           (353, 322, 3),
    "Kew Gardens":           (345, 327, 4),
    "West Brompton":         (448, 345, 2),
    "Fulham Broadway":       (452, 355, 2),
    "Parsons Green":         (456, 365, 2),
    "Putney Bridge":         (445, 372, 2),
    "East Putney":           (438, 368, 3),
    "Southfields":           (440, 378, 3),
    "Wimbledon Park":        (448, 388, 3),
    # Metropolitan branches
    "Chorleywood":           (292, 148, 6),
    "Rickmansworth":         (303, 153, 6),
    "Moor Park":             (312, 155, 6),
    "Northwood":             (320, 158, 6),
    "Northwood Hills":       (328, 163, 6),
    "Pinner":                (335, 167, 5),
    "North Harrow":          (340, 170, 5),
    "West Harrow":           (345, 177, 5),
    "Rayners Lane":          (348, 184, 5),
    "Eastcote":              (353, 192, 5),
    "Ruislip Manor":         (358, 198, 6),
    "Ruislip":               (362, 204, 6),
    "Ickenham":              (367, 208, 6),
    "Hillingdon":            (268, 252, 6),
    "Chesham":               (278, 130, 6),
    "Amersham":              (283, 138, 6),
    "Chalfont & Latimer":    (297, 147, 6),
    "Watford":               (378, 132, 7),
    "Watford High Street":   (380, 138, 7),
    "Watford Junction":      (376, 128, 7),
    "Northwick Park":        (365, 150, 5),
    "Preston Road":          (370, 155, 5),
    # Piccadilly NE
    "Caledonian Road":       (525, 213, 2),
    "Holloway Road":         (530, 208, 2),
    "Arsenal":               (535, 207, 2),
    # DLR
    "Island Gardens":        (605, 335, 2),
    "Beckton":               (648, 308, 3),
    "Woolwich Arsenal":      (640, 340, 4),
    # Overground
    "Canonbury":             (540, 218, 2),
    "Dalston Junction":      (555, 228, 2),
    "Dalston Kingsland":     (557, 232, 2),
    "Haggerston":            (558, 242, 2),
    "Hoxton":                (556, 250, 2),
    "Shoreditch High Street":(558, 262, 1),
    "Hackney Central":       (572, 215, 2),
    "Hackney Downs":         (566, 222, 2),
    "London Fields":         (568, 232, 2),
    "Cambridge Heath":       (572, 250, 2),
    "Rectory Road":          (560, 198, 3),
    "Stoke Newington":       (558, 192, 3),
    "Stamford Hill":         (557, 186, 3),
    "Crouch Hill":           (522, 180, 3),
    "Upper Holloway":        (516, 188, 3),
    "South Tottenham":       (557, 196, 3),
    "Blackhorse Road":       (570, 196, 3),
    "Walthamstow Queen's Road":(590, 198, 3),
    "Leyton Midland Road":   (607, 222, 3),
    "Wanstead Park":         (616, 227, 3),
    "Woodgrange Park":       (632, 227, 3),
    "Emerson Park":          (702, 295, 6),
    "Romford":               (712, 290, 6),
}

# Lines served at each station (for interchange detection + hover)
def _build_station_lines():
    lines_at = {s: [] for s in _TM_STATIONS}
    seqs = _tm_line_sequences()
    for line, seq in seqs.items():
        for stn in seq:
            if stn is None:
                continue
            if stn in lines_at and line not in lines_at[stn]:
                lines_at[stn].append(line)
    return lines_at

def _tm_line_sequences():
    # None = path break (used to draw branching lines without a connecting line)
    N = None
    return {
        # ── Single straight path ──────────────────────────────────────────────
        "Victoria": [
            "Brixton", "Stockwell", "Oval", "Kennington", "Vauxhall",
            "Pimlico", "Victoria", "Green Park", "Oxford Circus",
            "Warren Street", "Euston", "King's Cross",
            "Highbury & Islington", "Seven Sisters", "Tottenham Hale",
            "Blackhorse Road", "Walthamstow Central",
        ],
        # ── Two northern branches converge at Camden Town, two southern forks ─
        "Northern": [
            # Edgware branch
            "Edgware", "Burnt Oak", "Colindale", "Hendon Central",
            "Brent Cross", "Golders Green", "Hampstead", "Belsize Park",
            "Chalk Farm", "Camden Town",
            # trunk south (shared)
            "Mornington Crescent", "Euston", "Warren Street",
            "Goodge Street", "Tottenham Court Road", "Leicester Square",
            "Charing Cross", "Embankment", "Waterloo",
            # Bank branch south
            "London Bridge", "Borough", "Elephant & Castle",
            "Kennington", "Oval", "Stockwell",
            "Clapham North", "Clapham Common", "Clapham South",
            "Balham", "Tooting Bec", "Tooting Broadway",
            "Colliers Wood", "South Wimbledon", "Morden",
            N,
            # High Barnet branch (draws into Camden Town then joins trunk)
            "High Barnet", "Totteridge & Whetstone", "Woodside Park",
            "West Finchley", "Finchley Central", "East Finchley",
            "Highgate", "Archway", "Tufnell Park", "Kentish Town",
            "Camden Town", "Mornington Crescent",
        ],
        # ── West Ruislip branch + main east trunk ────────────────────────────
        "Central": [
            "West Ruislip", "Ruislip Gardens", "South Ruislip", "Northolt",
            "Greenford", "Perivale", "Hanger Lane", "North Acton",
            "East Acton", "White City", "Shepherd's Bush",
            "Holland Park", "Notting Hill Gate", "Queensway",
            "Lancaster Gate", "Marble Arch", "Bond Street",
            "Oxford Circus", "Tottenham Court Road", "Holborn",
            "Chancery Lane", "St Paul's", "Bank", "Liverpool Street",
            "Bethnal Green", "Mile End", "Stratford",
            "West Ham", "Plaistow", "Upton Park", "East Ham", "Barking",
        ],
        # ── Stanmore branch straight to Stratford ────────────────────────────
        "Jubilee": [
            "Stanmore", "Canons Park", "Queensbury", "Kingsbury",
            "Wembley Park", "Neasden", "Dollis Hill", "Kilburn",
            "West Hampstead", "Finchley Road", "Swiss Cottage",
            "St John's Wood", "Baker Street", "Bond Street",
            "Green Park", "Westminster", "Waterloo", "Southwark",
            "London Bridge", "Bermondsey", "Canada Water",
            "Canary Wharf", "North Greenwich", "Canning Town",
            "West Ham", "Stratford",
        ],
        # ── Heathrow branch + main NE spine; Uxbridge removed (Met only) ──────
        "Piccadilly": [
            # Heathrow terminal loop
            "Heathrow T5", "Heathrow Terminals 2&3", "Hatton Cross",
            N,
            "Heathrow T4", "Heathrow Terminals 2&3",
            N,
            # Main trunk west to east
            "Hatton Cross", "Hounslow West", "Hounslow Central",
            "Hounslow East", "Osterley", "Boston Manor",
            "Northfields", "South Ealing", "Acton Town",
            "Chiswick Park", "Turnham Green", "Hammersmith",
            "Barons Court", "Earl's Court", "Gloucester Road",
            "South Kensington", "Knightsbridge", "Hyde Park Corner",
            "Green Park", "Piccadilly Circus", "Leicester Square",
            "Covent Garden", "Holborn", "King's Cross",
            "Caledonian Road", "Holloway Road", "Arsenal",
            "Finsbury Park", "Manor House", "Turnpike Lane",
            "Wood Green", "Bounds Green", "Arnos Grove", "Southgate",
            "Oakwood", "Cockfosters",
        ],
        # ── Three western branches → main trunk east to Upminster ────────────
        "District": [
            # Richmond branch
            "Richmond", "Kew Gardens", "Gunnersbury", "Stamford Brook",
            "Turnham Green", "Ravenscourt Park", "Hammersmith",
            "Barons Court", "West Kensington", "Earl's Court",
            N,
            # Wimbledon branch
            "Wimbledon", "Wimbledon Park", "Southfields",
            "East Putney", "Putney Bridge", "Parsons Green",
            "Fulham Broadway", "West Brompton", "Earl's Court",
            N,
            # Main trunk (Ealing → Upminster via central London)
            "Earl's Court", "Gloucester Road", "South Kensington",
            "Sloane Square", "Victoria", "St James's Park",
            "Westminster", "Embankment", "Temple", "Blackfriars",
            "Mansion House", "Cannon Street", "Monument", "Tower Hill",
            "Aldgate East", "Whitechapel", "Stepney Green",
            "Mile End", "Bow Road", "Bromley-by-Bow", "West Ham",
            "Plaistow", "Upton Park", "East Ham", "Barking",
            "Upney", "Becontree", "Dagenham Heathway", "Dagenham East",
            "Elm Park", "Hornchurch", "Upminster Bridge", "Upminster",
        ],
        # ── Loop line (close it by repeating first station) ──────────────────
        "Circle": [
            "Hammersmith", "Goldhawk Road", "Shepherd's Bush Market",
            "Wood Lane", "Latimer Road", "Ladbroke Grove",
            "Westbourne Park", "Royal Oak", "Paddington", "Edgware Road",
            "Baker Street", "Great Portland Street", "Euston Square",
            "King's Cross", "Farringdon", "Barbican", "Moorgate",
            "Liverpool Street", "Aldgate", "Tower Hill", "Monument",
            "Cannon Street", "Blackfriars", "Temple", "Embankment",
            "Westminster", "St James's Park", "Victoria",
            "Sloane Square", "South Kensington", "Gloucester Road",
            "Earl's Court", "Notting Hill Gate", "Bayswater",
            "Paddington", "Edgware Road", "Hammersmith",
        ],
        # ── Hammersmith → Barking (straight path) ────────────────────────────
        "Hammersmith & City": [
            "Hammersmith", "Goldhawk Road", "Shepherd's Bush Market",
            "Wood Lane", "Latimer Road", "Ladbroke Grove",
            "Westbourne Park", "Royal Oak", "Paddington", "Edgware Road",
            "Baker Street", "Great Portland Street", "King's Cross",
            "Farringdon", "Barbican", "Moorgate", "Liverpool Street",
            "Whitechapel", "Stepney Green", "Mile End", "Bow Road",
            "Bromley-by-Bow", "West Ham", "Barking",
        ],
        # ── Multiple branches; each drawn separately ──────────────────────────
        "Metropolitan": [
            # Main: Aldgate → Harrow
            "Aldgate", "Liverpool Street", "Moorgate", "Farringdon",
            "King's Cross", "Baker Street", "Finchley Road",
            "Wembley Park", "Harrow-on-the-Hill",
            N,
            # Amersham / Chesham branch
            "Harrow-on-the-Hill", "Northwick Park", "Preston Road",
            "Moor Park", "Rickmansworth", "Chorleywood",
            "Chalfont & Latimer", "Amersham",
            N,
            "Chalfont & Latimer", "Chesham",
            N,
            # Watford branch
            "Moor Park", "Watford",
            N,
            # Uxbridge branch (via Rayners Lane)
            "Harrow-on-the-Hill", "North Harrow", "Pinner",
            "Northwood Hills", "Northwood", "Moor Park",
            N,
            "Harrow-on-the-Hill", "West Harrow", "Rayners Lane",
            "Eastcote", "Ruislip Manor", "Ruislip",
            "Ickenham", "Hillingdon", "Uxbridge",
        ],
        # ── Straight path Harrow & Wealdstone → Elephant & Castle ────────────
        "Bakerloo": [
            "Harrow & Wealdstone", "Kenton", "North Wembley",
            "Wembley Central", "Stonebridge Park", "Harlesden",
            "Willesden Junction", "Kensal Green", "Queen's Park",
            "Kilburn Park", "Maida Vale", "Warwick Avenue", "Paddington",
            "Edgware Road", "Marylebone", "Baker Street", "Regent's Park",
            "Oxford Circus", "Piccadilly Circus", "Charing Cross",
            "Embankment", "Waterloo", "Lambeth North", "Elephant & Castle",
        ],
        "Waterloo & City": ["Waterloo", "Bank"],
        # ── Elizabeth line: Heathrow/Reading + Abbey Wood branches ───────────
        "Elizabeth line": [
            # Heathrow spur
            "Heathrow T5", "Heathrow Terminals 2&3",
            "Ealing Broadway", "Paddington",
            "Bond Street", "Tottenham Court Road", "Farringdon",
            "Liverpool Street", "Whitechapel", "Stratford",
            N,
            # Abbey Wood branch
            "Whitechapel", "Woolwich Arsenal", "Abbey Wood",
        ],
        # ── DLR branches ─────────────────────────────────────────────────────
        "DLR": [
            # Stratford → Woolwich
            "Stratford", "West Ham", "Canning Town",
            "North Greenwich", "Woolwich Arsenal",
            N,
            # Canning Town → Beckton
            "Canning Town", "Beckton",
            N,
            # Canary Wharf spur
            "Canning Town", "Canary Wharf", "Island Gardens",
        ],
        # ── New Overground lines ──────────────────────────────────────────────
        "Lioness line": [
            "Euston", "Wembley Central", "Harrow & Wealdstone",
            "Watford High Street", "Watford Junction",
        ],
        "Mildmay line": [
            "Clapham Junction", "Canonbury", "Highbury & Islington",
            "Dalston Junction", "Haggerston", "Hoxton",
            "Shoreditch High Street", "Whitechapel", "Stratford",
            N,
            "Whitechapel", "Liverpool Street",
        ],
        "Suffragette line": [
            "Gospel Oak", "Upper Holloway", "Crouch Hill",
            "South Tottenham", "Blackhorse Road",
            "Walthamstow Queen's Road", "Leyton Midland Road",
            "Wanstead Park", "Woodgrange Park", "Barking",
            "Barking Riverside",
        ],
        # ── Weaver: Liverpool St → Chingford / Walthamstow ───────────────────
        "Weaver line": [
            "Liverpool Street", "Cambridge Heath", "London Fields",
            "Hackney Central", "Hackney Downs", "Rectory Road",
            "Stoke Newington", "Stamford Hill", "Seven Sisters",
            "Tottenham Hale", "Walthamstow Central",
            N,
            "Tottenham Hale", "Chingford",
        ],
        "Windrush line": [
            "Highbury & Islington", "Canonbury", "Dalston Kingsland",
            "Haggerston", "Hoxton", "Shoreditch High Street",
            "Whitechapel", "Canada Water", "Surrey Quays",
            "New Cross Gate", "Brockley", "Honor Oak Park",
            "Forest Hill", "Sydenham", "Crystal Palace",
            N,
            "Sydenham", "Norwood Junction", "West Croydon",
        ],
        "Liberty line": ["Romford", "Emerson Park", "Upminster"],
    }

# Canning Town not in _TM_STATIONS yet — add it
_TM_STATIONS["Canning Town"] = (618, 310, 3)
_TM_STATIONS["Euston Square"] = (470, 242, 1)


def render_tube_map(line_delays: Dict[str, float], ts_label: str, dark: bool) -> None:
    """
    Draw a schematic TfL-style Tube map using Plotly go.Scatter on a black
    canvas. Lines follow horizontal/vertical/45-degree routing.
    Clicking a line in the legend isolates it (all others hide).
    """
    def _delay_status(d: float) -> str:
        if d < 2:   return "Good Service"
        if d < 5:   return "Minor Delays"
        if d < 10:  return "Moderate Delays"
        return "Severe Delays"

    seqs       = _tm_line_sequences()
    lines_at   = _build_station_lines()

    fig = go.Figure()

    # ── Draw each line ────────────────────────────────────────────────────────
    for line_name, stations in seqs.items():
        xs, ys = [], []
        for stn in stations:
            if stn is None:
                xs.append(None)
                ys.append(None)
            elif stn in _TM_STATIONS:
                x, y, _ = _TM_STATIONS[stn]
                xs.append(x)
                ys.append(y)
            # silently skip unknown station names

        colour = _TM_COLORS.get(line_name, "#888888")
        delay  = line_delays.get(line_name, 0.0)
        status = _delay_status(delay)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=line_name,
            line=dict(color=colour, width=6),
            opacity=1.0,
            legendgroup=line_name,
            showlegend=True,
            hoverinfo="skip",
            customdata=[[line_name, delay, status]] * len(xs),
        ))

    # ── Station markers ───────────────────────────────────────────────────────
    inter_x, inter_y, inter_text = [], [], []
    std_groups: Dict[str, tuple] = {}   # line → (xs, ys, texts)

    for st_name, (sx, sy, zone) in _TM_STATIONS.items():
        served = lines_at.get(st_name, [])
        delay_vals = [line_delays.get(ln, 0.0) for ln in served]
        avg_delay  = sum(delay_vals) / len(delay_vals) if delay_vals else 0.0
        status     = _delay_status(avg_delay)
        lines_str  = ", ".join(served) if served else "—"
        tip = (
            f"<b>{st_name}</b><br>"
            f"Lines: {lines_str}<br>"
            f"Zone: {zone}<br>"
            f"Status: {status}<br>"
            f"Predicted delay: {avg_delay:.1f} min"
        )

        if len(served) >= 2:
            inter_x.append(sx)
            inter_y.append(sy)
            inter_text.append(tip)
        else:
            primary = served[0] if served else "Northern"
            if primary not in std_groups:
                std_groups[primary] = ([], [], [])
            std_groups[primary][0].append(sx)
            std_groups[primary][1].append(sy)
            std_groups[primary][2].append(tip)

    # Standard station dots (one trace per line for legend grouping)
    for line_name, (sxs, sys_, tips) in std_groups.items():
        colour = _TM_COLORS.get(line_name, "#888888")
        fig.add_trace(go.Scatter(
            x=sxs, y=sys_,
            mode="markers",
            name=line_name,
            legendgroup=line_name,
            showlegend=False,
            marker=dict(size=5, color="white", line=dict(width=2, color=colour)),
            text=tips,
            hovertemplate="%{text}<extra></extra>",
        ))

    # ── Station labels (key interchange stations only) ────────────────────────
    # Preferred label position per station — keeps labels off the lines
    _label_pos = {
        "King's Cross":           "top right",
        "Euston":                 "top left",
        "Oxford Circus":          "bottom left",
        "Bond Street":            "top left",
        "Paddington":             "bottom left",
        "Baker Street":           "top left",
        "Liverpool Street":       "top right",
        "Moorgate":               "top right",
        "Bank":                   "bottom right",
        "Waterloo":               "bottom left",
        "Westminster":            "bottom center",
        "Victoria":               "bottom left",
        "Green Park":             "bottom right",
        "London Bridge":          "bottom right",
        "Canary Wharf":           "bottom right",
        "Stratford":              "top right",
        "Hammersmith":            "bottom left",
        "Brixton":                "bottom center",
        "Wembley Park":           "top left",
        "Finsbury Park":          "top left",
        "Camden Town":            "top left",
        "Highbury & Islington":   "top right",
        "Canada Water":           "bottom right",
        "Elephant & Castle":      "bottom left",
        "Clapham Junction":       "bottom left",
        "Warren Street":          "top left",
        "Leicester Square":       "bottom right",
        "Charing Cross":          "bottom right",
        "Embankment":             "bottom right",
        "Farringdon":             "top right",
        "Aldgate":                "bottom right",
        "Tower Hill":             "bottom right",
        "Earl's Court":           "bottom left",
    }

    lx, ly, lt, lpos = [], [], [], []
    for st_name, (sx, sy, zone) in _TM_STATIONS.items():
        if st_name in _label_pos:
            served = lines_at.get(st_name, [])
            delay_vals = [line_delays.get(ln, 0.0) for ln in served]
            avg_delay  = sum(delay_vals) / len(delay_vals) if delay_vals else 0.0
            status     = _delay_status(avg_delay)
            lines_str  = ", ".join(served) if served else "—"
            tip = (
                f"<b>{st_name}</b><br>"
                f"Lines: {lines_str}<br>"
                f"Zone: {zone}<br>"
                f"Status: {status}<br>"
                f"Predicted delay: {avg_delay:.1f} min"
            )
            lx.append(sx);  ly.append(sy)
            lt.append(st_name);  lpos.append(_label_pos[st_name])

    fig.add_trace(go.Scatter(
        x=lx, y=ly,
        mode="text",
        text=lt,
        textfont=dict(size=8, color="#dddddd",
                      family="'Johnston 100', 'Helvetica Neue', Arial, sans-serif"),
        textposition=lpos,
        hoverinfo="skip",
        showlegend=False,
        name="Labels",
    ))

    # Interchange circles (always on top)
    fig.add_trace(go.Scatter(
        x=inter_x, y=inter_y,
        mode="markers",
        name="Interchange",
        showlegend=False,
        marker=dict(
            size=11,
            color="white",
            line=dict(width=2.5, color="#1a1a1a"),
            symbol="circle",
        ),
        text=inter_text,
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        height=720,
        margin=dict(l=10, r=10, t=50, b=130),
        title=dict(
            text="<b>London Underground</b>   Network Delay Status",
            font=dict(size=14, color="#e8edf5",
                      family="'Johnston 100', 'Helvetica Neue', Arial, sans-serif"),
            x=0.01, y=0.99,
        ),
        xaxis=dict(visible=False, range=[225, 765]),
        yaxis=dict(visible=False, range=[475, 95]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(color="#cccccc", size=9,
                      family="'Johnston 100', 'Helvetica Neue', Arial, sans-serif"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#2a2a2a",
            borderwidth=1,
            itemclick="toggleothers",
            itemdoubleclick="toggleothers",
            traceorder="normal",
        ),
        hovermode="closest",
        dragmode=False,
        hoverlabel=dict(
            bgcolor="#1c1f26",
            bordercolor="#3a4a6a",
            font_color="#e8edf5",
            font_size=12,
            font_family="'Helvetica Neue', Arial, sans-serif",
        ),
    )

    st.caption("Click a line in the legend to isolate it · Double-click to reset")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status_for_delay(delay: float):
    """Return (colour, label) for a delay value in minutes."""
    if delay < 2:
        return "#00B140", "Good Service"
    elif delay < 5:
        return "#FFD300", "Minor Delays"
    elif delay < 10:
        return "#FF6600", "Moderate Delays"
    else:
        return "#DC241F", "Severe Delays"


def _card(dark: bool, border_colour: str, content_html: str) -> str:
    """Render a styled card div."""
    bg  = "#111827" if dark else "#ffffff"
    bdr = "#1f2937" if dark else "#e5e7eb"
    return f"""
    <div style="background:{bg}; border:1px solid {bdr}; border-left:5px solid {border_colour};
                border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:0.8rem;
                box-shadow:0 2px 10px rgba(0,0,0,0.07); transition:transform .2s;
                font-family: 'Inter', sans-serif;">
        {content_html}
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Live Overview
# ─────────────────────────────────────────────────────────────────────────────

def render_live_overview_tab(
    artifacts: Dict,
    selected_line: str,
    model_col: str,
    model_choice: str,
    date_range,
    dark: bool,
    replay_ts=None,
) -> None:
    """
    Combine the Network Map and per-line Predictions into a single first tab
    so users immediately see the network-wide picture and their line's forecast.
    """
    test_preds = artifacts.get("test_predictions")
    metrics    = artifacts.get("metrics", {})

    if test_preds is None:
        st.warning("No test predictions found. Please run `python train.py` first.")
        return

    # ── Apply date filter ────────────────────────────────────────────────────
    filtered = test_preds.copy()
    if date_range and len(date_range) == 2:
        s, e = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= s) &
            (filtered["timestamp"].dt.date <= e)
        ]

    line_df = filtered[filtered["line"] == selected_line]

    # ── KPI row ──────────────────────────────────────────────────────────────
    m_key     = model_choice.lower()
    m_data    = metrics.get(m_key, {})
    naive_data = metrics.get("naive", {})

    mae  = m_data.get("test_mae",  0)
    rmse = m_data.get("test_rmse", 0)
    r2   = m_data.get("test_r2",   0)
    impr = (1 - mae / naive_data["test_mae"]) * 100 if naive_data.get("test_mae") else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Test MAE", f"{mae:.2f} min",
                  delta=f"{-impr:.1f}% vs Naive", delta_color="inverse")
    with k2:
        st.metric("Test RMSE", f"{rmse:.2f} min")
    with k3:
        st.metric("R² Score", f"{r2:.3f}")
    with k4:
        avg_delay = float(line_df["actual"].mean()) if not line_df.empty else 0.0
        st.metric(f"{selected_line} Avg Delay", f"{avg_delay:.1f} min")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Network Map ──────────────────────────────────────────────────────────
    st.markdown("### Network Status")

    # Determine snapshot
    if replay_ts is not None:
        snapshot = test_preds[test_preds["timestamp"] == replay_ts]
        ts_label = str(replay_ts)[:16]
    else:
        latest_ts = test_preds["timestamp"].max()
        snapshot  = test_preds[test_preds["timestamp"] == latest_ts]
        ts_label  = str(latest_ts)[:16]

    line_delays: Dict[str, float] = {}
    for line in ALL_LINES:
        ldf = snapshot[snapshot["line"] == line]
        if not ldf.empty:
            line_delays[line] = float(ldf[model_col].mean())
        else:
            fallback = test_preds[test_preds["line"] == line]
            line_delays[line] = float(fallback[model_col].mean()) if not fallback.empty else 0.0

    # Status summary strip
    text_col = "#e8edf5" if dark else "#111827"
    muted    = "#8b949e" if dark else "#6b7280"
    strip_bg = "#111827" if dark else "#f3f4f8"
    st.markdown(f"""
    <div style="background:{strip_bg}; border-radius:10px; padding:0.7rem 1.2rem;
                margin-bottom:1rem; display:flex; align-items:center;
                justify-content:space-between; font-family:'Inter',sans-serif;">
        <span style="font-weight:600; font-size:0.88rem; color:{text_col};">
            Snapshot: <span style="color:{muted}; font-weight:400;">{ts_label}</span>
        </span>
        <span style="font-weight:600; font-size:0.88rem; color:{text_col};">
            Lines with delays (&ge;5 min):
            <span style="color:#DC241F; font-weight:700;">
                {sum(1 for v in line_delays.values() if v >= 5)}
            </span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    render_tube_map(line_delays, ts_label, dark)

    # Status count cards row (4 statuses)
    statuses = [
        ("Good Service",     "#00B140", sum(1 for v in line_delays.values() if v < 2)),
        ("Minor Delays",     "#FFD300", sum(1 for v in line_delays.values() if 2 <= v < 5)),
        ("Moderate Delays",  "#FF6600", sum(1 for v in line_delays.values() if 5 <= v < 10)),
        ("Severe Delays",    "#DC241F", sum(1 for v in line_delays.values() if v >= 10)),
    ]
    card_bg  = "#111827" if dark else "#ffffff"
    card_bdr = "#1f2937" if dark else "#e5e7eb"
    muted2   = "#8b949e" if dark else "#6b7280"
    cols = st.columns(4)
    for i, (label, colour, count) in enumerate(statuses):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center; background:{card_bg}; border:1px solid {card_bdr};
                        border-radius:12px; padding:0.9rem; border-top:4px solid {colour};
                        font-family:'Inter',sans-serif;">
                <div style="font-size:2rem; font-weight:800; color:{colour}; line-height:1.2;">{count}</div>
                <div style="font-size:0.75rem; font-weight:600; color:{muted2}; margin-top:0.25rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Selected Line Forecast ────────────────────────────────────────────────
    st.markdown(f"### {selected_line} Line — 24-Hour Forecast")

    pred_col, gauge_col = st.columns([2, 1])

    with gauge_col:
        mean_pred = float(line_df[model_col].mean()) if not line_df.empty else 0.0
        badge_col, badge_txt = _status_for_delay(mean_pred)
        line_col = LINE_COLOURS.get(selected_line, "#003688")
        st.plotly_chart(
            create_gauge_chart(mean_pred, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(f"""
        <div style="text-align:center; margin-top:0.3rem; font-family:'Inter',sans-serif;">
            <span class="line-pill" style="background:{line_col};">{selected_line}</span>
            <span class="status-badge" style="background:{badge_col}20; color:{badge_col}; border:1.5px solid {badge_col};">
                {badge_txt}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with pred_col:
        st.plotly_chart(
            create_forecast_chart(filtered, selected_line, model_col, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Summary statistics (collapsed) ───────────────────────────────────────
    with st.expander(f"Summary Statistics — {selected_line}", expanded=False):
        if line_df.empty:
            st.info("No data in selected date range.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Actual Delays**")
                s = line_df["actual"]
                st.table(pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                    "Value":  [f"{s.mean():.2f} min", f"{s.median():.2f} min",
                               f"{s.std():.2f} min",  f"{s.min():.2f} min", f"{s.max():.2f} min"],
                }))
            with c2:
                st.markdown(f"**Predicted Delays ({model_choice})**")
                p = line_df[model_col]
                st.table(pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                    "Value":  [f"{p.mean():.2f} min", f"{p.median():.2f} min",
                               f"{p.std():.2f} min",  f"{p.min():.2f} min", f"{p.max():.2f} min"],
                }))

    # ── Hour-of-day heatmap (collapsed) ──────────────────────────────────────
    with st.expander("Hour-of-Day Delay Heatmap (all lines)", expanded=False):
        st.plotly_chart(
            create_line_heatmap(filtered, model_col, dark=dark),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Scenario Simulator (largely unchanged, minus emojis)
# ─────────────────────────────────────────────────────────────────────────────

def render_simulator_tab(artifacts: Dict, selected_line: str, model_col: str, dark: bool) -> None:
    """
    What-if scenario simulator: adjust feature values and see how the model's
    predicted delay responds.
    """
    from app.charts import create_sensitivity_chart
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config

    test_preds = artifacts.get("test_predictions")
    if test_preds is None:
        st.warning("No prediction data available. Please run `python train.py` first.")
        return

    st.markdown("### Scenario Simulator")
    st.caption(
        "Adjust the sliders below to explore how different conditions affect predicted delay. "
        "Uses the trained feature distributions as baselines."
    )

    line_df = test_preds[test_preds["line"] == selected_line]
    if line_df.empty:
        st.info(f"No data available for {selected_line}.")
        return

    # ── Slider controls ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temporal conditions**")
        hour = st.slider("Hour of day", 0, 23, 8, key="sim_hour")
        is_peak = st.checkbox("Peak hours", value=(7 <= hour <= 9 or 16 <= hour <= 19), key="sim_peak")
        is_weekend = st.checkbox("Weekend", value=False, key="sim_weekend")
        is_holiday = st.checkbox("Bank holiday", value=False, key="sim_holiday")

    with c2:
        st.markdown("**Weather conditions**")
        temp_c   = st.slider("Temperature (°C)", -5.0, 35.0, 15.0, step=0.5, key="sim_temp")
        rain_mm  = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, step=0.5, key="sim_rain")
        humidity = st.slider("Humidity (%)", 20, 100, 60, key="sim_humidity")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Estimate baseline delay from test data ────────────────────────────────
    # Filter to the chosen hour/peak conditions as a rough analogue
    mask = line_df["timestamp"].dt.hour == hour
    if is_peak:
        mask &= line_df[model_col] > line_df[model_col].median()
    sim_base = float(line_df.loc[mask, model_col].mean()) if mask.any() else float(line_df[model_col].mean())

    # Apply simple heuristic deltas on top of the data-driven baseline
    delta = 0.0
    if rain_mm > 5:
        delta += rain_mm * 0.08
    if temp_c > 28 or temp_c < 2:
        delta += abs(temp_c - 15) * 0.05
    if is_holiday:
        delta -= 0.5   # fewer trains → fewer cascading delays
    if is_weekend:
        delta -= 0.3
    sim_pred = max(0.0, sim_base + delta)

    badge_col, badge_txt = _status_for_delay(sim_pred)
    line_col = LINE_COLOURS.get(selected_line, "#003688")
    bg_card  = "#111827" if dark else "#ffffff"
    bdr_card = "#1f2937" if dark else "#e5e7eb"
    txt_col  = "#e8edf5" if dark else "#111827"
    muted    = "#8b949e" if dark else "#6b7280"

    st.markdown("#### Simulated Prediction")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Predicted Delay", f"{sim_pred:.2f} min", delta=f"{delta:+.2f} min vs baseline")
    with r2:
        st.metric("Baseline (this hour)", f"{sim_base:.2f} min")
    with r3:
        st.metric("Weather Adjustment", f"{delta:+.2f} min")

    st.markdown(f"""
    <div style="background:{bg_card}; border:1px solid {bdr_card}; border-left:5px solid {badge_col};
                border-radius:12px; padding:1.1rem 1.4rem; margin:1rem 0; font-family:'Inter',sans-serif;">
        <div style="display:flex; align-items:center; gap:1rem;">
            <span class="line-pill" style="background:{line_col}; font-size:0.9rem;">{selected_line}</span>
            <span style="font-size:1.5rem; font-weight:800; color:{badge_col};">{sim_pred:.1f} min</span>
            <span class="status-badge" style="background:{badge_col}20; color:{badge_col}; border:1.5px solid {badge_col};">
                {badge_txt}
            </span>
        </div>
        <div style="font-size:0.8rem; color:{muted}; margin-top:0.6rem;">
            Scenario: Hour={hour:02d}:00 &nbsp;|&nbsp;
            Peak={is_peak} &nbsp;|&nbsp;
            Weekend={is_weekend} &nbsp;|&nbsp;
            Temperature={temp_c}°C &nbsp;|&nbsp;
            Rain={rain_mm}mm
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sensitivity sweeps ───────────────────────────────────────────────────
    st.markdown("#### Sensitivity Analysis")
    st.caption("How does predicted delay change as each feature varies, holding others constant?")

    sweep_col1, sweep_col2 = st.columns(2)

    with sweep_col1:
        rain_vals   = [round(r, 1) for r in np.arange(0, 20.5, 0.5)]
        rain_delays = [max(0.0, sim_base + r * 0.08) for r in rain_vals]
        st.plotly_chart(
            create_sensitivity_chart("Rainfall (mm)", rain_vals, rain_delays, selected_line, dark=dark),
            use_container_width=True,
        )

    with sweep_col2:
        temp_vals   = [round(t, 1) for t in np.arange(-5, 35.5, 0.5)]
        temp_delays = [max(0.0, sim_base + abs(t - 15) * 0.05) for t in temp_vals]
        st.plotly_chart(
            create_sensitivity_chart("Temperature (°C)", temp_vals, temp_delays, selected_line, dark=dark),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Model Diagnostics (consolidated)
# ─────────────────────────────────────────────────────────────────────────────

def render_diagnostics_tab(
    artifacts: Dict,
    model_col: str,
    model_choice: str,
    selected_line: str,
    dark: bool,
) -> None:
    """
    Consolidated diagnostics tab replacing the old Performance, Line Comparison,
    and Trends tabs. Uses a sub-tab layout so users can drill down without clutter.
    """
    metrics    = artifacts.get("metrics", {})
    test_preds = artifacts.get("test_predictions")
    feat_imp   = artifacts.get("feature_importance")
    comp_df    = artifacts.get("model_comparison")

    if test_preds is None:
        st.warning("No prediction data available.")
        return

    sub_a, sub_b, sub_c = st.tabs([
        "Model Performance",
        "Line Analysis",
        "Trends",
    ])

    # ── Sub-tab A: Model Performance ──────────────────────────────────────────
    with sub_a:
        st.markdown("### Model Comparison")
        st.caption("Comparing the three trained models against each other and a naive baseline.")

        if metrics:
            st.plotly_chart(
                create_model_comparison_bar(metrics, dark=dark),
                use_container_width=True,
            )
        else:
            st.info("Metrics not available.")

        if comp_df is not None:
            if "Test MAE" in comp_df.columns and len(comp_df) > 1:
                naive_mae = comp_df[comp_df["Model"].str.lower() == "naive"]["Test MAE"].values
                if len(naive_mae):
                    comp_df = comp_df.copy()
                    comp_df["Improvement vs Naive"] = comp_df["Test MAE"].apply(
                        lambda x: f"{(1 - x / naive_mae[0]) * 100:+.1f}%"
                    )

            def _colour_row(row):
                if "best" in str(row.get("Model", "")).lower():
                    return ["background-color: rgba(0,177,64,0.15)"] * len(row)
                return [""] * len(row)

            with st.expander("Detailed Metrics Table", expanded=True):
                st.dataframe(
                    comp_df.style.apply(_colour_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Error Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                create_error_distribution(test_preds, model_col, dark=dark),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                create_scatter_actual_vs_pred(test_preds, model_col, dark=dark),
                use_container_width=True,
            )

        st.markdown("### Service Status Confusion Matrix")
        st.caption(
            "Shows how well the model predicts the categorical TfL service status "
            "derived from predicted delay minutes."
        )
        st.plotly_chart(
            create_confusion_matrix_chart(test_preds, model_col, dark=dark),
            use_container_width=True,
        )

        if feat_imp is not None:
            st.markdown("### Feature Importance (SHAP)")
            st.caption(
                "SHAP values quantify each feature's contribution to individual predictions. "
                "Higher absolute values indicate greater influence on the predicted delay."
            )
            st.plotly_chart(
                create_feature_importance_chart(feat_imp, dark=dark),
                use_container_width=True,
            )
        else:
            st.info("Feature importance data not found. Run `python explain.py` to generate it.")

    # ── Sub-tab B: Line Analysis ──────────────────────────────────────────────
    with sub_b:
        st.markdown("### Performance by Tube Line")
        st.caption("Mean absolute error per line — taller bars indicate harder-to-predict lines.")
        st.plotly_chart(
            create_line_perf_bar(test_preds, model_col, dark=dark),
            use_container_width=True,
        )

        st.markdown("### All Lines — Snapshot")
        st.caption("Mean predicted delay across the latest available data window for each line.")

        records = []
        for line in ALL_LINES:
            ldf = test_preds[test_preds["line"] == line]
            if ldf.empty:
                records.append({"line": line, "pred": 0.0, "actual": 0.0, "mae": 0.0, "n": 0})
            else:
                records.append({
                    "line":   line,
                    "pred":   float(ldf[model_col].mean()),
                    "actual": float(ldf["actual"].mean()),
                    "mae":    float(np.abs(ldf["actual"] - ldf[model_col]).mean()),
                    "n":      len(ldf),
                })
        records.sort(key=lambda x: x["pred"], reverse=True)

        bg_card  = "#111827" if dark else "#ffffff"
        bdr_card = "#1f2937" if dark else "#e5e7eb"
        txt_col  = "#e8edf5" if dark else "#111827"
        muted    = "#8b949e" if dark else "#6b7280"

        cols = st.columns(3)
        for i, rec in enumerate(records):
            line  = rec["line"]
            pred  = rec["pred"]
            mae   = rec["mae"]
            lc    = LINE_COLOURS.get(line, "#003688")
            sc, sl = _status_for_delay(pred)

            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:{bg_card}; border:1px solid {bdr_card}; border-left:5px solid {lc};
                            border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem;
                            font-family:'Inter',sans-serif;">
                    <div style="font-weight:700; font-size:0.92rem; color:{txt_col};">{line}</div>
                    <div style="font-size:1.9rem; font-weight:800; color:{lc}; margin:0.15rem 0; line-height:1.2;">
                        {pred:.1f}<span style="font-size:0.9rem; font-weight:400;"> min</span>
                    </div>
                    <span style="background:{sc}20; color:{sc}; border:1px solid {sc};
                                 border-radius:12px; padding:0.12rem 0.55rem;
                                 font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;">
                        {sl}
                    </span>
                    <div style="font-size:0.73rem; color:{muted}; margin-top:0.4rem;">
                        MAE: {mae:.2f} min &nbsp;·&nbsp; n={rec['n']:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("View as sortable table", expanded=False):
            df_view = pd.DataFrame(records).rename(columns={
                "line": "Line", "pred": "Avg Predicted (min)",
                "actual": "Avg Actual (min)", "mae": "MAE (min)", "n": "Records",
            })
            st.dataframe(
                df_view.style.background_gradient(subset=["Avg Predicted (min)"], cmap="RdYlGn_r"),
                use_container_width=True,
                hide_index=True,
            )

    # ── Sub-tab C: Trends ─────────────────────────────────────────────────────
    with sub_c:
        st.markdown(f"### Historical Accuracy — {selected_line} Line")

        line_df = test_preds[test_preds["line"] == selected_line].sort_values("timestamp")

        if line_df.empty:
            st.info(f"No data found for {selected_line}.")
            return

        lc       = LINE_COLOURS.get(selected_line, "#003688")
        paper_bg = "#0d1117" if dark else "#ffffff"
        plot_bg  = "#111827" if dark else "#fafbfc"
        font_col = "#e8edf5" if dark else "#111827"
        grid_col = "#1f2937" if dark else "#e5e7eb"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=line_df["timestamp"], y=line_df["actual"],
            mode="lines", name="Actual", line=dict(color="#6b7280", width=1.5, dash="dot"),
            hovertemplate="Actual: %{y:.1f} min<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=line_df["timestamp"], y=line_df[model_col],
            mode="lines", name="Predicted", line=dict(color=lc, width=2),
            hovertemplate="Predicted: %{y:.1f} min<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
            font=dict(color=font_col, family="'Inter', sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(text="Predictions vs Actuals Over Time", font=dict(size=15, color=font_col), x=0.01),
            xaxis=dict(gridcolor=grid_col),
            yaxis=dict(title="Delay (minutes)", gridcolor=grid_col),
            hovermode="x unified", height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        residuals = line_df["actual"] - line_df[model_col]
        fig2 = go.Figure()
        fig2.add_hline(y=0, line=dict(color="#FFD300", dash="dash", width=1.5))
        fig2.add_trace(go.Scatter(
            x=line_df["timestamp"], y=residuals,
            mode="markers", name="Residual",
            marker=dict(
                size=4, opacity=0.55,
                color=residuals,
                colorscale=[[0, "#00B140"], [0.5, "#FFD300"], [1, "#DC241F"]],
            ),
            hovertemplate="Residual: %{y:.1f} min<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
            font=dict(color=font_col, family="'Inter', sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(text="Residuals Over Time", font=dict(size=15, color=font_col), x=0.01),
            xaxis=dict(gridcolor=grid_col),
            yaxis=dict(title="Residual (min)", gridcolor=grid_col),
            height=280, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Average Delay by Hour of Day")
        hourly = line_df.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly_agg = hourly.groupby("hour").agg(
            actual_mean=("actual", "mean"),
            pred_mean=(model_col, "mean"),
        ).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=hourly_agg["hour"], y=hourly_agg["actual_mean"],
            mode="lines+markers", name="Actual (avg)",
            line=dict(color="#6b7280", width=2), marker=dict(size=5),
        ))
        fig3.add_trace(go.Scatter(
            x=hourly_agg["hour"], y=hourly_agg["pred_mean"],
            mode="lines+markers", name="Predicted (avg)",
            line=dict(color=lc, width=2), marker=dict(size=5),
        ))
        fig3.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
            font=dict(color=font_col, family="'Inter', sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(text="Average Delay by Hour of Day", font=dict(size=15, color=font_col), x=0.01),
            xaxis=dict(title="Hour", tickvals=list(range(0, 24, 2)), gridcolor=grid_col),
            yaxis=dict(title="Avg Delay (min)", gridcolor=grid_col),
            height=320, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def render_data_collection_tab(config, dark: bool) -> None:
    """
    Render the Data Pipeline tab with live progress metrics sourced from
    the merged CSV, showing collection rate, ETA, and per-line coverage.
    """
    st.markdown("### Data Collection Status")

    with st.spinner("Reading collection data…"):
        status = load_collection_status(str(config.paths.data_dir))

    is_active  = status["is_active"]
    h_col      = "#00B140" if is_active else "#DC241F"
    active_txt = "Active" if is_active else "Inactive"
    bg_hdr     = "#111827" if dark else "#f8f9fa"
    bdr_hdr    = "#1f2937" if dark else "#e5e7eb"
    txt_col    = "#e8edf5" if dark else "#111827"
    muted      = "#8b949e" if dark else "#6b7280"

    st.markdown(f"""
    <div style="background:{bg_hdr}; border:1px solid {bdr_hdr}; border-radius:12px;
                padding:1.2rem 1.5rem; margin-bottom:1rem; font-family:'Inter',sans-serif;">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <span style="font-size:1.3rem; font-weight:800; color:{txt_col};">Data Collection</span>
                <span style="margin-left:1rem; font-size:0.85rem; font-weight:700; color:{h_col};">
                    &bull; {active_txt}
                </span>
            </div>
            <div style="font-size:0.78rem; color:{muted};">
                Target: {DATA_COLLECTION_TARGET:,} records (2 weeks @ 15 min)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not status["has_data"]:
        st.info(
            "No data collected yet. Start collection with:\n"
            "```\npython data_collection.py\n```"
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records Collected", f"{status['record_count']:,}")
    with c2:
        rate = status["rate_per_hour"]
        st.metric("Collection Rate", f"{rate:.0f} rec/hr")
    with c3:
        if status["first_ts"] and status["last_ts"]:
            elapsed = status["last_ts"] - status["first_ts"]
            d, s = elapsed.days, elapsed.seconds
            elapsed_str = f"{d}d {s//3600}h {(s%3600)//60}m"
        else:
            elapsed_str = "—"
        st.metric("Time Elapsed", elapsed_str)
    with c4:
        if status["eta_hours"] is not None:
            eta_d = int(status["eta_hours"] // 24)
            eta_h = int(status["eta_hours"] % 24)
            eta_str = f"{eta_d}d {eta_h}h"
        else:
            eta_str = "Complete"
        st.metric("ETA to Target", eta_str)

    st.markdown("<br>", unsafe_allow_html=True)

    ring_col, bar_col = st.columns([1, 2])
    with ring_col:
        st.plotly_chart(
            create_collection_progress_chart(status, dark=dark),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with bar_col:
        st.markdown(f"**Progress: {status['record_count']:,} / {status['target']:,} records**")
        st.progress(min(status["pct"] / 100, 1.0))
        st.caption(f"{status['pct']:.1f}% complete")

        if status["first_ts"]:
            st.markdown(f"**First record:** {status['first_ts'].strftime('%Y-%m-%d %H:%M')}")
        if status["last_ts"]:
            st.markdown(f"**Last record:** {status['last_ts'].strftime('%Y-%m-%d %H:%M')}")

    if status["lines_present"]:
        st.markdown("**Lines with collected data:**")
        pills = "".join(
            f'<span class="line-pill" style="background:{LINE_COLOURS.get(l, "#003688")};">{l}</span>'
            for l in status["lines_present"]
        )
        st.markdown(f'<div style="margin:0.5rem 0;">{pills}</div>', unsafe_allow_html=True)

    with st.expander("How to run data collection", expanded=False):
        st.markdown("""
        **Start continuous collection (every 15 minutes):**
        ```bash
        python data_collection.py
        ```

        **Test a single collection cycle:**
        ```bash
        python data_collection.py --once
        ```

        **Monitor progress:**
        ```bash
        python scripts/check_collection_progress.py
        ```

        **Data is stored at:** `data/tfl_merged.csv`

        Collection requires:
        - `TFL_APP_KEY` (optional but recommended for higher rate limits)
        - `OPENWEATHERMAP_API_KEY` (required for weather data)

        Store keys in `.env` file (see `.env.example`).
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 – About
# ─────────────────────────────────────────────────────────────────────────────

def render_about_tab(artifacts: Dict) -> None:
    """
    Project overview: purpose, methodology, tech stack, and ethical considerations.
    """
    best    = artifacts.get("best_model_name", "lightgbm").upper()
    metrics = artifacts.get("metrics", {})
    best_key = artifacts.get("best_model_name", "best")
    mae  = metrics.get(best_key, {}).get("test_mae",  "—")
    r2   = metrics.get(best_key, {}).get("test_r2",   "—")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #002060 0%, #003688 50%, #005ab5 100%);
                border-radius: 16px; padding: 2rem; color: white; margin-bottom: 1.5rem;
                font-family:'Inter',sans-serif;">
        <h2 style="margin:0; font-weight:800; letter-spacing:-0.02em;">London Underground Delay Predictor</h2>
        <p style="margin:0.4rem 0 0; opacity:0.85; font-size:0.95rem;">
            COMP1682 Final Year Project &nbsp;·&nbsp; University of Greenwich &nbsp;·&nbsp; 2026
        </p>
        <div style="margin-top:1.2rem; display:flex; gap:2.5rem; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.7rem; opacity:0.75; text-transform:uppercase; letter-spacing:.06em;">Best Model</div>
                <div style="font-size:1.3rem; font-weight:700;">{best}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; opacity:0.75; text-transform:uppercase; letter-spacing:.06em;">Test MAE</div>
                <div style="font-size:1.3rem; font-weight:700;">{f'{mae:.2f} min' if isinstance(mae, float) else mae}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; opacity:0.75; text-transform:uppercase; letter-spacing:.06em;">R² Score</div>
                <div style="font-size:1.3rem; font-weight:700;">{f'{r2:.3f}' if isinstance(r2, float) else r2}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; opacity:0.75; text-transform:uppercase; letter-spacing:.06em;">Lines Modelled</div>
                <div style="font-size:1.3rem; font-weight:700;">11</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Project", "Methodology", "Tech Stack", "Ethics"])

    with tab_a:
        st.markdown("""
        ### Purpose
        This system predicts London Underground delay severity (in minutes) using machine learning,
        helping transport planners anticipate disruptions before they escalate.

        ### Research Questions
        1. Can ML models outperform a simple naive baseline for short-term delay prediction?
        2. Which features (weather, temporal, crowding) are most predictive?
        3. How does model performance vary across different tube lines?

        ### Key Contributions
        - **Rigorous temporal validation** — strict chronological 80/20 train/test split with no look-ahead.
        - **Multi-model comparison** — Naive baseline vs Ridge regression vs LightGBM.
        - **SHAP explainability** — every prediction backed by feature-attribution scores.
        - **Real data collection** — a 2-week TfL + weather data pipeline built from scratch.
        - **This dashboard** — interactive, production-quality interface for the dissertation demo.
        """)

    with tab_b:
        st.markdown("""
        ### Data Sources
        | Source | Content | Frequency |
        |--------|---------|-----------|
        | TfL Unified API | Line status, disruptions | Every 15 min |
        | OpenWeatherMap | Temperature, rain, humidity | Every 15 min |
        | `holidays` library | UK bank holiday calendar | Static |

        ### Feature Engineering
        Transformations applied with leakage protection:
        - **Lag features** — delay at t-1 h and t-3 h (per line)
        - **Rolling statistics** — mean & std over 3 h and 12 h windows
        - **Weather deltas** — rate of change in temperature and precipitation
        - **Temporal one-hots** — hour, day-of-week, month, peak/off-peak flag
        - **Crowding index** — proxy derived from known peak/off-peak patterns

        ### Models
        | Model | Description |
        |-------|-------------|
        | Naive | Persistence: last observed delay per line |
        | Ridge | L2-regularised linear regression |
        | LightGBM | Gradient-boosted trees with RandomisedSearchCV tuning |

        ### Evaluation
        - Primary metric: **MAE** (also RMSE, R²)
        - Validation: 5-fold `TimeSeriesSplit` cross-validation
        - Test: held-out final 20% of chronological data
        """)

    with tab_c:
        cols = st.columns(3)
        techs = [
            ("Python 3.x",       "Core language"),
            ("pandas / numpy",   "Data wrangling"),
            ("scikit-learn",     "Ridge, CV, preprocessing"),
            ("LightGBM",         "Best model candidate"),
            ("SHAP",             "Explainability"),
            ("Plotly",           "Interactive charts"),
            ("Streamlit",        "This dashboard"),
            ("FastAPI",          "Production REST API"),
            ("holidays",         "UK bank holiday calendar"),
            ("OpenWeatherMap",   "Weather API"),
            ("TfL API",          "Line status API"),
            ("joblib",           "Model serialisation"),
        ]
        bg_card  = "#1a2235"
        bdr_card = "#243050"
        for i, (name, desc) in enumerate(techs):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#f8f9fa; border:1px solid #e5e7eb; border-radius:10px;
                            padding:0.75rem 0.9rem; margin:0.3rem 0; font-family:'Inter',sans-serif;">
                    <div style="font-weight:700; font-size:0.88rem; color:#111827;">{name}</div>
                    <div style="color:#6b7280; font-size:0.78rem; margin-top:0.1rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_d:
        st.markdown("""
        ### Ethical Considerations

        **Privacy**
        - Zero personal passenger data is used. All metrics are fully aggregated at line level.

        **Transparency**
        - Every prediction is backed by SHAP feature attributions — no black-box decisions.
        - The methodology is fully documented and reproducible via the public codebase.

        **Limitations**
        - Predictions are estimates, not guarantees. Major incidents (strikes, engineering
          works) are not represented in training data and may degrade accuracy.
        - Model performance varies by line; consulting per-line MAE before operational use is recommended.

        **Human Oversight**
        - This system is designed as **decision support**, not automated action.
          Human review is required before acting on any prediction.

        **Data Quality**
        - Synthetic data was used for initial development; real data collection
          is ongoing. Dissertation results are clearly labelled by data source.
        """)
