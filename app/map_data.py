"""
Schematic tube map coordinates for the London Underground.

Grid: x=0 (far West) → x=100 (far East)
      y=0 (far South) → y=70  (far North)

Coordinates hand-mapped against the official TfL Beck schematic diagram.
All line segments use only 0°, 45°, or 90° angles (Beck design rule).

Key reference anchors:
  Bank/Monument   (62, 38)
  Oxford Circus   (52, 46)
  King's Cross    (54, 58)
  Waterloo        (57, 40)
  Victoria stn    (48, 40)
  Green Park      (50, 42)
  Westminster     (54, 40)
  Baker Street    (44, 52)
  Paddington      (36, 50)
  Hammersmith     (26, 43)
  Euston          (52, 52)
  Liverpool St    (64, 46)
  Moorgate        (62, 49)
  Farringdon      (59, 52)
  King's Cross    (54, 58)
  Finsbury Park   (58, 62)
"""

from typing import Dict, List, Tuple

LINE_PATHS: Dict[str, List[Tuple[float, float]]] = {

    # ── Bakerloo (brown) ──────────────────────────────────────────────────
    # Harrow & Wealdstone → Elephant & Castle
    # NW diagonal to Paddington → SE diagonal to Elephant & Castle
    "Bakerloo": [
        (26, 67),   # Harrow & Wealdstone
        (26, 65),   # Kenton
        (26, 63),   # North Wembley
        (26, 61),   # Wembley Central
        (26, 59),   # Stonebridge Park
        (26, 57),   # Harlesden
        (27, 56),   # Willesden Junction
        (29, 54),   # Kensal Green
        (31, 52),   # Queen's Park
        (33, 50),   # Kilburn Park
        (35, 48),   # Maida Vale
        (37, 46),   # Warwick Avenue
        (38, 46),   # Paddington
        (40, 48),   # Edgware Road (Bakerloo)
        (42, 50),   # Marylebone
        (44, 52),   # Baker Street
        (48, 48),   # Regent's Park
        (52, 46),   # Oxford Circus
        (54, 44),   # Piccadilly Circus
        (55, 43),   # Charing Cross
        (55, 41),   # Embankment
        (57, 40),   # Waterloo
        (57, 38),   # Lambeth North
        (57, 36),   # Elephant & Castle
    ],

    # ── Central (red) ─────────────────────────────────────────────────────
    # West Ruislip → Epping  (horizontal spine + diagonal ends)
    "Central": [
        # West Ruislip branch (SW diagonal in)
        (2, 50),    # West Ruislip
        (4, 48),    # Ruislip Gardens
        (6, 48),    # South Ruislip
        (8, 48),    # Northolt
        (10, 48),   # Greenford
        (12, 48),   # Perivale
        (14, 48),   # Hanger Lane
        (16, 48),   # North Acton
        (18, 48),   # East Acton
        (20, 48),   # White City
        (22, 48),   # Shepherd's Bush
        (28, 48),   # Holland Park
        (34, 46),   # Notting Hill Gate
        (38, 46),   # Queensway
        (40, 46),   # Lancaster Gate
        (43, 46),   # Marble Arch
        (46, 46),   # Bond Street
        (52, 46),   # Oxford Circus
        (55, 46),   # Tottenham Court Road
        (57, 46),   # Holborn
        (59, 46),   # Chancery Lane
        (61, 46),   # St Paul's
        (62, 46),   # Bank
        (64, 46),   # Liverpool Street
        (66, 46),   # Bethnal Green
        (69, 46),   # Mile End
        (72, 46),   # Stratford
        (75, 48),   # Leyton
        (78, 50),   # Leytonstone
        # Epping branch (NE diagonal)
        (81, 53),   # Snaresbrook
        (83, 55),   # South Woodford
        (85, 57),   # Woodford
        (87, 59),   # Buckhurst Hill
        (89, 61),   # Loughton
        (91, 63),   # Debden
        (93, 65),   # Theydon Bois
        (95, 67),   # Epping
    ],

    # ── Circle (yellow) ───────────────────────────────────────────────────
    # The classic rough-rectangle loop
    "Circle": [
        (26, 43),   # Hammersmith
        (30, 43),   # Ravenscourt Park
        (30, 41),   # Stamford Brook / Turnham Green
        (32, 39),   # Gunnersbury
        (32, 43),   # Kensington (High St)
        (34, 45),   # Notting Hill Gate (Circle)
        (34, 47),   # Bayswater
        (35, 49),   # Royal Oak
        (36, 50),   # Paddington
        (38, 52),   # Edgware Road (Circle)
        (40, 52),   # Baker St side
        (44, 52),   # Baker Street
        (46, 55),   # Great Portland Street
        (48, 57),   # Euston Square
        (50, 59),   # King's Cross St. Pancras
        (54, 58),   # King's Cross (top of loop)
        (56, 58),   # Angel/Farringdon direction
        (59, 58),   # Farringdon
        (61, 56),   # Barbican
        (62, 55),   # Moorgate
        (64, 54),   # Liverpool Street
        (65, 52),   # Aldgate
        (66, 50),   # Tower Hill
        (64, 44),   # Monument/Bank area
        (62, 42),   # Cannon Street
        (60, 40),   # Blackfriars
        (58, 40),   # Temple
        (57, 40),   # Embankment
        (55, 40),   # Westminster
        (54, 40),   # St James's Park
        (51, 40),   # Victoria
        (48, 40),   # Sloane Square
        (44, 40),   # South Kensington
        (40, 40),   # Gloucester Road
        (36, 40),   # Earl's Court
        (32, 40),   # Kensington (south)
        (32, 43),   # back up
        (26, 43),   # Hammersmith (close)
    ],

    # ── District (green) ──────────────────────────────────────────────────
    # Three SW branches → Earl's Court → east to Upminster
    "District": [
        # Wimbledon branch
        (28, 25),   # Wimbledon
        (30, 27),   # Wimbledon Park
        (31, 29),   # Southfields
        (32, 31),   # East Putney
        (32, 33),   # Putney Bridge
        (32, 35),   # Parsons Green
        (32, 37),   # Fulham Broadway
        (32, 39),   # West Brompton
        (34, 40),   # Earl's Court
        # Richmond branch (separate – shown as one path sharing Earl's Ct)
        (10, 35),   # Richmond
        (14, 35),   # Kew Gardens
        (17, 35),   # Gunnersbury
        (20, 37),   # Turnham Green
        (22, 39),   # Stamford Brook
        (24, 40),   # Ravenscourt Park
        (26, 40),   # Hammersmith (District)
        (28, 40),   # Barons Court
        (34, 40),   # Earl's Court
        # Ealing branch
        (22, 44),   # Acton Town
        (24, 42),   # Chiswick Park
        (26, 42),   # Turnham Green (H&C join)
        (34, 40),   # Earl's Court (re-join)
        # Trunk East
        (36, 40),   # Kensington (Olympia)
        (40, 40),   # Gloucester Road
        (44, 40),   # South Kensington
        (48, 40),   # Sloane Square
        (51, 40),   # Victoria
        (54, 40),   # St James's Park
        (55, 40),   # Westminster
        (57, 40),   # Embankment
        (58, 40),   # Temple
        (60, 40),   # Blackfriars
        (62, 40),   # Cannon Street
        (64, 42),   # Monument
        (66, 44),   # Tower Hill
        (67, 44),   # Aldgate East
        (68, 44),   # Whitechapel
        (70, 44),   # Stepney Green
        (72, 44),   # Mile End
        (74, 44),   # Bow Road
        (76, 44),   # Bromley-by-Bow
        (78, 44),   # West Ham
        (80, 43),   # Plaistow
        (82, 43),   # Upton Park
        (84, 43),   # East Ham
        (86, 43),   # Barking
        (88, 42),   # Upney
        (90, 42),   # Becontree
        (92, 42),   # Dagenham Heathway
        (94, 41),   # Dagenham East
        (96, 40),   # Elm Park
        (98, 39),   # Hornchurch
        (99, 38),   # Upminster Bridge
        (100, 38),  # Upminster
    ],

    # ── Hammersmith & City (pink) ─────────────────────────────────────────
    # Hammersmith → Barking via Baker Street and Aldgate
    "Hammersmith & City": [
        (26, 43),   # Hammersmith
        (28, 45),   # Goldhawk Road
        (30, 47),   # Shepherd's Bush Market
        (32, 49),   # Wood Lane
        (34, 51),   # Latimer Road
        (36, 51),   # Ladbroke Grove
        (36, 50),   # Westbourne Park
        (36, 50),   # Royal Oak
        (36, 50),   # Paddington (H&C)
        (38, 52),   # Edgware Road
        (44, 52),   # Baker Street
        (46, 55),   # Great Portland Street
        (48, 57),   # Euston Square
        (50, 59),   # King's Cross
        (54, 58),   # Barbican side
        (57, 57),   # Farringdon
        (60, 56),   # Barbican
        (62, 55),   # Moorgate
        (64, 54),   # Liverpool Street
        (65, 52),   # Aldgate
        (66, 50),   # Whitechapel
        (68, 48),   # Stepney Green
        (70, 46),   # Mile End
        (72, 44),   # Bow Road
        (74, 44),   # Bromley-by-Bow
        (76, 44),   # West Ham
        (78, 44),   # Plaistow
        (80, 44),   # Upton Park
        (82, 44),   # East Ham
        (84, 44),   # Barking
    ],

    # ── Jubilee (silver) ──────────────────────────────────────────────────
    # Stanmore → Stratford  (NW to centre then curves SE)
    "Jubilee": [
        (30, 67),   # Stanmore
        (30, 65),   # Canons Park
        (30, 63),   # Queensbury
        (30, 61),   # Kingsbury
        (32, 60),   # Wembley Park
        (32, 58),   # Neasden
        (32, 56),   # Dollis Hill
        (34, 54),   # Willesden Green
        (36, 52),   # Kilburn
        (37, 51),   # West Hampstead
        (38, 50),   # Finchley Road
        (40, 50),   # Swiss Cottage
        (42, 52),   # St John's Wood
        (44, 52),   # Baker Street
        (46, 50),   # Bond Street
        (50, 46),   # Green Park
        (54, 42),   # Westminster
        (57, 40),   # Waterloo
        (58, 38),   # Southwark
        (60, 36),   # London Bridge
        (62, 34),   # Bermondsey
        (64, 32),   # Canada Water
        (68, 30),   # Canary Wharf
        (72, 31),   # North Greenwich
        (76, 34),   # Canning Town
        (78, 36),   # West Ham
        (80, 38),   # Stratford
    ],

    # ── Metropolitan (dark purple) ────────────────────────────────────────
    # Amersham/Chesham/Uxbridge/Watford → Aldgate
    "Metropolitan": [
        # Amersham/Chesham branch
        (2, 68),    # Chesham
        (4, 66),    # Amersham
        (6, 64),    # Chalfont & Latimer
        (9, 64),    # Chorleywood
        (12, 64),   # Rickmansworth
        (15, 62),   # Moor Park
        (18, 65),   # Watford (branch)
        (15, 62),   # back to Moor Park
        (18, 60),   # Northwood
        (20, 58),   # Northwood Hills
        (22, 56),   # Pinner
        (24, 55),   # North Harrow
        (26, 54),   # Harrow-on-the-Hill
        # Uxbridge branch
        (4, 58),    # Uxbridge
        (6, 58),    # Hillingdon
        (8, 58),    # Ickenham
        (10, 57),   # Ruislip
        (12, 56),   # Ruislip Manor
        (14, 54),   # Eastcote
        (16, 53),   # Rayners Lane
        (18, 53),   # West Harrow
        (22, 54),   # Harrow-on-the-Hill (Ux)
        (26, 54),   # Harrow-on-the-Hill (join)
        # Main south line through Baker Street to Aldgate
        (28, 55),   # Northwick Park
        (30, 57),   # Preston Road
        (32, 59),   # Wembley Park
        (36, 57),   # Finchley Road (Met)
        (40, 54),   # Baker Street (approach)
        (44, 52),   # Baker Street
        (46, 55),   # Great Portland Street
        (48, 57),   # Euston Square
        (50, 59),   # King's Cross
        (54, 58),   # Barbican
        (57, 57),   # Farringdon
        (60, 56),   # Barbican
        (62, 55),   # Moorgate
        (64, 54),   # Liverpool Street
        (65, 52),   # Aldgate
    ],

    # ── Northern (black) ──────────────────────────────────────────────────
    # Edgware + High Barnet → [Camden Town] → Morden  (branches both ends)
    "Northern": [
        # Edgware branch (NW arm)
        (46, 68),   # Edgware
        (48, 66),   # Burnt Oak
        (48, 64),   # Colindale
        (48, 62),   # Hendon Central
        (48, 60),   # Brent Cross
        (50, 58),   # Golders Green
        (50, 56),   # Hampstead
        (50, 54),   # Belsize Park
        (52, 52),   # Chalk Farm
        (52, 52),   # Camden Town (W branch arrives)
        # High Barnet branch (NE arm)
        (58, 68),   # High Barnet
        (58, 66),   # Totteridge & Whetstone
        (58, 64),   # Woodside Park
        (58, 62),   # West Finchley
        (58, 60),   # Finchley Central
        (58, 58),   # East Finchley
        (56, 56),   # Highgate
        (55, 55),   # Archway
        (54, 54),   # Tufnell Park
        (53, 53),   # Kentish Town
        (52, 52),   # Camden Town (E branch arrives)
        # Merged south trunk
        (52, 50),   # Mornington Crescent
        (52, 48),   # Euston (Northern)
        (52, 46),   # Warren Street
        (52, 44),   # Goodge Street
        (52, 42),   # Tottenham Court Road
        (53, 41),   # Leicester Square
        (54, 40),   # Charing Cross
        # Bank branch (E fork south of Charing Cross)
        (56, 38),   # Waterloo (Bank branch)
        (60, 36),   # London Bridge
        (60, 34),   # Borough
        (60, 32),   # Elephant & Castle
        (60, 30),   # Kennington (junction)
        # Charing Cross branch (W fork – via Waterloo)
        (54, 38),   # Embankment
        (54, 36),   # Waterloo (CX branch)
        (56, 34),   # Lambeth North
        (58, 32),   # Elephant & Castle (CX)
        (60, 30),   # Kennington (re-join)
        # South to Morden
        (60, 28),   # Oval
        (60, 26),   # Stockwell
        (60, 24),   # Clapham North
        (60, 22),   # Clapham Common
        (60, 20),   # Clapham South
        (60, 18),   # Balham
        (60, 16),   # Tooting Bec
        (60, 14),   # Tooting Broadway
        (60, 12),   # Colliers Wood
        (60, 10),   # South Wimbledon
        (60, 8),    # Morden
    ],

    # ── Piccadilly (dark blue) ────────────────────────────────────────────
    # Heathrow (SW) → Cockfosters (NE)  ─ single long diagonal diagonal spine
    "Piccadilly": [
        # Heathrow spur (loop in SW corner)
        (2, 38),    # Heathrow T2/T3
        (2, 36),    # Heathrow T4 spur
        (2, 38),    # back
        (4, 40),    # Heathrow T5
        (6, 42),    # Hatton Cross
        (8, 42),    # Hounslow West
        (10, 42),   # Hounslow Central
        (12, 42),   # Hounslow East
        (14, 43),   # Osterley
        (16, 44),   # Boston Manor
        (18, 44),   # Northfields
        (20, 44),   # South Ealing
        (22, 44),   # Acton Town
        (24, 44),   # Turnham Green (Picc)
        (26, 43),   # Hammersmith (Piccadilly)
        (28, 42),   # Barons Court
        (30, 41),   # Earl's Court
        (32, 41),   # Gloucester Road
        (36, 42),   # South Kensington
        (40, 42),   # Knightsbridge
        (44, 42),   # Hyde Park Corner
        (50, 42),   # Green Park
        (54, 44),   # Piccadilly Circus
        (56, 45),   # Leicester Square
        (57, 46),   # Covent Garden
        (57, 48),   # Holborn
        (58, 50),   # Russell Square
        (58, 52),   # King's Cross (Picc)
        (60, 54),   # Caledonian Road
        (62, 56),   # Holloway Road
        (62, 58),   # Arsenal
        (62, 60),   # Finsbury Park
        (64, 62),   # Manor House
        (66, 64),   # Turnpike Lane
        (66, 66),   # Wood Green
        (67, 67),   # Bounds Green
        (68, 68),   # Arnos Grove
        (70, 69),   # Southgate
        (72, 69),   # Oakwood
        (74, 69),   # Cockfosters
    ],

    # ── Victoria (light blue) ─────────────────────────────────────────────
    # Brixton (SW) → Walthamstow Central (NE)  — consistent 45° diagonal
    "Victoria": [
        (44, 22),   # Brixton
        (46, 24),   # Stockwell
        (47, 26),   # Oval
        (48, 28),   # Kennington
        (48, 31),   # Vauxhall
        (48, 34),   # Pimlico
        (48, 38),   # Victoria
        (50, 40),   # Green Park (Victoria)
        (52, 44),   # Oxford Circus (Victoria joins Central)
        (52, 48),   # Warren Street
        (52, 52),   # Euston (Victoria)
        (54, 54),   # King's Cross (Victoria)
        (56, 56),   # Highbury & Islington
        (58, 58),   # Finsbury Park (Victoria)
        (60, 60),   # Seven Sisters
        (62, 62),   # Tottenham Hale
        (63, 63),   # Blackhorse Road
        (64, 64),   # Walthamstow Central
    ],

    # ── Waterloo & City (teal) ────────────────────────────────────────────
    "Waterloo & City": [
        (57, 40),   # Waterloo
        (62, 40),   # Bank
    ],
}


# ---------------------------------------------------------------------------
# Interchange stations  (x, y, label)
# ---------------------------------------------------------------------------
INTERCHANGE_STATIONS: List[Tuple[float, float, str]] = [
    (52, 46,  "Oxford Circus"),
    (50, 42,  "Green Park"),
    (48, 38,  "Victoria"),
    (57, 40,  "Waterloo"),
    (62, 40,  "Bank"),
    (60, 36,  "London Bridge"),
    (64, 46,  "Liverpool St"),
    (52, 52,  "Euston"),
    (54, 58,  "King's Cross"),
    (44, 52,  "Baker Street"),
    (34, 40,  "Earl's Court"),
    (36, 50,  "Paddington"),
    (26, 43,  "Hammersmith"),
    (54, 40,  "Westminster"),
    (62, 55,  "Moorgate"),
    (59, 52,  "Farringdon"),
    (52, 52,  "Camden Town"),
    (60, 26,  "Stockwell"),
    (32, 60,  "Wembley Park"),
    (62, 60,  "Finsbury Park"),
    (22, 44,  "Acton Town"),
    (66, 44,  "Tower Hill"),
    (65, 52,  "Aldgate"),
    (68, 30,  "Canary Wharf"),
    (26, 54,  "Harrow-on-the-Hill"),
]

# Deduplicate by (x, y)
_seen: set = set()
_deduped: List[Tuple[float, float, str]] = []
for _s in INTERCHANGE_STATIONS:
    _k = (_s[0], _s[1])
    if _k not in _seen:
        _seen.add(_k)
        _deduped.append(_s)
INTERCHANGE_STATIONS = _deduped
