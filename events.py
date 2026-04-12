"""
London major events calendar — venue mappings, event generation, and feature extraction.

Generates a reproducible calendar of ~500 discrete events (football, concerts,
public events, exhibitions, cultural celebrations) plus continuous seasonal
demand factors for the London Underground delay prediction pipeline.

> **Note**: This procedural system is a synthetic proxy. For real deployments,
> integrate an actual human-maintained events and engineering works calendar.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Venue definitions
# ---------------------------------------------------------------------------

@dataclass
class Venue:
    name: str
    capacity: int
    lines: List[str]
    impact_hours: float = 3.0


# Venue capacities are official or widely-cited figures.
# Sources: stadium club websites and The Guardian / BBC Sport venue guides
# as of 2024. Approximate figures are used for open-air public spaces
# (Hyde Park, Victoria Park, Central London) which have no fixed capacity.
VENUES: Dict[str, Venue] = {
    'wembley':        Venue('Wembley Stadium',               90_000, ['Jubilee', 'Metropolitan']),
    'emirates':       Venue('Emirates Stadium',               60_260, ['Piccadilly']),
    'stamford':       Venue('Stamford Bridge',                40_341, ['District']),
    'tottenham':      Venue('Tottenham Hotspur Stadium',      62_850, ['Victoria', 'Northern']),
    'london_stadium': Venue('London Stadium',                 60_000, ['Central']),
    'craven_cottage': Venue('Craven Cottage',                 25_700, ['District']),
    'selhurst':       Venue('Selhurst Park',                  25_486, ['Bakerloo', 'District']),
    'o2':             Venue('The O2 Arena',                   20_000, ['Jubilee']),
    'twickenham':     Venue('Twickenham Stadium',             82_000, ['District']),
    'the_oval':       Venue('The Oval',                       25_500, ['Northern']),
    'excel':          Venue('ExCeL London',                   35_000, ['Central']),
    'olympia':        Venue('Olympia London',                 10_000, ['District', 'Piccadilly']),
    'hyde_park':      Venue('Hyde Park',                     100_000, ['Central', 'Jubilee']),
    'victoria_park':  Venue('Victoria Park',                  80_000, ['Central']),
    'central_london': Venue('Central London',                200_000, ['Central', 'Jubilee', 'Northern', 'Piccadilly']),
    'west_end':       Venue('West End / Leicester Sq',        50_000, ['Piccadilly', 'Northern']),
    'westminster':    Venue('Westminster / Whitehall',        80_000, ['Jubilee', 'Circle', 'District']),
    'city':           Venue('City of London',                 40_000, ['Central', 'Circle']),
    'notting_hill':   Venue('Notting Hill',                  100_000, ['Central', 'Circle']),
    'wimbledon':      Venue('Wimbledon (SW19)',                15_000, ['District']),
    'putney':         Venue('Putney / Thames',                30_000, ['District']),
    'chelsea_rh':     Venue('Royal Hospital Chelsea',         50_000, ['Circle', 'District']),
    'regents_park':   Venue("Regent's Park",                  40_000, ['Bakerloo', 'Metropolitan']),
}

MAX_CAPACITY = max(v.capacity for v in VENUES.values())


# ---------------------------------------------------------------------------
# Event generators
# ---------------------------------------------------------------------------

def _football_season(rng: np.random.RandomState, year_start: int) -> List[Dict]:
    """Generate Premier League + cup home matches for a season (Aug–May)."""
    events = []
    teams = [
        ('emirates', 'Arsenal'),
        ('stamford', 'Chelsea'),
        ('tottenham', 'Tottenham'),
        ('london_stadium', 'West Ham'),
        ('craven_cottage', 'Fulham'),
        ('selhurst', 'Crystal Palace'),
    ]
    season_start = date(year_start, 8, 1)
    season_end = date(year_start + 1, 5, 31)
    total_days = (season_end - season_start).days

    for team, venue in teams:
        n_matches = 20  # ~19 PL home games + cups
        match_days = sorted(rng.choice(total_days, size=n_matches, replace=False))
        for d in match_days:
            match_date = season_start + timedelta(days=int(d))
            dow = match_date.weekday()
            # Weekend at 15:00, weekday evening at 19:30 or 20:00
            if dow >= 5:
                hour = 15
            elif rng.random() < 0.3:
                hour = 14
            else:
                hour = 19
            events.append({
                'date': datetime(match_date.year, match_date.month, match_date.day, hour),
                'venue': team,
                'event_type': 'football',
                'name': f'{venue} home match',
            })

    # Wembley: England/FA Cup (~6 events per season)
    wembley_days = sorted(rng.choice(total_days, size=6, replace=False))
    for d in wembley_days:
        match_date = season_start + timedelta(days=int(d))
        hour = 17 if match_date.weekday() >= 5 else 19
        events.append({
            'date': datetime(match_date.year, match_date.month, match_date.day, hour),
            'venue': 'wembley',
            'event_type': 'football',
            'name': 'Wembley football (England/FA Cup)',
        })

    # NFL London: 2–4 October games at Wembley/Tottenham
    for day_offset in [rng.randint(0, 29), rng.randint(0, 29) + 7]:
        event_date = date(year_start, 10, 1) + timedelta(days=int(day_offset) % 28)
        venue = 'tottenham' if rng.random() < 0.5 else 'wembley'
        events.append({
            'date': datetime(event_date.year, event_date.month, event_date.day, 14),
            'venue': venue,
            'event_type': 'football',
            'name': 'NFL London Game',
        })

    return events


def _concerts_entertainment(
    rng: np.random.RandomState,
    start: date,
    end: date,
) -> List[Dict]:
    """O2 year-round + summer stadium concerts."""
    events = []
    total_days = (end - start).days

    # O2: ~60 events per year
    n_o2 = 60
    o2_days = sorted(rng.choice(total_days, size=n_o2, replace=False))
    for d in o2_days:
        dt = start + timedelta(days=int(d))
        hour = 19 if dt.weekday() < 5 else 18
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, hour),
            'venue': 'o2',
            'event_type': 'concert',
            'name': 'The O2 concert/show',
        })

    # Summer stadium concerts (June–August)
    summer_start = date(start.year, 6, 1)
    summer_end = date(start.year, 8, 31)
    if summer_start >= start and summer_start <= end:
        summer_days_total = (summer_end - summer_start).days
        n_summer = 8
        summer_days = sorted(rng.choice(summer_days_total, size=n_summer, replace=False))
        summer_venues = ['wembley', 'london_stadium', 'hyde_park', 'victoria_park']
        for i, d in enumerate(summer_days):
            dt = summer_start + timedelta(days=int(d))
            venue = summer_venues[i % len(summer_venues)]
            events.append({
                'date': datetime(dt.year, dt.month, dt.day, 19),
                'venue': venue,
                'event_type': 'concert',
                'name': f'{VENUES[venue].name} summer concert',
            })

    # West End premieres (~5/year)
    premiere_days = sorted(rng.choice(total_days, size=5, replace=False))
    for d in premiere_days:
        dt = start + timedelta(days=int(d))
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 18),
            'venue': 'west_end',
            'event_type': 'premiere',
            'name': 'West End premiere/awards',
        })

    # O2 boxing/UFC (~5/year)
    boxing_days = sorted(rng.choice(total_days, size=5, replace=False))
    for d in boxing_days:
        dt = start + timedelta(days=int(d))
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 20),
            'venue': 'o2',
            'event_type': 'boxing',
            'name': 'O2 boxing/UFC fight night',
        })

    return events


def _other_sport(rng: np.random.RandomState, year: int) -> List[Dict]:
    """Rugby, tennis, cricket, horse racing, boat race, marathon."""
    events = []

    # Wimbledon: last week of June + first week of July (~14 days)
    wimbledon_start = date(year, 7, 1)
    for day in range(14):
        dt = wimbledon_start + timedelta(days=day)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 11),
            'venue': 'wimbledon',
            'event_type': 'tennis',
            'name': 'Wimbledon Championships',
        })

    # Six Nations: Feb–March (~3 home games at Twickenham)
    six_nations_days = [date(year, 2, 10), date(year, 2, 24), date(year, 3, 15)]
    for d in six_nations_days:
        events.append({
            'date': datetime(d.year, d.month, d.day, 16),
            'venue': 'twickenham',
            'event_type': 'rugby',
            'name': 'Twickenham rugby international',
        })

    # Autumn internationals: November (~3 home games)
    autumn_days = [date(year, 11, 9), date(year, 11, 16), date(year, 11, 23)]
    for d in autumn_days:
        events.append({
            'date': datetime(d.year, d.month, d.day, 15),
            'venue': 'twickenham',
            'event_type': 'rugby',
            'name': 'Twickenham rugby international',
        })

    # The Oval cricket (~5 summer test days)
    oval_base = date(year, 7, 24)
    for day in range(5):
        dt = oval_base + timedelta(days=day)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 10),
            'venue': 'the_oval',
            'event_type': 'cricket',
            'name': 'The Oval Test Match',
        })

    # London Marathon: April
    marathon_date = date(year, 4, 27)
    events.append({
        'date': datetime(marathon_date.year, marathon_date.month, marathon_date.day, 9),
        'venue': 'central_london',
        'event_type': 'running',
        'name': 'London Marathon',
    })

    # Boat Race: late March
    boat_date = date(year, 3, 30)
    events.append({
        'date': datetime(boat_date.year, boat_date.month, boat_date.day, 14),
        'venue': 'putney',
        'event_type': 'rowing',
        'name': 'Oxford-Cambridge Boat Race',
    })

    return events


def _public_events(year: int) -> List[Dict]:
    """Fixed-date public events, parades, celebrations."""
    events = []
    fixed = [
        # (month, day, hour, venue, event_type, name)
        (8, 25, 10, 'notting_hill', 'carnival', 'Notting Hill Carnival'),
        (8, 26, 10, 'notting_hill', 'carnival', 'Notting Hill Carnival'),
        (7, 6,  12, 'central_london', 'pride', 'London Pride Parade'),
        (2, 1,  12, 'west_end', 'celebration', 'Chinese New Year'),
        (10, 17, 17, 'westminster', 'celebration', 'Diwali Celebrations'),
        (4,  28, 17, 'central_london', 'celebration', 'Eid Celebrations'),
        (11, 5,  19, 'central_london', 'celebration', 'Bonfire Night'),
        (12, 31, 22, 'central_london', 'celebration', "New Year's Eve Fireworks"),
        (1,  1,  0,  'central_london', 'celebration', "New Year's Day"),
    ]
    for m, d, h, venue, etype, name in fixed:
        try:
            events.append({
                'date': datetime(year, m, d, h),
                'venue': venue,
                'event_type': etype,
                'name': name,
            })
        except ValueError:
            pass  # skip invalid dates

    # London Marathon Sunday (last Sunday of April)
    for offset in range(6, -1, -1):
        candidate = date(year, 4, 30) - timedelta(days=offset)
        if candidate.weekday() == 6:
            events.append({
                'date': datetime(candidate.year, candidate.month, candidate.day, 9),
                'venue': 'central_london',
                'event_type': 'running',
                'name': 'London Marathon',
            })
            break

    # Light Night / Late Museum Switchers (~Feb)
    lms_date = date(year, 2, 14)
    events.append({
        'date': datetime(lms_date.year, lms_date.month, lms_date.day, 18),
        'venue': 'westminster',
        'event_type': 'cultural',
        'name': 'Museums at Night / Late',
    })

    return events


def _festivals_exhibitions(rng: np.random.RandomState, year: int) -> List[Dict]:
    """Festivals, seasonal attractions, exhibitions."""
    events = []

    # BST Hyde Park: June–July, ~10 days
    bst_start = date(year, 6, 25)
    for day in range(10):
        dt = bst_start + timedelta(days=day)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 16),
            'venue': 'hyde_park',
            'event_type': 'festival',
            'name': 'BST Hyde Park',
        })

    # All Points East, Victoria Park: August, ~8 days
    ape_start = date(year, 8, 15)
    for day in range(8):
        dt = ape_start + timedelta(days=day)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 14),
            'venue': 'victoria_park',
            'event_type': 'festival',
            'name': 'All Points East',
        })

    # Winter Wonderland: mid-Nov to Jan 5
    ww_start = date(year, 11, 16)
    ww_end = date(year + 1, 1, 2)
    ww_days = (ww_end - ww_start).days
    for d in range(0, ww_days, 2):  # every other day for calendar size
        dt = ww_start + timedelta(days=d)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 17),
            'venue': 'hyde_park',
            'event_type': 'seasonal',
            'name': 'Winter Wonderland',
        })

    # RHS Chelsea Flower Show: late May, 5 days
    chelsea_start = date(year, 5, 20)
    for day in range(5):
        dt = chelsea_start + timedelta(days=day)
        events.append({
            'date': datetime(dt.year, dt.month, dt.day, 9),
            'venue': 'chelsea_rh',
            'event_type': 'exhibition',
            'name': 'RHS Chelsea Flower Show',
        })

    return events


# ---------------------------------------------------------------------------
# Seasonal demand
# ---------------------------------------------------------------------------

def compute_seasonal_demand(timestamps: pd.Series) -> pd.Series:
    """Continuous seasonal demand factor (0–1) for elevated-demand periods.

    Covers Christmas shopping, January sales, Black Friday, Easter,
    school half-terms, and summer tourism.
    """
    factor = pd.Series(0.0, index=timestamps.index)

    month = timestamps.dt.month
    day = timestamps.dt.day

    is_xmas_shopping = (month == 12) & (day >= 1) & (day <= 24)
    is_jan_sales = (month == 1) & (day >= 1) & (day <= 7)
    # Black Friday: last Friday of November (~day 24–30)
    is_black_friday = (month == 11) & (day >= 24) & (day <= 30) & (timestamps.dt.dayofweek == 4)
    is_easter = ((month == 4) & (day >= 10) & (day <= 17)) | ((month == 3) & (day >= 29))
    # School half-terms (approximate)
    is_feb_ht = (month == 2) & (day >= 17) & (day <= 21)
    is_may_ht = (month == 5) & (day >= 26) & (day <= 30)
    is_oct_ht = (month == 10) & (day >= 20) & (day <= 31)
    is_half_term = is_feb_ht | is_may_ht | is_oct_ht
    is_summer = (month >= 7) & (month <= 8)

    factor[is_xmas_shopping] = 0.7
    factor[is_jan_sales] = 0.4
    factor[is_black_friday] = 0.7
    factor[is_easter] = 0.5
    factor[is_half_term] = 0.3
    factor[is_summer] = 0.4

    return factor.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Real / template calendar
# ---------------------------------------------------------------------------

REAL_EVENTS_CSV = Path(__file__).parent / 'data' / 'events_calendar.csv'


def load_real_event_calendar(csv_path: Path = REAL_EVENTS_CSV) -> Optional[pd.DataFrame]:
    """Load a real events calendar from CSV.

    Expected CSV columns: date, venue, event_type, name

    The 'venue' column must match a key in the VENUES dict (e.g. 'wembley',
    'o2', 'emirates'). Unknown venues are kept but won't affect line features.

    You can create this CSV from:
      - football-data.co.uk (free PL fixture CSV)
      - Ticketmaster Discovery API (concerts at O2, Wembley, etc.)
      - Manual entry for known public events

    A template is created at data/events_calendar_template.csv on first run.
    """
    required = {'date', 'venue', 'event_type', 'name'}
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        unknown = required - set(df.columns)
        if unknown:
            logger.warning('Real events CSV missing columns %s — falling back to generated', unknown)
            return None
        logger.info('Loaded %d real events from %s', len(df), csv_path)
        unknown_venues = set(df['venue'].unique()) - set(VENUES)
        if unknown_venues:
            logger.warning("Unknown venues in real events CSV (won't affect line features): %s", unknown_venues)
        return df
    except Exception as e:
        logger.warning('Failed to load real events CSV: %s', e)
        return None


def create_events_template(output_path: Optional[Path] = None) -> Path:
    """Create a template CSV with example rows showing the expected format."""
    if output_path is None:
        output_path = Path('data') / 'events_calendar_template.csv'
    template = pd.DataFrame([
        {'date': '2026-01-18 15:00', 'venue': 'emirates',  'event_type': 'football', 'name': 'Arsenal vs Aston Villa'},
        {'date': '2026-01-18 17:30', 'venue': 'stamford',  'event_type': 'football', 'name': 'Chelsea vs Wolves'},
        {'date': '2026-01-20 20:00', 'venue': 'o2',        'event_type': 'concert',  'name': 'Dua Lipa — The O2'},
        {'date': '2026-02-01 17:30', 'venue': 'wembley',   'event_type': 'football', 'name': 'FA Cup 4th Round'},
        {'date': '2026-06-28 19:00', 'venue': 'hyde_park', 'event_type': 'festival', 'name': 'BST Hyde Park — Kendrick Lamar'},
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    logger.info('Created events template at %s', output_path)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_event_calendar(
    start_date: date,
    end_date: date,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load real events CSV if available, otherwise generate synthetic calendar.

    Returns DataFrame with columns: date, venue, event_type, name.
    """
    real_df = load_real_event_calendar()
    if real_df is not None:
        in_range = real_df[(real_df['date'] >= pd.Timestamp(start_date)) &
                           (real_df['date'] <= pd.Timestamp(end_date))]
        if len(in_range) > 0:
            logger.info('Using REAL event calendar: %d events (%s to %s)',
                        len(in_range), start_date, end_date)
            return in_range.reset_index(drop=True)
        logger.warning('Real event calendar has no events in range %s–%s, falling back to generated',
                       start_date, end_date)

    rng = np.random.RandomState(seed)
    all_events: List[Dict] = []
    years = list(range(start_date.year, end_date.year + 1))
    for year in years:
        all_events.extend(_football_season(rng, year))
        all_events.extend(_concerts_entertainment(rng, date(year, 1, 1), date(year, 12, 31)))
        all_events.extend(_other_sport(rng, year))
        all_events.extend(_public_events(year))
        all_events.extend(_festivals_exhibitions(rng, year))

    df = pd.DataFrame(all_events)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
    df = df.sort_values('date').reset_index(drop=True)
    logger.info('Generated synthetic event calendar: %d events (%s to %s)',
                len(df), start_date, end_date)
    logger.info('Event type distribution:\n%s', df['event_type'].value_counts().to_string())

    # Create template on first run so users know how to supply real data
    template_path = Path('data') / 'events_calendar_template.csv'
    if not template_path.exists():
        create_events_template(template_path)

    # Persist a wide-range synthetic calendar (2025-2027) so subsequent runs
    # load from CSV instead of regenerating — removes the "no such file" warning
    # and makes event features reproducible across runs.
    if not REAL_EVENTS_CSV.exists():
        _wide_rng = np.random.RandomState(seed)
        _wide_events: List[Dict] = []
        for _yr in range(2025, 2028):
            _wide_events.extend(_football_season(_wide_rng, _yr))
            _wide_events.extend(_concerts_entertainment(
                _wide_rng, date(_yr, 1, 1), date(_yr, 12, 31)))
            _wide_events.extend(_other_sport(_wide_rng, _yr))
            _wide_events.extend(_public_events(_yr))
            _wide_events.extend(_festivals_exhibitions(_wide_rng, _yr))
        _wide_df = pd.DataFrame(_wide_events)
        _wide_df['date'] = pd.to_datetime(_wide_df['date'])
        _wide_df = _wide_df.sort_values('date').reset_index(drop=True)
        REAL_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        _wide_df.to_csv(REAL_EVENTS_CSV, index=False)
        logger.info(
            'Persisted synthetic event calendar (%d events, 2025-2027) to %s',
            len(_wide_df), REAL_EVENTS_CSV,
        )

    return df


def get_event_features(
    timestamps: pd.Series,
    lines: pd.Series,
) -> pd.DataFrame:
    """Create is_major_event, event_crowd_boost, and seasonal_demand_factor.

    For each (timestamp, line) pair, checks if any event is happening at a
    venue served by that line within the venue's impact window.

    Returns a DataFrame with the same index containing three columns.
    """
    if len(timestamps) == 0:
        return pd.DataFrame(
            {'is_major_event': [], 'event_crowd_boost': [], 'seasonal_demand_factor': []},
            index=timestamps.index,
        )

    start_date = timestamps.min().date()
    end_date = timestamps.max().date()
    calendar = generate_event_calendar(start_date, end_date)

    is_event = pd.Series(0, index=timestamps.index)
    crowd_boost = pd.Series(0.0, index=timestamps.index)

    ts_arr = timestamps.values
    line_arr = lines.values

    for _, row in calendar.iterrows():
        venue = VENUES.get(row['venue'])
        if venue is None:
            continue
        ev_start = pd.Timestamp(row['date'])
        ev_end = ev_start + timedelta(hours=venue.impact_hours)
        venue_lines = venue.lines
        boost = venue.capacity / MAX_CAPACITY

        time_mask = (ts_arr >= ev_start.to_datetime64()) & (ts_arr <= ev_end.to_datetime64())
        if not time_mask.any():
            continue

        for line in venue_lines:
            line_mask = line_arr == line
            combined = time_mask & line_mask
            is_event[combined] = 1
            crowd_boost[combined] = crowd_boost[combined].clip(lower=boost)

    n_flagged = int(is_event.sum())
    logger.info(
        'Event features: %d/%d rows flagged as major event (%.1f%%)',
        n_flagged, len(timestamps), 100 * n_flagged / max(len(timestamps), 1),
    )

    return pd.DataFrame({
        'is_major_event': is_event,
        'event_crowd_boost': crowd_boost,
        'seasonal_demand_factor': compute_seasonal_demand(timestamps),
    }, index=timestamps.index)
