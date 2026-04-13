"""
CSS injection for the TfL-branded Streamlit dashboard.

Uses Inter from Google Fonts, CSS custom properties for theming,
and glassmorphic card styles for a modern, premium look.
"""

import streamlit as st


def apply_custom_css(dark_mode: bool = False) -> None:
    """Inject TfL-branded CSS. Dark mode swaps root colour variables."""
    if dark_mode:
        bg_primary   = "#0b0f19"
        bg_secondary = "#111827"
        bg_card      = "rgba(255,255,255,0.04)"
        bg_card_solid = "#111827"
        text_primary = "#e8edf5"
        text_muted   = "#8b949e"
        border_col   = "rgba(255,255,255,0.08)"
        metric_bg    = "rgba(255,255,255,0.05)"
        glass_bg     = "rgba(17,24,39,0.8)"
    else:
        bg_primary   = "#f3f4f8"
        bg_secondary = "#ffffff"
        bg_card      = "rgba(255,255,255,0.85)"
        bg_card_solid = "#ffffff"
        text_primary = "#111827"
        text_muted   = "#6b7280"
        border_col   = "rgba(0,0,0,0.08)"
        metric_bg    = "rgba(255,255,255,0.9)"
        glass_bg     = "rgba(255,255,255,0.75)"

    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        /* ── Root variables ── */
        :root {{
            --bg-primary:    {bg_primary};
            --bg-secondary:  {bg_secondary};
            --bg-card:       {bg_card};
            --bg-card-solid: {bg_card_solid};
            --text-primary:  {text_primary};
            --text-muted:    {text_muted};
            --border:        {border_col};
            --accent:        #003688;
            --accent2:       #0098D4;
            --radius:        14px;
            --shadow:        0 4px 20px rgba(0,0,0,0.08);
            --shadow-hover:  0 10px 36px rgba(0,0,0,0.16);
            --glass-bg:      {glass_bg};
        }}

        /* ── Typography & Global Base Overrides ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        .stApp, .stApp .main {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }}
        
        /* Force Streamlit native typography to use our text-primary colour */
        .stApp .main p:not(.hero-title):not(.hero-subtitle),
        .stApp .main h1, .stApp .main h2, .stApp .main h3,
        .stApp .main h4, .stApp .main h5, .stApp .main h6,
        .stApp .main li, .stApp .main label,
        .stApp .main .stMarkdown, .stApp .main .stMarkdown p,
        .stApp .main span, .stApp .main div.stText {{
            color: var(--text-primary) !important;
        }}
        /* Selectbox and input text */
        .stApp .main .stSelectbox div[data-baseweb="select"] span,
        .stApp .main .stSelectbox div[data-baseweb="select"] div,
        .stApp .main input, .stApp .main textarea {{
            color: var(--text-primary) !important;
        }}

        /* ── Main content padding ── */
        .block-container {{
            padding: 1.5rem 2.5rem 4rem;
            max-width: 1440px;
        }}

        /* ── Metric cards (glassmorphic) ── */
        div[data-testid="metric-container"] {{
            background: {metric_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.3rem 1.5rem;
            box-shadow: var(--shadow);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-hover);
        }}
        div[data-testid="metric-container"] label,
        div[data-testid="metric-container"] [data-testid="stMetricLabel"],
        div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
            color: var(--text-muted) !important;
            font-size: 0.73rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.07em !important;
            text-transform: uppercase !important;
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"],
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div,
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] p {{
            color: var(--text-primary) !important;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}
        div[data-testid="metric-container"] [data-testid="stMetricDelta"],
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] p,
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg {{
            color: var(--text-muted) !important;
        }}
        /* Catch Streamlit generated class names for metric values */
        div[data-testid="metric-container"] > div > div > div {{
            color: var(--text-primary) !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            width: 100%;
            border-radius: 10px;
            height: 2.8rem;
            font-weight: 600;
            font-size: 0.88rem;
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            color: white !important;
            border: none;
            box-shadow: 0 2px 10px rgba(0,54,136,0.25);
            transition: all 0.22s ease;
            letter-spacing: 0.02em;
        }}
        .stButton > button p {{
            color: white !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(0,54,136,0.4);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #002060 0%, #003688 60%, #004aad 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] * {{
            color: #dce8ff !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stDateInput label,
        section[data-testid="stSidebar"] .stCheckbox label {{
            color: #a8c4f0 !important;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.12) !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox > div > div {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 8px;
            color: white !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {{
            border-color: rgba(255,255,255,0.45) !important;
            box-shadow: 0 0 0 2px rgba(0,152,212,0.4);
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background: transparent !important;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
            margin-bottom: 1.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--text-muted);
            background: transparent !important;
            border: none;
            transition: all 0.18s;
            letter-spacing: 0.01em;
        }}
        .stTabs [data-baseweb="tab"] p {{
            color: inherit !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: var(--text-primary) !important;
            background: var(--bg-card) !important;
        }}
        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"]:hover {{
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%) !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(0,54,136,0.3);
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background-color: var(--bg-card-solid) !important;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--text-primary) !important;
            transition: background 0.2s;
        }}
        .streamlit-expanderHeader p {{
            color: inherit !important;
        }}
        .streamlit-expanderContent {{
            background-color: var(--bg-card-solid) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 var(--radius) var(--radius);
        }}

        /* ── Dataframe ── */
        .stDataFrame {{
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow);
        }}

        /* ── Progress bar ── */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #003688, #0098D4);
            border-radius: 4px;
        }}

        /* ── Hero card ── */
        .hero-card {{
            background: linear-gradient(135deg, #002060 0%, #003688 50%, #005ab5 100%);
            background-size: 200% 200%;
            animation: gradientShift 8s ease infinite;
            border-radius: 18px;
            padding: 2rem 2.5rem;
            color: white !important;
            box-shadow: 0 16px 48px rgba(0,54,136,0.3);
            margin-bottom: 1.8rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-card p {{
            color: white !important;
        }}
        @keyframes gradientShift {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .hero-card::before {{
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 240px; height: 240px;
            background: rgba(255,255,255,0.05);
            border-radius: 50%;
        }}
        .hero-card::after {{
            content: '';
            position: absolute;
            bottom: -80px; left: -40px;
            width: 300px; height: 300px;
            background: rgba(255,255,255,0.03);
            border-radius: 50%;
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }}
        .hero-subtitle {{
            font-size: 0.95rem;
            opacity: 0.8;
            margin-top: 0.4rem;
            font-weight: 400;
            letter-spacing: 0.01em;
        }}

        /* ── Section divider ── */
        .section-divider {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 1.5rem 0;
        }}

        /* ── Status badge ── */
        .status-badge {{
            display: inline-block;
            padding: 0.28rem 0.85rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ── Line pill ── */
        .line-pill {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            color: white;
            margin: 2px;
        }}

        /* ── Info card ── */
        .info-card {{
            background: var(--bg-card-solid);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.4rem 1.6rem;
            box-shadow: var(--shadow);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}
        .info-card:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-hover);
        }}

        /* ── Fade-in animation ── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{
            animation: fadeInUp 0.4s ease-out both;
        }}

        /* ── Footer ── */
        .dashboard-footer {{
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 2.5rem;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{
            background: #003688;
            border-radius: 4px;
        }}

        /* ── Alert boxes ── */
        .stAlert {{
            border-radius: var(--radius);
        }}

        /* ── Subheadings ── */
        h2, h3 {{
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* ── Caption text ── */
        .stCaption, small {{
            color: var(--text-muted) !important;
            font-size: 0.82rem;
        }}

        /* ── st.info ── */
        div[data-testid="stNotification"] {{
            border-radius: var(--radius);
        }}
    </style>
    """, unsafe_allow_html=True)
