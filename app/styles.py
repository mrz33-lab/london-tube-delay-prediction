"""
CSS injection for the TfL-branded Streamlit dashboard.
"""

import streamlit as st


def apply_custom_css(dark_mode: bool = False) -> None:
    """Inject TfL-branded CSS. Dark mode swaps root colour variables."""
    if dark_mode:
        bg_primary   = "#0d1117"
        bg_secondary = "#161b22"
        bg_card      = "#21262d"
        text_primary = "#e6edf3"
        text_muted   = "#8b949e"
        border_col   = "#30363d"
        metric_bg    = "#161b22"
    else:
        bg_primary   = "#f0f4f8"
        bg_secondary = "#ffffff"
        bg_card      = "#ffffff"
        text_primary = "#1a1a2e"
        text_muted   = "#6c757d"
        border_col   = "#dee2e6"
        metric_bg    = "#f8f9fa"

    st.markdown(f"""
    <style>
        /* ── Root variables ── */
        :root {{
            --bg-primary:   {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-card:      {bg_card};
            --text-primary: {text_primary};
            --text-muted:   {text_muted};
            --border:       {border_col};
            --accent:       #003688;
            --accent2:      #0098D4;
            --radius:       12px;
            --shadow:       0 4px 16px rgba(0,0,0,0.10);
            --shadow-hover: 0 8px 28px rgba(0,0,0,0.18);
        }}

        /* ── App background ── */
        .stApp {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, Oxygen, Ubuntu, sans-serif;
        }}

        /* ── Main content padding ── */
        .block-container {{
            padding: 1.5rem 2rem 3rem;
            max-width: 1400px;
        }}

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {{
            background-color: {metric_bg};
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            box-shadow: var(--shadow);
            transition: box-shadow 0.25s;
        }}
        div[data-testid="metric-container"]:hover {{
            box-shadow: var(--shadow-hover);
        }}
        div[data-testid="metric-container"] label {{
            color: var(--text-muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            width: 100%;
            border-radius: 8px;
            height: 2.8rem;
            font-weight: 600;
            font-size: 0.9rem;
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            color: white;
            border: none;
            box-shadow: 0 2px 8px rgba(0,54,136,0.3);
            transition: all 0.25s ease;
            letter-spacing: 0.02em;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,54,136,0.45);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #003688 0%, #002060 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: #e0e8ff !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stDateInput label,
        section[data-testid="stSidebar"] .stCheckbox label {{
            color: #b8ccff !important;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15) !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox > div > div {{
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: white !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: transparent;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--text-muted);
            background: transparent;
            border: none;
            transition: all 0.2s;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            color: white !important;
            box-shadow: 0 4px 12px rgba(0,54,136,0.25);
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background-color: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            font-weight: 600;
            color: var(--text-primary);
        }}
        .streamlit-expanderContent {{
            background-color: var(--bg-card);
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

        /* ── Custom hero card ── */
        .hero-card {{
            background: linear-gradient(135deg, #003688 0%, #0098D4 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            color: white;
            box-shadow: 0 12px 40px rgba(0,54,136,0.35);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-card::before {{
            content: '';
            position: absolute;
            top: -40px; right: -40px;
            width: 200px; height: 200px;
            background: rgba(255,255,255,0.06);
            border-radius: 50%;
        }}
        .hero-card::after {{
            content: '';
            position: absolute;
            bottom: -60px; left: -30px;
            width: 250px; height: 250px;
            background: rgba(255,255,255,0.04);
            border-radius: 50%;
        }}
        .hero-title {{
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 0.3rem;
        }}

        /* ── Status badge ── */
        .status-badge {{
            display: inline-block;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ── Line pill ── */
        .line-pill {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            color: white;
            margin: 2px;
        }}

        /* ── Section card ── */
        .section-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }}

        /* ── Fade-in animation ── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{
            animation: fadeInUp 0.45s ease-out both;
        }}

        /* ── Footer ── */
        .dashboard-footer {{
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.82rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{
            background: #003688;
            border-radius: 4px;
        }}

        /* ── Info / warning / success boxes ── */
        .stAlert {{
            border-radius: var(--radius);
        }}
    </style>
    """, unsafe_allow_html=True)
