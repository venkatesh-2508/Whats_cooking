import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PIL import Image
import config, rec_sys
from ingredient_parser import ingredient_parser
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="What's Cooking? 🍳",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0d0f14;
    --bg-secondary:  #13161e;
    --bg-card:       rgba(255,255,255,0.04);
    --accent-orange: #ff6b35;
    --accent-gold:   #f7c948;
    --accent-green:  #4ade80;
    --accent-pink:   #f472b6;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --border:        rgba(255,255,255,0.08);
    --glow-orange:   rgba(255,107,53,0.25);
    --glow-gold:     rgba(247,201,72,0.15);
    --radius:        16px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── App Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, rgba(255,107,53,0.12) 0%, transparent 55%),
                radial-gradient(ellipse at 80% 100%, rgba(247,201,72,0.08) 0%, transparent 55%),
                radial-gradient(ellipse at 50% 50%, rgba(19,22,30,1) 0%, rgba(13,15,20,1) 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main Block ── */
.block-container {
    padding: 2rem 3rem 4rem 3rem !important;
    max-width: 1100px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(255,107,53,0.15) 0%, rgba(247,201,72,0.10) 50%, rgba(244,114,182,0.10) 100%);
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(255,107,53,0.08) 0%, transparent 60%);
    animation: pulse-glow 4s ease-in-out infinite alternate;
}
@keyframes pulse-glow {
    from { opacity: 0.5; transform: scale(1); }
    to   { opacity: 1;   transform: scale(1.05); }
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c948 50%, #f472b6 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.1 !important;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,107,53,0.15);
    border: 1px solid rgba(255,107,53,0.4);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #ff6b35;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: rgba(255,107,53,0.4);
    background: rgba(255,107,53,0.06);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b35, #f7c948);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Search Card ── */
.search-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.search-label {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Inputs ── */
.stTextInput > div {
    background: transparent !important;
}
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
.stTextInput > div > div,
.stTextInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="base-input"] {
    background-color: #141722 !important;
    background: #141722 !important;
    border: 1.5px solid rgba(255,107,53,0.4) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within,
.stTextInput > div > div:focus-within {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.25), 0 0 20px rgba(255,107,53,0.18) !important;
    background-color: #181c2b !important;
    background: #181c2b !important;
}
.stTextInput input,
input[aria-label="ingredients_input"],
div[data-baseweb="input"] input {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 14px !important;
    color: #ffb03a !important; /* Premium golden-amber entered text */
    -webkit-text-fill-color: #ffb03a !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 1.1rem !important;
    caret-color: #ff6b35 !important;
}
.stTextInput input::placeholder,
input[aria-label="ingredients_input"]::placeholder,
div[data-baseweb="input"] input::placeholder {
    color: rgba(148,163,184,0.6) !important;
    -webkit-text-fill-color: rgba(148,163,184,0.6) !important;
    font-weight: 400 !important;
}
.stTextInput div[data-testid="InputInstructions"],
div[data-testid="InputInstructions"] {
    color: rgba(148,163,184,0.65) !important;
}
.stTextInput div[data-testid="InputInstructions"] span,
div[data-testid="InputInstructions"] span {
    color: rgba(148,163,184,0.65) !important;
    -webkit-text-fill-color: rgba(148,163,184,0.65) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #141722 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #f1f5f9 !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary:hover {
    color: #ff6b35 !important;
}
[data-testid="stExpander"] > div[role="region"] {
    background: #0f121a !important;
    color: #cbd5e1 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff6b35 0%, #f7a035 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(255,107,53,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,107,53,0.5) !important;
    background: linear-gradient(135deg, #ff8c5a 0%, #f7b555 100%) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div,
div[data-baseweb="select"] {
    background: #141722 !important;
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
.stSelectbox > div > div:focus-within,
div[data-baseweb="select"]:focus-within {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
}

/* ── Recipe Results Card ── */
.results-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.5rem 0 1rem 0;
}
.results-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
}
.results-badge {
    background: rgba(74,222,128,0.15);
    border: 1px solid rgba(74,222,128,0.4);
    color: #4ade80;
    border-radius: 50px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Recipe Table ── */
.recipe-table-wrapper {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1rem;
}
.recipe-table-wrapper table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.recipe-table-wrapper th {
    background: rgba(255,107,53,0.12);
    color: #ff6b35;
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid rgba(255,107,53,0.2);
    text-align: left;
}
.recipe-table-wrapper td {
    padding: 0.9rem 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: var(--text-muted);
    vertical-align: top;
    line-height: 1.5;
}
.recipe-table-wrapper tr:last-child td { border-bottom: none; }
.recipe-table-wrapper tr:hover td {
    background: rgba(255,107,53,0.05);
    color: var(--text-primary);
}
.recipe-table-wrapper a {
    color: #f7c948 !important;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}
.recipe-table-wrapper a:hover { color: #ff6b35 !important; text-decoration: underline; }

/* ── Single Recipe Card ── */
.recipe-detail-card {
    background: linear-gradient(135deg, rgba(255,107,53,0.08) 0%, rgba(247,201,72,0.05) 100%);
    border: 1px solid rgba(255,107,53,0.25);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
}
.recipe-detail-row {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.recipe-detail-row:last-child { border-bottom: none; }
.recipe-detail-key {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #ff6b35;
    min-width: 110px;
    padding-top: 2px;
}
.recipe-detail-val {
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.6;
    flex: 1;
}
.recipe-detail-val a {
    color: #f7c948 !important;
    font-weight: 600;
    text-decoration: none;
}
.recipe-detail-val a:hover { color: #ff6b35 !important; }

/* ── Score Pill ── */
.score-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(74,222,128,0.2), rgba(74,222,128,0.1));
    border: 1px solid rgba(74,222,128,0.4);
    color: #4ade80;
    border-radius: 50px;
    padding: 2px 14px;
    font-size: 0.9rem;
    font-weight: 700;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161e 0%, #0f1117 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    background: linear-gradient(135deg, #ff6b35, #f7c948);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}
[data-testid="stSidebar"] a { color: #f7c948 !important; }
[data-testid="stSidebar"] a:hover { color: #ff6b35 !important; }
.streamlit-expanderHeader {
    background: rgba(255,107,53,0.08) !important;
    border: 1px solid rgba(255,107,53,0.2) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Info / Success boxes ── */
.stAlert {
    background: rgba(255,107,53,0.08) !important;
    border: 1px solid rgba(255,107,53,0.25) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #ff6b35 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,107,53,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #ff6b35; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Sidebar steps ── */
.how-step {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 1.1rem;
    padding: 0.8rem;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}
.how-step-num {
    min-width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #ff6b35, #f7a035);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 800;
    color: white;
    flex-shrink: 0;
}
.how-step-text {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def make_clickable(name, link):
    return f'<a target="_blank" href="{link}">{name}</a>'


def render_recipe_table(html_table: str):
    st.markdown(
        f'<div class="recipe-table-wrapper">{html_table}</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
            <div style="font-size:3rem;">🍳</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:800;
                        background:linear-gradient(135deg,#ff6b35,#f7c948);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">
                What's Cooking?
            </div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
                ML-Powered Recipe Finder
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.06); margin:0 0 1.5rem 0;">
        """, unsafe_allow_html=True)

        with st.expander("⚙️  How it works", expanded=True):
            st.markdown("""
            <div class="how-step">
                <div class="how-step-num">1</div>
                <div class="how-step-text"><b>Enter ingredients</b> you have at home, separated by commas.</div>
            </div>
            <div class="how-step">
                <div class="how-step-num">2</div>
                <div class="how-step-text"><b>TF-IDF + Cosine Similarity</b> matches your ingredients to 4500+ recipes.</div>
            </div>
            <div class="how-step">
                <div class="how-step-num">3</div>
                <div class="how-step-text"><b>Top 5 recipes</b> are returned ranked by similarity score.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.06); margin:1.5rem 0;">
        <div style="font-size:0.75rem; color:#475569; text-align:center;">
            Built by lORD KVR · Powered by sklearn & Streamlit
        </div>
        """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero Banner ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🤖 ML Powered · 4500+ Recipes</div>
        <div class="hero-title">What's Cooking?</div>
        <div class="hero-subtitle">
            Tell me what ingredients you have and I'll find the perfect recipes for you.
            Powered by <strong style="color:#ff6b35;">TF-IDF embeddings</strong> and
            <strong style="color:#f7c948;">Cosine Similarity</strong> across 4,500+ curated recipes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats Row ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stats-row">
        <div class="stat-card">
            <span class="stat-number">4,500+</span>
            <span class="stat-label">Recipes Indexed</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">TF-IDF</span>
            <span class="stat-label">ML Algorithm</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">Top 5</span>
            <span class="stat-label">Recommendations</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">⚡ Fast</span>
            <span class="stat-label">Real-Time Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session State ────────────────────────────────────────────────────────────
    for key, default in [
        ("recipe_df", ""), ("recipes", ""), ("model_computed", False),
        ("execute_recsys", False), ("recipe_df_clean", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Search Section ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:0.6rem;">
        <span style="font-size:1.15rem; font-weight:700; color:#f8fafc; letter-spacing:0.01em;">
            🥕 Enter your ingredients
        </span><br>
        <span style="font-size:0.88rem; color:#94a3b8; font-weight:400; margin-top:3px; display:inline-block;">
            Separate multiple ingredients with commas — e.g. <span style="color:#f7c948; font-weight:500;">chicken, garlic, tomato, onion</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ingredients = st.text_input(
            label="ingredients_input",
            placeholder="e.g. chicken, garlic, tomato, pasta, basil...",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Find Recipes", use_container_width=True)

    st.session_state.execute_recsys = search_clicked

    # ── Recommendation Trigger ───────────────────────────────────────────────────
    if st.session_state.execute_recsys:
        if not ingredients.strip():
            st.warning("⚠️  Please enter at least one ingredient to search.")
        else:
            with st.spinner("🍳 Finding the best recipes for you..."):
                recipe = rec_sys.RecSys(ingredients)

            st.session_state.recipe_df_clean = recipe.copy()
            recipe["url"] = recipe.apply(
                lambda row: make_clickable(row["recipe"], row["url"]), axis=1
            )
            recipe_display = recipe[["recipe", "url", "ingredients"]]
            st.session_state.recipe_display = recipe_display.to_html(
                escape=False, index=False,
                classes="",
            )
            st.session_state.recipes = recipe.recipe.values.tolist()
            st.session_state.model_computed = True
            st.session_state.execute_recsys = False

    # ── Results ─────────────────────────────────────────────────────────────────
    if st.session_state.model_computed:
        st.markdown("""
        <div class="results-header">
            <span class="results-title">✨ Recipe Recommendations</span>
            <span class="results-badge">✅ Ready</span>
        </div>
        """, unsafe_allow_html=True)

        view_mode = st.selectbox(
            "View mode",
            ["🍽️  Show All Top 5 Recipes", "🔎  Explore a Single Recipe"],
            label_visibility="collapsed",
        )

        if "Show All" in view_mode:
            render_recipe_table(st.session_state.recipe_display)

        else:
            selection = st.selectbox(
                "Pick a recipe:",
                options=st.session_state.recipes,
            )
            details = st.session_state.recipe_df_clean.loc[
                st.session_state.recipe_df_clean.recipe == selection
            ]
            recipe_name = details.recipe.values[0]
            ingredients_val = details.ingredients.values[0]
            url_val = details.url.values[0]
            score_val = details.score.values[0]

            st.markdown(f"""
            <div class="recipe-detail-card">
                <div style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700;
                            color:#f1f5f9; margin-bottom:1.2rem;">
                    🍴 {recipe_name}
                </div>
                <div class="recipe-detail-row">
                    <div class="recipe-detail-key">Ingredients</div>
                    <div class="recipe-detail-val">{ingredients_val}</div>
                </div>
                <div class="recipe-detail-row">
                    <div class="recipe-detail-key">Full Recipe</div>
                    <div class="recipe-detail-val">
                        <a href="{url_val}" target="_blank">🔗 View Full Recipe →</a>
                    </div>
                </div>
                <div class="recipe-detail-row">
                    <div class="recipe-detail-key">Match Score</div>
                    <div class="recipe-detail-val">
                        <span class="score-pill">⭐ {score_val}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
