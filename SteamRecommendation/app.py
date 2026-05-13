# STEAMVAULT — Modern Steam-like Recommendation Dashboard

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="SteamVault",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "steam_top_games_2026.csv"

# ==============================================================
# CUSTOM CSS
# ==============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b0e14, #101826);
        color: #e2e8f0;
    }
    .hero {
        background: linear-gradient(135deg, rgba(26,159,255,.15), rgba(104,211,145,.06));
        border: 1px solid rgba(99,179,237,.2);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0;
        font-size: 3rem;
        color: #63b3ed;
    }
    .hero p {
        margin-top: 6px;
        color: #94a3b8;
    }
    .metric-card {
        background: #161d2e;
        border: 1px solid rgba(99,179,237,.12);
        border-radius: 16px;
        padding: 18px;
    }
    .game-card {
        background: #161d2e;
        border: 1px solid rgba(99,179,237,.12);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .score-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #1a9fff;
        color: white;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================
# DATA LOADING
# ==============================================================
def parse_owners(s):
    if pd.isna(s) or not str(s).strip():
        return np.nan
    nums = re.findall(r"[\d,]+", str(s))
    nums = [float(x.replace(",", "")) for x in nums]
    if len(nums) < 2:
        return np.nan
    return round(np.mean(nums) / 1e6, 2)


@st.cache_data
def load_steam(path):
    if not Path(path).exists():
        return pd.DataFrame(
            {
                "app_id": [1],
                "name": ["Demo Game"],
                "release_date": ["Jan 1, 2020"],
                "price_usd": [19.99],
                "is_free": [False],
                "discount_pct": [0],
                "developer": ["Demo Studio"],
                "publisher": ["Demo Publisher"],
                "genres": ["Action"],
                "categories": ["Single-player"],
                "tags": ["Action, Indie"],
                "platforms_win": [True],
                "platforms_mac": [False],
                "platforms_linux": [False],
                "metacritic_score": [80],
                "recommendations": [5000],
                "positive_reviews": [4500],
                "negative_reviews": [500],
                "avg_playtime_forever": [600],
                "avg_playtime_2weeks": [120],
                "median_playtime": [400],
                "peak_ccu": [200],
                "required_age": [0],
                "dlc_count": [0],
                "achievements": [20],
                "short_description": ["A demo game."],
                "header_image": [""],
                "estimated_owners": ["100,000 .. 200,000"],
            }
        )

    df = pd.read_csv(path)

    # year extraction
    df["year"] = (
        df["release_date"].astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
    )

    numeric_cols = [
        "price_usd", "discount_pct", "metacritic_score", "recommendations",
        "positive_reviews", "negative_reviews", "avg_playtime_forever",
        "avg_playtime_2weeks", "median_playtime", "peak_ccu",
        "required_age", "dlc_count", "achievements"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    chr_cols = [
        "name", "genres", "categories", "tags", "developer",
        "publisher", "short_description", "header_image", "estimated_owners"
    ]

    for col in chr_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").replace("nan", "")

    df["total_reviews"] = df["positive_reviews"] + df["negative_reviews"]
    df["positivity"] = np.where(
        df["total_reviews"] > 0,
        df["positive_reviews"] / df["total_reviews"] * 100,
        np.nan,
    )
    df["positivity"] = df["positivity"].round(1)

    df["genre_primary"] = df["genres"].astype(str).str.split(",").str[0].str.strip()

    cat = df["categories"].str.lower()
    df["is_multiplayer"] = cat.str.contains("multi-player|multiplayer", na=False)
    df["is_coop"] = cat.str.contains("co-op", na=False)
    df["is_singleplayer"] = cat.str.contains("single-player", na=False)

    df["score_raw"] = df["positivity"] * np.log1p(df["recommendations"].fillna(0))
    mn, mx = df["score_raw"].min(), df["score_raw"].max()
    if pd.notna(mn) and pd.notna(mx) and mx > mn:
        df["score"] = ((df["score_raw"] - mn) / (mx - mn) * 100).round(1)
    else:
        df["score"] = 50

    df["owners_m"] = df["estimated_owners"].apply(parse_owners)

    return df


games = load_steam(CSV_PATH)
# Pastikan kolom genre_primary ada
if "genre_primary" not in games.columns:
    if "genres" in games.columns:
        games["genre_primary"] = (
            games["genres"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )
    else:
        games["genre_primary"] = ""
# Pastikan kolom name ada
if "name" not in games.columns:
    games["name"] = ""

# List genre dan title
all_genres = sorted(
    [g for g in games["genre_primary"].dropna().unique() if str(g).strip()]
)

all_titles = sorted(
    [t for t in games["name"].dropna().unique() if str(t).strip()]
)

# ==============================================================
# RECOMMENDATION FUNCTIONS
# ==============================================================
def wr_score(df):
    C = df["positivity"].mean()
    m = df["recommendations"].quantile(0.70)
    v = df["recommendations"].fillna(0)
    R = df["positivity"].fillna(C)
    return ((v / (v + m)) * R + (m / (v + m)) * C).round(2)


def rec_rule(df, genres, max_price, min_pos, min_recs, mode, top_n):
    res = df.copy()
    if genres:
        res = res[res["genre_primary"].isin(genres)]
    res = res[(res["price_usd"].fillna(9999) <= max_price) | (res.get("is_free", False))]
    res = res[res["positivity"].fillna(0) >= min_pos]
    res = res[res["recommendations"].fillna(0) >= min_recs]

    if mode == "multiplayer":
        res = res[res["is_multiplayer"]]
    elif mode == "coop":
        res = res[res["is_coop"]]
    elif mode == "singleplayer":
        res = res[res["is_singleplayer"]]

    return res.sort_values("score", ascending=False).head(top_n)


def rec_cb(df, ref_name, top_n):
    if not ref_name:
        return pd.DataFrame()

    ref = df[df["name"].str.lower() == ref_name.lower()]
    if ref.empty:
        return pd.DataFrame()
    ref = ref.iloc[0]

    def feat(row):
        return f"{row.get('genre_primary','')} {row.get('genres','')} {row.get('tags','')} {row.get('developer','')}".lower()

    t_tokens = set([t for t in re.split(r"[,\s]+", feat(ref)) if len(t) > 1])

    scores = []
    for _, row in df.iterrows():
        r_tokens = set([t for t in re.split(r"[,\s]+", feat(row)) if len(t) > 1])
        if not r_tokens or not t_tokens:
            scores.append(0)
        else:
            scores.append(len(t_tokens & r_tokens) / math.sqrt(len(t_tokens) * len(r_tokens)))

    res = df.copy()
    res["cb_score"] = scores
    res = res[res["name"].str.lower() != ref_name.lower()]
    return res.sort_values("cb_score", ascending=False).head(top_n)


def rec_cf(df, genres, min_pos, top_n):
    res = df.copy()
    if genres:
        res = res[res["genre_primary"].isin(genres)]
    res = res[res["positivity"].fillna(0) >= min_pos]
    res["wr"] = wr_score(res)
    return res.sort_values("wr", ascending=False).head(top_n)


def rec_hybrid(df, ref_name, top_n, cb_w=0.6):
    cb = rec_cb(df, ref_name, len(df))
    if cb.empty:
        return cb

    cb["wr"] = wr_score(cb)

    scaler = MinMaxScaler()
    cb["cb_norm"] = scaler.fit_transform(cb[["cb_score"]].fillna(0))
    cb["wr_norm"] = scaler.fit_transform(cb[["wr"]].fillna(0))
    cb["hybrid_score"] = cb_w * cb["cb_norm"] + (1 - cb_w) * cb["wr_norm"]

    return cb.sort_values("hybrid_score", ascending=False).head(top_n)


# ==============================================================
# FILTER SIDEBAR
# ==============================================================
st.sidebar.title("⚙️ Filters")

if not games.empty:
    min_year = int(games["year"].dropna().min()) if games["year"].notna().any() else 2000
    max_year = int(games["year"].dropna().max()) if games["year"].notna().any() else 2026
else:
    min_year, max_year = 2000, 2026

f_year = st.sidebar.slider("Release Year", min_year, max_year, (max(min_year, 2015), max_year))
f_price = st.sidebar.slider("Max Price ($)", 0, 150, 60)
f_pos = st.sidebar.slider("Min Positivity (%)", 0, 100, 60)
f_genre = st.sidebar.selectbox("Genre", ["All"] + all_genres)
f_mode = st.sidebar.selectbox(
    "Mode", ["any", "singleplayer", "multiplayer", "coop"]
)

# ==============================================================
# FILTERED DATA
# ==============================================================
def get_filtered_data():
    df = games.copy()
    df = df[df["year"].fillna(0).between(f_year[0], f_year[1])]
    df = df[(df["price_usd"].fillna(9999) <= f_price) | (df.get("is_free", False))]
    df = df[df["positivity"].fillna(0) >= f_pos]

    if f_genre != "All":
        df = df[df["genre_primary"] == f_genre]

    if f_mode == "singleplayer":
        df = df[df["is_singleplayer"]]
    elif f_mode == "multiplayer":
        df = df[df["is_multiplayer"]]
    elif f_mode == "coop":
        df = df[df["is_coop"]]

    return df


filt = get_filtered_data()

# ==============================================================
# HEADER
# ==============================================================
st.markdown(
    """
    <div class='hero'>
        <h1>🎮 STEAMVAULT</h1>
        <p>Modern game discovery & recommendation dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================
# NAVIGATION
# ==============================================================
tab_discover, tab_detail, tab_recommend, tab_analytics = st.tabs(
    ["⚡ Discover", "🔬 Detail", "🎯 Recommend", "📡 Analytics"]
)

# ==============================================================
# DISCOVER TAB
# ==============================================================
with tab_discover:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles", f"{len(filt):,}")
    c2.metric("Free Games", int(filt.get("is_free", pd.Series(dtype=bool)).sum()))
    c3.metric("Positivity ≥ 90%", int((filt["positivity"] >= 90).sum()))
    c4.metric("Avg Metacritic", f"{filt['metacritic_score'].mean():.0f}" if filt['metacritic_score'].notna().any() else "—")

    search = st.text_input("Search game titles")
    sort_by = st.selectbox(
        "Sort By",
        ["score", "recommendations", "positivity", "avg_playtime_forever", "peak_ccu", "year", "metacritic_score"],
    )
    n_show = st.number_input("Number of Games", 8, 200, 24, step=8)

    browse = filt.copy()
    if search:
        browse = browse[browse["name"].str.contains(search, case=False, na=False)]

    browse = browse.sort_values(sort_by, ascending=False, na_position="last").head(n_show)

    for _, row in browse.iterrows():
        st.markdown(f"""
        <div class='game-card'>
            <div class='score-pill'>Score {row.get('score', 0):.1f}</div>
            <h4>{row['name']}</h4>
            <p>{row.get('genre_primary', '')} · {int(row['year']) if pd.notna(row['year']) else '—'}</p>
            <p>👍 {row.get('positivity', 0):.1f}% | 👥 {int(row.get('recommendations', 0)):,}</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================
# DETAIL TAB
# ==============================================================
with tab_detail:
    selected_game = st.selectbox("Select Game", all_titles)

    if selected_game:
        g = games[games["name"] == selected_game].iloc[0]

        st.subheader(g["name"])
        st.write(g.get("short_description", "No description available."))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positivity", f"{g.get('positivity', np.nan):.1f}%")
        col2.metric("Recommendations", f"{int(g.get('recommendations', 0)):,}")
        col3.metric("Metacritic", g.get("metacritic_score", "—"))
        col4.metric("Playtime", f"{g.get('avg_playtime_forever', 0)/60:.1f} h")

        st.markdown("### Similar Games")
        sim = rec_cb(games, selected_game, 8)
        if not sim.empty:
            st.dataframe(sim[["name", "genre_primary", "cb_score"]], use_container_width=True)

# ==============================================================
# RECOMMEND TAB
# ==============================================================
with tab_recommend:
    method = st.selectbox(
        "Algorithm",
        ["Rule-Based", "Content-Based", "Collaborative Filtering", "Hybrid"],
    )

    recs = pd.DataFrame()

    if method == "Rule-Based":
        genres = st.multiselect("Genres", all_genres)
        max_price = st.slider("Max Price", 0, 150, 30)
        min_pos2 = st.slider("Min Positivity", 0, 100, 70)
        min_recs = st.slider("Min Recommendations", 0, 100000, 5000, step=1000)
        mode = st.selectbox("Mode", ["any", "singleplayer", "multiplayer", "coop"])
        top_n = st.number_input("Top N", 5, 30, 10)

        if st.button("Run Engine"):
            recs = rec_rule(games, genres, max_price, min_pos2, min_recs, mode, top_n)

    elif method == "Content-Based":
        ref = st.selectbox("Reference Game", all_titles, key="cb")
        top_n = st.number_input("Top N", 5, 30, 10, key="cbn")
        if st.button("Run Engine"):
            recs = rec_cb(games, ref, top_n)

    elif method == "Collaborative Filtering":
        genres = st.multiselect("Genres", all_genres, key="cf")
        min_pos2 = st.slider("Min Positivity", 0, 100, 75, key="cfp")
        top_n = st.number_input("Top N", 5, 30, 10, key="cfn")
        if st.button("Run Engine"):
            recs = rec_cf(games, genres, min_pos2, top_n)

    else:
        ref = st.selectbox("Reference Game", all_titles, key="hy")
        cb_w = st.slider("CB Weight", 0.0, 1.0, 0.6, 0.05)
        top_n = st.number_input("Top N", 5, 30, 10, key="hyn")
        if st.button("Run Engine"):
            recs = rec_hybrid(games, ref, top_n, cb_w)

    if not recs.empty:
        show_cols = [c for c in ["name", "genre_primary", "score", "cb_score", "wr", "hybrid_score"] if c in recs.columns]
        st.dataframe(recs[show_cols], use_container_width=True)

# ==============================================================
# ANALYTICS TAB
# ==============================================================
with tab_analytics:
    if not filt.empty:
        c1, c2 = st.columns(2)

        with c1:
            genre_count = (
                filt[filt["genre_primary"] != ""]
                .groupby("genre_primary")
                .size()
                .sort_values(ascending=False)
                .head(12)
                .reset_index(name="n")
            )
            fig = px.bar(genre_count, x="n", y="genre_primary", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(filt, x="price_usd", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            filt,
            x=filt["avg_playtime_forever"] / 60,
            y="score",
            color="positivity",
            hover_name="name",
            labels={"x": "Avg Playtime (hours)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filt, use_container_width=True)

        csv = filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export CSV",
            data=csv,
            file_name="steamvault_export.csv",
            mime="text/csv",
        )
