# SteamVault — Steam Game Discovery & Recommendation Dashboard

import math
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SteamVault",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# FIX: path relatif terhadap lokasi file ini, bukan working directory
CSV_PATH = Path(__file__).parent / "steam_top_games_2026.csv"
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0b0e14, #101826); color: #e2e8f0; }
    .hero {
        background: linear-gradient(135deg, rgba(26,159,255,.15), rgba(104,211,145,.06));
        border: 1px solid rgba(99,179,237,.2);
        border-radius: 20px; padding: 28px; margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 3rem; color: #63b3ed; }
    .hero p  { margin-top: 6px; color: #94a3b8; }
    .game-card {
        background: #161d2e; border: 1px solid rgba(99,179,237,.12);
        border-radius: 16px; padding: 16px; margin-bottom: 14px;
        display: flex; gap: 14px; align-items: flex-start;
    }
    .game-card img {
        width: 120px; height: 56px; object-fit: cover;
        border-radius: 8px; flex-shrink: 0;
    }
    .game-card-body { flex: 1; min-width: 0; }
    .score-pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: #1a9fff; color: white; font-size: 12px; font-weight: 700;
        margin-bottom: 8px;
    }
    .free-pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: #38a169; color: white; font-size: 12px; font-weight: 700;
        margin-left: 6px;
    }
    .discount-pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: #e53e3e; color: white; font-size: 12px; font-weight: 700;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ── Data loading ──────────────────────────────────────────────────────────────
def parse_owners(s: str) -> float:
    """Parse '1,000,000 .. 2,000,000' → mean in millions."""
    if pd.isna(s) or not str(s).strip():
        return np.nan
    nums = [float(x.replace(",", "")) for x in re.findall(r"[\d,]+", str(s))]
    return round(np.mean(nums) / 1_000_000, 2) if len(nums) >= 2 else np.nan
 
 
@st.cache_data(ttl=3600)  # FIX: cache invalidation setelah 1 jam
def load_steam(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"File CSV tidak ditemukan: {path}")
        st.stop()
 
    df = pd.read_csv(path)
 
    # ── Tahun ─────────────────────────────────────────────────────────────────
    df["year"] = (
        df["release_date"].astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
    )
 
    # ── Numerik ───────────────────────────────────────────────────────────────
    numeric_cols = [
        "price_usd", "discount_pct", "metacritic_score", "recommendations",
        "positive_reviews", "negative_reviews", "avg_playtime_forever",
        "avg_playtime_2weeks", "median_playtime", "peak_ccu",
        "required_age", "dlc_count", "achievements",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # FIX: outlier sentinel di median_playtime (nilai > 99th percentile yang aneh)
    p99_median = df["median_playtime"].quantile(0.99)
    df["median_playtime"] = df["median_playtime"].where(
        df["median_playtime"] <= p99_median, np.nan
    )
 
    # FIX: avg_playtime_2weeks — ganti 0 dengan NaN supaya tidak menyesatkan
    df["avg_playtime_2weeks"] = df["avg_playtime_2weeks"].replace(0, np.nan)
 
    # ── String ────────────────────────────────────────────────────────────────
    str_cols = [
        "name", "genres", "categories", "tags", "developer",
        "publisher", "short_description", "header_image", "estimated_owners",
    ]
    for col in str_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).replace("nan", "")
 
    # FIX: is_free harus boolean bersih, bukan string "True"/"False"
    if df["is_free"].dtype == object:
        df["is_free"] = df["is_free"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)
    else:
        df["is_free"] = df["is_free"].fillna(False).astype(bool)
 
    # ── Derived columns ───────────────────────────────────────────────────────
    df["total_reviews"] = df["positive_reviews"].fillna(0) + df["negative_reviews"].fillna(0)
    df["positivity"] = np.where(
        df["total_reviews"] > 0,
        df["positive_reviews"] / df["total_reviews"] * 100,
        np.nan,
    ).round(1)  # type: ignore[arg-type]
 
    df["genre_primary"] = df["genres"].str.split(",").str[0].str.strip()
 
    cat = df["categories"].str.lower()
    df["is_multiplayer"] = cat.str.contains("multi-player|multiplayer", na=False)
    df["is_coop"]        = cat.str.contains("co-op", na=False)
    df["is_singleplayer"]= cat.str.contains("single-player", na=False)
 
    # FIX: clip outlier sebelum scoring supaya tidak didominasi 1-2 game
    recs_clipped = df["recommendations"].clip(upper=df["recommendations"].quantile(0.99))
    df["score_raw"] = df["positivity"] * np.log1p(recs_clipped.fillna(0))
    mn, mx = df["score_raw"].min(), df["score_raw"].max()
    df["score"] = ((df["score_raw"] - mn) / (mx - mn) * 100).round(1) if mx > mn else 50.0
 
    df["owners_m"] = df["estimated_owners"].apply(parse_owners)
 
    # FIX: harga game gratis yang price_usd = 0 pastikan is_free = True juga
    df.loc[df["price_usd"] == 0, "is_free"] = True
 
    # ── TF-IDF matrix untuk content-based ────────────────────────────────────
    df["_cb_text"] = (
        df["genre_primary"].fillna("") + " " +
        df["genres"].fillna("") + " " +
        df["tags"].fillna("") + " " +
        df["developer"].fillna("")
    ).str.lower()
 
    return df
 
 
games = load_steam(CSV_PATH)
all_genres = sorted(g for g in games["genre_primary"].dropna().unique() if g)
all_titles = sorted(t for t in games["name"].dropna().unique() if t)
 
 
# ── TF-IDF (dicompute sekali, di-cache di session) ────────────────────────────
@st.cache_resource
def build_tfidf(texts: list[str]):
    """FIX: ganti Jaccard token matching dengan TF-IDF + cosine similarity."""
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)
    mat = vec.fit_transform(texts)
    return mat
 
 
cb_matrix = build_tfidf(games["_cb_text"].tolist())
 
 
# ── Recommendation engines ────────────────────────────────────────────────────
def wr_score(df: pd.DataFrame) -> pd.Series:
    """Bayesian weighted rating (mirip formula IMDb)."""
    C = df["positivity"].mean()
    m = df["recommendations"].quantile(0.70)
    v = df["recommendations"].fillna(0)
    R = df["positivity"].fillna(C)
    return ((v / (v + m)) * R + (m / (v + m)) * C).round(2)
 
 
def rec_rule(
    df: pd.DataFrame,
    genres: list[str],
    max_price: float,
    min_pos: float,
    min_recs: int,
    mode: str,
    top_n: int,
) -> pd.DataFrame:
    res = df.copy()
    if genres:
        res = res[res["genre_primary"].isin(genres)]
 
    # FIX: is_free sekarang boolean bersih, bisa langsung dipakai
    price_ok = (res["price_usd"].fillna(9999) <= max_price) | res["is_free"]
    res = res[price_ok]
    res = res[res["positivity"].fillna(0) >= min_pos]
    res = res[res["recommendations"].fillna(0) >= min_recs]
 
    if mode == "multiplayer":
        res = res[res["is_multiplayer"]]
    elif mode == "coop":
        res = res[res["is_coop"]]
    elif mode == "singleplayer":
        res = res[res["is_singleplayer"]]
 
    return res.sort_values("score", ascending=False).head(top_n)
 
 
def rec_cb(df: pd.DataFrame, ref_name: str, top_n: int) -> pd.DataFrame:
    """FIX: TF-IDF cosine similarity, jauh lebih akurat."""
    if not ref_name:
        return pd.DataFrame()
 
    idx = df.index[df["name"].str.lower() == ref_name.lower()]
    if idx.empty:
        return pd.DataFrame()
 
    ref_idx = idx[0]
    ref_vec = cb_matrix[df.index.get_loc(ref_idx)]
    sims = cosine_similarity(ref_vec, cb_matrix).flatten()
 
    res = df.copy()
    res["cb_score"] = sims
    res = res[res["name"].str.lower() != ref_name.lower()]
    return res.sort_values("cb_score", ascending=False).head(top_n)
 
 
def rec_cf(
    df: pd.DataFrame, genres: list[str], min_pos: float, top_n: int
) -> pd.DataFrame:
    res = df.copy()
    if genres:
        res = res[res["genre_primary"].isin(genres)]
    res = res[res["positivity"].fillna(0) >= min_pos]
    # FIX: filter game dengan terlalu sedikit review
    res = res[res["recommendations"].fillna(0) >= 50]
    res["wr"] = wr_score(res)
    return res.sort_values("wr", ascending=False).head(top_n)
 
 
def rec_hybrid(
    df: pd.DataFrame, ref_name: str, top_n: int, cb_w: float = 0.6
) -> pd.DataFrame:
    cb = rec_cb(df, ref_name, len(df))
    if cb.empty:
        return cb
 
    cb["wr"] = wr_score(cb)
 
    scaler = MinMaxScaler()
    cb["cb_norm"] = scaler.fit_transform(cb[["cb_score"]].fillna(0))
    cb["wr_norm"] = scaler.fit_transform(cb[["wr"]].fillna(0))
    cb["hybrid_score"] = cb_w * cb["cb_norm"] + (1 - cb_w) * cb["wr_norm"]
 
    return cb.sort_values("hybrid_score", ascending=False).head(top_n)
 
 
# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Filters")
 
min_year = int(games["year"].dropna().min()) if games["year"].notna().any() else 2000
max_year = int(games["year"].dropna().max()) if games["year"].notna().any() else 2026
 
f_year   = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))
f_price  = st.sidebar.slider("Max Price ($)", 0, 150, 60)
f_pos    = st.sidebar.slider("Min Positivity (%)", 0, 100, 0)
f_genre  = st.sidebar.selectbox("Genre", ["All"] + all_genres)
f_mode   = st.sidebar.selectbox("Mode", ["any", "singleplayer", "multiplayer", "coop"])
 
# NOTE: filter sidebar berlaku di Discover & Analytics.
# Tab Recommend punya filter sendiri yang independen (by design).
st.sidebar.info("💡 Filter di atas berlaku untuk tab Discover & Analytics.")
 
 
def get_filtered_data() -> pd.DataFrame:
    df = games.copy()
    df = df[df["year"].fillna(0).between(f_year[0], f_year[1])]
 
    # FIX: is_free boolean bersih → tidak ada lagi ValueError
    price_ok = (df["price_usd"].fillna(9999) <= f_price) | df["is_free"]
    df = df[price_ok]
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
 
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero'><h1>🎮 STEAMVAULT</h1>"
    "<p>Modern game discovery & recommendation dashboard</p></div>",
    unsafe_allow_html=True,
)
 
tab_discover, tab_detail, tab_recommend, tab_analytics = st.tabs(
    ["⚡ Discover", "🔬 Detail", "🎯 Recommend", "📡 Analytics"]
)
 
 
# ── DISCOVER ──────────────────────────────────────────────────────────────────
with tab_discover:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles", f"{len(filt):,}")
    c2.metric("Free Games", int(filt["is_free"].sum()))  # FIX: is_free boolean
    c3.metric("Positivity ≥ 90%", int((filt["positivity"] >= 90).sum()))
    meta_mean = filt["metacritic_score"].mean()
    c4.metric("Avg Metacritic", f"{meta_mean:.0f}" if pd.notna(meta_mean) else "—")
 
    search  = st.text_input("Search game titles")
    sort_by = st.selectbox(
        "Sort By",
        ["score", "recommendations", "positivity", "avg_playtime_forever", "peak_ccu", "year", "metacritic_score"],
    )
    n_show = st.number_input("Number of Games", 8, 2000, 24, step=8)
 
    browse = filt.copy()
    if search:
        browse = browse[browse["name"].str.contains(search, case=False, na=False)]
    browse = browse.sort_values(sort_by, ascending=False, na_position="last").head(n_show)
 
    for _, row in browse.iterrows():
        price_tag = ""
        if row["is_free"]:
            price_tag = "<span class='free-pill'>FREE</span>"
        elif row.get("discount_pct", 0) > 0:
            price_tag = (
                f"<span class='discount-pill'>-{int(row['discount_pct'])}%</span>"
                f" ${row['price_usd']:.2f}"
            )
        else:
            price_tag = f"${row['price_usd']:.2f}" if pd.notna(row["price_usd"]) else "—"
 
        # FIX: tampilkan header_image dari CSV
        img_html = ""
        if row.get("header_image", ""):
            img_html = f"<img src='{row['header_image']}' onerror=\"this.style.display='none'\">"
 
        steam_url = f"https://store.steampowered.com/app/{int(row['app_id'])}/"
        st.markdown(
            f"""
            <div class='game-card'>
                <a href='{steam_url}' target='_blank' style='display:contents'>
                    {img_html}
                </a>
                <div class='game-card-body'>
                    <span class='score-pill'>Score {row.get('score', 0):.1f}</span>
                    <a href='{steam_url}' target='_blank'
                       style='text-decoration:none;color:inherit'>
                        <h4 style='margin:0 0 4px;display:inline'>{row['name']}</h4>
                        <span style='font-size:12px;color:#4a9eff;margin-left:6px'>↗ Steam</span>
                    </a>
                    <p style='margin:4px 0 0;color:#94a3b8;font-size:13px'>
                        {row.get('genre_primary', '')} ·
                        {int(row['year']) if pd.notna(row['year']) else '—'} ·
                        {price_tag}
                    </p>
                    <p style='margin:4px 0 0;font-size:13px'>
                        👍 {row.get('positivity', 0):.1f}% &nbsp;|&nbsp;
                        👥 {int(row.get('recommendations', 0)):,} recs &nbsp;|&nbsp;
                        🕐 {row.get('avg_playtime_forever', 0)/60:.0f}h avg
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
 
# ── DETAIL ────────────────────────────────────────────────────────────────────
with tab_detail:
    selected_game = st.selectbox("Select Game", all_titles)
 
    if selected_game:
        g = games[games["name"] == selected_game].iloc[0]
 
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if g.get("header_image", ""):
                st.image(g["header_image"], use_container_width=True)
 
        with col_info:
            steam_url = f"https://store.steampowered.com/app/{int(g['app_id'])}/"
            st.subheader(g["name"])
            st.markdown(f"[🔗 Buka di Steam Store]({steam_url})", unsafe_allow_html=False)
            desc = g.get("short_description", "")
            if desc:
                st.write(desc)
 
            col1, col2, col3, col4 = st.columns(4)
            pos = g.get("positivity")
            col1.metric("Positivity", f"{pos:.1f}%" if pd.notna(pos) else "—")
            col2.metric("Recommendations", f"{int(g.get('recommendations', 0)):,}")
            meta = g.get("metacritic_score")
            col3.metric("Metacritic", f"{int(meta)}" if pd.notna(meta) else "—")
            pt = g.get("avg_playtime_forever", 0)
            col4.metric("Avg Playtime", f"{pt/60:.1f} h" if pt else "—")
 
            price_display = "Free" if g["is_free"] else f"${g.get('price_usd', 0):.2f}"
            st.caption(
                f"**Developer:** {g.get('developer','—')} &nbsp;|&nbsp; "
                f"**Publisher:** {g.get('publisher','—')} &nbsp;|&nbsp; "
                f"**Price:** {price_display}"
            )
            if g.get("tags"):
                tags = g["tags"].split(",")[:8]
                st.write(" ".join(f"`{t.strip()}`" for t in tags))
 
        st.markdown("### 🎮 Similar Games")
        sim = rec_cb(games, selected_game, 8)
        if not sim.empty:
            sim["steam_url"] = sim["app_id"].apply(
                lambda x: f"https://store.steampowered.com/app/{int(x)}/"
            )
            st.dataframe(
                sim[["name", "genre_primary", "cb_score", "score", "positivity", "steam_url"]].rename(
                    columns={
                        "name": "Game",
                        "genre_primary": "Genre",
                        "cb_score": "Similarity",
                        "score": "Score",
                        "positivity": "Positivity %",
                        "steam_url": "Steam",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                column_config={"Steam": st.column_config.LinkColumn("Steam", display_text="↗ Buka")},
            )
 
 
# ── RECOMMEND ────────────────────────────────────────────────────────────────
with tab_recommend:
    method = st.selectbox(
        "Algorithm",
        ["Rule-Based", "Content-Based (TF-IDF)", "Collaborative Filtering", "Hybrid"],
    )
 
    recs = pd.DataFrame()
 
    if method == "Rule-Based":
        genres   = st.multiselect("Genres", all_genres)
        max_price = st.slider("Max Price ($)", 0, 150, 30)
        min_pos2  = st.slider("Min Positivity (%)", 0, 100, 70)
        min_recs  = st.slider("Min Recommendations", 0, 100_000, 5_000, step=1_000)
        mode      = st.selectbox("Mode", ["any", "singleplayer", "multiplayer", "coop"])
        top_n     = st.number_input("Top N", 5, 30, 10)
 
        if st.button("🚀 Run Engine"):
            recs = rec_rule(games, genres, max_price, min_pos2, min_recs, mode, int(top_n))
 
    elif method == "Content-Based (TF-IDF)":
        ref   = st.selectbox("Reference Game", all_titles, key="cb")
        top_n = st.number_input("Top N", 5, 30, 10, key="cbn")
 
        if st.button("🚀 Run Engine"):
            recs = rec_cb(games, ref, int(top_n))
 
    elif method == "Collaborative Filtering":
        genres   = st.multiselect("Genres", all_genres, key="cf")
        min_pos2  = st.slider("Min Positivity (%)", 0, 100, 75, key="cfp")
        top_n     = st.number_input("Top N", 5, 30, 10, key="cfn")
 
        if st.button("🚀 Run Engine"):
            recs = rec_cf(games, genres, min_pos2, int(top_n))
 
    else:  # Hybrid
        ref   = st.selectbox("Reference Game", all_titles, key="hy")
        cb_w  = st.slider("Content-Based Weight", 0.0, 1.0, 0.6, 0.05)
        top_n = st.number_input("Top N", 5, 30, 10, key="hyn")
 
        if st.button("🚀 Run Engine"):
            recs = rec_hybrid(games, ref, int(top_n), cb_w)
 
    if not recs.empty:
        show_cols = [c for c in ["name", "genre_primary", "score", "positivity",
                                  "recommendations", "cb_score", "wr", "hybrid_score"]
                     if c in recs.columns]
        recs["steam_url"] = recs["app_id"].apply(
            lambda x: f"https://store.steampowered.com/app/{int(x)}/"
        )
        st.dataframe(
            recs[show_cols + ["steam_url"]],
            use_container_width=True,
            hide_index=True,
            column_config={"steam_url": st.column_config.LinkColumn("Steam", display_text="↗ Buka")},
        )
 
 
# ── ANALYTICS ─────────────────────────────────────────────────────────────────
with tab_analytics:
    if filt.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
    else:
        c1, c2 = st.columns(2)
 
        with c1:
            genre_count = (
                filt[filt["genre_primary"] != ""]
                .groupby("genre_primary").size()
                .sort_values(ascending=False)
                .head(12)
                .reset_index(name="n")
            )
            fig = px.bar(
                genre_count, x="n", y="genre_primary", orientation="h",
                title="Top 12 Genres", labels={"n": "Jumlah Game", "genre_primary": "Genre"},
                color="n", color_continuous_scale="Blues",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
 
        with c2:
            price_data = filt[filt["price_usd"] > 0]
            fig = px.histogram(
                price_data, x="price_usd", nbins=20,
                title="Distribusi Harga (exclude free)",
                labels={"price_usd": "Harga ($)", "count": "Jumlah"},
            )
            st.plotly_chart(fig, use_container_width=True)
 
        # FIX: filter outlier playtime supaya scatter tidak rusak
        scatter_data = filt[
            filt["avg_playtime_forever"].notna() &
            (filt["avg_playtime_forever"] > 0) &
            (filt["avg_playtime_forever"] < filt["avg_playtime_forever"].quantile(0.99))
        ].copy()
        scatter_data["playtime_h"] = scatter_data["avg_playtime_forever"] / 60
 
        fig = px.scatter(
            scatter_data,
            x="playtime_h", y="score",
            color="positivity",
            hover_name="name",
            hover_data={"playtime_h": ":.1f", "score": ":.1f", "positivity": ":.1f"},
            title="Playtime vs Score (warna = positivity)",
            labels={"playtime_h": "Avg Playtime (jam)", "score": "Score", "positivity": "Positivity %"},
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)
 
        # Tabel lengkap
        st.markdown("### 📋 Data Lengkap")
        display_cols = [
            "name", "genre_primary", "year", "price_usd", "is_free",
            "positivity", "recommendations", "score", "metacritic_score",
            "avg_playtime_forever", "peak_ccu",
        ]
        st.dataframe(filt[display_cols], use_container_width=True, hide_index=True)
 
        csv_bytes = filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export CSV",
            data=csv_bytes,
            file_name="steamvault_export.csv",
            mime="text/csv",
        )
