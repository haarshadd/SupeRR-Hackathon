from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
DATA = BASE / "data"

st.set_page_config(page_title="SupeRR Selector | Pressure Intelligence", page_icon="🏏", layout="wide")

# Rajasthan Royals-inspired visual system; no official team marks are used.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f6f8; color: #172033; }
.block-container { padding-top: 1.5rem; max-width: 1440px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; }
.hero { background: linear-gradient(120deg,#14263d 0%,#253f5b 68%,#b88a2d 160%); padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 20px; }
.hero-kicker { color: #e8bd62; text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 700; }
.hero h1 { color: white; font-size: 34px; margin: 6px 0 4px 0; }
.hero p { color: #dce5ee; margin: 0; font-size: 14px; }
.metric-card { background: white; border: 1px solid #e6e9ee; border-radius: 14px; padding: 15px 17px; min-height: 90px; }
.metric-label { color:#697586; font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
.metric-value { color:#14263d; font-family:'Space Grotesk'; font-size:25px; font-weight:700; margin-top:4px; }
.metric-note { color:#8a94a3; font-size:11px; margin-top:1px; }
.section-kicker { color:#b18428; font-size:11px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
div[data-testid="stDataFrame"] { border:1px solid #e6e9ee; border-radius:12px; overflow:hidden; }
div[data-testid="stTabs"] button { font-weight:700; }
.note { background:#fff8e9; border-left:4px solid #d5a43b; padding:12px 14px; border-radius:8px; color:#59451e; font-size:13px; }
.footer { color:#7c8795; font-size:11px; border-top:1px solid #e1e5ea; padding-top:12px; margin-top:28px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)

bat = load_csv("ratings_bat.csv")
bowl = load_csv("ratings_bowl.csv")
bat_top = load_csv("top10_bat_all_players.csv")
bowl_top = load_csv("top10_bowl_all_players.csv")
comparison = load_csv("rating_method_comparison.csv")
validation = load_csv("validation_summary.csv")
reliability = load_csv("validation_reliability.csv")
wp = load_csv("wp_model_metrics.csv")
coverage = load_csv("pressure_coverage.csv")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">SupeRR Selector · Season 2 · Round 1</div>
  <h1>Pressure Intelligence</h1>
  <p>Finding T20 players whose performance holds up when the match is on the line.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("## Explore the shortlist")
role = st.sidebar.radio("Player role", ["Batters", "Bowlers"], horizontal=False)
source = bat if role == "Batters" else bowl
top = bat_top if role == "Batters" else bowl_top
is_bat = role == "Batters"
search = st.sidebar.text_input("Find a player", placeholder="Start typing a name…")
min_balls = st.sidebar.slider("Minimum pressure balls", 0, int(source.n_p.max()), 0, 10)
filtered = source[(source.n_p >= min_balls) & source.name.str.contains(search, case=False, na=False)].copy()
filtered = filtered.sort_values("rank")

# KPI row
n_eligible = len(source)
mean_balls = int(source.n_p.mean())
role_cmp = comparison[comparison.role == ("bat" if is_bat else "bowl")].iloc[0]
role_rel = reliability[(reliability.role == ("bat" if is_bat else "bowl")) & reliability.rating.str.contains("configured", case=False)]
if role_rel.empty:
    role_rel = reliability[(reliability.role == ("bat" if is_bat else "bowl")) & reliability.rating.str.contains("25% pressure" if is_bat else "50% pressure", case=False)]
rel_row = role_rel.iloc[0]
cols = st.columns(4)
metrics = [
    ("Eligible players", f"{n_eligible}", "Met the configured ranking eligibility"),
    ("Top 10 pressure sample", f"{int(top.n_p.sum()):,}", "Pressure balls across the ranked ten"),
    ("Mean player exposure", f"{mean_balls:,}", "Pressure balls per eligible player"),
    ("100-split reliability", f"{rel_row.split_half_reliability:.3f}", f"{int(rel_row.splits)}/100 successful splits"),
]
for c,(label,value,note) in zip(cols,metrics):
    c.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.25, .75], gap="large")
with left:
    st.markdown('<div class="section-kicker">Ranked shortlist</div>', unsafe_allow_html=True)
    st.subheader(f"Top 10 {role.lower()}")
    display = top.copy().sort_values("rank")
    if is_bat:
        cols_show = ["rank", "name", "n_p", "matches_p", "sr_p", "dis_per100_p", "blend_lcb"]
        rename = {"rank":"Rank", "name":"Player", "n_p":"Pressure balls", "matches_p":"Matches", "sr_p":"Pressure SR", "dis_per100_p":"Dismissals / 100", "blend_lcb":"Rating LCB"}
    else:
        cols_show = ["rank", "name", "n_p", "matches_p", "eco_p", "wkts_per100_p", "blend_lcb"]
        rename = {"rank":"Rank", "name":"Player", "n_p":"Pressure balls", "matches_p":"Matches", "eco_p":"Pressure economy", "wkts_per100_p":"Wickets / 100", "blend_lcb":"Rating LCB"}
    st.dataframe(display[cols_show].rename(columns=rename).round(2), hide_index=True, use_container_width=True, height=390)
    st.download_button("Download Top 10 CSV", display[cols_show].rename(columns=rename).to_csv(index=False), file_name=f"superr_top10_{'batters' if is_bat else 'bowlers'}.csv", mime="text/csv")
with right:
    st.markdown('<div class="section-kicker">What drives the rank?</div>', unsafe_allow_html=True)
    st.subheader("Pressure performance")
    metric = "sr_p" if is_bat else "eco_p"
    metric_label = "Pressure strike rate" if is_bat else "Pressure economy"
    fig = px.bar(display.sort_values(metric, ascending=is_bat), x=metric, y="name", orientation="h", text=metric,
                 labels={metric:metric_label,"name":"Player"}, color_discrete_sequence=["#bd922f"])
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=390, margin=dict(l=5,r=30,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#435064", yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

st.divider()
st.markdown('<div class="section-kicker">Player explorer</div>', unsafe_allow_html=True)
st.subheader("Explore all eligible players")
if filtered.empty:
    st.info("No players match the current filters.")
else:
    if is_bat:
        viewcols = ["rank", "name", "n_p", "matches_p", "sr_p", "dis_per100_p", "sr_o", "blend_lcb", "blend_sd"]
        labels = {"rank":"Rank", "name":"Player", "n_p":"Pressure balls", "matches_p":"Matches", "sr_p":"Pressure SR", "dis_per100_p":"Dismissals / 100 (pressure)", "sr_o":"Ordinary SR", "blend_lcb":"Rating LCB", "blend_sd":"Bootstrap SD"}
    else:
        viewcols = ["rank", "name", "n_p", "matches_p", "eco_p", "wkts_per100_p", "eco_o", "blend_lcb", "blend_sd"]
        labels = {"rank":"Rank", "name":"Player", "n_p":"Pressure balls", "matches_p":"Matches", "eco_p":"Pressure economy", "wkts_per100_p":"Wickets / 100 (pressure)", "eco_o":"Ordinary economy", "blend_lcb":"Rating LCB", "blend_sd":"Bootstrap SD"}
    st.dataframe(filtered[viewcols].rename(columns=labels).round(3), hide_index=True, use_container_width=True, height=430)

st.divider()
st.markdown('<div class="section-kicker">Model transparency</div>', unsafe_allow_html=True)
st.subheader("How the ranking is built")
method_cols = st.columns([1,1,1])
method_cols[0].markdown("**Pressure component**\n\nMeasures player credit in defined high-pressure deliveries, with shrinkage to reduce noisy small samples.")
method_cols[1].markdown("**Ordinary component**\n\nProvides a broader performance baseline. Components are standardized before blending.")
method_cols[2].markdown("**Confidence-aware rank**\n\nMatch-level bootstrap estimates uncertainty; players are ranked using the 20th-percentile lower confidence bound.")
weights = pd.DataFrame({"Role":["Batters","Bowlers"],"Pressure weight":[25,50],"Ordinary weight":[75,50]})
figw = px.bar(weights, x="Role", y=["Pressure weight","Ordinary weight"], barmode="stack", text_auto=True,
              color_discrete_sequence=["#bd922f","#233f5b"], labels={"value":"Weight (%)","variable":"Component"})
figw.update_layout(height=260, margin=dict(l=5,r=5,t=15,b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0,100], legend_title_text="")
st.plotly_chart(figw, use_container_width=True, config={"displayModeBar":False})

with st.expander("Pressure-event definitions"):
    st.markdown("- **First innings:** collapse pressure when at least four wickets have already fallen within the first 84 legal balls.\n- **Chases:** high-leverage deliveries in the top 25% of the chase-leverage distribution.\n\nThe rules are operational definitions for this analysis; alternate thresholds can change the player list.")
with st.expander("Model comparison: hierarchical vs legacy"):
    cmp = comparison.copy()
    cmp["role"] = cmp.role.map({"bat":"Batters","bowl":"Bowlers"})
    st.dataframe(cmp[["role","hierarchical_eligible_players","legacy_eligible_players","top10_overlap","spearman_rank_correlation"]].rename(columns={"role":"Role","hierarchical_eligible_players":"Eligible (new)","legacy_eligible_players":"Eligible (legacy)","top10_overlap":"Top-10 overlap","spearman_rank_correlation":"Spearman correlation"}).round(3), hide_index=True, use_container_width=True)
    st.caption("Top-10 overlap is membership overlap. Spearman correlation compares rankings across eligible players.")
with st.expander("Repeated split-half reliability"):
    relshow = reliability.copy()
    relshow["role"] = relshow.role.map({"bat":"Batters","bowl":"Bowlers"})
    st.dataframe(relshow.rename(columns={"role":"Role","rating":"Model","predicts_held_out_pressure":"Held-out pressure Spearman","split_half_reliability":"Split-half reliability","players_per_split":"Players per split","splits":"Successful splits"}).round(3), hide_index=True, use_container_width=True)
    st.caption("Matches are split into halves. Predictive validity measures agreement with held-out pressure performance; reliability measures agreement between ratings from both halves. This is repeated holdout evidence on the same dataset, not independent external validation.")
with st.expander("Win-probability model diagnostics"):
    st.dataframe(wp.rename(columns={"innings":"Innings","rows":"Rows","matches":"Matches","auc":"AUC","logloss":"Log loss","baseline_logloss":"Baseline log loss"}).round(3), hide_index=True, use_container_width=True)
    st.caption("First-innings AUC is approximately 0.696; chase AUC approximately 0.898. Diagnostics are for the fitted, match-cross-fitted historical dataset.")
with st.expander("Pressure-data coverage"):
    st.dataframe(coverage.rename(columns={"scope":"Scope","pressure_tier":"Delivery category","deliveries":"Delivery rows","share":"Share","matches":"Matches","scope_total_deliveries":"Scope total"}), hide_index=True, use_container_width=True)

st.markdown("""
<div class="note"><b>How to interpret this dashboard:</b> rankings are historical, model-based scouting signals—not guaranteed forecasts or automatic auction recommendations. Final decisions also require availability, role fit, fitness, overseas-slot balance, price and purse checks.</div>
<div class="footer">SupeRR Selector · Round 1 · Uses derived competition outputs only. Proprietary raw ball-by-ball data is not bundled or exposed in this dashboard. Unofficial analytical project; not affiliated with or endorsed by Rajasthan Royals.</div>
""", unsafe_allow_html=True)
