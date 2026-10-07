"""PlayerLoad-Risk dashboard (Streamlit).

    streamlit run dashboard/app.py            # http://localhost:8501

Presentation only – all numbers come from the API / RiskService.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.backend import BackendError, make_backend  # noqa: E402

st.set_page_config(page_title="PlayerLoad-Risk", page_icon="📈", layout="wide")

# ---- palette (validated reference palette; status colours always paired with a text label)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"LOW": "#0ca30c", "MODERATE": "#fab219", "HIGH": "#ec835a", "VERY_HIGH": "#d03b3b"}
SEQ_BLUE = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
GRID, MUTED = "#e1e0d9", "#898781"
CATEGORIES = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]

DISCLAIMER = ("**Statistical availability-risk estimates from workload patterns – not a medical diagnosis.** "
              "Estimates describe associations in historical data and are uncertain. All data shown is **synthetic**.")


def style(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0), font=dict(size=12))
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def badge(cat: str | None) -> str:
    if not cat:
        return "–"
    return (f"<span style='background:{STATUS[cat]};color:#0b0b0b;padding:2px 10px;border-radius:10px;"
            f"font-weight:600'>{cat.replace('_', ' ')}</span>")


@st.cache_resource(show_spinner="Connecting to the risk service…")
def backend():
    return make_backend()


@st.cache_data(ttl=300, show_spinner=False)
def cached(name: str, *args):
    return getattr(backend(), name)(*args)


def threshold_lines(fig, thresholds: dict, x0, x1):
    for cat, val in thresholds.items():
        fig.add_hline(y=val, line=dict(color=STATUS[cat], width=1, dash="dot"),
                      annotation_text=cat.replace("_", " "), annotation_position="top left",
                      annotation_font=dict(size=10, color=MUTED))


# =============================================================================================== pages
def page_squad(as_of: str, team: str | None):
    data = cached("squad", as_of, team)
    st.subheader(f"Squad overview · {data['date']}" + (f" · team {team}" if team else ""))
    c = st.columns(5)
    c[0].metric("Athletes", data["n_athletes"])
    c[1].metric("Available", data["n_available"])
    c[2].metric("Unavailable", data["n_unavailable"])
    c[3].metric("High / very high risk", data["n_high_risk"])
    c[4].metric("Average 7-day risk", f"{100 * (data['mean_risk'] or 0):.1f}%",
                help=f"Reference base rate {100 * data['base_rate']:.1f}%")

    left, right = st.columns([1, 2])
    with left:
        cats = data["categories"]
        fig = go.Figure(go.Bar(x=[cats[c] for c in CATEGORIES], y=[c.replace("_", " ") for c in CATEGORIES],
                               orientation="h", marker_color=[STATUS[c] for c in CATEGORIES],
                               text=[cats[c] for c in CATEGORIES], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x} athletes<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 260, "Available athletes by risk category"), use_container_width=True)
        th = data["thresholds"]
        st.caption("Category thresholds (multiples of the reference base rate): "
                   + " · ".join(f"{k.replace('_', ' ')} ≥ {100 * v:.1f}%" for k, v in th.items()))
    with right:
        tr = pd.DataFrame(data["trend"])
        fig = go.Figure(go.Scatter(x=tr["date"], y=tr["mean_risk"], mode="lines", line=dict(color=SERIES[0], width=2),
                                   name="mean risk", hovertemplate="%{y:.1%}<extra>mean risk</extra>"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 260, "Average predicted 7-day risk (last 8 weeks)"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=tr["date"], y=tr["player_load_7d"], mode="lines",
                                   line=dict(color=SERIES[1], width=2), name="7-day player load"))
        st.plotly_chart(style(fig, 240, "Squad mean 7-day player load (AU)"), use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=tr["date"], y=tr["unavailable"], marker_color=SERIES[6], name="unavailable"))
        st.plotly_chart(style(fig, 240, "Unavailable athletes per day"), use_container_width=True)

    st.markdown("##### Highest estimated risk today")
    top = pd.DataFrame(data["highest_risk"])
    if not top.empty:
        top["risk"] = (100 * top["risk_probability"]).round(1).astype(str) + "%"
        st.dataframe(top[["athlete_id", "team_id", "position", "risk", "risk_category", "acwr_total_load",
                          "sprint_distance_vs_baseline_pct"]], hide_index=True, use_container_width=True)


def page_profile(aid: str, as_of: str):
    try:
        prof = cached("athlete", aid)
    except BackendError as exc:
        st.error(str(exc))
        return
    st.subheader(f"Athlete {aid} · {prof['position']} · team {prof['team_id']} · age {prof['age']}")
    try:
        pred = cached("predict", aid, as_of)
    except BackendError as exc:
        pred = None
        st.warning(str(exc))
    if pred:
        c = st.columns(4)
        c[0].metric("7-day availability risk", f"{100 * pred['risk_probability']:.1f}%",
                    help=f"Window {pred['risk_window']['start']} → {pred['risk_window']['end']}")
        c[1].markdown(f"**Category**<br>{badge(pred['risk_category'])}", unsafe_allow_html=True)
        c[2].metric("Relative to base rate", f"{pred['relative_risk']:.1f}×")
        c[3].metric("Percentile", f"{pred['risk_percentile']:.0f}th")
        st.info(pred["summary"])
        cal = pred.get("calibration", {})
        if cal.get("interpretation"):
            st.caption(f"Calibration: {cal['interpretation']} Test ROC-AUC {cal.get('test_roc_auc')} · "
                       f"Brier {cal.get('test_brier')}.")
        for w in pred.get("warnings", []):
            st.warning(w)
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("##### ▲ Factors associated with higher risk")
            for f in pred["top_risk_factors"] or [{"explanation": "none above threshold", "impact": 0}]:
                st.markdown(f"- {f['explanation']}  \n  <small style='color:{MUTED}'>SHAP impact "
                            f"{f['impact']:+.3f}</small>", unsafe_allow_html=True)
        with f2:
            st.markdown("##### ▼ Factors associated with lower risk")
            for f in pred["protective_factors"] or [{"explanation": "none above threshold", "impact": 0}]:
                st.markdown(f"- {f['explanation']}  \n  <small style='color:{MUTED}'>SHAP impact "
                            f"{f['impact']:+.3f}</small>", unsafe_allow_html=True)

    start = str(pd.Timestamp(as_of) - pd.Timedelta(days=120))
    hist = pd.DataFrame(cached("risk_history", aid, start, as_of)["history"])
    rh = cached("risk_history", aid, start, as_of)
    fig = go.Figure(go.Scatter(x=hist["date"], y=hist["risk_probability"], mode="lines", connectgaps=False,
                               line=dict(color=SERIES[0], width=2), name="predicted risk",
                               hovertemplate="%{y:.1%}<extra>risk</extra>"))
    on = hist[hist["absence_onset"]]
    if not on.empty:
        fig.add_trace(go.Scatter(x=on["date"], y=[0] * len(on), mode="markers", name="absence onset",
                                 marker=dict(symbol="x", size=10, color=SERIES[7])))
    threshold_lines(fig, rh["thresholds"], hist["date"].min(), hist["date"].max())
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 300, "Risk history (gaps = athlete unavailable)"), use_container_width=True)

    wl = pd.DataFrame(cached("workload", aid, start, as_of)["days"])
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=wl["date"], y=wl["player_load"], name="daily player load", marker_color=SERIES[0]))
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["player_load_7d"] / 7, name="7-day mean", mode="lines",
                                 line=dict(color=SERIES[1], width=2)))
        st.plotly_chart(style(fig, 280, "Workload history (player load, AU)"), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["sprint_distance_7d"], name="sprint distance (7 d)", mode="lines",
                                 line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["sprint_distance_baseline_weekly"], name="personal baseline",
                                 mode="lines", line=dict(color=MUTED, width=2, dash="dash")))
        st.plotly_chart(style(fig, 280, "Sprint exposure vs personal baseline (m / week)"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for i, (col, name) in enumerate([("acwr_total_load", "ACWR (rolling)"), ("ewma_acwr_total_load", "ACWR (EWMA)"),
                                         ("acwr_sprint_load", "ACWR sprint")]):
            fig.add_trace(go.Scatter(x=wl["date"], y=wl[col], name=name, mode="lines",
                                     line=dict(color=SERIES[i], width=2)))
        st.plotly_chart(style(fig, 260, "Acute:chronic workload ratios"), use_container_width=True)
    with c4:
        fig = go.Figure(go.Bar(x=wl["date"], y=wl["rest_days_7d"], marker_color=SERIES[2], name="rest days (7 d)"))
        st.plotly_chart(style(fig, 260, "Recovery: rest days in the previous 7 days"), use_container_width=True)

    st.markdown("##### Current week vs personal baseline")
    base = pd.DataFrame(prof["baseline"]).T.reset_index().rename(columns={"index": "metric"})
    st.dataframe(base, hide_index=True, use_container_width=True)
    st.markdown("##### Absence history")
    hist_inj = pd.DataFrame(prof["injury_history"])
    st.dataframe(hist_inj if not hist_inj.empty else pd.DataFrame({"info": ["no absences recorded"]}),
                 hide_index=True, use_container_width=True)
    st.caption(f"Prior injuries recorded before the dataset: {prof['prior_injury_count_before_dataset']}")


def page_workload(aid: str, as_of: str):
    st.subheader(f"Workload monitor · {aid}")
    days = st.slider("Window (days)", 28, 365, 120, step=7)
    start = str(pd.Timestamp(as_of) - pd.Timedelta(days=days))
    wl = pd.DataFrame(cached("workload", aid, start, as_of)["days"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl["date"], y=wl["player_load_7d"], name="7-day load (acute)", mode="lines",
                             line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=wl["date"], y=wl["player_load_28d"] / 4, name="28-day load ÷ 4 (chronic, weekly)",
                             mode="lines", line=dict(color=SERIES[1], width=2)))
    st.plotly_chart(style(fig, 300, "Acute vs chronic player load (AU per week)"), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl["date"], y=wl["acwr_total_load"], name="ACWR rolling (coupled)", mode="lines",
                             line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=wl["date"], y=wl["ewma_acwr_total_load"], name="ACWR EWMA", mode="lines",
                             line=dict(color=SERIES[1], width=2)))
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor="#e1e0d9", opacity=0.35, line_width=0,
                  annotation_text="commonly cited 0.8–1.3 band (contested)", annotation_position="top left",
                  annotation_font=dict(size=10, color=MUTED))
    st.plotly_chart(style(fig, 280, "ACWR – shown for context; see README for its limitations"),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["sprint_distance_7d"], name="sprint (7 d)", mode="lines",
                                 line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["high_speed_distance_7d"], name="high-speed running (7 d)",
                                 mode="lines", line=dict(color=SERIES[1], width=2)))
        st.plotly_chart(style(fig, 280, "Sprint & high-speed running exposure (m)"), use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["accelerations_7d"], name="accelerations (7 d)", mode="lines",
                                 line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=wl["date"], y=wl["decelerations_7d"], name="decelerations (7 d)", mode="lines",
                                 line=dict(color=SERIES[1], width=2)))
        st.plotly_chart(style(fig, 280, "Accelerations & decelerations"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Scatter(x=wl["date"], y=wl["monotony_7d"], mode="lines", line=dict(color=SERIES[2], width=2),
                                   name="monotony"))
        st.plotly_chart(style(fig, 240, "Training monotony (Foster)"), use_container_width=True)
    with c4:
        fig = go.Figure(go.Bar(x=wl["date"], y=wl["total_distance"], marker_color=SERIES[0], name="daily distance"))
        st.plotly_chart(style(fig, 240, "Daily total distance (m)"), use_container_width=True)
    with st.expander("Data table"):
        st.dataframe(wl, hide_index=True, use_container_width=True)


def page_heatmap(as_of: str, team: str | None):
    st.subheader("Risk heatmap · athletes × dates")
    days = st.slider("Days", 14, 84, 28, step=7)
    start = str(pd.Timestamp(as_of) - pd.Timedelta(days=days - 1))
    hm = cached("heatmap", start, as_of, team)
    z = pd.DataFrame(hm["values"], index=hm["athletes"], columns=hm["dates"])
    order = z.iloc[:, -1].fillna(-1).sort_values(ascending=False).index
    z = z.loc[order]
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=SEQ_BLUE, zmin=0,
                               zmax=max(0.3, float(pd.DataFrame(z).max().max() or 0.3)),
                               colorbar=dict(title="risk", tickformat=".0%"), xgap=1, ygap=1,
                               hovertemplate="%{y} · %{x}<br>risk %{z:.1%}<extra></extra>"))
    fig.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(style(fig, max(400, 16 * len(z) + 80)), use_container_width=True)
    st.caption("Sorted by the latest day's risk. Empty cells = athlete unavailable (absent). Colour intensity is "
               "the calibrated 7-day probability.")


def page_simulator(aid: str, as_of: str):
    st.subheader(f"What-if simulator · {aid}")
    st.warning("**Model simulation, not medical advice or causal evidence.** It shows how the model's estimate "
               "responds to a hypothetical change in tomorrow's session.")
    c = st.columns(5)
    total = c[0].slider("Total load %", -50, 100, 0, 5)
    sprint = c[1].slider("Sprint load %", -100, 150, 20, 5)
    hsr = c[2].slider("High-speed running %", -100, 150, 0, 5)
    dur = c[3].slider("Duration %", -80, 80, 0, 5)
    rest = c[4].checkbox("Make it a recovery day")
    custom = {"name": "Custom scenario", "total_load_pct": total, "sprint_load_pct": sprint, "hsr_pct": hsr,
              "duration_pct": dur, "rest_day": rest}
    presets = [{"name": "+10% total load", "total_load_pct": 10}, {"name": "+20% sprint load", "sprint_load_pct": 20},
               {"name": "-20% sprint load", "sprint_load_pct": -20}, {"name": "+30% HSR", "hsr_pct": 30},
               {"name": "Reduced session (-30%)", "duration_pct": -30},
               {"name": "Additional recovery day", "rest_day": True}]
    try:
        res = backend().simulate(aid, as_of, [custom] + presets)
    except BackendError as exc:
        st.error(str(exc))
        return
    base = res["baseline"]
    m = st.columns(3)
    m[0].metric("Risk today (current data)", f"{100 * res['current_risk']['risk_probability']:.1f}%")
    m[1].metric(f"Planned session on {res['simulated_day']}", f"{100 * base['risk_probability']:.1f}%")
    cs = res["scenarios"][0]
    m[2].metric("Custom scenario", f"{100 * cs['risk_probability']:.1f}%",
                delta=f"{100 * cs['delta_probability']:+.1f} pp", delta_color="inverse")
    rows = [{"scenario": "Planned session (no change)", "risk": base["risk_probability"],
             "category": base["risk_category"]}] + [
        {"scenario": s["name"], "risk": s["risk_probability"], "category": s["risk_category"]}
        for s in res["scenarios"]]
    df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=df["risk"], y=df["scenario"], orientation="h",
                           marker_color=[STATUS[c] for c in df["category"]],
                           text=[f"{100 * r:.1f}% · {c.replace('_', ' ')}" for r, c in zip(df["risk"], df["category"])],
                           textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(df["risk"]) * 1.35])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 360, "Simulated 7-day risk by scenario"), use_container_width=True)
    with st.expander("Planned session used as the baseline"):
        st.json(res["planned_session"])
    with st.expander("Feature changes in the custom scenario"):
        st.dataframe(pd.DataFrame(cs["feature_changes"]).T, use_container_width=True)
    st.caption(res["disclaimer"])


def page_model():
    st.subheader("Model & monitoring")
    info, met = cached("model_info"), cached("model_metrics")
    c = st.columns(4)
    c[0].metric("Model", f"{info['model_type']} v{info['model_version']}")
    c[1].metric("Test ROC-AUC", f"{met['test']['roc_auc']:.3f}")
    c[2].metric("Test PR-AUC", f"{met['test']['pr_auc']:.3f}", help=f"base rate {met['test']['base_rate']:.3f}")
    c[3].metric("Test Brier", f"{met['test']['brier']:.4f}")
    comp = pd.DataFrame({k: {**{f"val_{m}": v["validation"][m] for m in ("roc_auc", "pr_auc", "brier")},
                             **{f"test_{m}": v["test"][m] for m in ("roc_auc", "pr_auc", "brier")}}
                         for k, v in met["model_comparison"].items()}).T
    st.dataframe(comp.round(4), use_container_width=True)
    cal = pd.DataFrame(met["calibration_table_test"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["observed_rate"], mode="lines+markers", name="model",
                             line=dict(color=SERIES[0], width=2), marker=dict(size=8)))
    lim = float(max(cal["mean_predicted"].max(), cal["observed_rate"].max()))
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect calibration",
                             line=dict(color=MUTED, dash="dash", width=1)))
    st.plotly_chart(style(fig, 300, "Calibration on the held-out test period"), use_container_width=True)
    try:
        d = cached("drift")
        st.markdown(f"##### Drift report · overall status **{d['summary']['overall_status'].upper()}** · window "
                    f"{d['current']['start']} → {d['current']['end']}")
        st.caption(d["summary"]["recommendation"])
        st.dataframe(pd.DataFrame(d["feature_drift"])[["feature", "psi", "status", "reference_mean", "current_mean",
                                                       "mean_change_pct"]], hide_index=True, use_container_width=True)
    except BackendError as exc:
        st.info(f"Drift report unavailable: {exc}")
    with st.expander("Model lineage"):
        st.json(info)


# =============================================================================================== layout
def main():
    st.title("PlayerLoad-Risk")
    st.caption("Explainable injury & availability risk modelling from GPS-style workload data")
    st.markdown(DISCLAIMER)
    try:
        h = cached("health")
        athletes = pd.DataFrame(cached("athletes"))
    except BackendError as exc:
        st.error(f"Cannot reach the risk service: {exc}")
        st.stop()
    latest = date.fromisoformat(h["data_range"][1])
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Section", ["Squad overview", "Athlete profile", "Workload monitor", "Risk heatmap",
                                    "What-if simulator", "Model & monitoring"])
        as_of = st.date_input("Prediction date (end of day)", latest, min_value=latest - timedelta(days=700),
                              max_value=latest)
        teams = ["All"] + sorted(athletes["team_id"].unique())
        team = st.selectbox("Team", teams)
        team = None if team == "All" else team
        pool = athletes if team is None else athletes[athletes["team_id"] == team]
        # default: the available athlete with the highest current estimate (most informative demo)
        pool = pool.sort_values(["available", "latest_risk_probability"], ascending=[False, False])
        labels = {r.athlete_id: f"{r.athlete_id} · {r.position} · {r.team_id}"
                  + ("" if r.available else " · unavailable") for r in pool.itertuples()}
        aid = st.selectbox("Athlete", list(labels), format_func=labels.get)
        st.caption(f"Backend: {backend().name} · model {h['model_version']} (build {h['build_id']}) · data up to "
                   f"{h['data_range'][1]}")
    as_of = str(as_of)
    if page == "Squad overview":
        page_squad(as_of, team)
    elif page == "Athlete profile":
        page_profile(aid, as_of)
    elif page == "Workload monitor":
        page_workload(aid, as_of)
    elif page == "Risk heatmap":
        page_heatmap(as_of, team)
    elif page == "What-if simulator":
        page_simulator(aid, as_of)
    else:
        page_model()


main()
