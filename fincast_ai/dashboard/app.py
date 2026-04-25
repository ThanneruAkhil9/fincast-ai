"""
FinCast AI - Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
import streamlit as st
 
# ── Inject Streamlit secrets into environment (MUST be before agent imports) ──
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
st.set_page_config(
    page_title="FinCast AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
 
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }
 
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
 
.fincast-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.fincast-header h1 { margin:0; font-size: 2rem; font-weight: 700; }
.fincast-header p  { margin: 0.25rem 0 0; font-size: 0.9rem; opacity: 0.7; }
 
.badge-CRITICAL { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-HIGH     { background:#fff7ed; color:#ea580c; border:1px solid #fed7aa; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-MEDIUM   { background:#fefce8; color:#ca8a04; border:1px solid #fde68a; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-LOW      { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
 
.chat-user { background:#eff6ff; border-radius:12px 12px 2px 12px; padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.9rem; }
.chat-ai   { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px 12px 12px 2px; padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.9rem; }
 
.section-title { font-size:1.1rem; font-weight:600; color:#0f172a; margin:1.5rem 0 0.75rem; }
 
.agent-log { background:#0f172a; color:#7dd3fc; font-family:'DM Mono',monospace; font-size:0.72rem; padding:1rem; border-radius:8px; max-height:200px; overflow-y:auto; }
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource(show_spinner=False)
def run_pipeline():
    from orchestrator import Orchestrator
    return Orchestrator().run()
 
 
with st.sidebar:
    st.markdown("## 💹 FinCast AI")
    st.markdown("*Autonomous FP&A Multi-Agent System*")
    st.divider()
 
    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "📈 Forecasting",
            "🚨 Anomaly Radar",
            "📝 AI Narratives",
            "💬 Chat with Data",
            "⚙️ Agent Logs",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Stack:** Python · SQLite · Plotly · Streamlit")
    st.caption("**Agents:** Data · Forecast · Anomaly · Narrative · Q&A")
    st.caption("**LLM:** Groq (free) / Template fallback")
    st.divider()
    if st.button("🔄 Clear Cache & Restart"):
        st.cache_resource.clear()
        st.rerun()
 
 
with st.spinner("🤖 Running FinCast AI pipeline..."):
    pipeline = run_pipeline()
 
data       = pipeline["data"]
forecast   = pipeline["forecast"]
anomaly    = pipeline["anomaly"]
narrative  = pipeline["narrative"]
qa_agent   = pipeline["qa"]
 
merged     = data["merged"]
actuals    = data["actuals"]
fc_df      = forecast.get("forecasts", pd.DataFrame())
an_df      = anomaly.get("anomalies", pd.DataFrame())
narratives = narrative["narratives"]
 
COLORS = {
    "primary": "#4f46e5",
    "success": "#16a34a",
    "danger":  "#dc2626",
    "warning": "#d97706",
    "info":    "#0284c7",
    "muted":   "#64748b",
}
SEV_COLORS = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#d97706", "LOW": "#16a34a"}
 
 
# ── PAGE 1: EXECUTIVE DASHBOARD ──────────────────────────────
if page == "🏠 Executive Dashboard":
    st.markdown("""
    <div class="fincast-header">
        <h1>💹 FinCast AI — Executive Dashboard</h1>
        <p>Autonomous FP&A Intelligence · Powered by Multi-Agent AI</p>
    </div>
    """, unsafe_allow_html=True)
 
    stats = data["stats"]
    col1, col2, col3, col4, col5 = st.columns(5)
    total_var_pct = (stats["total_variance"] / stats["total_budget"]) * 100 if stats["total_budget"] else 0
 
    col1.metric("Total Actual Spend", f"${stats['total_actual']/1e6:.1f}M")
    col2.metric("Total Budget",       f"${stats['total_budget']/1e6:.1f}M")
    col3.metric("Total Variance",     f"${stats['total_variance']/1e6:.1f}M",
                f"{total_var_pct:.1f}%", delta_color="inverse")
    col4.metric("Anomalies Detected", anomaly["summary"].get("total_anomalies", 0),
                f"{anomaly['summary'].get('critical_count', 0)} critical", delta_color="inverse")
    col5.metric("6-Month Forecast",   f"${forecast.get('summary', {}).get('total_forecast_6m', 0)/1e6:.1f}M")
 
    st.markdown("<div class='section-title'>Actual vs Budget by Department</div>", unsafe_allow_html=True)
 
    dept_summary = (
        merged.groupby("department")
        .agg(actual=("actual", "sum"), budget=("budget", "sum"))
        .reset_index()
    )
    dept_summary["variance"] = dept_summary["actual"] - dept_summary["budget"]
 
    fig = go.Figure()
    fig.add_bar(x=dept_summary["department"], y=dept_summary["budget"],
                name="Budget", marker_color="#e2e8f0")
    fig.add_bar(x=dept_summary["department"], y=dept_summary["actual"],
                name="Actual", marker_color=COLORS["primary"])
    fig.update_layout(
        barmode="group", height=350,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_tickprefix="$", yaxis_tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True)
 
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown("<div class='section-title'>Spend by Category</div>", unsafe_allow_html=True)
        cat_sum = actuals.groupby("category")["actual"].sum().sort_values(ascending=False)
        fig2 = px.pie(
            values=cat_sum.values, names=cat_sum.index,
            hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        fig2.update_traces(textinfo="percent+label", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    with col_r:
        st.markdown("<div class='section-title'>Monthly Spend Trend</div>", unsafe_allow_html=True)
        monthly = actuals.groupby(["year", "month"])["actual"].sum().reset_index()
        monthly["period"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
        fig3 = px.area(monthly, x="period", y="actual",
                       color_discrete_sequence=[COLORS["primary"]])
        fig3.update_layout(
            height=300, margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_tickprefix="$", yaxis_tickformat=".2s",
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig3, use_container_width=True)
 
    if not an_df.empty:
        st.markdown("<div class='section-title'>🚨 Top Anomalies</div>", unsafe_allow_html=True)
        top_an = an_df[an_df["severity"].isin(["CRITICAL", "HIGH"])].head(5)
        for _, row in top_an.iterrows():
            sev = row["severity"]
            col_s, col_d = st.columns([1, 8])
            col_s.markdown(f"<span class='badge-{sev}'>{sev}</span>", unsafe_allow_html=True)
            col_d.caption(row["description"])
 
 
# ── PAGE 2: FORECASTING ───────────────────────────────────────
elif page == "📈 Forecasting":
    st.markdown("## 📈 6-Month Forecast")
 
    if fc_df.empty:
        st.warning("No forecast data available.")
        st.stop()
 
    col_f1, col_f2 = st.columns(2)
    dept_sel = col_f1.selectbox("Department", ["All"] + sorted(fc_df["department"].unique().tolist()))
    cat_sel  = col_f2.selectbox("Category",   ["All"] + sorted(fc_df["category"].unique().tolist()))
 
    act_f = actuals.copy()
    fc_f  = fc_df.copy()
    if dept_sel != "All":
        act_f = act_f[act_f["department"] == dept_sel]
        fc_f  = fc_f[fc_f["department"]  == dept_sel]
    if cat_sel != "All":
        act_f = act_f[act_f["category"] == cat_sel]
        fc_f  = fc_f[fc_f["category"]  == cat_sel]
 
    act_monthly = act_f.groupby(["year", "month"])["actual"].sum().reset_index()
    act_monthly["period"] = pd.to_datetime(act_monthly[["year", "month"]].assign(day=1))
    fc_monthly = fc_f.groupby(["year", "month"]).agg(
        forecast=("forecast", "sum"),
        lower=("lower_ci", "sum"),
        upper=("upper_ci", "sum"),
    ).reset_index()
    fc_monthly["period"] = pd.to_datetime(fc_monthly[["year", "month"]].assign(day=1))
 
    fig = go.Figure()
    fig.add_scatter(x=act_monthly["period"], y=act_monthly["actual"],
                    name="Actuals", line=dict(color=COLORS["primary"], width=2))
    fig.add_scatter(x=fc_monthly["period"], y=fc_monthly["forecast"],
                    name="Forecast", line=dict(color=COLORS["warning"], width=2, dash="dot"))
    fig.add_scatter(
        x=pd.concat([fc_monthly["period"], fc_monthly["period"][::-1]]),
        y=pd.concat([fc_monthly["upper"], fc_monthly["lower"][::-1]]),
        fill="toself", fillcolor="rgba(217,119,6,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval",
    )
    fig.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        yaxis_tickprefix="$", yaxis_tickformat=".2s",
        xaxis_title="", yaxis_title="",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("<div class='section-title'>Forecast Detail by Department</div>", unsafe_allow_html=True)
    fc_dept = fc_df.groupby("department").agg(
        total_forecast=("forecast", "sum"),
        avg_monthly=("forecast", "mean"),
    ).round(0).reset_index().sort_values("total_forecast", ascending=False)
    fc_dept["total_forecast"] = fc_dept["total_forecast"].apply(lambda x: f"${x:,.0f}")
    fc_dept["avg_monthly"]    = fc_dept["avg_monthly"].apply(lambda x: f"${x:,.0f}")
    fc_dept.columns = ["Department", "6-Month Total", "Avg Monthly"]
    st.dataframe(fc_dept, use_container_width=True, hide_index=True)
 
 
# ── PAGE 3: ANOMALY RADAR ─────────────────────────────────────
elif page == "🚨 Anomaly Radar":
    st.markdown("## 🚨 Anomaly Radar")
 
    if an_df.empty:
        st.info("No anomalies detected.")
        st.stop()
 
    summ = anomaly["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Anomalies", summ.get("total_anomalies", 0))
    c2.metric("Critical",        summ.get("critical_count", 0), delta_color="inverse")
    c3.metric("High",            summ.get("high_count", 0), delta_color="inverse")
    c4.metric("Methods Used",    len(summ.get("by_method", {})))
 
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>By Severity</div>", unsafe_allow_html=True)
        sev_counts = an_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig = px.bar(sev_counts, x="severity", y="count",
                     color="severity", color_discrete_map=SEV_COLORS, text="count")
        fig.update_layout(height=260, showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
 
    with col_r:
        st.markdown("<div class='section-title'>By Department</div>", unsafe_allow_html=True)
        dept_counts = an_df["department"].value_counts().head(7).reset_index()
        dept_counts.columns = ["department", "count"]
        fig2 = px.bar(dept_counts, x="count", y="department",
                      orientation="h", color_discrete_sequence=[COLORS["primary"]])
        fig2.update_layout(height=260, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("<div class='section-title'>Anomaly Table</div>", unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    sev_filter    = col_f1.multiselect("Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                        default=["CRITICAL", "HIGH"])
    dept_filter   = col_f2.selectbox("Department", ["All"] + sorted(an_df["department"].unique().tolist()))
    method_filter = col_f3.selectbox("Method", ["All"] + sorted(an_df["method"].unique().tolist()))
 
    filtered = an_df.copy()
    if sev_filter:             filtered = filtered[filtered["severity"].isin(sev_filter)]
    if dept_filter != "All":   filtered = filtered[filtered["department"] == dept_filter]
    if method_filter != "All": filtered = filtered[filtered["method"] == method_filter]
 
    display = filtered[["date", "department", "category", "severity", "method", "variance_pct", "description"]].copy()
    display["variance_pct"] = display["variance_pct"].apply(lambda x: f"{x:.1f}%")
    display.columns = ["Date", "Department", "Category", "Severity", "Method", "Variance %", "Description"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)
 
 
# ── PAGE 4: AI NARRATIVES ─────────────────────────────────────
elif page == "📝 AI Narratives":
    st.markdown("## 📝 AI-Generated FP&A Narratives")
 
    tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Department Commentary", "Anomaly Alerts", "Forecast Narrative"])
 
    with tab1:
        st.markdown("### Executive Summary")
        exec_text = narratives.get("executive_summary", "")
        for para in exec_text.split("\n\n"):
            if para.strip():
                st.markdown(para)
 
    with tab2:
        st.markdown("### Department Commentary")
        dept_comm = narratives.get("dept_commentary", {})
        for dept, text in dept_comm.items():
            with st.expander(f"📁 {dept}"):
                st.write(text)
 
    with tab3:
        st.markdown("### Anomaly Alerts")
        alerts = narratives.get("anomaly_commentary", [])
        for alert in alerts:
            st.warning(alert)
 
    with tab4:
        st.markdown("### Forecast Narrative")
        fc_text = narratives.get("forecast_narrative", "")
        st.info(fc_text)
        st.caption("💡 Set GROQ_API_KEY in your environment to enable LLM-powered narratives.")
 
 
# ── PAGE 5: CHAT WITH DATA ────────────────────────────────────
elif page == "💬 Chat with Data":
    st.markdown("## 💬 Chat with Your FP&A Data")
    st.caption("Ask anything about your financials — powered by AI + SQLite")
 
    st.markdown("**Quick questions:**")
    suggestions = [
        "Which department has the highest spend?",
        "Which departments are over budget?",
        "What is the total salary cost?",
        "Show me the 6-month forecast",
        "What is the headcount by department?",
        "Show me the marketing spend trend",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"sugg_{i}", use_container_width=True):
            st.session_state.setdefault("chat_history", [])
            st.session_state["pending_question"] = s
 
    st.divider()
 
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
 
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("data") is not None and not msg["data"].empty:
                with st.expander("📊 View data table"):
                    st.dataframe(msg["data"].head(15), use_container_width=True)
            if msg.get("sql"):
                with st.expander("🔍 View SQL"):
                    st.code(msg["sql"], language="sql")
 
    user_q = st.chat_input("Ask about your FP&A data...")
 
    if "pending_question" in st.session_state:
        user_q = st.session_state.pop("pending_question")
 
    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.spinner("🤖 Analysing..."):
            result = qa_agent.ask(user_q)
        st.session_state["chat_history"].append({
            "role":    "assistant",
            "content": result["answer"],
            "data":    result.get("data"),
            "sql":     result.get("sql"),
        })
        st.rerun()
 
 
# ── PAGE 6: AGENT LOGS ────────────────────────────────────────
elif page == "⚙️ Agent Logs":
    st.markdown("## ⚙️ Agent Execution Logs")
 
    timings = pipeline.get("timings", {})
    if timings:
        st.markdown("**Pipeline Timings:**")
        c1, c2, c3, c4 = st.columns(4)
        for i, (agent, t) in enumerate(timings.items()):
            [c1, c2, c3, c4][i % 4].metric(f"{agent.title()} Agent", f"{t}s")
 
    st.divider()
    all_logs = []
    for src in ["data", "forecast", "anomaly", "narrative"]:
        logs = pipeline.get(src, {}).get("log", [])
        all_logs += logs
 
    log_html = "<div class='agent-log'>" + "<br>".join(all_logs) + "</div>"
    st.markdown(log_html, unsafe_allow_html=True)
