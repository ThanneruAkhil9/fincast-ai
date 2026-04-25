# LinkedIn Post — FinCast AI

---

## POST OPTION 1 — Story-driven (Recommended)

I'm a Data Analyst supporting FP&A teams. Every month I watched finance spend **3 days** doing this manually:

→ Pull actuals from SQL
→ Compare to budget in Excel
→ Hunt for anomalies row by row
→ Write variance commentary department by department
→ Answer the same 20 questions on repeat

So I built an AI system that does all of it in **under 1 second**.

Introducing **FinCast AI** — a 5-agent autonomous FP&A system:

🤖 **Agent 1 — Data Agent**
Loads, validates and merges actuals + budget into a live SQL database automatically

📈 **Agent 2 — Forecast Agent**
Holt-Winters time-series model forecasts the next 6 months per department × category — with confidence intervals

🚨 **Agent 3 — Anomaly Agent**
3 detection methods (Z-score, statistical outliers, MoM change) flag every budget spike before your team notices

📝 **Agent 4 — Narrative Agent**
Writes professional variance commentary for every department in plain English — powered by LLaMA-3 (free)

💬 **Agent 5 — Q&A Agent**
Ask your financial data anything in plain English. "Which departments are over budget?" → instant answer

**Built entirely with free tools:**
Python · DuckDB · Pandas · Streamlit · Plotly · Groq API (free tier)

No paid APIs. No subscriptions. Just AI solving a real problem.

The result? Finance teams get a full FP&A briefing every morning — automatically.

Full code on GitHub 👇
[github link]

---

🔁 What manual FP&A task would YOU automate first?

#DataScience #FPA #FinanceAI #MultiAgent #Python #LLM #AI #Analytics #FinTech #Automation

---

## POST OPTION 2 — Technical (for DS audience)

Built a multi-agent FP&A system from scratch using only free tools. Here's the architecture:

**The Problem:**
FP&A teams spend 60-70% of their time on data prep, variance analysis, and writing commentary. This is pure automation opportunity.

**The Solution — 5 Autonomous Agents:**

```
Orchestrator
    │
    ├── Agent 1: Data Agent (Pandas + SQLite)
    │   └── Loads, cleans, merges actuals vs budget
    │
    ├── Agent 2: Forecast Agent (Pure NumPy Holt-Winters)  
    │   └── 6-month forecast per dept × category
    │
    ├── Agent 3: Anomaly Agent (Z-score + MoM detection)
    │   └── 654 anomalies detected across 1,512 data points
    │
    ├── Agent 4: Narrative Agent (LLaMA-3 via Groq — free)
    │   └── Auto-writes executive summary + dept commentary
    │
    └── Agent 5: Q&A Agent (NL → SQL → Answer)
        └── "Which dept is over budget?" → instant answer
```

**Results on synthetic dataset:**
- 1,512 data points processed
- 654 anomalies flagged (349 high severity)
- 6-month forecast: $47.7M projected spend
- Pipeline runs in **0.5 seconds**
- 0 paid APIs used

**Stack:** Python · Pandas · NumPy · Streamlit · Plotly · Groq (free)

GitHub link in comments 👇

What would you add as Agent 6?

#Python #DataScience #FPA #MultiAgent #LLM #MLOps #FinanceAI

---

## DEMO SCRIPT (for screen recording)

1. Open Streamlit dashboard (http://localhost:8501)
2. Show Executive Dashboard — KPIs load instantly
3. Navigate to Forecasting — show the 6-month chart with CI bands
4. Navigate to Anomaly Radar — filter by CRITICAL/HIGH
5. Navigate to AI Narratives — show the executive summary
6. Navigate to Chat — type: "Which department is over budget?"
7. Show the SQL it generated underneath
8. Type: "Show me the headcount by department"
9. Navigate to Agent Logs — show the pipeline ran in 0.5s

**Record at 1080p, keep under 90 seconds**
