# 💹 FinCast AI — Autonomous FP&A Multi-Agent System

> *An end-to-end AI system that forecasts, explains, detects anomalies, and answers questions about your financials — fully automated. Built with 100% free tools.*

---

## 🎯 What It Does

FinCast AI is a **production-grade multi-agent system** purpose-built for FP&A teams. It replaces hours of manual analysis with an autonomous pipeline that runs in seconds.

| Capability | How |
|---|---|
| ✅ **Saves Time** | Auto-pulls data, forecasts, writes commentary — zero manual work |
| ✅ **Catches Errors** | Flags budget overruns, GL spikes, duplicate patterns |
| ✅ **Better Insights** | 6-month forward forecast with confidence intervals |
| ✅ **Smart Q&A** | Chat with your data: *"Why did OpEx spike in March?"* |

---

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────┐
│              ORCHESTRATOR                        │
└──────┬──────────┬──────────┬──────────┬─────────┘
       ▼          ▼          ▼          ▼
  [Agent 1]  [Agent 2]  [Agent 3]  [Agent 4]  [Agent 5]
  Data Prep  Forecast   Anomaly    Narrative   Q&A Chat
  DuckDB     H-W / Lin  Z-score    Groq/Tmpl   SQL + LLM
```

### Agent Details

| # | Agent | Technology | Output |
|---|---|---|---|
| 1 | **Data Agent** | Pandas, DuckDB | Cleaned & merged FP&A dataset |
| 2 | **Forecast Agent** | Holt-Winters (statsmodels) | 6-month forecast + CI |
| 3 | **Anomaly Agent** | Z-Score, IQR, MoM analysis | Ranked anomaly list |
| 4 | **Narrative Agent** | Groq LLaMA-3 / Templates | Human-readable commentary |
| 5 | **Q&A Agent** | DuckDB + NL patterns | Natural language answers |

---

## 🛠️ 100% Free Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data Layer | DuckDB (in-memory SQL) |
| Forecasting | statsmodels Holt-Winters |
| Anomaly Detection | scikit-learn + pandas |
| LLM (optional) | Groq API — **free tier** (LLaMA 3) |
| Dashboard | Streamlit + Plotly |
| Data Format | CSV / can connect to any SQL DB |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/fincast-ai.git
cd fincast-ai
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python data/generate_data.py
```

### 3. Run the Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. (Optional) Enable LLM Narratives — Free
```bash
# Get free API key at: https://console.groq.com
export GROQ_API_KEY=your_key_here
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

| Page | What You See |
|---|---|
| 🏠 Executive Dashboard | KPIs, Actual vs Budget, Spend Trends |
| 📈 Forecasting | 6-month forecast chart with confidence intervals |
| 🚨 Anomaly Radar | Severity-ranked anomaly table with filters |
| 📝 AI Narratives | Auto-generated variance commentary per department |
| 💬 Chat with Data | Natural language Q&A over your financials |
| ⚙️ Agent Logs | Full pipeline execution trace |

---

## 🔌 Connect Your Real Data

Replace CSV files in `/data/` with your actual data exports:
- **actuals.csv** — Monthly actual spend by dept + category
- **budget.csv** — Monthly budget by dept + category
- **headcount.csv** — Monthly headcount by dept

Or connect directly to Azure SQL / PostgreSQL by modifying `DataAgent`.

---

## 📁 Project Structure

```
fincast-ai/
├── data/
│   ├── generate_data.py      ← Synthetic data generator
│   ├── actuals.csv
│   ├── budget.csv
│   └── headcount.csv
├── agents/
│   ├── data_agent.py         ← Agent 1: Load & clean
│   ├── forecast_agent.py     ← Agent 2: Holt-Winters
│   ├── anomaly_agent.py      ← Agent 3: Z-score + MoM
│   ├── narrative_agent.py    ← Agent 4: LLM commentary
│   └── qa_agent.py           ← Agent 5: NL → SQL → Answer
├── dashboard/
│   └── app.py                ← Streamlit UI
├── orchestrator.py           ← Pipeline controller
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Built By

*Thanneru Akhil*.

*Built with ❤️ to solve real problems I face every day at work.*

---

## 🗺️ Roadmap

- [ ] Azure SQL / PostgreSQL connector
- [ ] Power BI embedded reports
- [ ] Email digest scheduler (Apache Airflow)
- [ ] AP/AR reconciliation agent
- [ ] HR headcount cost attribution agent
- [ ] Multi-currency support

---

## ⭐ If this helped you, give it a star!
