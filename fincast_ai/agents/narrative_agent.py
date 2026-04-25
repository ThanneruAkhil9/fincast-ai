import os
import pandas as pd
from datetime import datetime
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

class NarrativeAgent:
    def __init__(self):
        self.log  = []
        # Read key at runtime (not module load time) so Streamlit secrets are available
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.mode = "groq" if GROQ_API_KEY else "template"
        if self.mode == "groq":
            try:
                self.client = Groq(
                    api_key=GROQ_API_KEY,
                    default_headers={"User-Agent": "Mozilla/5.0"}
                )
            except Exception as e:
                self._log(f"⚠️ Failed to init Groq: {e}")
                self.mode = "template"
        self._log(f"🟢 NarrativeAgent starting (mode={self.mode}) …")

    def _log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log.append(msg)

    def _generate(self, prompt: str, fallback_template: str) -> str:
        if self.mode != "groq":
            return fallback_template
        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior FP&A Director. Write professional, multi-paragraph commentary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            self._log(f"⚠️ Groq error: {e}")
            return fallback_template

    def run(self, merged, anomalies, forecast_summary):
        narratives = {}
        narratives["executive_summary"] = self._executive_summary(merged, anomalies, forecast_summary)
        narratives["dept_commentary"] = {d: self._dept_commentary(merged, d) for d in sorted(merged["department"].unique())}
        narratives["anomaly_commentary"] = [self._generate(f"Summarize this anomaly professionally: {r['description']}", f"⚠️ {r['description']}") for _, r in anomalies.head(5).iterrows()]
        narratives["forecast_narrative"] = self._generate(f"Comment on the 6-month forecast of ${forecast_summary.get('total_forecast_6m', 0):,.0f}.", "Forecast analyzed.")
        self._log("✅ NarrativeAgent complete")
        return {"narratives": narratives, "log": self.log}

    def _executive_summary(self, merged, anomalies, forecast_summary):
        actual, budget = merged["actual"].sum(), merged["budget"].sum()
        var = actual - budget
        fc = forecast_summary.get("total_forecast_6m", 0)
        prompt = f"Write a professional 3-paragraph summary. Actual: ${actual:,.0f}, Budget: ${budget:,.0f}, Var: ${var:,.0f}, Forecast: ${fc:,.0f}."
        return self._generate(prompt, f"Performance: ${var:,.0f} variance vs budget.")

    def _dept_commentary(self, merged, dept):
        df = merged[merged["department"] == dept]
        actual, budget = df["actual"].sum(), df["budget"].sum()
        prompt = f"Write 2 sentences on {dept} department. Actual: ${actual:,.0f}, Budget: ${budget:,.0f}."
        return self._generate(prompt, f"{dept} variance: ${actual-budget:,.0f}")
