"""
Orchestrator — FinCast AI
Coordinates all 5 agents in the correct sequence.
Exposes a single .run() method that returns the full pipeline output.
Now supports interactive chat and command-line questions.
"""

import time
import sys
import os
import argparse
from datetime import datetime

class Orchestrator:
    """
    Pipeline:
      DataAgent → ForecastAgent + AnomalyAgent
                → NarrativeAgent → QAAgent (ready to serve)
    """

    def __init__(self):
        self.log       = []
        self.results   = {}
        self.timings   = {}
        self._log("🚀 FinCast AI Orchestrator initialised")

    def run(self) -> dict:
        self._log("=" * 50)
        self._log("  FinCast AI — Multi-Agent FP&A System")
        self._log("=" * 50)

        # ── Agent 1: Data ─────────────────────────────
        t0 = time.time()
        from agents.data_agent import DataAgent
        data_result = DataAgent().run()
        self.results["data"]   = data_result
        self.timings["data"]   = round(time.time() - t0, 2)
        self._log(f"✅ DataAgent done in {self.timings['data']}s")

        # ── Agent 2: Forecast ─────────────────────────
        t0 = time.time()
        from agents.forecast_agent import ForecastAgent
        fc_result = ForecastAgent().run(data_result["actuals"])
        self.results["forecast"] = fc_result
        self.timings["forecast"] = round(time.time() - t0, 2)
        self._log(f"✅ ForecastAgent done in {self.timings['forecast']}s")

        # ── Agent 3: Anomaly ──────────────────────────
        t0 = time.time()
        from agents.anomaly_agent import AnomalyAgent
        an_result = AnomalyAgent().run(data_result["merged"])
        self.results["anomaly"] = an_result
        self.timings["anomaly"] = round(time.time() - t0, 2)
        self._log(f"✅ AnomalyAgent done in {self.timings['anomaly']}s")

        # ── Agent 4: Narrative ────────────────────────
        t0 = time.time()
        from agents.narrative_agent import NarrativeAgent
        fc_summary = fc_result.get("summary", {})
        nr_result  = NarrativeAgent().run(
            data_result["merged"],
            an_result["anomalies"],
            fc_summary,
        )
        self.results["narrative"] = nr_result
        self.timings["narrative"] = round(time.time() - t0, 2)
        self._log(f"✅ NarrativeAgent done in {self.timings['narrative']}s")

        # ── Agent 5: Q&A ──────────────────────────────
        from agents.qa_agent import QAAgent
        qa_agent = QAAgent(data_result["con"])
        self.results["qa"] = qa_agent
        self._log("✅ QAAgent ready")

        # ── Pipeline summary ──────────────────────────
        total = sum(self.timings.values())
        self._log(f"\n🏁 Pipeline complete in {total:.1f}s")
        self._log(f"   Anomalies found:  {an_result['summary'].get('total_anomalies', 0)}")
        self._log(f"   Forecast rows:    {len(fc_result.get('forecasts', []))}")
        self._log(f"   Narratives ready: {len(nr_result['narratives'])}")

        return {
            "data":      data_result,
            "forecast":  fc_result,
            "anomaly":   an_result,
            "narrative": nr_result,
            "qa":        qa_agent,
            "timings":   self.timings,
            "log":       self.log,
        }

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        print(entry)


# ──────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    
    # 1. Setup argument parser for --question
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Ask a specific question immediately")
    args = parser.parse_args()

    # 2. Run the main pipeline
    orch = Orchestrator()
    out  = orch.run()

    # 3. Print the AI-generated Summary
    print("\n" + "="*50)
    print("📝 EXECUTIVE SUMMARY")
    print("="*50)
    print(out["narrative"]["narratives"]["executive_summary"])

    # 4. Handle Q&A Logic
    qa = out["qa"]
    
    if args.question:
        # If user passed a question via --question in terminal
        print(f"\n🔍 Querying: {args.question}")
        res = qa.ask(args.question)
        print(f"🤖 AI: {res['answer']}\n")
    else:
        # Enter Interactive Chat Mode
        print("\n" + "="*50)
        print("💬 FINCAST AI INTERACTIVE CHAT")
        print("Type 'exit' or 'quit' to stop.")
        print("="*50)
        
        while True:
            try:
                user_q = input("\n❓ Ask a question about your data: ").strip()
                if user_q.lower() in ['exit', 'quit', 'bye']:
                    print("Exiting chat. Goodbye!")
                    break
                if not user_q:
                    continue
                    
                res = qa.ask(user_q)
                print(f"\n🤖 AI: {res['answer']}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
