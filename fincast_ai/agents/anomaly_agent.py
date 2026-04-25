"""
Agent 3 — Anomaly Detection Agent
Detects budget overruns, spending spikes, and unusual patterns
using Z-score, IQR, and rolling-window methods.
100% free — pure scikit-learn + pandas.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List


@dataclass
class Anomaly:
    date:         str
    department:   str
    category:     str
    actual:       float
    budget:       float
    variance:     float
    variance_pct: float
    method:       str
    severity:     str   # LOW / MEDIUM / HIGH / CRITICAL
    description:  str


class AnomalyAgent:
    """
    Detects three types of anomalies in FP&A data:
      1. Budget variance spikes (> threshold %)
      2. Statistical outliers (Z-score on rolling 12-month window)
      3. Month-over-month jumps (> 50% change vs prior month)
    """

    VARIANCE_THRESHOLDS = {
        "LOW":      15,
        "MEDIUM":   25,
        "HIGH":     40,
        "CRITICAL": 60,
    }
    Z_THRESHOLD   = 2.5
    MOM_THRESHOLD = 0.50   # 50% month-over-month change

    def __init__(self):
        self.log = []

    # ──────────────────────────────────────────────────
    def run(self, merged: pd.DataFrame) -> dict:
        self._log("🟢 AnomalyAgent starting …")

        all_anomalies: List[Anomaly] = []
        all_anomalies += self._detect_variance(merged)
        all_anomalies += self._detect_zscore(merged)
        all_anomalies += self._detect_mom(merged)

        df = self._to_dataframe(all_anomalies)
        summary = self._summarize(df)

        self._log(f"✅ AnomalyAgent — {len(df)} anomalies found")
        return {"anomalies": df, "summary": summary, "log": self.log}

    # ──────────────────────────────────────────────────
    # Method 1 — Budget Variance
    # ──────────────────────────────────────────────────
    def _detect_variance(self, df: pd.DataFrame) -> List[Anomaly]:
        out = []
        mask = df["budget"].notna() & (df["budget"] > 0)
        sub  = df[mask].copy()
        sub["var_pct_abs"] = sub["variance_pct"].abs()

        for _, row in sub[sub["var_pct_abs"] > self.VARIANCE_THRESHOLDS["LOW"]].iterrows():
            sev = self._severity_from_pct(row["var_pct_abs"])
            direction = "over" if row["variance"] > 0 else "under"
            out.append(Anomaly(
                date=str(row["date"].date()),
                department=row["department"],
                category=row["category"],
                actual=row["actual"],
                budget=row["budget"],
                variance=round(row["variance"], 2),
                variance_pct=round(row["variance_pct"], 2),
                method="Budget Variance",
                severity=sev,
                description=(
                    f"{row['department']} / {row['category']} is "
                    f"{abs(row['variance_pct']):.1f}% {direction} budget "
                    f"(${abs(row['variance']):,.0f})."
                ),
            ))
        self._log(f"   Budget variance anomalies: {len(out)}")
        return out

    # ──────────────────────────────────────────────────
    # Method 2 — Z-Score (rolling 12-month)
    # ──────────────────────────────────────────────────
    def _detect_zscore(self, df: pd.DataFrame) -> List[Anomaly]:
        out = []
        for (dept, cat), grp in df.groupby(["department", "category"]):
            grp = grp.sort_values("date")
            if len(grp) < 6:
                continue
            roll_mean = grp["actual"].rolling(12, min_periods=4).mean()
            roll_std  = grp["actual"].rolling(12, min_periods=4).std()
            z = (grp["actual"] - roll_mean) / roll_std.replace(0, 1)

            for idx, (_, row) in enumerate(grp.iterrows()):
                if abs(z.iloc[idx]) >= self.Z_THRESHOLD:
                    direction = "spike" if z.iloc[idx] > 0 else "dip"
                    out.append(Anomaly(
                        date=str(row["date"].date()),
                        department=dept,
                        category=cat,
                        actual=row["actual"],
                        budget=row.get("budget", 0) or 0,
                        variance=row.get("variance", 0) or 0,
                        variance_pct=row.get("variance_pct", 0) or 0,
                        method="Z-Score",
                        severity="HIGH" if abs(z.iloc[idx]) > 3.5 else "MEDIUM",
                        description=(
                            f"Statistical {direction} detected in {dept}/{cat} "
                            f"(Z={z.iloc[idx]:.2f}). Value ${row['actual']:,.0f} "
                            f"is {abs(z.iloc[idx]):.1f}σ from rolling mean."
                        ),
                    ))
        self._log(f"   Z-score anomalies: {len(out)}")
        return out

    # ──────────────────────────────────────────────────
    # Method 3 — Month-over-Month jumps
    # ──────────────────────────────────────────────────
    def _detect_mom(self, df: pd.DataFrame) -> List[Anomaly]:
        out = []
        for (dept, cat), grp in df.groupby(["department", "category"]):
            grp = grp.sort_values("date").reset_index(drop=True)
            if len(grp) < 2:
                continue
            grp["mom_pct"] = grp["actual"].pct_change()
            spikes = grp[grp["mom_pct"].abs() > self.MOM_THRESHOLD].iloc[1:]  # skip first NaN row

            for _, row in spikes.iterrows():
                direction = "increased" if row["mom_pct"] > 0 else "decreased"
                out.append(Anomaly(
                    date=str(row["date"].date()),
                    department=dept,
                    category=cat,
                    actual=row["actual"],
                    budget=row.get("budget", 0) or 0,
                    variance=row.get("variance", 0) or 0,
                    variance_pct=row.get("variance_pct", 0) or 0,
                    method="MoM Change",
                    severity="HIGH" if abs(row["mom_pct"]) > 0.80 else "MEDIUM",
                    description=(
                        f"{dept}/{cat} {direction} by "
                        f"{abs(row['mom_pct']) * 100:.1f}% month-over-month "
                        f"to ${row['actual']:,.0f}."
                    ),
                ))
        self._log(f"   MoM anomalies: {len(out)}")
        return out

    # ──────────────────────────────────────────────────
    def _severity_from_pct(self, pct: float) -> str:
        if pct >= self.VARIANCE_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif pct >= self.VARIANCE_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif pct >= self.VARIANCE_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def _to_dataframe(self, anomalies: List[Anomaly]) -> pd.DataFrame:
        if not anomalies:
            return pd.DataFrame()
        rows = [a.__dict__ for a in anomalies]
        df = pd.DataFrame(rows)
        # de-duplicate
        df = df.drop_duplicates(subset=["date", "department", "category", "method"])
        df = df.sort_values(["severity", "variance_pct"], ascending=[False, False])
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        df["_sev"] = df["severity"].map(severity_order)
        df = df.sort_values("_sev").drop(columns="_sev")
        return df.reset_index(drop=True)

    def _summarize(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        return {
            "total_anomalies":   len(df),
            "by_severity":       df["severity"].value_counts().to_dict(),
            "by_method":         df["method"].value_counts().to_dict(),
            "top_departments":   df["department"].value_counts().head(3).to_dict(),
            "critical_count":    int((df["severity"] == "CRITICAL").sum()),
            "high_count":        int((df["severity"] == "HIGH").sum()),
        }

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        print(msg)


# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agents.data_agent import DataAgent

    data   = DataAgent().run()
    agent  = AnomalyAgent()
    result = agent.run(data["merged"])

    print("\n📊 Anomaly Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print("\nTop 5 anomalies:")
    print(result["anomalies"][["date","department","category","severity","description"]].head(5).to_string())
