"""
Agent 1 — Data Agent
Loads, cleans, merges and validates all FP&A data.
Returns a unified SQLite in-memory database for downstream agents.
"""
 
import os
import sqlite3
import pandas as pd
from datetime import datetime
 
 
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
 
 
class FakeCon:
    """SQLite-backed connection that mimics DuckDB's fetchdf() API."""
    def __init__(self, con):
        self._con = con
        self._tables = {}
 
    def register(self, name, df):
        self._tables[name] = df
        df.to_sql(name, self._con, if_exists="replace", index=False)
 
    def execute(self, sql):
        return _SQLiteResult(self._con, sql)
 
 
class _SQLiteResult:
    def __init__(self, con, sql):
        self._con = con
        self._sql = sql
 
    def fetchdf(self):
        return pd.read_sql_query(self._sql, self._con)
 
 
class DataAgent:
    """Loads raw CSVs → validates → registers in SQLite."""
 
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        # ✅ Fix: check_same_thread=False allows SQLite to be used across Streamlit threads
        self.con = FakeCon(sqlite3.connect(":memory:", check_same_thread=False))
        self.log = []
 
    # ──────────────────────────────────────────────────
    def run(self) -> dict:
        self._log("🟢 DataAgent starting …")
        actuals   = self._load("actuals.csv")
        budget    = self._load("budget.csv")
        forecast  = self._load("forecast_seed.csv")
        headcount = self._load("headcount.csv")
 
        actuals   = self._clean_actuals(actuals)
        budget    = self._clean_budget(budget)
        forecast  = self._clean_forecast(forecast)
        headcount = self._clean_headcount(headcount)
 
        merged = self._merge(actuals, budget)
 
        self._register("actuals",   actuals)
        self._register("budget",    budget)
        self._register("forecast",  forecast)
        self._register("headcount", headcount)
        self._register("merged",    merged)
 
        # Log registered columns for debugging
        self._log(f"   headcount columns: {list(headcount.columns)}")
        self._log(f"   merged columns: {list(merged.columns)}")
 
        stats = self._stats(actuals, budget, merged)
        self._log(f"✅ DataAgent complete — {len(merged):,} merged rows")
 
        return {
            "con":       self.con,
            "actuals":   actuals,
            "budget":    budget,
            "forecast":  forecast,
            "headcount": headcount,
            "merged":    merged,
            "stats":     stats,
            "log":       self.log,
        }
 
    # ──────────────────────────────────────────────────
    def _load(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, parse_dates=["date"])
        self._log(f"   Loaded {filename}: {len(df):,} rows, cols={list(df.columns)}")
        return df
 
    def _clean_actuals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["actual"])
        df = df[df["actual"] >= 0]
        df["actual"] = df["actual"].round(2)
        return df
 
    def _clean_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["budget"])
        df = df[df["budget"] >= 0]
        df["budget"] = df["budget"].round(2)
        return df
 
    def _clean_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["forecast"])
        df["forecast"] = df["forecast"].round(2)
        return df
 
    def _clean_headcount(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle both possible column names: 'headcount' or 'employee_id'
        if "headcount" in df.columns:
            df = df.dropna(subset=["headcount"])
            df["headcount"] = pd.to_numeric(df["headcount"], errors="coerce").fillna(0).astype(int)
        elif "employee_id" in df.columns:
            # If CSV has employee_id rows (one row per employee), aggregate to headcount
            df = df.dropna(subset=["employee_id"])
            df["headcount"] = 1  # each row = 1 employee
        else:
            self._log("⚠️  headcount.csv has no 'headcount' or 'employee_id' column — check your CSV!")
        return df
 
    def _merge(self, actuals: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
        keys = ["date", "year", "month", "quarter", "department", "cost_center", "category"]
        # Only merge on keys that exist in both dataframes
        valid_keys = [k for k in keys if k in actuals.columns and k in budget.columns]
        merged = pd.merge(actuals, budget[valid_keys + ["budget"]], on=valid_keys, how="left")
        merged["variance"]     = merged["actual"] - merged["budget"]
        merged["variance_pct"] = (merged["variance"] / merged["budget"].replace(0, 1)) * 100
        return merged
 
    def _register(self, name: str, df: pd.DataFrame):
        self.con.register(name, df)
 
    def _stats(self, actuals, budget, merged) -> dict:
        total_actual = actuals["actual"].sum()
        total_budget = budget["budget"].sum()
        over_budget  = merged[merged["variance"] > 0]
        return {
            "total_actual":    round(total_actual, 2),
            "total_budget":    round(total_budget, 2),
            "total_variance":  round(total_actual - total_budget, 2),
            "over_budget_rows": len(over_budget),
            "date_range":      f"{actuals['date'].min().date()} → {actuals['date'].max().date()}",
            "departments":     sorted(actuals["department"].unique().tolist()),
            "categories":      sorted(actuals["category"].unique().tolist()),
        }
 
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        print(msg)
 
 
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DataAgent()
    result = agent.run()
    print("\n📊 Stats:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")