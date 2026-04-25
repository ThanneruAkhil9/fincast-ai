"""
Agent 2 — Forecast Agent
Uses Holt-Winters-style exponential smoothing implemented in pure NumPy.
No paid APIs. No external libraries beyond pandas + numpy.
"""

import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")


def _holt_winters(series: np.ndarray, periods: int, horizon: int):
    """Simple additive Holt-Winters in pure NumPy."""
    n = len(series)
    if n < periods * 2:
        # linear fallback
        x = np.arange(n)
        m, b = np.polyfit(x, series, 1)
        return np.array([m * (n + i) + b for i in range(horizon)])

    alpha, beta, gamma = 0.3, 0.1, 0.2

    # Init
    level = np.mean(series[:periods])
    trend = (np.mean(series[periods:2*periods]) - np.mean(series[:periods])) / periods
    season = [series[i] - level for i in range(periods)]

    levels, trends, seasons = [level], [trend], season[:]
    fitted = []

    for t in range(n):
        s_idx = t % periods
        prev_level = levels[-1]
        prev_trend = trends[-1]
        prev_season = seasons[t] if t < periods else seasons[t - periods + periods]

        new_level  = alpha * (series[t] - prev_season) + (1 - alpha) * (prev_level + prev_trend)
        new_trend  = beta  * (new_level - prev_level)  + (1 - beta)  * prev_trend
        new_season = gamma * (series[t] - new_level)   + (1 - gamma) * prev_season

        levels.append(new_level)
        trends.append(new_trend)
        seasons.append(new_season)
        fitted.append(new_level + prev_trend + prev_season)

    # Forecast
    forecasts = []
    for h in range(1, horizon + 1):
        s_idx   = (n - 1 + h) % periods
        s_val   = seasons[-(periods - s_idx) if (periods - s_idx) <= periods else -periods]
        fc_val  = levels[-1] + h * trends[-1] + seasons[-(periods - (h % periods)) if (periods - (h % periods)) <= periods else -1]
        forecasts.append(max(0, fc_val))

    return np.array(forecasts)


class ForecastAgent:
    """
    Receives the merged DataFrame from DataAgent and produces
    a 6-month forward forecast for every dept × category pair.
    """

    HORIZON = 6   # months ahead

    def __init__(self):
        self.log = []

    # ──────────────────────────────────────────────────
    def run(self, actuals: pd.DataFrame) -> dict:
        self._log("🟢 ForecastAgent starting …")

        forecasts = []
        pairs = actuals[["department", "category"]].drop_duplicates().values

        for dept, cat in pairs:
            series = self._build_series(actuals, dept, cat)
            if series is None or len(series) < 6:
                continue
            fc = self._forecast(series, dept, cat)
            forecasts.append(fc)

        if not forecasts:
            self._log("⚠️  No forecasts generated (not enough data)")
            return {"forecasts": pd.DataFrame(), "log": self.log}

        fc_df = pd.concat(forecasts, ignore_index=True)
        summary = self._summary(fc_df)

        self._log(f"✅ ForecastAgent complete — {len(fc_df):,} forecast rows across {len(pairs)} series")
        return {"forecasts": fc_df, "summary": summary, "log": self.log}

    # ──────────────────────────────────────────────────
    def _build_series(self, actuals, dept, cat) -> pd.Series | None:
        df = actuals[
            (actuals["department"] == dept) &
            (actuals["category"]   == cat)
        ].sort_values("date").set_index("date")["actual"]

        if df.empty:
            return None
        df = df.resample("MS").sum()      # monthly start
        df = df.ffill()
        return df

    def _forecast(self, series: pd.Series, dept: str, cat: str) -> pd.DataFrame:
        try:
            vals   = series.values.astype(float)
            n      = len(vals)
            method = "Holt-Winters" if n >= 12 else "Linear"

            if n >= 12:
                pred_vals = _holt_winters(vals, periods=12, horizon=self.HORIZON)
            else:
                x = np.arange(n)
                m, b = np.polyfit(x, vals, 1)
                pred_vals = np.array([max(0, m * (n + i) + b) for i in range(self.HORIZON)])

            last_date    = series.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.HORIZON, freq="MS",
            )

            rows = []
            for date, val in zip(future_dates, pred_vals):
                val = max(0, float(val))
                rows.append({
                    "date":       date,
                    "year":       date.year,
                    "month":      date.month,
                    "quarter":    f"Q{(date.month - 1) // 3 + 1}",
                    "department": dept,
                    "category":   cat,
                    "forecast":   round(val, 2),
                    "lower_ci":   round(val * 0.90, 2),
                    "upper_ci":   round(val * 1.10, 2),
                    "method":     method,
                })
            return pd.DataFrame(rows)

        except Exception as e:
            self._log(f"   ⚠️  Error forecasting {dept}/{cat}: {e}")
            return pd.DataFrame()


    def _summary(self, fc_df: pd.DataFrame) -> dict:
        total_forecast = fc_df["forecast"].sum()
        by_dept = (
            fc_df.groupby("department")["forecast"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )
        by_cat = (
            fc_df.groupby("category")["forecast"]
            .sum()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )
        return {
            "total_forecast_6m":   round(total_forecast, 2),
            "forecast_by_dept":    by_dept,
            "forecast_by_category":by_cat,
            "horizon_months":      self.HORIZON,
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
    agent  = ForecastAgent()
    result = agent.run(data["actuals"])

    print("\n📊 Forecast Summary:")
    for k, v in result["summary"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in list(v.items())[:5]:
                print(f"    {kk}: {vv:,.0f}")
        else:
            print(f"  {k}: {v}")
