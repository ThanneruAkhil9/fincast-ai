import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Operations", "Finance", "IT"]
COST_CENTERS = {
    "Engineering": "CC-1001",
    "Sales":       "CC-2001",
    "Marketing":   "CC-3001",
    "HR":          "CC-4001",
    "Operations":  "CC-5001",
    "Finance":     "CC-6001",
    "IT":          "CC-7001",
}
CATEGORIES = ["Salaries", "OPEX", "Capex", "T&E", "Marketing Spend", "Software", "Utilities", "Headcount"]

START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2025, 3, 31)

# ─────────────────────────────────────────────
# ACTUALS  (monthly)
# ─────────────────────────────────────────────
def generate_actuals():
    rows = []
    date = START_DATE
    while date <= END_DATE:
        for dept in DEPARTMENTS:
            for cat in CATEGORIES:
                base = {
                    "Salaries":        random.uniform(400_000, 900_000),
                    "OPEX":            random.uniform(50_000,  200_000),
                    "Capex":           random.uniform(20_000,  150_000),
                    "T&E":             random.uniform(5_000,   40_000),
                    "Marketing Spend": random.uniform(30_000,  300_000),
                    "Software":        random.uniform(10_000,  80_000),
                    "Utilities":       random.uniform(5_000,   30_000),
                    "Headcount":       random.uniform(10,      120),
                }[cat]

                # seasonal noise
                seasonal = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
                # occasional anomaly
                anomaly  = random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1.6, 0.4])

                actual = round(base * seasonal * anomaly * random.uniform(0.9, 1.1), 2)
                rows.append({
                    "date":        date.strftime("%Y-%m-%d"),
                    "year":        date.year,
                    "month":       date.month,
                    "quarter":     f"Q{(date.month - 1) // 3 + 1}",
                    "department":  dept,
                    "cost_center": COST_CENTERS[dept],
                    "category":    cat,
                    "actual":      actual,
                })
        date = date + pd.DateOffset(months=1)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# BUDGET  (monthly)
# ─────────────────────────────────────────────
def generate_budget(actuals_df):
    rows = []
    for dept in DEPARTMENTS:
        for cat in CATEGORIES:
            for year in [2023, 2024, 2025]:
                for month in range(1, 13):
                    if year == 2025 and month > 3:
                        continue
                    subset = actuals_df[
                        (actuals_df["department"] == dept) &
                        (actuals_df["category"]   == cat) &
                        (actuals_df["year"]        == year) &
                        (actuals_df["month"]       == month)
                    ]
                    if subset.empty:
                        continue
                    # budget = actual ± 10-15%
                    budget = round(subset["actual"].values[0] * random.uniform(0.88, 1.14), 2)
                    rows.append({
                        "date":        f"{year}-{month:02d}-01",
                        "year":        year,
                        "month":       month,
                        "quarter":     f"Q{(month - 1) // 3 + 1}",
                        "department":  dept,
                        "cost_center": COST_CENTERS[dept],
                        "category":    cat,
                        "budget":      budget,
                    })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# FORECAST SEED  (future 6 months)
# ─────────────────────────────────────────────
def generate_forecast_seed():
    rows = []
    start = datetime(2025, 4, 1)
    for i in range(6):
        date = start + pd.DateOffset(months=i)
        for dept in DEPARTMENTS:
            for cat in CATEGORIES:
                base = {
                    "Salaries":        random.uniform(400_000, 900_000),
                    "OPEX":            random.uniform(50_000,  200_000),
                    "Capex":           random.uniform(20_000,  150_000),
                    "T&E":             random.uniform(5_000,   40_000),
                    "Marketing Spend": random.uniform(30_000,  300_000),
                    "Software":        random.uniform(10_000,  80_000),
                    "Utilities":       random.uniform(5_000,   30_000),
                    "Headcount":       random.uniform(10,      120),
                }[cat]
                rows.append({
                    "date":        date.strftime("%Y-%m-%d"),
                    "year":        date.year,
                    "month":       date.month,
                    "quarter":     f"Q{(date.month - 1) // 3 + 1}",
                    "department":  dept,
                    "cost_center": COST_CENTERS[dept],
                    "category":    cat,
                    "forecast":    round(base * random.uniform(0.95, 1.05), 2),
                })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# HEADCOUNT
# ─────────────────────────────────────────────
def generate_headcount():
    rows = []
    date = START_DATE
    base_hc = {d: random.randint(30, 200) for d in DEPARTMENTS}
    while date <= END_DATE:
        for dept in DEPARTMENTS:
            base_hc[dept] += random.randint(-2, 4)
            base_hc[dept]  = max(20, base_hc[dept])
            rows.append({
                "date":       date.strftime("%Y-%m-%d"),
                "year":       date.year,
                "month":      date.month,
                "department": dept,
                "headcount":  base_hc[dept],
            })
        date = date + pd.DateOffset(months=1)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.dirname(__file__)

    actuals  = generate_actuals()
    budget   = generate_budget(actuals)
    forecast = generate_forecast_seed()
    hc       = generate_headcount()

    actuals.to_csv(f"{out}/actuals.csv",  index=False)
    budget.to_csv( f"{out}/budget.csv",   index=False)
    forecast.to_csv(f"{out}/forecast_seed.csv", index=False)
    hc.to_csv(     f"{out}/headcount.csv",index=False)

    print(f"✅ Actuals:   {len(actuals):,} rows")
    print(f"✅ Budget:    {len(budget):,} rows")
    print(f"✅ Forecast:  {len(forecast):,} rows")
    print(f"✅ Headcount: {len(hc):,} rows")
    print("All synthetic data saved to /data/")
