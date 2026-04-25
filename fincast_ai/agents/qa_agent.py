import os
import pandas as pd
from datetime import datetime
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

class QAAgent:
    def __init__(self, con):
        self.con = con
        self.log = []
        # Read key at runtime (not module load time) so Streamlit secrets are available
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.mode = "groq" if GROQ_API_KEY else "pattern"

        if self.mode == "groq":
            try:
                self.client = Groq(
                    api_key=GROQ_API_KEY
                )
            except Exception as e:
                self._log(f"⚠️ Groq init failed: {e}")
                self.mode = "pattern"

        self._log(f"🟢 QAAgent ready (mode={self.mode})")

    def _log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log.append(msg)

    def _groq_call(self, system, prompt):
        try:
            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            self._log(f"⚠️ Groq error: {e}")
            return None

    def ask(self, question: str):
        self._log(f"❓ Q: {question}")

        if self.mode == "groq":
            try:
                sql_prompt = f"""
Convert this question into SQL compatible with SQLite syntax.

Question: {question}

Tables available:
1. merged(date, year, month, quarter, department, cost_center, category, actual, budget, variance, variance_pct)
2. forecast(date, year, month, quarter, department, cost_center, category, forecast)
3. headcount(date, year, month, quarter, department, cost_center, category, headcount)
4. actuals(date, year, month, quarter, department, cost_center, category, actual)
5. budget(date, year, month, quarter, department, cost_center, category, budget)

Rules:
- Use SQLite-compatible syntax (no DuckDB-specific functions)
- For headcount questions → use SUM(headcount) or just SELECT headcount FROM headcount
- For employee count by department → SELECT department, SUM(headcount) AS headcount FROM headcount GROUP BY department
- For spend/cost questions → use SUM(actual) from merged or actuals
- For budget questions → use SUM(budget) from merged or budget table
- For variance → use SUM(variance) from merged
- Use proper GROUP BY when using aggregations
- Return ONLY the SQL query, no explanation
"""
                sql = self._groq_call(
                    "You are a SQLite SQL expert. Output only a valid SQL query, nothing else.",
                    sql_prompt
                )

                if not sql:
                    raise Exception("SQL generation failed")

                sql = sql.replace("```sql", "").replace("```", "").strip()
                self._log(f"📝 Generated SQL: {sql}")

                try:
                    res = self.con.execute(sql).fetchdf()
                except Exception as e:
                    self._log(f"❌ SQL execution failed: {e}")
                    return {
                        "answer": f"I couldn't execute the query ({str(e)}). Try rephrasing your question.",
                        "sql": sql,
                        "data": None
                    }

                if res is None or res.empty:
                    return {
                        "answer": "No data found for this query.",
                        "sql": sql,
                        "data": res
                    }

                preview = res.head(5).to_dict(orient="records")

                answer_prompt = f"""
Answer the question using this data.

Question: {question}

Data:
{preview}

Rules:
- If it's a headcount/employee count → return as integer (no $ sign)
- If it's money/spend/budget/variance → format with $ and commas
- Be concise, 1-2 sentences only
- Do NOT make up data not in the result
"""
                ans = self._groq_call(
                    "You are a senior FP&A analyst. Answer concisely based only on the data provided.",
                    answer_prompt
                )

                if not ans:
                    ans = f"Here is the result: {preview}"

                return {
                    "answer": ans,
                    "sql": sql,
                    "data": res
                }

            except Exception as e:
                self._log(f"⚠️ Failure: {e}")

        return {
            "answer": "I couldn't process that question. Try something simpler.",
            "sql": None,
            "data": None
        }
