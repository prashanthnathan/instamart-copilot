"""Part A: audit, clean, store, validate, SQL, pandas analysis.
Run from the repo root:  python src/part_a.py"""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)
os.makedirs("sql", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================================================
# TASK 2: AUDIT THE RAW DATA
# =====================================================================
customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

print("=" * 60, "\nAUDIT\n" + "=" * 60)
for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n--- {name}: shape {df.shape}")
    print(df.dtypes.to_string())
    miss = df.isna().sum()
    print(pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)}).to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
audit = {
    "raw order rows (13,084)": len(orders),
    "exact duplicate rows (119)": int(orders.duplicated().sum()),
    "pending rows (656)": int((orders["status"] == "pending").sum()),
    "distinct order_id (12,309)": int(orders["order_id"].nunique()),
    "non-numeric amounts (120)": int(amt_num.isna().sum()),
    "negative delivery_minutes (106)": int((dm < 0).sum()),
    "blank delivery_minutes (2,541)": int(dm.isna().sum()),
    "blank ages (61)": int(customers["age"].isna().sum()),
    "ages outside 10-100 (6)": int(((customers["age"] < 10) | (customers["age"] > 100)).sum()),
    "raw city spellings (24)": int(customers["city"].nunique()),
    "raw gender spellings (10)": int(customers["gender"].dropna().nunique()),
    "blank genders (30)": int(customers["gender"].isna().sum()),
}
print("\nAUDIT FIGURES (reference in brackets)")
for k, v in audit.items():
    print(f"  {k}: {v}")

# =====================================================================
# TASK 3: CLEAN CUSTOMERS
# =====================================================================
c = customers.copy()
c["city"] = c["city"].str.strip().str.title()                      # (a)
g = c["gender"].fillna("").str.strip().str.lower()                 # (b)
c["gender"] = g.map({"m": "male", "male": "male",
                     "f": "female", "female": "female"}).fillna("unknown")
bad_age = (c["age"] < 10) | (c["age"] > 100)                       # (c)
c.loc[bad_age, "age"] = np.nan
n_age_missing = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])                # (d)
print("\nCUSTOMERS CLEANED")
print("  ages set missing (outside 10-100):", int(bad_age.sum()))
print("  ages filled with city median:", n_age_missing)
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())

# =====================================================================
# TASK 4: CLEAN ORDERS (in the brief's order)
# =====================================================================
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                      # rule 1

n = len(o)
o = o.drop_duplicates()                                            # rule 2
print("\nORDERS CLEANED\n  rule 2: exact duplicates removed:", n - len(o))

n = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")   # rule 3
print("  rule 3: older records per order_id removed:", n - len(o))

o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")    # rule 4
n_bad_amt = int(o["order_amount"].isna().sum())
fill_amt = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill_amt)
print("  rule 4: amounts filled from product price:", n_bad_amt)

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
print("  rule 5: negative delivery_minutes set to missing:", int(neg.sum()))
done = o["status"] == "completed"
cat_median = o[done].groupby("category")["delivery_minutes"].transform("median")
n_fill = int(o.loc[done, "delivery_minutes"].isna().sum())
o.loc[done, "delivery_minutes"] = o.loc[done, "delivery_minutes"].fillna(cat_median)
print("  rule 5: completed orders filled with category median:", n_fill)
print("  cancelled orders left missing (never delivered):",
      int(o.loc[o["status"] == "cancelled", "delivery_minutes"].isna().sum()))

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print(o["status"].value_counts().to_string())

# =====================================================================
# TASK 5: STORE (CSV + SQLite) AND VALIDATE
# =====================================================================
c.to_csv(f"{CLEAN}/customers.csv", index=False)
products.to_csv(f"{CLEAN}/products.csv", index=False)
o.to_csv(f"{CLEAN}/orders.csv", index=False)

c_db, o_db = c.copy(), o.copy()
c_db["signup_date"] = c_db["signup_date"].dt.strftime("%Y-%m-%d")
o_db["order_ts"] = o_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
con = sqlite3.connect(DB)
c_db.to_sql("customers", con, if_exists="replace", index=False)
products.to_sql("products", con, if_exists="replace", index=False)
o_db.to_sql("orders", con, if_exists="replace", index=False)

# --- validation: 12+ reference checkpoints ---
cc = pd.read_csv(f"{CLEAN}/customers.csv")
oc = pd.read_csv(f"{CLEAN}/orders.csv")
dn, cn = oc[oc["status"] == "completed"], oc[oc["status"] == "cancelled"]
assert len(oc) == 12309 and oc["order_id"].is_unique, "row count / unique ids"
assert len(dn) == 11110, "completed count"
assert len(cn) == 1199, "cancelled count"
assert not (oc["status"] == "pending").any(), "pending rows remain"
assert oc["order_amount"].isna().sum() == 0, "missing amounts"
assert dn["delivery_minutes"].isna().sum() == 0, "missing delivery (completed)"
assert cn["delivery_minutes"].isna().sum() == 1199, "cancelled delivery should stay missing"
assert abs(dn["order_amount"].sum() - 2998668.80) < 0.01, "completed revenue"
assert cc["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                               "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "cities"
assert cc["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "genders"
assert cc["age"].isna().sum() == 0 and cc["age"].between(18, 60).all(), "ages"
for t in ["customers", "products", "orders"]:
    assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0, f"table {t} empty"
print("\nALL CHECKPOINTS PASSED")

# =====================================================================
# TASK 6: SQL ANALYSIS (writes sql/queries.sql, then runs it)
# =====================================================================
QUERIES = {
"Q1 WHERE then HAVING (expect 30 customers)": """
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000""",
"Q2 three-table join (expect 30 rows, total 2998668.80, max Bengaluru/Electronics 233909.45)": """
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category""",
"Q3 scalar subquery (expect 11 products)": """
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)""",
"Q4 LEFT JOIN (expect 25 customers)": """
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL""",
"Q5 CTEs (expect 550 customers, average 2552.06)": """
WITH spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status = 'completed' GROUP BY customer_id
), avg_spend AS (
  SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend, ROUND(a.avg_total, 2) AS avg_total
FROM spend s, avg_spend a
WHERE s.total_spend > a.avg_total
ORDER BY s.total_spend DESC""",
"Q6 ranking windows (expect 8 rows where rnk != row_num)": """
WITH spend AS (
  SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  WHERE o.status = 'completed'
  GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, rnk""",
"Q7 running total (expect 183 rows, last cumulative 2998668.80)": """
WITH daily AS (
  SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
  FROM orders WHERE status = 'completed' GROUP BY day
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day""",
}
with open("sql/queries.sql", "w", encoding="utf-8") as f:
    for title, q in QUERIES.items():
        f.write(f"-- {title}\n{q.strip()};\n\n")

print("\n" + "=" * 60 + "\nSQL RESULTS (run from sql/queries.sql)\n" + "=" * 60)
sql_text = open("sql/queries.sql", encoding="utf-8").read()
stmts = [s.strip() for s in sql_text.split(";") if s.strip()]
sql_total = None
for i, stmt in enumerate(stmts, 1):
    df = pd.read_sql_query(stmt, con)
    print(f"\nQ{i}: {len(df)} rows")
    print(df.head(6).to_string(index=False))
    if i == 2:
        sql_total = round(df["revenue"].sum(), 2)
        print("  revenue total:", sql_total, "| largest cell:", df.loc[df["revenue"].idxmax()].tolist())
    if i == 5:
        print("  average spend:", df["avg_total"].iloc[0])
    if i == 6:
        print("  top customer per city:")
        print(df[df["rnk"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
        print("  rows with RANK != ROW_NUMBER:", int((df["rnk"] != df["row_num"]).sum()))
    if i == 7:
        print("  last cumulative value:", df["cumulative_revenue"].iloc[-1])
con.close()

# =====================================================================
# TASK 7: PANDAS CROSS-CHECK AND EXPLORATORY ANALYSIS
# =====================================================================
print("\n" + "=" * 60 + "\nPANDAS CROSS-CHECK AND EDA\n" + "=" * 60)
d = (dn.merge(cc[["customer_id", "city"]], on="customer_id")
       .merge(products[["product_id", "category"]], on="product_id"))

pt = pd.pivot_table(d, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True)
print(pt.round(2).to_string())
pandas_total = round(pt.loc["All", "All"], 2)
print("pivot grand total:", pandas_total, "| SQL Q2 total:", sql_total)
assert abs(pandas_total - sql_total) < 0.01, "pivot total must equal SQL total"

print("\nPer-city sum / mean / count of completed order_amount:")
print(d.groupby("city").agg(total=("order_amount", "sum"),
                            mean=("order_amount", "mean"),
                            n=("order_amount", "count")).round(2).to_string())

s = d["order_amount"]
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {s.mean():.2f} | median {s.median():.2f} | skewness {s.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | fences {lo:.2f} / {hi:.2f}")
print(f"outliers below lower fence: {int((s < lo).sum())} | above upper fence: {int((s > hi).sum())}")

corr = d[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
top = pairs.abs().idxmax()
print("largest |correlation| pair:", top, "=", round(pairs[top], 3))