"""Part B: customer segments (RFM + K-means + PCA) and lapse prediction.
Run from the repo root:  python src/part_b.py
Reads:  data/clean/orders.csv, customers.csv, products.csv
Writes: outputs/customer_segments.csv, outputs/lapse_model.joblib, outputs/retention_list.csv"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.makedirs("outputs", exist_ok=True)
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")

# =====================================================================
# TASK 1: RFM TABLE
# =====================================================================
print("=" * 60, "\nTASK 1: RFM\n" + "=" * 60)
done = orders[orders["status"] == "completed"]
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

rfm = (done.groupby("customer_id")
       .agg(last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"))
       .reset_index())
rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers (1175):", len(rfm))
print("recency median/min/max (19/0/180):", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())
print("frequency mean/max (9.46/33):", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
print("monetary mean/max (2552.06/14766.55):", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))

# =====================================================================
# TASK 2: K-MEANS, K = 2..8
# =====================================================================
print("\n" + "=" * 60, "\nTASK 2: K-MEANS\n" + "=" * 60)
X_rfm = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_rfm)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X_rfm, km.labels_), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("Reference: K=3 1268/0.445 | K=4 743/0.526 | K=5 617/0.481 (inertia +-5%, silhouette +-0.05)")

FINAL_K = 4   # chosen from {3,4,5}: best silhouette, clear elbow after K=4, four actionable segments

# =====================================================================
# TASK 3: PROFILE, NAME, PCA
# =====================================================================
print("\n" + "=" * 60, "\nTASK 3: SEGMENTS\n" + "=" * 60)
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X_rfm)
rfm["segment_id"] = km.labels_

prof = (rfm.groupby("segment_id")
        .agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
             mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
        .round(2))
print(prof.to_string())
print("Reference K=4: sizes ~436/138/419/182; 138-segment monetary ~7130, frequency ~6.8; "
      "419-segment frequency ~19.4, recency ~8.8")

# Cluster numbers are only labels with no ordering, so names come from the profile.
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest = rest.drop(index=loyal_id)
risk_id = rest["mean_recency"].idxmax()
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars", risk_id: "At-Risk / Fading"}
for sid in prof.index:
    names.setdefault(sid, "Occasional Buyers")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("segment names:", names)
print("CHECK these names against the profile table. Rename if they do not fit the numbers.")

pca = PCA(n_components=2).fit(X_rfm)
evr = pca.explained_variance_ratio_
print("PCA explained variance:", evr.round(3), "total:", round(evr.sum(), 3), "(ref ~0.634, 0.218, 0.852)")

rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)

# =====================================================================
# TASK 4: DEFINE THE LAPSE PROBLEM
# =====================================================================
print("\n" + "=" * 60, "\nTASK 4: LAPSE PROBLEM\n" + "=" * 60)
CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
pre = orders[orders["order_ts"] < CUTOFF]                 # ONLY orders before the cutoff feed features
pre_done = pre[pre["status"] == "completed"]

feat = (pre_done.groupby("customer_id")
        .agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
             monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
             avg_delivery=("delivery_minutes", "mean"))
        .reset_index())
feat["recency"] = (CUTOFF - feat["last_order"]).dt.days

n_cancel = pre[pre["status"] == "cancelled"].groupby("customer_id").size()
n_final = pre[pre["status"].isin(["completed", "cancelled"])].groupby("customer_id").size()
feat["cancel_rate"] = (feat["customer_id"].map(n_cancel).fillna(0) / feat["customer_id"].map(n_final)).values

fav = (pre_done.merge(prod[["product_id", "category"]], on="product_id")
       .groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id")[["customer_id", "category"]]
       .rename(columns={"category": "fav_category"}))
feat = (feat.merge(fav, on="customer_id")
            .merge(cust[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id"))
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days

sept_buyers = set(orders[(orders["status"] == "completed") &
                         (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(sept_buyers)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat.set_index("customer_id")[NUM + CAT]
y = feat.set_index("customer_id")["lapsed"]
print(f"cohort (1156): {len(feat)} | lapsed (443): {int(y.sum())} ({y.mean():.2%}, ref 38.32%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train/test (924/232): {len(X_train)}/{len(X_test)}")

# =====================================================================
# TASK 5: PREPROCESS WITHOUT LEAKAGE
# =====================================================================
print("\n" + "=" * 60, "\nTASK 5: PREPROCESSING\n" + "=" * 60)
assert X.isna().sum().sum() == 0, "feature columns contain missing values"
print("no missing values in any feature column: OK")

def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first"), CAT)])

demo = make_prep().fit(X_train)       # fitted on training rows only
scaler_mean = demo.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"scaler mean of frequency (train only): {scaler_mean:.3f} | full-cohort mean: {X['frequency'].mean():.3f}")
print("They differ because the scaler learned its mean from the training rows only; "
      "the test rows never influenced it.")

# =====================================================================
# TASK 6: BASELINE, TUNING, TEST EVALUATION
# =====================================================================
print("\n" + "=" * 60, "\nTASK 6: TRAIN, TUNE, EVALUATE\n" + "=" * 60)
def metrics(y_true, pred):
    return {"precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, pred, zero_division=0), 3)}

baseline = np.zeros(len(y_test), dtype=int)
print("baseline accuracy (ref 0.616):", round(accuracy_score(y_test, baseline), 3),
      "| lapsed F1 (ref 0):", f1_score(y_test, baseline, zero_division=0))

lr_pipe = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))])
gs_lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)

rf_pipe = Pipeline([("prep", make_prep()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
gs_rf = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)

print("\nLogReg best params:", gs_lr.best_params_, "| best CV F1:", round(gs_lr.best_score_, 4))
print("RandForest best params:", gs_rf.best_params_, "| best CV F1:", round(gs_rf.best_score_, 4))

for name, gs in [("LogReg", gs_lr), ("RandForest", gs_rf)]:
    pred = gs.best_estimator_.predict(X_test)
    print(f"\n{name} test confusion matrix [[TN FP],[FN TP]]:\n{confusion_matrix(y_test, pred)}")
    print(f"{name} test metrics (lapsed class):", metrics(y_test, pred))

# Select using cross-validated F1 so the test set plays no part in the choice.
if gs_rf.best_score_ >= gs_lr.best_score_:
    best_name, best_model = "RandForest", gs_rf.best_estimator_
else:
    best_name, best_model = "LogReg", gs_lr.best_estimator_
sel_f1 = f1_score(y_test, best_model.predict(X_test), zero_division=0)
print(f"\nselected model: {best_name} | test F1 = {sel_f1:.3f} (must be >= 0.55)")
assert sel_f1 >= 0.55, "selected model must reach test F1 >= 0.55"

# =====================================================================
# TASK 7: CLASS-BALANCE EXPERIMENT (training set only)
# =====================================================================
print("\n" + "=" * 60, "\nTASK 7: OVERSAMPLING\n" + "=" * 60)
n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
extra = y_train[y_train == 1].sample(n=n_extra, replace=True, random_state=42).index
X_bal = pd.concat([X_train, X_train.loc[extra]])
y_bal = pd.concat([y_train, y_train.loc[extra]])
print("balanced training counts (570 each):", y_bal.value_counts().to_dict())

lr_bal = Pipeline([("prep", make_prep()),
                   ("clf", LogisticRegression(max_iter=1000, C=gs_lr.best_params_["clf__C"]))]).fit(X_bal, y_bal)
before = metrics(y_test, gs_lr.best_estimator_.predict(X_test))
after = metrics(y_test, lr_bal.predict(X_test))
print("LogReg un-balanced:", before)
print("LogReg balanced   :", after)
print("Reference: recall ~0.64 -> ~0.85, precision stays ~0.6")

# =====================================================================
# TASK 8: INTERPRET, PERSIST, USE
# =====================================================================
print("\n" + "=" * 60, "\nTASK 8: INTERPRET, SAVE, RETENTION LIST\n" + "=" * 60)
rf_best, lr_best = gs_rf.best_estimator_, gs_lr.best_estimator_
imp = pd.Series(rf_best["clf"].feature_importances_,
                index=rf_best["prep"].get_feature_names_out()).sort_values(ascending=False)
print("Top-5 random forest importances:\n", imp.head(5).round(4).to_string())

coef = pd.Series(lr_best["clf"].coef_[0], index=lr_best["prep"].get_feature_names_out())
print("\nLogReg coefficients (sign = direction):\n",
      coef[["num__recency", "num__frequency", "num__cancel_rate"]].round(3).to_string())

joblib.dump(best_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(best_model.predict(X_test), reloaded.predict(X_test))
print("\nreloaded model reproduces the original test predictions exactly: OK")

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
res = pd.DataFrame({"lapse_probability": best_model.predict_proba(X_test)[:, 1],
                    "actually_lapsed": y_test.values}, index=X_test.index)
res["is_premium"] = seg["is_premium"].reindex(res.index).fillna(False).astype(bool)
ret = (res[res["is_premium"]].sort_values("lapse_probability", ascending=False)
       .head(10).reset_index()[["customer_id", "lapse_probability", "actually_lapsed"]])
ret["lapse_probability"] = ret["lapse_probability"].round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print(ret.to_string(index=False))
print("truly lapsed among them:", int(ret["actually_lapsed"].sum()), "of", len(ret))
