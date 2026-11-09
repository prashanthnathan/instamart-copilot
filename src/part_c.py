"""Part C: support-ticket triage with guardrails.
Run from the repo root:  python src/part_c.py
Reads:  data/raw/tickets.csv, outputs/customer_segments.csv, mock_llm.py
Writes: outputs/triage_budget_demo.csv, outputs/triage_results.csv"""
import os, sys, re, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so mock_llm.py (repo root) imports
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

os.makedirs("outputs", exist_ok=True)

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)   # empty texts stay ""
print("tickets loaded (164):", len(tickets))

# =====================================================================
# TASK 2: REDACTION (order: email, phone, order id)
# =====================================================================
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counter=None):
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        if counter is not None:
            counter[name] += n
    return text

def has_pii(text):
    return any(pat.search(text) for _, pat, _ in PATTERNS)

print("\n" + "=" * 60, "\nREDACTION REPORT (all 164 tickets)\n" + "=" * 60)
counter = {"email": 0, "phone": 0, "order_id": 0}
redacted_all = [redact(t, counter) for t in tickets["text"]]
print("substitutions:", counter, "| total:", sum(counter.values()), "(ref 55 / 91 / 132 / 278)")
assert not any(has_pii(t) for t in redacted_all), "PII left after redaction"
print("re-applying the three patterns to redacted texts finds nothing: OK")

# input guardrail check
n_empty = int((tickets["text"].str.strip() == "").sum())
n_long = int(tickets["text"].apply(lambda t: est_tokens(t) > MAX_INPUT_TOKENS).sum())
long_tok = tickets.loc[tickets["text"].apply(lambda t: est_tokens(t) > MAX_INPUT_TOKENS), "text"].apply(est_tokens).unique()
print(f"empty tickets (4): {n_empty} | long tickets (4): {n_long} | their est. tokens (734): {list(long_tok)}")

# =====================================================================
# TASK 3: TWO PROMPTS
# =====================================================================
ZERO_SHOT = ('You CLASSIFY support tickets into billing, delivery, technical or fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')

FEW_SHOT = ('You CLASSIFY support tickets into billing, delivery, technical or fraud. '
            'Reply with JSON only: {"category": "...", "reason": "..."}\n'
            'EXAMPLES:\n'
            'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My order has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "missing order"}\n'
            'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
            'Ticket: Someone hacked my account and used my wallet. -> {"category": "fraud", "reason": "unauthorized use"}')

assert "CLASSIFY" in ZERO_SHOT and "CLASSIFY" in FEW_SHOT
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
assert all(c in ZERO_SHOT and c in FEW_SHOT for c in CATEGORIES)
assert not has_pii(ZERO_SHOT) and not has_pii(FEW_SHOT)
print("\nprompt sizes (est. tokens): zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

# =====================================================================
# TASK 4: RESILIENT CALLS
# =====================================================================
def call_with_retry(messages, ticket_id, stats):
    """Up to MAX_RETRIES retries; wait before retry n = BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if retry == MAX_RETRIES:
                raise                       # permanent failure (should not happen with this mock)
            wait = BASE_DELAY * (2 ** retry) * random.uniform(0.9, 1.15)
            stats["retries"] += 1
            print(f"  [retry] ticket={ticket_id} retry={retry} wait={wait:.4f}s")
            time.sleep(wait)

# =====================================================================
# TASK 5: VALIDATE EVERY ANSWER
# =====================================================================
def parse_answer(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))   # 'billing-related' -> billing, related
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""        # valid JSON but category outside the list: never guess

# =====================================================================
# THE PIPELINE (guardrails -> redact -> call -> validate), with optional token budget
# =====================================================================
def run_pipeline(system_prompt, label, budget=None):
    print(f"\n--- running pipeline: {label} (budget={budget}) ---")
    mock_llm.reset()                         # reset the mock's counters before every full run
    random.seed(42)
    stats = {"transient": 0, "retries": 0}
    total_tokens, out = 0, []
    for _, r in tickets.iterrows():
        tokens, category = 0, ""
        text = r["text"]
        if not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif budget is not None and total_tokens >= budget:
            status = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(text)}]
            assert not any(has_pii(m["content"]) for m in messages), "PII reached a prompt"
            try:
                resp = call_with_retry(messages, r["ticket_id"], stats)
                tokens = resp.usage.total_tokens
                total_tokens += tokens
                status, category = parse_answer(resp.choices[0].message.content)
            except TransientLLMError:
                status = "failed_transient"
        out.append({"ticket_id": r["ticket_id"], "customer_id": r["customer_id"], "status": status,
                    "category": category, "true_category": r["true_category"], "tokens_used": tokens})
    df = pd.DataFrame(out)
    print("transient errors:", stats["transient"], "| retries:", stats["retries"], "(ref 19, all recovered)")
    print("status counts:", df["status"].value_counts().to_dict())
    return df, total_tokens

# =====================================================================
# TASK 6: ZERO-SHOT VS FEW-SHOT (no budget)
# =====================================================================
print("\n" + "=" * 60, "\nTASK 6: ZERO-SHOT VS FEW-SHOT\n" + "=" * 60)
summary = []
zero_df = None
for label, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
    df, total = run_pipeline(prompt, label)
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean()
    summary.append({"prompt": label, "classified": len(ok), "accuracy_%": round(acc * 100, 1),
                    "total_tokens": total, "tokens_per_classified": round(total / len(ok), 1)})
    if label == "zero-shot":
        zero_df, zero_ok = df, ok
print("\n", pd.DataFrame(summary).to_string(index=False))
print("Reference: 147 classified | zero-shot 85.7% (126/147) | few-shot 100%")
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zero_ok["true_category"], zero_ok["category"]).to_string())
print("Reference: 11 fraud->billing and 10 technical->billing errors")

# =====================================================================
# TASK 7: BUDGET, ROUTING, OUTPUT
# =====================================================================
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
seg["is_premium"] = seg["is_premium"].astype(str).str.lower() == "true"
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "failed_transient"}

def route_tickets(df):
    rows = []
    for _, r in df.iterrows():
        cid = r["customer_id"]
        seg_name = seg.loc[cid, "segment_name"] if cid in seg.index else ""
        premium = bool(seg.loc[cid, "is_premium"]) if cid in seg.index else False   # no row -> not premium
        if r["status"] in HUMAN:
            route, priority = "human_review", "NORMAL"
        elif r["category"] == "fraud":
            route, priority = "security_team", "HIGH"
        elif premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        rows.append({"ticket_id": r["ticket_id"], "customer_id": cid, "status": r["status"],
                     "category": r["category"], "segment_name": seg_name,
                     "priority": priority, "route": route, "tokens_used": r["tokens_used"]})
    return pd.DataFrame(rows)

print("\n" + "=" * 60, "\nTASK 7: BUDGET DEMO + FINAL ROUTING\n" + "=" * 60)
DEMO_BUDGET = 6000
demo_df, demo_total = run_pipeline(ZERO_SHOT, "zero-shot budget demo", budget=DEMO_BUDGET)
n_ex = int((demo_df["status"] == "budget_exhausted").sum())
print(f"budget used: {DEMO_BUDGET} | budget_exhausted tickets: {n_ex}")
assert n_ex >= 1, "lower DEMO_BUDGET (e.g. 5000) so at least one ticket is exhausted"
route_tickets(demo_df).to_csv("outputs/triage_budget_demo.csv", index=False)

final_df, final_total = run_pipeline(FEW_SHOT, "few-shot FINAL", budget=100000)
final_routed = route_tickets(final_df)
final_routed.to_csv("outputs/triage_results.csv", index=False)
print("\nfinal rows (164):", len(final_routed))
print("route counts:", final_routed["route"].value_counts().to_dict())
print("priority counts:", final_routed["priority"].value_counts().to_dict())
