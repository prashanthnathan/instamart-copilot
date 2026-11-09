"""Part D: policy RAG assistant + refund agent.
Run from the repo root:  python src/part_d.py
Reads:  data/policies/*.txt, data/raw/policy_eval.csv, data/raw/agent_scenarios.csv, data/instamart.db
Writes: outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, outputs/agent_demos.jsonl, state/*.json"""
import os, sys, json, time, shutil, sqlite3, uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # so mock_llm.py imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

DB, POLICIES = "data/instamart.db", "data/policies"
LEDGER, RUNS, DEMOS, STATE = "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl", "state"

# start every run from a clean slate so the script is reproducible
os.makedirs("outputs", exist_ok=True)
shutil.rmtree(STATE, ignore_errors=True)
os.makedirs(STATE, exist_ok=True)
for p in (LEDGER, RUNS, DEMOS):
    if os.path.exists(p):
        os.remove(p)

# =====================================================================
# TASK 1: CHUNK THE DOCUMENTS (40 words, overlap 10 -> step 30)
# =====================================================================
def chunk_words(words, size=40, overlap=10):
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):          # stop after the first chunk that reaches the last word
            break
        start += step
    return chunks

def build_chunks(size=40, overlap=10):
    out = []
    for fn in sorted(os.listdir(POLICIES)):
        if fn.endswith(".txt"):
            words = open(os.path.join(POLICIES, fn), encoding="utf-8").read().split()
            out += [{"source": fn, "text": t} for t in chunk_words(words, size, overlap)]
    return out

print("=" * 60, "\nTASK 1: CHUNKING\n" + "=" * 60)
chunks10 = build_chunks(40, 10)
counts = pd.Series([c["source"] for c in chunks10]).value_counts().sort_index()
print(counts.to_string())
print("total chunks (ref 15; refund 4, delivery 3, escalation 3, payment 3, privacy 2):", len(chunks10))

# =====================================================================
# TASK 2: TF-IDF RETRIEVAL
# =====================================================================
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer()
        self.mat = self.vec.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        q = self.vec.transform([query])                        # same fitted vectorizer
        sims = cosine_similarity(q, self.mat).ravel()
        top = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in top]

RET = Retriever(chunks10)

def retrieve(query, k=3):
    return RET.retrieve(query, k)

print("\n" + "=" * 60, "\nTASK 2: RETRIEVAL\n" + "=" * 60)
for r in retrieve("refund policy late delivery"):
    print(f"  {r['source']}  score={r['score']:.3f}")
print("(ref: all three from refund_policy.txt)")

# =====================================================================
# TASK 3: AUGMENT AND ANSWER
# =====================================================================
POLICY_SYSTEM = ("You are a POLICY ASSISTANT for support staff. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say that you could not find the information.")

def answer(question, retriever=None):
    retriever = retriever or RET
    top = retriever.retrieve(question, 3)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in top)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": POLICY_SYSTEM},
                          {"role": "user", "content": user}])
    return {"answer": resp.choices[0].message.content, "sources": [c["source"] for c in top]}

# =====================================================================
# TASK 4: EVALUATE RETRIEVAL AND GROUNDING
# =====================================================================
def evaluate(retriever, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    hit = fact = fallback = n_ans = n_unans = 0
    failures = []
    for _, r in ev.iterrows():
        res = answer(r["question"], retriever)
        if r["expected_doc"] == "NONE":
            n_unans += 1
            if res["answer"].startswith("I could not find"):
                fallback += 1
            else:
                failures.append((r["question"], "should have been the fallback answer"))
        else:
            n_ans += 1
            h = r["expected_doc"] in res["sources"]
            f = r["expected_fact"] in res["answer"]          # case-sensitive substring
            hit += h; fact += f
            if not (h and f):
                failures.append((r["question"], f"doc in top-3: {h} | fact in answer: {f} | answer: {res['answer']}"))
    print(f"\n[{label}] answerable: expected_doc in top-3 {hit}/{n_ans} | answer has expected_fact {fact}/{n_ans} "
          f"| unanswerable fallback {fallback}/{n_unans}")
    for q, why in failures:
        print("   FAILS:", q, "->", why)

print("\n" + "=" * 60, "\nTASK 4: EVALUATION\n" + "=" * 60)
evaluate(RET, "overlap 10")
print("(ref: 12/12, 12/12, 2/2)")
RET0 = Retriever(build_chunks(40, 0))
print("chunks with overlap 0:", len(RET0.chunks))
evaluate(RET0, "overlap 0")
print("(ref: at least one answerable question fails: an answer split across a chunk boundary)")

# =====================================================================
# TASK 5: TOOLS AND TOOL MAP
# =====================================================================
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or (isinstance(row[3], float) and np.isnan(row[3])) else int(round(row[3]))
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]), "delivery_minutes": dm}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# =====================================================================
# TASKS 6-8: REACT LOOP, STOP SAFELY, RESUMABLE
# =====================================================================
AGENT_SYSTEM = ("You are a refund AGENT for support staff. Use the tools to look up the order, check the refund rules, "
                "calculate the refund, then issue it or escalate it. Finish with one line of the form STATUS: message.")

class SimulatedCrash(Exception):
    pass

def save_checkpoint(st):
    with open(os.path.join(STATE, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, default=str)

def _handoff(st, status):
    done = ", ".join(st["tools_executed"]) or "none"
    return (f"HANDOFF TO HUMAN ({status}): stopped after {st['llm_calls']} model calls and {len(st['steps'])} "
            f"tool steps (tools run: {done}). Request: {st['user_message']}")

def _loop(st, crash_after_tools=None):
    t0 = time.monotonic()

    def finish(status, final):
        return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final,
                "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"],
                "tools_executed": st["tools_executed"]}

    while True:
        if time.monotonic() - t0 >= st["time_limit_seconds"]:                 # (c) timeout, top of iteration
            return finish("FALLBACK_TIMEOUT", _handoff(st, "FALLBACK_TIMEOUT"))
        if st["llm_calls"] >= st["max_iterations"]:                           # (a) too many model calls
            return finish("FALLBACK_MAX_ITERATIONS", _handoff(st, "FALLBACK_MAX_ITERATIONS"))
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                             # (b) token limit, tool NOT executed
            return finish("FALLBACK_TOKEN_LIMIT", _handoff(st, "FALLBACK_TOKEN_LIMIT"))
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                # final answer
            final = (msg.content or "").strip()
            status = final.split(":")[0].strip().upper() or "UNKNOWN"
            return finish(status, final)
        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        fn = TOOL_MAP.get(tc.name)
        try:
            result = fn(**args) if fn else {"error": f"unknown tool {tc.name}"}
        except Exception as e:
            result = {"error": str(e)}
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(result)})
        st["steps"].append({"tool": tc.name, "arguments": args, "observation": result})
        st["tools_executed"].append(tc.name)
        st["step"] += 1
        save_checkpoint(st)                                                   # checkpoint after every observation
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution #{crash_after_tools} ({tc.name})")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:6]}", "scenario_id": scenario_id,
          "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": [],
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st)

def refund_of(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0

def write_run(path, scenario_id, res):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": scenario_id, "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                            "steps": res["steps"]}) + "\n")

# ---- run all 7 scenarios ----
print("\n" + "=" * 60, "\nTASKS 6-7: SEVEN SCENARIOS\n" + "=" * 60)
sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
table = []
for _, r in sc.iterrows():
    res = run_agent(r["message"], r["scenario_id"])
    write_run(RUNS, r["scenario_id"], res)
    refund = refund_of(res)
    exp = r["expected_outcome"]
    ok_status = res["status"] == exp or (exp == "FALLBACK" and res["status"] == "FALLBACK_MAX_ITERATIONS")
    ok_amt = abs(refund - float(r["expected_refund_inr"])) <= 0.01
    table.append({"scenario": r["scenario_id"], "status": res["status"], "refund_inr": refund,
                  "model_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                  "expected": exp, "matches": ok_status and ok_amt})
print(pd.DataFrame(table).to_string(index=False))
print("(ref: S1 REFUND 128.25 in 5 calls | S2 REFUND 135.00 | S3 ESCALATE 1999.00 | S4, S5 DENIED | "
      "S6 NOT_FOUND | S7 FALLBACK_MAX_ITERATIONS)")
assert all(t["matches"] for t in table), "a scenario does not match its expected outcome"

# ---- stop-safely demonstrations ----
print("\n" + "=" * 60, "\nTASK 7: STOP-SAFELY DEMOS\n" + "=" * 60)
s1_msg = sc.loc[sc["scenario_id"] == "S1", "message"].iloc[0]
s1_amount = float(sc.loc[sc["scenario_id"] == "S1", "expected_refund_inr"].iloc[0])

demo_tok = run_agent(s1_msg, "S1", token_limit=200)
write_run(DEMOS, "S1_token_limit_200", demo_tok)
print("S1 with token_limit=200 ->", demo_tok["status"], "| tokens:", demo_tok["tokens_used"],
      "| tools run:", demo_tok["tools_executed"])
assert demo_tok["status"] == "FALLBACK_TOKEN_LIMIT"

demo_time = run_agent(s1_msg, "S1", time_limit_seconds=0)
write_run(DEMOS, "S1_time_limit_0", demo_time)
print("S1 with time_limit_seconds=0 ->", demo_time["status"], "| model calls:", demo_time["llm_calls"])
assert demo_time["status"] == "FALLBACK_TIMEOUT"

# ---- crash and resume ----
print("\n" + "=" * 60, "\nTASK 8: CRASH AND RESUME (S1)\n" + "=" * 60)
try:
    run_agent(s1_msg, "S1", run_id="S1_crash_demo", crash_after_tools=2)
except SimulatedCrash as e:
    print("crash:", e)
res = resume_agent("S1_crash_demo")
write_run(DEMOS, "S1_crash_resume", res)
print("tools executed across crash + resume, in order:", res["tools_executed"])
print("final status:", res["status"], "| refund:", refund_of(res))
assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - s1_amount) <= 0.01

print("\nledger lines:", sum(1 for _ in open(LEDGER, encoding="utf-8")),
      "| agent_runs lines (7):", sum(1 for _ in open(RUNS, encoding="utf-8")),
      "| checkpoint files:", len(os.listdir(STATE)))
