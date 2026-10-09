"""Re-grade every stored row of the traces corpus after the label audit of 8 October 2026.

1. Applies data/label-audit.json: new gold for relabelled items; excluded items are left in the
   files (stats.py drops them by id).
2. Re-grades every answer with a grader that normalises math expressions (\\frac, \\sqrt, $…$,
   \\( \\), trailing '≈ …' / '(or approximately …)') before comparing; numeric answers compared
   with relative tolerance 1e-6. The LLM arbiter is used only through its existing cache; no calls.
3. Recomputes the derived correctness fields of the judge, endorse and aggregate stages.
4. Rebuilds matrix-v2.csv from the <layer>.jsonl files only (never from samples-/budget- files).

    python3 regrade_all.py            # report only
    python3 regrade_all.py --apply    # rewrite files in place
"""
import json, re, sys, collections, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent; RAW = HERE / "raw"; DATA = HERE / "data"
sys.path.insert(0, str(HERE))
from grader import norm as gnorm, _num
APPLY = "--apply" in sys.argv

audit = json.load((DATA / "label-audit.json").open())
RELABEL = {k: v["new"] for k, v in audit["relabels"].items()}

items = {}
for fn in ("items.jsonl", "items-gpqa.jsonl"):
    for l in (DATA / fn).open():
        if l.strip():
            r = json.loads(l); items[r["id"]] = r
for i, g in RELABEL.items():
    assert i in items, i
    items[i]["gold"] = g

_cache = {}
cp = HERE / "grader-cache.jsonl"
if cp.exists():
    for l in cp.read_text().splitlines():
        if l.strip():
            r = json.loads(l); _cache[(r["gold"], r["answer"])] = r["equivalent"]

def preclean(a: str) -> str:
    a = a.strip().replace("$", "")
    a = re.sub(r"^\\\(|\\\)$", "", a).strip()
    a = re.sub(r"(?<=\d),(?=\d{3})", "", a)
    a = re.sub(r"\s*\((?:or|approximately|approx\.?|≈)[^)]*\)\s*$", "", a, flags=re.I)
    a = re.sub(r"\s*(≈|\\approx|or approximately|or about)\s.*$", "", a, flags=re.I)
    a = re.sub(r"\s*,?\s*(?:or|i\.e\.|which is)\s+(?:approximately|about)\s.*$", "", a, flags=re.I)
    return a.strip().rstrip(".")

def grade(item, ans):
    """True/False, None when there is nothing to grade."""
    if ans is None or not str(ans).strip():
        return None
    ans = str(ans); t = item.get("answer_type"); gold = str(item["gold"]); ds = item.get("dataset") or item.get("layer")
    if t == "letter" or ds in ("mmlupro", "mmlu-pro", "gpqa", "gpqa-diamond") or (t is None and re.fullmatch(r"[A-J]", gold)):
        m = re.match(r"^\(?([A-J])\)?", ans.strip().upper()); return (m.group(1) == gold.upper()) if m else None
    if t == "yesno":
        m = re.search(r"\b(yes|no)\b", ans.lower()); return (m.group(1) == gold) if m else None
    if t == "set":
        got = {x.strip().strip(".").lower() for x in re.split(r"[,;]|\band\b", ans) if x.strip()} - {""}
        gs = {x.lower() for x in gold.split(",")} if gold != "none" else set()
        if "none" in got: got = set()
        return got == gs
    a = preclean(ans)
    ga, gg = gnorm(gold), gnorm(a)
    if ga == gg: return True
    na, ng = _num(gg), _num(ga)
    if na is not None and ng is not None:
        return abs(na - ng) < 1e-6 * max(1, abs(ng))
    if (gold, ans) in _cache: return _cache[(gold, ans)]
    if (gold, a) in _cache: return _cache[(gold, a)]
    return False

JUDGE_DIR = {"claude:haiku": "claude_haiku", "claude:sonnet55": "claude_sonnet55", "claude:opus": "claude_opus",
             "deepseek/deepseek-v3.2": "deepseek_deepseek-v3.2", "qwen/qwen3.7-plus": "qwen_qwen3.7-plus",
             "z-ai/glm-5.3-flash": "z-ai_glm-5.3-flash", "mistralai/mistral-large-2512": "mistralai_mistral-large-2512"}

flips = collections.Counter(); detail = collections.defaultdict(list)
def setf(r, key, new, tag, rid):
    old = r.get(key)
    if old is None and new is None: return
    if bool(old) != bool(new) or (old is None) != (new is None):
        flips[(tag, key, f"{old}->{new}")] += 1; detail[tag].append((rid, key, old, new))
    r[key] = new

matrix_correct = {}  # (id, dir) -> correct, after regrade
def pass_answers():
    for p in sorted(RAW.rglob("*.jsonl")):
        d = p.parent.name
        if d in ("judge", "endorse", "aggregate"): continue
        rows = [json.loads(l) for l in p.read_text().split("\n") if l.strip()]
        tag = f"{d}/{p.name}"
        for r in rows:
            it = items.get(r["id"])
            if it is None: continue
            if "gold" in r and r["id"] in RELABEL: r["gold"] = RELABEL[r["id"]]
            if r.get("answer") not in (None, "") and r.get("correct") is not None:  # truncated rows (correct: null) stay missing
                setf(r, "correct", grade(it, r["answer"]), tag, r["id"])
            if d not in ("debate",) and not d.startswith("debate-homo") and not p.name.startswith(("samples-", "budget-")):
                matrix_correct[(r["id"], d)] = r.get("correct")
        if APPLY: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def pass_derived():
    for p in sorted(RAW.rglob("*.jsonl")):
        d = p.parent.name
        if d not in ("judge", "endorse", "aggregate"): continue
        rows = [json.loads(l) for l in p.read_text().split("\n") if l.strip()]
        tag = f"{d}/{p.name}"
        for r in rows:
            it = items.get(r["id"])
            if it is None: continue
            if "gold" in r and r["id"] in RELABEL: r["gold"] = RELABEL[r["id"]]
            if d == "judge":
                for c in r.get("candidates", []):
                    setf(c, "correct", bool(grade(it, c["answer"])), tag + ":cand", r["id"])
                setf(r, "any_correct", any(c["correct"] for c in r.get("candidates", [])), tag, r["id"])
                if r.get("pick_answer") not in (None, "") and r.get("pick_correct") is not None:
                    setf(r, "pick_correct", bool(grade(it, r["pick_answer"])), tag, r["id"])
                if r.get("own_answer") not in (None, "") and r.get("own_correct") is not None:
                    setf(r, "own_correct", bool(grade(it, r["own_answer"])), tag, r["id"])
            elif d == "endorse":
                setf(r, "cand_correct", bool(grade(it, r["cand_answer"])), tag, r["id"])
                oc = matrix_correct.get((r["id"], JUDGE_DIR.get(r["judge"], "")))
                if oc is not None and r.get("own_correct") is not None: setf(r, "own_correct", oc, tag, r["id"])
                if r.get("verdict") in ("CORRECT", "INCORRECT"):
                    setf(r, "right", (r["verdict"] == "CORRECT") == r["cand_correct"], tag, r["id"])
            elif d == "aggregate":
                for c in r.get("inputs", []):
                    setf(c, "correct", grade(it, c.get("answer")), tag + ":input", r["id"])
                setf(r, "any_input_correct", any(bool(c.get("correct")) for c in r.get("inputs", [])), tag, r["id"])
                if r.get("answer") not in (None, ""): setf(r, "correct", grade(it, r.get("answer")), tag, r["id"])
        if APPLY: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def build_matrix():
    cols = {}
    for d in sorted(RAW.iterdir()):
        if not d.is_dir() or d.name in ("judge", "endorse", "aggregate") or d.name.startswith("debate"): continue
        rows = {}
        for p in d.glob("*.jsonl"):
            if p.name.startswith(("samples-", "budget-")): continue
            for l in p.read_text().split("\n"):
                if not l.strip(): continue
                r = json.loads(l); rows[r["id"]] = r.get("correct")
        cols[d.name] = rows
    its = [json.loads(l) for fn in ("items.jsonl", "items-gpqa.jsonl") for l in (DATA / fn).open() if l.strip()]
    for it in its: it["layer"] = it.get("layer") or it.get("dataset") or "gpqa-diamond"
    out = ["item_id,dataset,gold," + ",".join(cols)]
    for it in its:
        out.append(",".join([it["id"], it["layer"], json.dumps(str(it["gold"]))] + ["" if cols[c].get(it["id"]) is None else str(int(cols[c][it["id"]])) for c in cols]))
    if APPLY: (HERE / "matrix-v2.csv").write_text("\n".join(out) + "\n")
    return len(its), list(cols)

pass_answers(); pass_derived(); n, cols = build_matrix()
if APPLY:
    for fn, keys in (("items.jsonl", [i for i in items if not i.startswith("gpqa")]), ("items-gpqa.jsonl", [i for i in items if i.startswith("gpqa")])):
        order = [json.loads(l)["id"] for l in (DATA / fn).open() if l.strip()]
        (DATA / fn).write_text("".join(json.dumps(items[i], ensure_ascii=False) + "\n" for i in order))
print(f"{'APPLIED' if APPLY else 'DRY RUN'}: matrix {n} items x {len(cols)} cols")
by_tag = collections.defaultdict(collections.Counter)
for (tag, key, ch), c in flips.items(): by_tag[tag][f"{key} {ch}"] += c
for tag in sorted(by_tag):
    print(f"{tag:45s} " + ", ".join(f"{k}: {v}" for k, v in sorted(by_tag[tag].items())))
sus = [(t, x) for t, v in detail.items() for x in v if (x[2] and not x[3]) or x[3] is None]
print("suspicious flips (True->False or ->None):", len(sus))
import collections as _c
print(_c.Counter((x[0] in RELABEL) for t, x in sus))
for t, x in sus:
    if x[0] not in RELABEL: print("  ", t, x)
json.dump({k: v for k, v in detail.items()}, (DATA / "regrade-log.json").open("w"), indent=0, ensure_ascii=False, default=str) if APPLY else None
