"""Axis tables from the traces corpus (library; run.py calls the sections)

A. matrix: per-layer accuracy per model, majority vote, best single, ceiling (1 - all wrong),
   mean pairwise phi, n_eff (Kish-style: n / (1 + (n-1)*phi)).
A'. samples (T=1, k per item): per model p, self-consistency MV@k, pass@k, within-model phi, n_eff_self.
B. debate: per round mean agent acc, majority, unanimity, items fixed/broken.
G. budget: accuracy and mean reasoning tokens per budget per model per layer.
Hard layers only unless --all. Rows torn by concurrent writers are skipped.
"""
import re, argparse, collections, csv, itertools, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
HARD = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]
MAIN = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512", "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]
DEBATE_AGENTS = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "qwen/qwen3.7-plus", "claude:haiku", "claude:sonnet55"]

# Truncation policy. A row with correct == None is a truncated answer (the model hit its completion cap, 8,192 tokens on
# OpenRouter, before writing a final line). False (default): such a row is MISSING in every count below (not an agent, not a
# vote, not a dissenter). True: the policy the first two versions of the note used by accident: the row is present and wrong.
TRUNC_AS_WRONG = False


def present(r):
    """Does this row count as an answer under the current truncation policy?"""
    return bool(r) and (TRUNC_AS_WRONG or r.get("correct") is not None)


def pairwise_phi(cols):
    """Mean pairwise φ over columns of 0/1/None; each pair is computed on the rows where both are not None."""
    ph = []
    for a, b in itertools.combinations(list(cols), 2):
        x = [(u, v) for u, v in zip(cols[a], cols[b]) if u is not None and v is not None]
        if len(x) < 5: continue
        q = phi([u for u, _ in x], [v for _, v in x])
        if q is not None: ph.append(q)
    return (sum(ph) / len(ph) if ph else None), len(ph)


def pairwise_tetra(cols):
    te = []
    for a, b in itertools.combinations(list(cols), 2):
        x = [(u, v) for u, v in zip(cols[a], cols[b]) if u is not None and v is not None]
        if len(x) < 5: continue
        t = tetra([u for u, _ in x], [v for _, v in x])
        if t is not None: te.append(t)
    return (sum(te) / len(te) if te else None), len(te)


def cval(r):
    """0/1 correctness of a row, or None when the row is a truncated answer and the policy treats it as missing."""
    if not present(r): return None
    return bool(r.get("correct"))


_AUDIT = json.load((HERE / "label-audit.json").open()) if (HERE / "label-audit.json").exists() else {}
EXCLUDED = set(_AUDIT.get("excluded", {}))   # items dropped after the label audit (ambiguous or ill-posed); see label-audit.json


def jl(path):
    for line in path.open():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("id") in EXCLUDED: continue
        yield r


def canon(a):
    """Canonical answer identity: numbers compared as numbers ('24', '24.00', '$24', '5,072,530' are one answer),
    simple LaTeX normalised; letters and hashes unchanged. Used for every plurality and unanimity count."""
    if a in (None, ""): return a
    t = str(a).strip().replace("$", "").replace(" ", "")
    t = re.sub(r"^\\\(|\\\)$", "", t); t = re.sub(r"(?<=\d),(?=\d{3})", "", t).rstrip(".")
    try: return repr(float(f"{float(t):.7g}"))   # 7 significant digits ≈ the grader's 1e-6 relative tolerance
    except ValueError: pass
    t = t.replace("\\dfrac", "\\frac"); t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
    t = t.replace("\\sqrt{", "√").replace("\\sqrt", "√").replace("{", "").replace("}", "").lower()
    return t


def phi(a, b):
    n = len(a); p1 = sum(a) / n; p2 = sum(b) / n
    if p1 in (0, 1) or p2 in (0, 1): return None
    p12 = sum(x and y for x, y in zip(a, b)) / n
    return (p12 - p1 * p2) / math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))


def n_eff(n, ph):
    return n / (1 + (n - 1) * ph) if ph is not None else None


def majority(answers):
    c = collections.Counter(canon(a) for a in answers if a not in (None, ""))
    if not c: return None, 0
    top, cnt = c.most_common(1)[0]
    return top, cnt


def maj_correct(rs):
    """Is the plurality answer of these rows correct? False when nobody answered. Truncated rows do not vote;
    ties go to the first row in the order given (the fixed agent order everywhere in this file)."""
    rs = [r for r in rs if present(r)]
    top, _ = majority([r.get("answer") for r in rs])
    if top is None: return False
    return bool(next((r for r in rs if canon(r.get("answer")) == top), {}).get("correct"))


def matrix_ids(layer):
    """Items of raw/<m>/<layer>.jsonl answered (not truncated, non-empty) by all seven main models: the A0 item set."""
    got = collections.defaultdict(set)
    for m in MAIN:
        f = RAW / m / f"{layer}.jsonl"
        if not f.exists(): return set()
        for r in jl(f):
            if r.get("correct") is not None and r.get("answer"): got[r["id"]].add(m)
    return {i for i, ms in got.items() if len(ms) == len(MAIN)}


def raw_matrix(layer):
    """item -> {model: row} for the items every main model answered (the A0 set), with answers, from the raw files."""
    rows = collections.defaultdict(dict)
    for m in MAIN:
        f = RAW / m / f"{layer}.jsonl"
        if not f.exists(): return {}
        for r in jl(f):
            if r.get("correct") is not None and r.get("answer"): rows[r["id"]][m] = r
    return {i: d for i, d in rows.items() if len(d) == len(MAIN)}


def spearman(x, y):
    """Spearman rank correlation with tie-averaged ranks (Pearson on ranks)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = ranks(x), ranks(y); n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((u - ma) ** 2 for u in a)); sb = math.sqrt(sum((u - mb) ** 2 for u in b))
    return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def matrix_rows():
    """matrix-v2.csv rows restricted to the per-layer files' item sets (the csv also folds in an older MMLU-Pro run)."""
    ids = {layer: matrix_ids(layer) for layer in HARD}
    return [r for r in csv.DictReader((HERE / "matrix-v2.csv").open()) if r["item_id"] not in EXCLUDED and (r["dataset"] not in ids or r["item_id"] in ids[r["dataset"]])]


def section_matrix(layers):
    rows = matrix_rows()
    models = [m for m in MAIN if m in rows[0]]
    out = ["## A. Matrix (one answer, T=0): accuracy, majority, ceiling, correlation", "",
           "| layer | n | " + " | ".join(models) + " | best | MV | ceiling | mean φ | n_eff |", "|" + "---|" * (len(models) + 7)]
    for layer in layers:
        rs = [r for r in rows if r["dataset"] == layer and all(r.get(m) not in ("", None) for m in models)]
        if not rs: continue
        acc = {m: sum(int(r[m]) for r in rs) / len(rs) for m in models}
        mv = 0; allw = 0
        for r in rs:
            v = [int(r[m]) for m in models]
            mv += sum(v) > len(v) / 2; allw += sum(v) == 0
        phis = [p for a, b in itertools.combinations(models, 2) if (p := phi([int(r[a]) for r in rs], [int(r[b]) for r in rs])) is not None]
        mp = sum(phis) / len(phis) if phis else None
        out.append(f"| {layer} | {len(rs)} | " + " | ".join(f"{acc[m]:.3f}" for m in models) + f" | {max(acc.values()):.3f} | {mv/len(rs):.3f} | {1-allw/len(rs):.3f} | {mp:.2f} | {n_eff(len(models), mp):.2f} |")
    out.append(""); out.append("MV = majority of the listed models (correctness only, not answer identity); ceiling = 1 − P(all wrong).")
    return out


def section_samples(layers):
    out = ["## A'. Samples (T=1, k per item): self-consistency vs single draw", "",
           "| model | layer | items | k | p (mean draw) | MV@k | pass@k | all-wrong | φ within | n_eff self | all items: n / MV@k / pass@k (cut-off draws absent) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(RAW.iterdir()):
        for layer in layers:
            f = d / f"samples-{layer}.jsonl"
            if not f.exists(): continue
            by = collections.defaultdict(dict); allby = collections.defaultdict(dict)
            for r in jl(f):
                allby[r["id"]][r["sample"]] = r
                if r.get("correct") is not None: by[r["id"]][r["sample"]] = r
            ks = [len(v) for v in by.values()]
            if not ks: continue
            k = 16 if sum(all(j in v for j in range(16)) for v in by.values()) >= 10 else 8
            full = {i: {j: v[j] for j in range(k)} for i, v in by.items() if all(j in v for j in range(k))}
            if len(full) < 10: continue
            p = sum(bool(r["correct"]) for v in full.values() for r in v.values()) / (k * len(full))
            mv = sum(bool(next(r for r in v.values() if canon(r.get("answer")) == majority([r.get("answer") for r in v.values()])[0])["correct"]) for v in full.values()) / len(full)
            passk = sum(any(r["correct"] for r in v.values()) for v in full.values()) / len(full)
            allw = sum(not any(r["correct"] for r in v.values()) for v in full.values()) / len(full)
            # within-model phi: mean over sample pairs
            phis = []
            for s1, s2 in itertools.combinations(range(k), 2):
                a = [bool(v[s1]["correct"]) for v in full.values() if s1 in v and s2 in v]
                b = [bool(v[s2]["correct"]) for v in full.values() if s1 in v and s2 in v]
                ph = phi(a, b)
                if ph is not None: phis.append(ph)
            mp = sum(phis) / len(phis) if phis else None
            # every item of the layer, the first k draws by index, a cut-off draw a missing vote (the note's policy)
            amv = apk = 0
            for i, v in allby.items():
                rs_ = [v[s] for s in range(k) if s in v]
                amv += maj_correct(rs_); apk += any(bool(r.get("correct")) for r in rs_)
            out.append(f"| {d.name} | {layer} | {len(full)} | {k} | {p:.3f} | {mv:.3f} | {passk:.3f} | {allw:.3f} | {mp:.2f} | {n_eff(k, mp):.2f} | {len(allby)} / {amv/len(allby):.3f} / {apk/len(allby):.3f} |")
    out += ["", "Items, k, p, MV@k, pass@k, φ: items with k complete draws (a model that truncated any of its first k draws on an item is not counted there). The last column keeps every item and scores a cut-off draw as a missing vote."]
    return out


def section_debate(layers):
    items = load_debate(layers)
    pol = "truncated answers counted as wrong" if TRUNC_AS_WRONG else "truncated answers are missing"
    out = [f"## B. Debate (3 synchronous rounds, round 0 = matrix answers; {pol})", ""]
    for agents, name in ((DEBATE_AGENTS, "6 agents"), ([a for a in DEBATE_AGENTS if "mistral" not in a], "5 agents (no Mistral)")):
        full = [i for i, d in items.items() if all((a, rd) in d for a in agents for rd in range(4))]
        if not full: continue
        out += [f"### {name}: {len(full)} items", "", "| round | agent acc | majority | unanimous | fixed | broken | truncated rows |", "|---|---|---|---|---|---|---|"]
        wrong0 = set()
        for rd in range(4):
            acc_n = acc_d = maj = una = fixed = broken = trunc = 0
            for i in full:
                rs = [items[i][(a, rd)] for a in agents]
                pr = [r for r in rs if present(r)]
                acc_n += sum(bool(r.get("correct")) for r in pr); acc_d += len(pr)
                trunc += sum(r.get("correct") is None for r in rs)
                ok = maj_correct(rs); maj += ok
                top, cnt = majority([r.get("answer") for r in rs if present(r)])
                una += len(pr) >= 2 and cnt == len(pr)
                if rd == 0 and not ok: wrong0.add(i)
                if rd and ok and i in wrong0: fixed += 1
                if rd and not ok and i not in wrong0: broken += 1
            n = len(full)
            out.append(f"| {rd} | {acc_n/max(acc_d,1):.3f} | {maj/n:.3f} | {una/n:.3f} | {fixed} | {broken} | {trunc} |")
        out.append("")
    out += ["Agent acc is over answered rows; majority is the plurality by answer identity of the answered rows; unanimous = every answered agent gave the same answer; fixed/broken are cumulative against round 0.", ""]
    return out


def section_personas(layers):
    """Axis D: error correlation across persona prompts vs within the plain model (same model, same items)."""
    base = RAW / "deepseek_deepseek-v3.2"
    pers = sorted(d for d in RAW.iterdir() if "__persona-" in d.name)
    if not pers: return []
    out = ["## D. Personas (DeepSeek, 4 role prompts x 4 draws): does a role buy independence?", "",
           "| layer | items | p plain | p personas (mean) | φ within plain (draws) | φ within a persona (draws) | φ across personas | n_eff 4 personas | MV plain@4 | MV 4 personas | across − within plain, bootstrap 95% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        data = {}; rows_ = {}
        for d in [base, *pers]:
            f = d / f"samples-{layer}.jsonl"
            if not f.exists(): continue
            by = collections.defaultdict(dict); byr = collections.defaultdict(dict)
            for r in jl(f):
                if r.get("correct") is not None and r["sample"] < 4: by[r["id"]][r["sample"]] = bool(r["correct"]); byr[r["id"]][r["sample"]] = r
            data[d.name] = {i: v for i, v in by.items() if len(v) == 4}; rows_[d.name] = byr
        names = [n for n in data if "persona" in n]
        if len(names) < 2 or base.name not in data: continue
        ids = set(data[base.name])
        for n in names: ids &= set(data[n])
        ids = sorted(ids)
        if len(ids) < 10: continue
        def mphi(cols):
            ph = [p for a, b in itertools.combinations(cols, 2) if (p := phi(a, b)) is not None]
            return sum(ph) / len(ph) if ph else None
        plain = [[data[base.name][i][s] for i in ids] for s in range(4)]
        within_p = [mphi([[data[n][i][s] for i in ids] for s in range(4)]) for n in names]
        within_p = [x for x in within_p if x is not None]
        def across_phi(sel):
            """Mean φ over pairs of different personas, every draw of one against every draw of the other."""
            ph = []
            for n1, n2 in itertools.combinations(names, 2):
                for s1 in range(4):
                    for s2 in range(4):
                        q = phi([data[n1][i][s1] for i in sel], [data[n2][i][s2] for i in sel])
                        if q is not None: ph.append(q)
            return sum(ph) / len(ph) if ph else None
        def within_phi(sel):
            ph = [q for s1, s2 in itertools.combinations(range(4), 2) if (q := phi([data[base.name][i][s1] for i in sel], [data[base.name][i][s2] for i in sel])) is not None]
            return sum(ph) / len(ph) if ph else None
        across = across_phi(ids)
        p_plain = sum(sum(c) for c in plain) / (4 * len(ids))
        p_pers = sum(data[n][i][s] for n in names for i in ids for s in range(4)) / (4 * len(names) * len(ids))
        mv_plain = sum(maj_correct([rows_[base.name][i][s] for s in range(4)]) for i in ids) / len(ids)
        mv_pers = sum(maj_correct([rows_[n][i][s] for n in names]) for i in ids for s in range(4)) / (4 * len(ids))
        wp = within_phi(ids); wpp = sum(within_p) / len(within_p) if within_p else None
        import random
        rng = random.Random(0); diffs = []; undefined = 0
        for _ in range(2000):
            sel = [rng.choice(ids) for _ in ids]
            a_, w_ = across_phi(sel), within_phi(sel)
            if a_ is None or w_ is None: undefined += 1; continue
            diffs.append(a_ - w_)
        diffs.sort(); lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        out.append(f"| {layer} | {len(ids)} | {p_plain:.3f} | {p_pers:.3f} | {wp:.2f} | {wpp:.2f} | {across:.2f} | {n_eff(len(names), across):.2f} | {mv_plain:.3f} | {mv_pers:.3f} | {across-wp:+.2f} [{lo:+.2f}, {hi:+.2f}] ({undefined} undefined) |")
    out += ["", "Across-persona φ: mean over pairs of different personas and over all draw pairs (16 per persona pair); within: pairs of draws of the same prompt. p and the pluralities (answer identity, first draw breaks ties) average over the four draws. Last column: across − within(plain) with a percentile bootstrap over items (2,000 resamples, seed 0); resamples on which φ is undefined on one side are dropped and counted. Equal φ ⇒ the role prompt adds no independence beyond temperature."]
    return out


def _ncand(r):
    """Number of distinct candidate answers after canonicalisation ('24' and '24.00' are one candidate)."""
    return len({canon(c["answer"]) for c in r.get("_cands", r.get("candidates", []))})


_MATRIX_PRESENT = {}


def _matrix_present():
    """(item, agent) -> matrix row for the six debate agents, under the current truncation policy (present() decides)."""
    key = TRUNC_AS_WRONG
    if key in _MATRIX_PRESENT: return _MATRIX_PRESENT[key]
    m = {}
    for a in DEBATE_AGENTS:
        for layer in HARD:
            f = RAW / a.replace("/", "_").replace(":", "_") / f"{layer}.jsonl"
            if f.exists():
                for r in jl(f): m[(r["id"], a)] = r
    _MATRIX_PRESENT[key] = m
    return m


def judge_prepare(rs):
    """Annotate judge rows in place. _cands: candidates whose proposer answered (a cut-off proposer's junk string is not a
    candidate); _contested: the distinct _cands disagree and at least one is correct and one is wrong; _parsed: the pick
    maps to a candidate or is NONE (a letter outside the list is not a pick); _mapped: the pick names a candidate;
    _plur: the solvers' plurality by answer identity over the proposers that answered, ties to the first agent in
    DEBATE_AGENTS order; _agent: per-agent matrix correctness (None = cut off)."""
    M = _matrix_present()
    for r in rs:
        i = r["id"]
        cands = []
        for c in r.get("candidates", []):
            by = [a for a in c.get("by", []) if present(M.get((i, a)))]
            if by: cands.append(dict(c, by=by))
        r["_cands"] = cands
        ids_ = {canon(c["answer"]) for c in cands}
        r["_contested"] = len(ids_) > 1 and any(c["correct"] for c in cands) and not all(c["correct"] for c in cands)
        r["_mapped"] = r.get("pick_answer") is not None
        r["_parsed"] = r["_mapped"] or r.get("pick") == "NONE"
        votes = []
        for a in DEBATE_AGENTS:
            for c in cands:
                if a in c["by"]: votes.append((canon(c["answer"]), bool(c["correct"]))); break
        cnt = collections.Counter(v for v, _ in votes)
        if cnt:
            top = max(cnt.values()); first = next(v for v, _ in votes if cnt[v] == top)
            r["_plur"] = any(ok for v, ok in votes if v == first)
        else: r["_plur"] = False
        r["_agent"] = {a: cval(M.get((i, a))) if M.get((i, a)) else None for a in DEBATE_AGENTS}
    return rs


def section_judge(layers):
    rs = judge_prepare([r for layer in layers if (RAW / "judge" / f"{layer}.jsonl").exists() for r in jl(RAW / "judge" / f"{layer}.jsonl")])
    if not rs: return []
    by = collections.defaultdict(list)
    for r in rs: by[r["judge"]].append(r)
    pol = "truncated answers counted as wrong" if TRUNC_AS_WRONG else "truncated answers are missing"
    out = [f"## C. Judge panel: each model picks among the matrix candidates (contested items = the answering proposers' candidates disagree by answer identity, at least one is correct and one is wrong; {pol})", "",
           "| judge | contested | pick acc (unparsed pick = wrong) | parsed picks | pick acc on parsed picks | own solve acc | own answered | picked own | said NONE when none correct |", "|---|---|---|---|---|---|---|---|---|"]
    items = {}
    L = [chr(65 + i) for i in range(10)]
    unparsed = collections.Counter()
    for j, v in sorted(by.items()):
        ac = [r for r in v if r["_contested"]]; nc = [r for r in v if not r["any_correct"]]
        for r in ac: items[r["id"]] = r
        parsed = [r for r in ac if r["_parsed"]]
        for r in ac:
            if not r["_parsed"]: unparsed[(j, "cut off" if r.get("finish") == "length" else "letter outside the list")] += 1
        ans = [r for r in ac if TRUNC_AS_WRONG or r.get("own_correct") is not None]
        out.append(f"| {j} | {len(ac)} | {sum(bool(r['pick_correct']) for r in ac)/max(len(ac),1):.2f} | {len(parsed)} | {sum(bool(r['pick_correct']) for r in parsed)/max(len(parsed),1):.2f} | {sum(bool(r['own_correct']) for r in ans)/max(len(ans),1):.2f} | {len(ans)} | {sum(r['picked_own'] for r in ac)/max(len(ac),1):.2f} | {sum(r['pick']=='NONE' for r in nc)}/{len(nc)} |")
    mv = sum(r["_plur"] for r in items.values())
    lay = collections.Counter(i.split(":")[0] for i in items)
    cut = sum(1 for r in items.values() if any(v is None for v in r["_agent"].values()))
    out += ["", f"Plurality of the solvers on the same {len(items)} contested items (ties to the first proposer in the fixed agent order): {mv/max(len(items),1):.3f}. Oracle selection = 1.00 by construction. Contested items by layer: " + ", ".join(f"{k} {n}" for k, n in sorted(lay.items())) + f". Items with at least one cut-off proposer: {cut} of {len(items)}. Unparsed picks: " + "; ".join(f"{j} {n} ({why})" for (j, why), n in sorted(unparsed.items())) + "."]
    return out


def section_budget(layers):
    out = ["## G. Budget forcing: accuracy vs visible reasoning budget", "",
           "| model | layer | budget | n | acc | mean think tok |", "|---|---|---|---|---|---|"]
    for d in sorted(RAW.iterdir()):
        for layer in layers:
            f = d / f"budget-{layer}.jsonl"
            if not f.exists(): continue
            by = collections.defaultdict(list)
            for r in jl(f):
                if r.get("correct") is not None: by[r["budget"]].append(r)
            for b in sorted(by):
                rs = by[b]
                tt = [r.get("reasoning_tokens") or (r.get("tok") or {}).get("out") or 0 for r in rs]
                out.append(f"| {d.name} | {layer} | {b} | {len(rs)} | {sum(bool(r['correct']) for r in rs)/len(rs):.3f} | {sum(tt)/len(tt):.0f} |")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); a = ap.parse_args()
    layers = HARD if not a.all else sorted({p.stem.replace("samples-", "").replace("budget-", "") for d in RAW.iterdir() for p in d.glob("*.jsonl")})
    md = ["# Traces corpus: axis tables", "", f"Generated by stats.py; layers: {', '.join(layers)}.", ""]
    md += section_matrix(layers) + [""] + section_samples(layers) + [""] + section_debate(layers) + section_judge(layers) + [""] + section_personas(layers) + [""] + section_budget(layers)
    (HERE / "STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))




# ---------- deeper sections requested by the verdict pass (all from existing raw data) ----------

def tetra(a, b):
    """Tetrachoric correlation, Pearson cos-pi approximation: r = cos(pi / (1 + sqrt(ad/bc))) on the 2x2 table."""
    n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum((not x) and y for x, y in zip(a, b))
    if n10 * n01 == 0 or n11 * n00 == 0: return None   # undefined (the cos-π estimate degenerates to ±1); dropped from averages
    return math.cos(math.pi / (1 + math.sqrt(n11 * n00 / (n10 * n01))))


def cp_interval(k, n, z=1.96):
    """Wilson interval as a cheap stand-in for Clopper-Pearson."""
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def icc1(mat):
    """One-way ICC(1) on an items x raters 0/1 matrix (list of rows)."""
    n = len(mat); k = len(mat[0])
    gm = sum(sum(r) for r in mat) / (n * k)
    msb = sum(k * (sum(r) / k - gm) ** 2 for r in mat) / (n - 1)
    msw = sum((x - sum(r) / k) ** 2 for r in mat for x in r) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) else None


def load_debate(layers):
    items = collections.defaultdict(dict)
    for layer in layers:
        f = RAW / "debate" / f"{layer}.jsonl"
        if not f.exists(): continue
        for r in jl(f): items[r["id"]][(r["agent"], r["round"])] = dict(r, layer=layer)
        for a in DEBATE_AGENTS:
            mf = RAW / a.replace("/", "_").replace(":", "_") / f"{layer}.jsonl"
            if mf.exists():
                for r in jl(mf):
                    if r["id"] in items: items[r["id"]][(a, 0)] = dict(r, layer=layer)
    return items


def section_debate_deep(layers):
    items = load_debate(layers); A = DEBATE_AGENTS
    full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    if not full: return []
    pol = "truncated answers counted as wrong" if TRUNC_AS_WRONG else "truncated answers are missing"
    out = [f"## B2. Debate, deeper cuts (6 agents, {len(full)} items complete in all rounds; {pol})", ""]
    out += ["| round | mean pairwise φ (pairwise-complete) | mean tetrachoric ρ (defined pairs) | ICC(1) (items with 6 answers) | n_eff (φ) | unanimous | unanimous & wrong | all answered wrong | items with 6 answers |", "|---|---|---|---|---|---|---|---|---|"]
    for rd in range(4):
        cols = {a: [cval(full[i][(a, rd)]) for i in full] for a in A}
        mp, _ = pairwise_phi(cols); mt, nt = pairwise_tetra(cols)
        mat = [[cols[a][j] for a in A] for j in range(len(full)) if all(cols[a][j] is not None for a in A)]
        una = unw = allw = 0
        for i in full:
            rs = [full[i][(a, rd)] for a in A]; pr = [r for r in rs if present(r)]
            top, cnt = majority([r.get("answer") for r in rs if present(r)])
            if len(pr) >= 2 and cnt == len(pr):
                una += 1; unw += not bool(next(r for r in rs if canon(r.get("answer")) == top).get("correct"))
            allw += bool(pr) and not any(r.get("correct") for r in pr)
        out.append(f"| {rd} | {mp:.2f} | {mt:.2f} ({nt}/15) | {icc1(mat):.2f} | {n_eff(len(A), mp):.2f} | {una} | {unw} | {allw} | {len(mat)} |")
    out += ["", "### Items every answering agent got wrong at round 0 (the 1 − β ceiling) and what debate did with them", "",
            "| layer | items | all answered wrong r0 | correct majority r3 among them | Wilson 95% | r0 majority | r3 majority | best single r0 (truncated = 0) | best single r0 (own answered) |", "|---|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        ids = [i for i in full if full[i][(A[0], 1)]["layer"] == layer]
        if not ids: continue
        aw = [i for i in ids if (pr := [full[i][(a, 0)] for a in A if present(full[i][(a, 0)])]) and not any(r.get("correct") for r in pr)]
        fx = sum(maj_correct([full[i][(a, 3)] for a in A]) for i in aw)
        def mj(rd): return sum(maj_correct([full[i][(a, rd)] for a in A]) for i in ids) / len(ids)
        best = max((sum(bool(full[i][(a, 0)].get("correct")) for i in ids if present(full[i][(a, 0)])) / max(1, sum(present(full[i][(a, 0)]) for i in ids))) for a in A)
        lo, hi = cp_interval(fx, len(aw))
        best0 = max(sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in A)
        out.append(f"| {layer} | {len(ids)} | {len(aw)} | {fx} | [{lo:.2f}, {hi:.2f}] | {mj(0):.3f} | {mj(3):.3f} | {best0:.3f} | {best:.3f} |")
    out += ["", "### Round-0 correct count among answering agents → round-3 majority (transition table)", "", "| correct at r0 (of answered) | items | of which all 6 answered | r0 majority correct | r3 majority correct | mean correct at r3 (answered) |", "|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for i in full: by[sum(bool(full[i][(a, 0)].get("correct")) for a in A if present(full[i][(a, 0)]))].append(i)
    for c in sorted(by):
        ids = by[c]
        def mjc(rd): return sum(maj_correct([full[i][(a, rd)] for a in A]) for i in ids) / len(ids)
        pr3 = [full[i][(a, 3)] for i in ids for a in A if present(full[i][(a, 3)])]
        mc = sum(bool(r.get("correct")) for r in pr3) / max(1, len(pr3))
        six = sum(all(present(full[i][(a, 0)]) for a in A) for i in ids)
        out.append(f"| {c} | {len(ids)} | {six} | {mjc(0):.2f} | {mjc(3):.2f} | {mc:.2f} |")
    out += ["", "### Flips between rounds (agent correct at round t and answering at t+1): P(wrong at t+1) by number of answering peers holding a different answer (answer identity)", "", "| peers disagreeing | cases | flipped to wrong |", "|---|---|---|"]
    fl = collections.defaultdict(lambda: [0, 0])
    keep = collections.defaultdict(lambda: [0, 0])
    for i in full:
        for rd in range(3):
            for a in A:
                r = full[i][(a, rd)]
                if not r.get("correct"): continue
                nxt = full[i][(a, rd + 1)]
                if not present(nxt): continue   # cut off at t+1: no answer, so neither kept nor abandoned
                peers = [o for o in A if o != a and present(full[i][(o, rd)])]
                opp = sum(1 for o in peers if canon(full[i][(o, rd)].get("answer")) != canon(r.get("answer")))
                fl[opp][0] += 1; fl[opp][1] += not nxt.get("correct")
                if opp == 5:
                    keep[a][0] += 1; keep[a][1] += bool(nxt.get("correct"))
    for k in sorted(fl): out.append(f"| {k} | {fl[k][0]} | {fl[k][1]/fl[k][0]:.2f} |")
    out += ["", "Per agent, when it alone was correct against 5 answering, disagreeing peers: kept the correct answer " + ", ".join(f"{a} {keep[a][1]}/{keep[a][0]}" for a in A if keep[a][0]) + "."]
    tk = collections.defaultdict(list)
    for i in full:
        for rd in range(1, 4):
            for a in A:
                t = (full[i][(a, rd)].get("tok") or {}).get("out")
                if t: tk[rd].append(t)
    if tk: out += ["", "Mean completion tokens per agent-turn: " + ", ".join(f"round {rd} {sum(v)/len(v):.0f}" for rd, v in sorted(tk.items())) + f"; 3 rounds × 6 agents ≈ {sum(sum(v)/len(v) for v in tk.values())*6:.0f} tokens per item on top of round 0."]
    return out


def section_matrix_deep(layers):
    rows = matrix_rows()
    models = [m for m in MAIN if m in rows[0]]
    out = ["## A2. Matrix, deeper cuts", "", "| layer | n | mean tetrachoric ρ | ρ range | wrong-count 95% quantile (of 7) | P(all wrong) observed | P(all wrong) if independent | MV gain over best: best 3-subset / worst / mean of 35 | Spearman(gain, φ̄) over 3-subsets |", "|---|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        rs = [r for r in rows if r["dataset"] == layer and all(r.get(m) not in ("", None) for m in models)]
        if len(rs) < 20: continue
        cols = {m: [bool(int(r[m])) for r in rs] for m in models}
        te = [t for a, b in itertools.combinations(models, 2) if (t := tetra(cols[a], cols[b])) is not None]
        wc = sorted(sum(not cols[m][j] for m in models) for j in range(len(rs)))
        q95 = wc[int(0.95 * (len(wc) - 1))]
        allw = sum(1 for j in range(len(rs)) if all(not cols[m][j] for m in models)) / len(rs)
        ind = 1
        for m in models: ind *= 1 - sum(cols[m]) / len(rs)
        raw = raw_matrix(layer); ids = [r["item_id"] for r in rs if r["item_id"] in raw]
        gains = []; phis = []
        for sub in itertools.combinations(models, 3):
            acc = [sum(cols[m]) / len(rs) for m in sub]
            ph = [p for a, b in itertools.combinations(sub, 2) if (p := phi(cols[a], cols[b])) is not None]
            if not ph: continue   # a triple with a constant member has no φ
            mv = sum(maj_correct([raw[i][m] for m in sub]) for i in ids) / len(ids)
            gains.append(mv - max(acc)); phis.append(sum(ph) / len(ph))
        n = len(gains)
        sp = spearman(gains, phis) if n > 2 else float("nan")
        out.append(f"| {layer} | {len(rs)} | {sum(te)/len(te):.2f} | {min(te):.2f}–{max(te):.2f} | {q95} | {allw:.3f} | {ind:.4f} | {max(gains):+.3f} / {min(gains):+.3f} / {sum(gains)/n:+.3f} ({n} triples) | {sp:+.2f} |")
    out += ["", "Independent all-wrong = ∏(1−acc_m); the observed/independent ratio is the correlation penalty. 3-subset gain is the plurality by answer identity of the three (ties to the first in the fixed order) minus its best member, over the triples on which φ̄ is defined; Spearman uses tie-averaged ranks. Tetrachoric ρ here and below is Pearson's cos-π approximation, biased upward when accuracies are high; read it as 'latent correlation is far above φ', not as a precise value."]
    return out


def section_samples_deep(layers):
    out = ["## A'2. Samples, deeper cuts: MV@k curve, ICC, Kish prediction", "",
           "| model | layer | items | ICC(1) | tetra ρ (draw pairs) | MV@1 | MV@4 | MV@8 | MV@16 | Kish-predicted MV@k (exact binomial at n_eff, interpolated) | pass@4 → pass@16 (observed) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(RAW.iterdir()):
        if "persona" in d.name: continue
        for layer in layers:
            f = d / f"samples-{layer}.jsonl"
            if not f.exists(): continue
            by = collections.defaultdict(dict)
            for r in jl(f):
                if r.get("correct") is not None: by[r["id"]][r["sample"]] = r
            k = 16 if sum(len(v) >= 16 for v in by.values()) >= 10 else 8
            full = {i: v for i, v in by.items() if all(s in v for s in range(k))}
            if len(full) < 10: continue
            mat = [[bool(v[s]["correct"]) for s in range(k)] for v in full.values()]
            te = [t for s1, s2 in itertools.combinations(range(k), 2) if (t := tetra([m[s1] for m in mat], [m[s2] for m in mat])) is not None]
            def mvk(kk):
                c = 0
                for v in full.values():
                    top, _ = majority([v[s].get("answer") for s in range(kk)])
                    c += bool(next(v[s] for s in range(kk) if canon(v[s].get("answer")) == top).get("correct"))
                return c / len(full)
            p = sum(sum(m) for m in mat) / (k * len(full))
            ph = [q for s1, s2 in itertools.combinations(range(k), 2) if (q := phi([m[s1] for m in mat], [m[s2] for m in mat])) is not None]
            ne = n_eff(k, sum(ph) / len(ph)) if ph else None
            # Kish: P(majority correct) of n_eff independent votes at accuracy p. n_eff is not an integer, so the exact
            # binomial majority (ties split) is taken at floor and ceil of n_eff and interpolated linearly. (v2 of the note
            # used a normal approximation, which at n < 3 is off by up to 4 points in the direction of p − 0.85; v1 rounded n_eff.)
            def pmaj_exact(n, p):
                return sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * (1.0 if 2 * c > n else 0.5 if 2 * c == n else 0.0) for c in range(n + 1))
            def pmaj(n, p):
                lo = math.floor(n); hi = lo + 1
                return pmaj_exact(lo, p) + (n - lo) * (pmaj_exact(hi, p) - pmaj_exact(lo, p))
            kish = pmaj(ne, p) if ne else None
            pk = lambda kk: sum(any(m[:kk]) for m in mat) / len(mat)
            tstr = f"{sum(te)/len(te):.2f}" if te else "–"
            out.append(f"| {d.name} | {layer} | {len(full)} | {icc1(mat):.2f} | {tstr} | {mvk(1):.3f} | {mvk(4):.3f} | {mvk(8):.3f} | {mvk(k):.3f} | {kish:.3f} | {pk(4):.3f} → {pk(k):.3f} |")
    out += ["", "Kish prediction = P(majority correct) of n_eff independent votes at the mean draw accuracy: exact binomial majority with ties split, at floor and ceil of n_eff, interpolated linearly. For 1 ≤ n_eff ≤ 2 this equals the mean draw accuracy itself (a single vote, or two votes with the tie split), so the prediction is 'voting buys nothing'. Compare with observed MV@k. (v1 rounded n_eff to an integer; v2 used a normal approximation, which is off by up to 4 points at n < 3.)"]
    return out


def section_judge_deep(layers):
    rs = judge_prepare([r for layer in layers if (RAW / "judge" / f"{layer}.jsonl").exists() for r in jl(RAW / "judge" / f"{layer}.jsonl")])
    if not rs: return []
    by = collections.defaultdict(dict)
    for r in rs: by[r["id"]][r["judge"]] = r
    judges = sorted({r["judge"] for r in rs})
    out = ["## C2. Judge panel, deeper cuts", "", "| judge | FPR: picked a wrong candidate, over the picks that named a candidate | self-preference: picked own when own wrong | picked own when own right | pick acc on items where own was wrong |", "|---|---|---|---|---|"]
    for j in judges:
        v = [d[j] for d in by.values() if j in d and d[j]["_contested"]]
        fpr = sum(1 for r in v if r["_mapped"] and not r["pick_correct"]) / max(sum(r["_mapped"] for r in v), 1)
        ow = [r for r in v if r.get("own_correct") is False]; orr = [r for r in v if r.get("own_correct")]
        out.append(f"| {j} | {fpr:.2f} | {sum(r['picked_own'] for r in ow)/max(len(ow),1):.2f} (n={len(ow)}) | {sum(r['picked_own'] for r in orr)/max(len(orr),1):.2f} (n={len(orr)}) | {sum(bool(r['pick_correct']) for r in ow)/max(len(ow),1):.2f} |")
    # judge-majority accuracy vs k, tetrachoric among judges on pick-correct
    ids = [i for i, d in by.items() if all(j in d for j in judges) and d[judges[0]]["_contested"]]
    cols = {j: [bool(by[i][j]["pick_correct"]) for i in ids] for j in judges}
    te = [t for a, b in itertools.combinations(judges, 2) if (t := tetra(cols[a], cols[b])) is not None]
    ph = [p for a, b in itertools.combinations(judges, 2) if (p := phi(cols[a], cols[b])) is not None]
    out += ["", f"On {len(ids)} contested items with all 5 judges: mean pairwise φ of 'pick correct' {sum(ph)/len(ph):.2f}, tetrachoric {sum(te)/len(te):.2f}, n_eff of 5 judges {n_eff(5, sum(ph)/len(ph)):.2f}.", "",
            "| judge-majority of k | k=1 (mean) | k=3 (mean over triples) | k=5 |", "|---|---|---|---|"]
    def jm(sub, ties="first"):
        """Plurality of the judges' picks. ties='first': tie broken by the first judge in the list (the rule used before
        8 Oct 2026); 'abstain': a tie counts as no answer; 'strict': only a pick shared by more than half the judges counts."""
        c = 0
        for i in ids:
            picks = [canon(by[i][j].get("pick_answer")) for j in sub]
            cnt = collections.Counter(p for p in picks if p not in (None, ""))
            if not cnt: continue
            mc = cnt.most_common(); top, n = mc[0]; tied = [a for a, m in mc if m == n]
            if ties == "strict" and n <= len(sub) / 2: continue
            if ties == "abstain" and len(tied) > 1: continue
            c += any(canon(by[i][j].get("pick_answer")) == top and by[i][j]["pick_correct"] for j in sub)
        return c / len(ids)
    def covered(sub, ties):
        n_ = 0
        for i in ids:
            picks = [canon(by[i][j].get("pick_answer")) for j in sub]
            cnt = collections.Counter(p for p in picks if p not in (None, ""))
            if not cnt: continue
            mc = cnt.most_common(); top, n = mc[0]; tied = [a for a, m in mc if m == n]
            if ties == "strict" and n <= len(sub) / 2: continue
            if ties == "abstain" and len(tied) > 1: continue
            n_ += 1
        return n_
    k1 = sum(sum(cols[j]) / len(ids) for j in judges) / len(judges)
    k3 = sum(jm(s) for s in itertools.combinations(judges, 3)) / 10
    out.append(f"| ties broken by first judge | {k1:.3f} | {k3:.3f} | {jm(judges):.3f} |")
    out.append(f"| ties count as wrong | {k1:.3f} | {sum(jm(s, 'abstain') for s in itertools.combinations(judges, 3))/10:.3f} | {jm(judges, 'abstain'):.3f} |")
    cs = covered(judges, 'strict')
    out.append(f"| strict majority only (> half the judges agree), else no answer | | | {jm(judges, 'strict') * len(ids) / max(cs, 1):.3f} on the {cs} of {len(ids)} items answered |")
    ties_n = sum(1 for i in ids if (lambda mc: len(mc) > 1 and mc[0][1] == mc[1][1])(collections.Counter(p for p in [canon(by[i][j].get("pick_answer")) for j in judges] if p).most_common()))
    small = collections.Counter(sum(1 for j in judges if by[i][j].get("pick_answer") not in (None, "")) for i in ids)
    sp = sum(by[i][judges[0]]["_plur"] for i in ids)
    sp_ab = 0
    for i in ids:
        r = by[i][judges[0]]; votes = []
        for a in DEBATE_AGENTS:
            for c in r["_cands"]:
                if a in c["by"]: votes.append((canon(c["answer"]), bool(c["correct"]))); break
        cnt = collections.Counter(v for v, _ in votes); mc = cnt.most_common()
        if len(mc) > 1 and mc[0][1] == mc[1][1]: continue
        sp_ab += any(ok for v, ok in votes if v == mc[0][0])
    disc = [(sum(1 for i in ids if jm_item(by, judges, i) and not by[i][judges[0]]["_plur"]), sum(1 for i in ids if by[i][judges[0]]["_plur"] and not jm_item(by, judges, i)))]
    out += ["", f"Items with a tie at the top of the five judges' picks: {ties_n} of {len(ids)}. Panels with fewer than five named picks: {sum(n for k, n in small.items() if k < 5)} of {len(ids)} (" + ", ".join(f"{n} with {k}" for k, n in sorted(small.items())) + ").",
            f"Solvers' plurality on the same {len(ids)} items, ties to the first proposer in the fixed agent order: {sp/len(ids):.3f}; a tie counting as wrong: {sp_ab/len(ids):.3f}. Panel (first-judge ties) against solvers' plurality: {disc[0][0]} items won, {disc[0][1]} lost, exact McNemar p = {mcnemar(disc[0][0], disc[0][1]):.2f}."]
    # whole-set pipeline: unanimous items keep the common answer, contested items go to the panel, none-correct items are wrong
    allids = [i for i, d in by.items() if all(j in d for j in judges)]
    def plur(i): return by[i][judges[0]]["_plur"]
    def panel_ok(i, ties):
        picks = [canon(by[i][j].get("pick_answer")) for j in judges]
        cnt = collections.Counter(p for p in picks if p not in (None, ""))
        if not cnt: return False
        mc = cnt.most_common(); top, n = mc[0]; tied = [a for a, m in mc if m == n]
        if ties == "abstain" and len(tied) > 1: return plur(i)  # fall back to the solvers' plurality
        return any(canon(by[i][j].get("pick_answer")) == top and by[i][j]["pick_correct"] for j in judges)
    contested = set(ids)
    rows_ = []
    for ties in ("first", "abstain"):
        acc = sum((panel_ok(i, ties) if i in contested else plur(i)) for i in allids) / len(allids)
        rows_.append((ties, acc))
    plu_all = sum(plur(i) for i in allids) / len(allids)
    acc0 = {a: sum(bool(by[i][judges[0]]["_agent"][a]) for i in allids) / len(allids) for a in DEBATE_AGENTS}
    accA = {a: (sum(bool(by[i][judges[0]]["_agent"][a]) for i in allids if by[i][judges[0]]["_agent"][a] is not None), sum(by[i][judges[0]]["_agent"][a] is not None for i in allids)) for a in DEBATE_AGENTS}
    b0 = max(acc0, key=acc0.get); bA = max(DEBATE_AGENTS, key=lambda a: accA[a][0] / max(accA[a][1], 1))
    subA = [i for i in allids if by[i][judges[0]]["_agent"][bA] is not None]
    out += ["", f"Whole-set accounting over the {len(allids)} items with all five judges (unanimous items keep their answer, contested items go to the panel, none-correct items are wrong): "
            f"solvers' plurality {plu_all:.3f}; plurality + panel {rows_[0][1]:.3f} (ties to first judge) / {rows_[1][1]:.3f} (ties fall back to the plurality); "
            f"best single solver of the six with a cut-off answer scoring 0: {b0} {acc0[b0]:.3f}; best single solver on the items it answered: {bA} {accA[bA][0]/max(accA[bA][1],1):.3f} ({accA[bA][1]} items), on which the solvers' plurality is {sum(plur(i) for i in subA)/len(subA):.3f} and plurality + panel {sum((panel_ok(i, 'first') if i in contested else plur(i)) for i in subA)/len(subA):.3f} / {sum((panel_ok(i, 'abstain') if i in contested else plur(i)) for i in subA)/len(subA):.3f} (section T has every solver)."]
    return out


def jm_item(by, judges, i):
    """Panel plurality on one item, ties to the first judge in the list; cut-off and NONE picks absent."""
    picks = [canon(by[i][j].get("pick_answer")) for j in judges]
    cnt = collections.Counter(p for p in picks if p not in (None, ""))
    if not cnt: return False
    top = cnt.most_common(1)[0][0]
    return any(canon(by[i][j].get("pick_answer")) == top and by[i][j]["pick_correct"] for j in judges)


def mcnemar(b, c):
    """Exact two-sided McNemar p on b vs c discordant items."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def section_tokens(layers):
    out = ["## G2. Tokens per stage (cost-matching material; completion tokens incl. reasoning, mean per call)", "", "| stage | model | layer | calls | mean out tok | mean think tok | acc |", "|---|---|---|---|---|---|---|"]
    for d in sorted(RAW.iterdir()):
        if d.name == "judge": continue
        for layer in layers:
            for stage, f in ((("debate", d / f"{layer}.jsonl"),) if d.name == "debate" else (("matrix", d / f"{layer}.jsonl"), ("samples", d / f"samples-{layer}.jsonl"))):
                if not f or not f.exists(): continue
                rs = [r for r in jl(f) if (r.get("tok") or {}).get("out")]
                if len(rs) < 10: continue
                acc = [bool(r["correct"]) for r in rs if r.get("correct") is not None]
                out.append(f"| {stage if d.name != 'debate' else 'debate r1-3'} | {d.name} | {layer} | {len(rs)} | {sum(r['tok']['out'] for r in rs)/len(rs):.0f} | {sum((r['tok'].get('think') or 0) for r in rs)/len(rs):.0f} | {sum(acc)/max(len(acc),1):.3f} |")
    return out


def main2():
    import sys
    layers = HARD
    md = ["# Traces corpus: axis tables", "", f"Generated by stats.py; layers: {', '.join(layers)}.", ""]
    md += section_matrix(layers) + [""] + section_matrix_deep(layers) + [""] + section_samples(layers) + [""] + section_samples_deep(layers) + [""]
    md += section_debate(layers) + section_debate_deep(layers) + [""] + section_judge(layers) + [""] + section_judge_deep(layers) + [""]
    md += section_personas(layers) + [""] + section_budget(layers) + [""] + section_tokens(layers)
    (HERE / "STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))




# ---------- offline analyses for the unresolved bridges (G1/G2 in runs/missing-runs-2026-10-07.md) ----------

def _norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def _norm_ppf(p):
    # Acklam's approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def onefactor_sim(accs, rho, n_sim=20000, seed=0):
    """One-factor Gaussian copula: latent z_i = sqrt(rho)*F + sqrt(1-rho)*e_i; correct iff z_i < Phi^-1(acc_i).
    Returns (P(all wrong), P(majority correct))."""
    import random
    rnd = random.Random(seed); th = [_norm_ppf(a) for a in accs]; n = len(accs)
    allw = maj = 0
    for _ in range(n_sim):
        F = rnd.gauss(0, 1)
        ok = [math.sqrt(rho) * F + math.sqrt(1 - rho) * rnd.gauss(0, 1) < t for t in th]
        allw += not any(ok); maj += sum(ok) > n / 2
    return allw / n_sim, maj / n_sim


def fit_rho_tetra(cols):
    te = [t for a, b in itertools.combinations(list(cols), 2) if (t := tetra(cols[a], cols[b])) is not None]
    return sum(te) / len(te) if te else None


def section_copula(layers):
    rows = matrix_rows()
    models = [m for m in MAIN if m in rows[0]]
    out = ["## A3. Copula check: can a one-factor latent correlation reproduce the all-wrong rate and the vote?", "",
           "| layer | n | mean acc | tetra ρ (cos-π) | observed all-wrong | copula all-wrong (ρ tetra) | ρ needed to match all-wrong | observed MV | copula MV (ρ tetra) | independent MV |", "|---|---|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        rs = [r for r in rows if r["dataset"] == layer and all(r.get(m) not in ("", None) for m in models)]
        if len(rs) < 20: continue
        cols = {m: [bool(int(r[m])) for r in rs] for m in models}
        accs = [sum(cols[m]) / len(rs) for m in models]
        obs_aw = sum(1 for j in range(len(rs)) if all(not cols[m][j] for m in models)) / len(rs)
        obs_mv = sum(1 for j in range(len(rs)) if sum(cols[m][j] for m in models) > len(models) / 2) / len(rs)
        rho = fit_rho_tetra(cols)
        if rho is None: continue
        rho_c = min(max(rho, 0.0), 0.995)
        aw_t, mv_t = onefactor_sim(accs, rho_c)
        _, mv_0 = onefactor_sim(accs, 0.0)
        # rho needed: scan
        best = None
        for r_ in [i / 100 for i in range(0, 100, 5)]:
            aw, _ = onefactor_sim(accs, r_, n_sim=6000, seed=1)
            if best is None or abs(aw - obs_aw) < abs(best[1] - obs_aw): best = (r_, aw)
        out.append(f"| {layer} | {len(rs)} | {sum(accs)/len(accs):.3f} | {rho:.2f} | {obs_aw:.3f} | {aw_t:.3f} | {best[0]:.2f} | {obs_mv:.3f} | {mv_t:.3f} | {mv_0:.3f} |")
    out += ["", "One-factor Gaussian copula with per-model accuracies as thresholds. If the ρ needed to match the all-wrong rate is close to the tetrachoric estimate, a single latent 'item difficulty' factor explains the shared errors."]
    return out


def section_betabin(layers):
    """Two-point mixture on per-item correct counts of the k=16 samples: share of 'hard-state' items (p≈0) vs a Binomial easy state."""
    out = ["## A'3. Hard-state mass from the k=16 samples (two-component fit: items are either 'solvable at rate q' or 'unsolvable')", "",
           "| model | layer | items | items with 0/16 correct | items with 16/16 | fitted hard mass w | fitted q (easy state) | predicted MV@16 | observed MV@16 | Good–Toulmin pass@16 from first 4 | observed pass@16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(RAW.iterdir()):
        if "persona" in d.name: continue
        for layer in layers:
            f = d / f"samples-{layer}.jsonl"
            if not f.exists(): continue
            by = collections.defaultdict(dict)
            for r in jl(f):
                if r.get("correct") is not None: by[r["id"]][r["sample"]] = r
            full = {i: v for i, v in by.items() if all(s in v for s in range(16))}
            if len(full) < 10: continue
            cnt = [sum(bool(v[s]["correct"]) for s in range(16)) for v in full.values()]
            zeros = sum(c == 0 for c in cnt); full16 = sum(c == 16 for c in cnt)
            # EM-ish grid fit: w in [0,1], q in (0,1); likelihood of counts under w*delta0 + (1-w)*Binom(16,q)
            from math import comb, log
            best = None
            for w in [i / 100 for i in range(0, 100)]:
                for q in [i / 100 for i in range(1, 100)]:
                    ll = 0
                    for c in cnt:
                        pb = comb(16, c) * q ** c * (1 - q) ** (16 - c)
                        ll += log(max(w * (c == 0) + (1 - w) * pb, 1e-300))
                    if best is None or ll > best[0]: best = (ll, w, q)
            _, w, q = best
            pm = (1 - w) * sum(comb(16, j) * q ** j * (1 - q) ** (16 - j) for j in range(9, 17))
            obs_mv = 0
            for v in full.values():
                top, _ = majority([v[s].get("answer") for s in range(16)])
                obs_mv += bool(next(v[s] for s in range(16) if canon(v[s].get("answer")) == top).get("correct"))
            obs_mv /= len(full)
            # Good-Toulmin style extrapolation of pass@k from the first 4 draws (Chao-style lower bound via f1,f2 on per-item first-success)
            n4 = [sum(bool(v[s]["correct"]) for s in range(4)) for v in full.values()]
            p4 = sum(c > 0 for c in n4) / len(full)
            # per-item p estimate from 4 draws, predicted pass@16 = 1 - E[(1-p)^16] with p ~ Beta posterior (Jeffreys)
            pred16 = 0
            for c in n4:
                a_, b_ = c + 0.5, 4 - c + 0.5
                # E[(1-p)^16] under Beta(a,b) = B(a, b+16)/B(a,b)
                num = math.lgamma(a_ + b_) + math.lgamma(b_ + 16) - math.lgamma(b_) - math.lgamma(a_ + b_ + 16)
                pred16 += 1 - math.exp(num)
            pred16 /= len(full)
            obs16 = sum(c > 0 for c in cnt) / len(full)
            out.append(f"| {d.name} | {layer} | {len(full)} | {zeros} | {full16} | {w:.2f} | {q:.2f} | {pm:.3f} | {obs_mv:.3f} | {pred16:.3f} (pass@4 {p4:.3f}) | {obs16:.3f} |")
    out += ["", "Hard mass w = items the model essentially never solves at T=1; MV@k cannot exceed 1−w. The pass@16 forecast uses a Beta(c+½, 4−c+½) posterior per item from the first 4 draws."]
    return out


def section_debate_offline(layers):
    items = load_debate(layers); A = DEBATE_AGENTS
    full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    if not full: return []
    pol = "truncated answers counted as wrong" if TRUNC_AS_WRONG else "truncated answers are missing"
    out = [f"## B3. Debate, offline cuts: agent subsets, difficulty, stopping rules, per-layer gain vs n_eff ({pol})", ""]
    # per layer: n_eff at r0, gain r3-majority minus best single, r3 majority minus r0 majority
    out += ["| layer | items | r0 mean φ (pairwise-complete) | r0 n_eff | r0 majority | r3 majority | best single r0 (a truncated answer scores 0) | gain vs that | best single r0 on its own answered items | gain vs that, r3 majority on the same items | gain vs r0 MV | r0 ICC (6-answer items) | r3 ICC (6-answer items) | items with 6 answers r0 / r3 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def mj(ids, rd):
        m = 0
        for i in ids:
            m += maj_correct([full[i][(a, rd)] for a in A])
        return m / len(ids)
    for layer in layers:
        ids = [i for i in full if full[i][(A[0], 1)]["layer"] == layer]
        if len(ids) < 10: continue
        cols = {a: [cval(full[i][(a, 0)]) for i in ids] for a in A}
        mp, _ = pairwise_phi(cols)
        bestA = max(A, key=lambda a: sum(c for c in cols[a] if c) / max(1, sum(c is not None for c in cols[a])))
        best = sum(c for c in cols[bestA] if c) / max(1, sum(c is not None for c in cols[bestA]))
        bestA_ids = [i for i, c in zip(ids, cols[bestA]) if c is not None]   # the majority is compared on the same items
        best0 = max(sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in A)
        mat0 = [[cols[a][j] for a in A] for j in range(len(ids)) if all(cols[a][j] is not None for a in A)]
        cols3 = {a: [cval(full[i][(a, 3)]) for i in ids] for a in A}
        mat3 = [[cols3[a][j] for a in A] for j in range(len(ids)) if all(cols3[a][j] is not None for a in A)]
        f2 = lambda m: f"{icc1(m):.2f}" if len(m) >= 5 and icc1(m) is not None else "–"
        out.append(f"| {layer} | {len(ids)} | {mp:.2f} | {n_eff(6, mp):.2f} | {mj(ids,0):.3f} | {mj(ids,3):.3f} | {best0:.3f} | {mj(ids,3)-best0:+.3f} | {best:.3f} ({len(bestA_ids)} items) | {mj(bestA_ids,3)-best:+.3f} | {mj(ids,3)-mj(ids,0):+.3f} | {f2(mat0)} | {f2(mat3)} | {len(mat0)} / {len(mat3)} |")
    # agent subsets k=2..6: r0 majority vs r3 majority (mean over subsets)
    out += ["", "### Agent subsets (mean over all subsets of size k): majority accuracy at round 0 and round 3", "", "| k | subsets | r0 majority | r3 majority | r3 − r0 |", "|---|---|---|---|---|"]
    ids = list(full)
    for k in range(2, 7):
        subs = list(itertools.combinations(A, k)); m0 = m3 = 0
        for sub in subs:
            for rd, acc in ((0, 'm0'), (3, 'm3')):
                c = 0
                for i in ids:
                    c += maj_correct([full[i][(a, rd)] for a in sub])
                if rd == 0: m0 += c / len(ids)
                else: m3 += c / len(ids)
        out.append(f"| {k} | {len(subs)} | {m0/len(subs):.3f} | {m3/len(subs):.3f} | {(m3-m0)/len(subs):+.3f} |")
    # stopping rules: stop at first unanimous round; accuracy and rounds used
    out += ["", "### Stopping rule: stop at the first unanimous round (else take round 3)", ""]
    acc = rounds_used = 0; by_stop = collections.Counter(); wrong_by_stop = collections.Counter()
    for i in ids:
        for rd in range(0, 4):
            rs = [full[i][(a, rd)] for a in A]; top, cnt = majority([r.get("answer") for r in rs if present(r)]); npr = sum(present(r) for r in rs)
            if (npr >= 2 and cnt == npr) or rd == 3:
                ok = maj_correct(rs)
                acc += ok; rounds_used += rd; by_stop[rd] += 1; wrong_by_stop[rd] += not ok; break
    out.append(f"Accuracy {acc/len(ids):.3f} using {rounds_used/len(ids):.2f} rounds on average (vs 3). Stops: " + ", ".join(f"round {rd}: {by_stop[rd]} items, {wrong_by_stop[rd]} wrong" for rd in sorted(by_stop)) + ".")
    # difficulty bins from samples (DeepSeek k=16 p_i) joined to debate outcome
    p_i = {}
    for layer in layers:
        f = RAW / "deepseek_deepseek-v3.2" / f"samples-{layer}.jsonl"
        if f.exists():
            by = collections.defaultdict(list)
            for r in jl(f):
                if r.get("correct") is not None: by[r["id"]].append(bool(r["correct"]))
            for i, v in by.items():
                if len(v) >= 8: p_i[i] = sum(v) / len(v)
    out += ["", "### Debate outcome by item difficulty (DeepSeek T=1 solve rate p_i from the samples)", "", "| p_i bin | items | r0 majority | r3 majority | fixed | broken |", "|---|---|---|---|---|---|"]
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in bins:
        sel = [i for i in ids if i in p_i and lo <= p_i[i] < hi]
        if not sel: continue
        fx = br = 0
        for i in sel:
            o0 = maj_correct([full[i][(a, 0)] for a in A]); o3 = maj_correct([full[i][(a, 3)] for a in A])
            fx += (not o0) and o3; br += o0 and not o3
        out.append(f"| [{lo:.2f}, {hi:.2f}) | {len(sel)} | {mj(sel,0):.3f} | {mj(sel,3):.3f} | {fx} | {br} |")
    # self-vs-peer disagreement as error predictor (AUROC) for DeepSeek
    out += ["", "### Error prediction for DeepSeek's matrix answer: peer disagreement vs own-sample disagreement (AUROC)", ""]
    def auroc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
        if not pos or not neg: return None
        return sum((p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg) / (len(pos) * len(neg))
    tgt = "deepseek/deepseek-v3.2"; labels = []; peer = []; peerf = []; sonnet = []; own = {3: [], 5: [], 15: []}
    byS = collections.defaultdict(dict)
    for layer in layers:
        f = RAW / "deepseek_deepseek-v3.2" / f"samples-{layer}.jsonl"
        if f.exists():
            for r in jl(f): byS[r["id"]][r["sample"]] = r
    for i in ids:
        r = full[i][(tgt, 0)]
        if i not in byS or len(byS[i]) < 3: continue
        if not present(r): continue
        labels.append(not bool(r.get("correct")))
        peers = [a for a in A if a != tgt and present(full[i][(a, 0)])]
        dis = sum(1 for a in peers if canon(full[i][(a, 0)].get("answer")) != canon(r.get("answer")))
        peer.append(dis); peerf.append(dis / max(1, len(peers)))
        sr = full[i][("claude:sonnet55", 0)]
        sonnet.append(int(present(sr) and canon(sr.get("answer")) != canon(r.get("answer"))))
        for k in own:   # own draws: share of the first k draws that answered and disagree; a cut-off draw is absent
            ds = [byS[i][s] for s in range(k) if s in byS[i] and present(byS[i][s])]
            own[k].append(sum(1 for d in ds if canon(d.get("answer")) != canon(r.get("answer"))) / max(1, len(ds)))
    if labels:
        import random
        rng = random.Random(0); n = len(labels); d5 = []; dS = []
        for _ in range(2000):
            sel = [rng.randrange(n) for _ in range(n)]
            L = [labels[j] for j in sel]
            a15 = auroc([own[15][j] for j in sel], L); a5 = auroc([peer[j] for j in sel], L); aS = auroc([sonnet[j] for j in sel], L)
            if None in (a15, a5, aS): continue
            d5.append(a5 - a15); dS.append(aS - a15)
        d5.sort(); dS.sort()
        ci = lambda d: f"[{d[int(0.025*len(d))]:+.3f}, {d[int(0.975*len(d))-1]:+.3f}]"
        out.append(f"{len(labels)} items; error rate {sum(labels)/len(labels):.3f}. AUROC from the number of disagreeing answering peers (of up to 5 other models): {auroc(peer, labels):.3f} (as a fraction of answering peers: {auroc(peerf, labels):.3f}); from Sonnet alone: {auroc(sonnet, labels):.3f}; "
                   f"from the share of disagreeing own T=1 draws among the first 3 / 5 / 15 that answered: {auroc(own[3], labels):.3f} / {auroc(own[5], labels):.3f} / {auroc(own[15], labels):.3f}. "
                   f"Five peers − fifteen own draws: {auroc(peer, labels)-auroc(own[15], labels):+.3f}, percentile bootstrap over items (2,000, seed 0) {ci(d5)}; Sonnet alone − fifteen own draws: {auroc(sonnet, labels)-auroc(own[15], labels):+.3f} {ci(dS)}.")
    return out


def main3():
    layers = HARD
    md = (HERE / "STATS.md").read_text().rstrip("\n").split("\n")
    md += ["", *section_copula(layers), "", *section_betabin(layers), "", *section_debate_offline(layers)]
    (HERE / "STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-60:]))




# ---------- wave 2 sections ----------

def section_endorse(layers):
    rs = [r for layer in layers if (RAW / "endorse" / f"{layer}.jsonl").exists() for r in jl(RAW / "endorse" / f"{layer}.jsonl") if r.get("verdict")]
    if not rs: return []
    out = ["## C3. Endorse: each judge says CORRECT/INCORRECT for every candidate (contested items + one decoy per item)", "",
           "| judge | verdicts | acc | FPR (accepted a wrong cand) | FNR (rejected a right cand) | accepted decoy | accepted own wrong | accepted others' wrong | rejected own right |", "|---|---|---|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in rs: by[r["judge"]].append(r)
    for j, v in sorted(by.items()):
        wrong = [r for r in v if not r["cand_correct"]]; right = [r for r in v if r["cand_correct"]]
        dec = [r for r in wrong if r["decoy"]]; ow = [r for r in wrong if r["is_own"]]; otw = [r for r in wrong if not r["is_own"] and not r["decoy"]]
        orr = [r for r in right if r["is_own"]]
        f = lambda sel, cond: f"{sum(cond(r) for r in sel)/len(sel):.2f} (n={len(sel)})" if sel else "–"
        out.append(f"| {j} | {len(v)} | {sum(bool(r['right']) for r in v)/len(v):.2f} | {f(wrong, lambda r: r['verdict']=='CORRECT')} | {f(right, lambda r: r['verdict']=='INCORRECT')} | {f(dec, lambda r: r['verdict']=='CORRECT')} | {f(ow, lambda r: r['verdict']=='CORRECT')} | {f(otw, lambda r: r['verdict']=='CORRECT')} | {f(orr, lambda r: r['verdict']=='INCORRECT')} |")
    # committee: candidate accepted by >= q of 5 judges; precision/recall over candidates with all 5 verdicts
    # duplicates by format ('24' / '24.00', '475,501,268,359' / '475501268359') are one candidate. A judge that saw an answer in
    # several formats and ruled differently is scored by the stated rule: 'any' = accepted if it accepted any format (the
    # lenient reading, what the committee rows below use); 'all' = accepted only if it accepted every format.
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs: groups[(r["id"], canon(r["cand_answer"]))][r["judge"]].append(r)
    conflicts = sum(1 for g in groups.values() for v in g.values() if len({r["verdict"] for r in v}) > 1)
    multi = sum(1 for g in groups.values() for v in g.values() if len(v) > 1)
    def merged(rule):
        full = {}
        for k, g in groups.items():
            if len(g) < 5: continue
            full[k] = {j: {"cand_correct": any(r["cand_correct"] for r in v), "accept": (any if rule == "any" else all)(r["verdict"] == "CORRECT" for r in v)} for j, v in g.items()}
        return full
    full = merged("any")
    out += ["", f"Committee over {len(full)} candidates judged by all 5: accept if ≥q judges say CORRECT. {multi} (candidate, judge) pairs were presented in more than one format and {conflicts} of them got conflicting verdicts; a judge counts as accepting when it accepted any format.", "", "| q | accepted | precision | recall of correct cands | wrong cands accepted | if a judge must accept every format: wrong accepted / recall |", "|---|---|---|---|---|---|"]
    strict = merged("all")
    for q in range(1, 6):
        acc = [(k, v) for k, v in full.items() if sum(r["accept"] for r in v.values()) >= q]
        tp = sum(next(iter(v.values()))["cand_correct"] for k, v in acc); ncor = sum(next(iter(v.values()))["cand_correct"] for v in full.values())
        accs = [(k, v) for k, v in strict.items() if sum(r["accept"] for r in v.values()) >= q]
        tps = sum(next(iter(v.values()))["cand_correct"] for k, v in accs)
        out.append(f"| {q} | {len(acc)} | {tp/max(len(acc),1):.2f} | {tp/max(ncor,1):.2f} | {len(acc)-tp} | {len(accs)-tps} / {tps/max(ncor,1):.2f} |")
    # judge error correlation on wrong candidates (false accepts)
    wrongc = {k: v for k, v in full.items() if not next(iter(v.values()))["cand_correct"]}
    judges = sorted(by)
    cols = {j: [wrongc[k][j]["accept"] for k in wrongc] for j in judges}
    ph = [p for a, b in itertools.combinations(judges, 2) if (p := phi(cols[a], cols[b])) is not None]
    if ph: out += ["", f"False-accept φ between judges over {len(wrongc)} wrong candidates: mean {sum(ph)/len(ph):.2f}, n_eff of 5 judges {n_eff(5, sum(ph)/len(ph)):.2f}."]
    return out


def section_aggregate(layers):
    out = ["## B4/C4. Aggregators: one DeepSeek call reads several answers and writes one final", "",
           "| mode | inputs | items | input majority correct | aggregator correct | any input correct | aggregator correct when NO input was correct | novel answers (not among inputs) | novel & correct |", "|---|---|---|---|---|---|---|---|---|"]
    for mode in ("synth", "moa"):
        rs = [r for layer in layers if (RAW / "aggregate" / f"{mode}-{layer}.jsonl").exists() for r in jl(RAW / "aggregate" / f"{mode}-{layer}.jsonl") if r.get("correct") is not None]
        if not rs: continue
        mv = 0
        for r in rs:
            top, _ = majority([x["answer"] for x in r["inputs"] if x.get("correct") is not None])
            mv += any(canon(x["answer"]) == top and x["correct"] for x in r["inputs"])
        none = [r for r in rs if not r["any_input_correct"]]
        nov = [r for r in rs if r["novel"]]
        out.append(f"| {mode} | {'6 debate round-3 answers' if mode=='synth' else '5 own T=1 samples'} | {len(rs)} | {mv/len(rs):.3f} | {sum(bool(r['correct']) for r in rs)/len(rs):.3f} | {sum(r['any_input_correct'] for r in rs)/len(rs):.3f} | {sum(bool(r['correct']) for r in none)}/{len(none)} | {len(nov)} | {sum(bool(r['correct']) for r in nov)} |")
    return out


def section_homodebate(layers):
    pol = "truncated answers counted as wrong" if TRUNC_AS_WRONG else "truncated answers are missing"
    out = [f"## B5. Homogeneous debate: 6 DeepSeek copies (round 0 = its own T=1 samples), 3 rounds, three arms ({pol})", ""]
    out += ["| arm | items complete | r0 agent acc | r0 majority | r3 agent acc | r3 majority | r0 unanimous | r3 unanimous | unanimous&wrong r0→r3 | all-wrong r0 | fixed | broken | r0 φ | r3 φ |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("plain", "selfonly", "pressure"):
        d = RAW / f"debate-homo-{arm}"
        if not d.exists(): continue
        items = collections.defaultdict(dict)
        for layer in layers:
            f = d / f"{layer}.jsonl"
            if not f.exists(): continue
            for r in jl(f): items[r["id"]][(r["agent"], r["round"])] = r
            sf = RAW / "deepseek_deepseek-v3.2" / f"samples-{layer}.jsonl"
            for r in jl(sf):
                if r["id"] in items and r["sample"] < 6: items[r["id"]][(f"deepseek/deepseek-v3.2#{r['sample']}", 0)] = r
        A = [f"deepseek/deepseek-v3.2#{i}" for i in range(6)]
        full = [i for i, dd in items.items() if all((a, rd) in dd for a in A for rd in range(4))]
        if len(full) < 10: continue
        def stat(rd):
            acc_n = acc_d = maj = una = unw = allw = 0
            for i in full:
                rs = [items[i][(a, rd)] for a in A]; pr = [r for r in rs if present(r)]
                acc_n += sum(bool(r.get("correct")) for r in pr); acc_d += len(pr); ok = maj_correct(rs); maj += ok
                top, cnt = majority([r.get("answer") for r in rs if present(r)]); u = len(pr) >= 2 and cnt == len(pr); una += u; unw += (u and not ok); allw += bool(pr) and not any(r.get("correct") for r in pr)
            cols = {a: [cval(items[i][(a, rd)]) for i in full] for a in A}
            mp, _ = pairwise_phi(cols)
            return acc_n / max(acc_d, 1), maj / len(full), una, unw, allw, (mp if mp is not None else float('nan'))
        a0, m0, u0, w0, aw0, p0 = stat(0); a3, m3, u3, w3, aw3, p3 = stat(3)
        fixed = sum(1 for i in full if not maj_correct([items[i][(a, 0)] for a in A]) and maj_correct([items[i][(a, 3)] for a in A]))
        broken = sum(1 for i in full if maj_correct([items[i][(a, 0)] for a in A]) and not maj_correct([items[i][(a, 3)] for a in A]))
        out.append(f"| {arm} | {len(full)} | {a0:.3f} | {m0:.3f} | {a3:.3f} | {m3:.3f} | {u0} | {u3} | {w0}→{w3} | {aw0} | {fixed} | {broken} | {p0:.2f} | {p3:.2f} |")
    # the three arms on their common item set
    common = None; arms_ = {}
    for arm in ("plain", "selfonly", "pressure"):
        d = RAW / f"debate-homo-{arm}"
        if not d.exists(): continue
        items = collections.defaultdict(dict)
        for layer in layers:
            f = d / f"{layer}.jsonl"
            if not f.exists(): continue
            for r in jl(f): items[r["id"]][(r["agent"], r["round"])] = r
            for r in jl(RAW / "deepseek_deepseek-v3.2" / f"samples-{layer}.jsonl"):
                if r["id"] in items and r["sample"] < 6: items[r["id"]][(f"deepseek/deepseek-v3.2#{r['sample']}", 0)] = r
        A = [f"deepseek/deepseek-v3.2#{i}" for i in range(6)]
        full = {i for i, dd in items.items() if all((a, rd) in dd for a in A for rd in range(4))}
        common = full if common is None else common & full
        arms_[arm] = items
    if common:
        A = [f"deepseek/deepseek-v3.2#{i}" for i in range(6)]
        m0 = sum(maj_correct([arms_["plain"][i][(a, 0)] for a in A]) for i in common) / len(common)
        parts = []
        for arm, items in arms_.items():
            m3 = sum(maj_correct([items[i][(a, 3)] for a in A]) for i in common) / len(common)
            fx = sum(1 for i in common if not maj_correct([items[i][(a, 0)] for a in A]) and maj_correct([items[i][(a, 3)] for a in A]))
            br = sum(1 for i in common if maj_correct([items[i][(a, 0)] for a in A]) and not maj_correct([items[i][(a, 3)] for a in A]))
            parts.append(f"{arm} {m3:.3f} ({m3-m0:+.3f}; fixed {fx} / broken {br})")
        out += ["", f"On the {len(common)} items all three arms completed: round-0 majority {m0:.3f}; round-3 majority " + ", ".join(parts) + "."]
    out += ["", "selfonly = each copy reconsiders its own answer without seeing peers (controls for 'thinking again'); pressure = told to defer to the majority. Compare with the mixed-family debate in B/B2."]
    return out


def main4():
    layers = HARD
    md = (HERE / "STATS.md").read_text().rstrip("\n").split("\n")
    md += ["", *section_endorse(layers), "", *section_aggregate(layers), "", *section_homodebate(layers)]
    (HERE / "STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-45:]))

