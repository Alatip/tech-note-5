"""Extra table for the note: matrix plurality by ANSWER IDENTITY (not correctness count), with answer-level
unanimity, from the raw matrix rows; and a per-layer summary of tokens per item per operator."""
import collections, itertools, json
from pathlib import Path
import stats as S

HERE = Path(__file__).resolve().parent; RAW = HERE / "raw"
MODELS = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512", "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]


def section_audit(layers):
    a = json.load((HERE / "label-audit.json").open()) if (HERE / "label-audit.json").exists() else {}
    out = ["## 0. Label audit (8 October 2026): gold labels and grading", "",
           "Every item on which all seven models (or all six debate agents) were scored wrong was re-solved (Claude Opus 5.5, spot-checked by hand). "
           "Items whose gold answer was wrong were relabelled and every stored row re-graded; ambiguous or ill-posed items are excluded from every table below. "
           "Olympiad expressions are now graded with the normalising grader (\\frac, \\sqrt, $…$, trailing '≈ …'), which the matrix stage had not used.", "",
           "| layer | relabelled (gold was wrong) | excluded (ambiguous / ill-posed) | kept although all models erred |", "|---|---|---|---|"]
    def lay(i): return {"mmlupro": "mmlu-pro", "gsmhard": "gsm-hard", "gpqa": "gpqa-diamond"}.get(i.split(":")[0], "olympiad")
    for layer in layers:
        rl = [i for i in a.get("relabels", {}) if lay(i) == layer]; ex = [i for i in a.get("excluded", {}) if lay(i) == layer]; kp = [i for i in a.get("kept_models_wrong", {}) if lay(i) == layer]
        out.append(f"| {layer} | {len(rl)} | {len(ex)} | {len(kp)} |")
    out += ["", "Details, with the reason for each decision, in label-audit.json; the re-grading script is regrade_all.py (no model calls)."]
    return out


def section_plurality(layers):
    out = ["## A0. Matrix plurality by answer identity (7 models, items answered by all 7)", "",
           "| layer | items | best single | plurality (answer identity) | plurality − best | unanimous | unanimous & wrong | any correct (ceiling) |", "|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        rows = collections.defaultdict(dict)
        for m in MODELS:
            f = RAW / m / f"{layer}.jsonl"
            if not f.exists(): continue
            for r in S.jl(f):
                if r.get("correct") is not None and r.get("answer"): rows[r["id"]][m] = r
        ids = [i for i, d in rows.items() if len(d) == len(MODELS)]
        if len(ids) < 20: continue
        best = max(sum(bool(rows[i][m]["correct"]) for i in ids) / len(ids) for m in MODELS)
        plu = una = unw = anyc = 0
        for i in ids:
            rs = [rows[i][m] for m in MODELS]
            ok = S.maj_correct(rs); plu += ok
            top, cnt = S.majority([r["answer"] for r in rs]); una += cnt == 7; unw += cnt == 7 and not ok
            anyc += any(r["correct"] for r in rs)
        out.append(f"| {layer} | {len(ids)} | {best:.3f} | {plu/len(ids):.3f} | {plu/len(ids)-best:+.3f} | {una} | {unw} | {anyc/len(ids):.3f} |")
    return out


def section_cost(layers):
    out = ["## G3. Tokens per item per operator (completion tokens incl. reasoning; DeepSeek where available; the same items in every row of a layer)", "",
           "| operator | layer | items | tokens per item | accuracy |", "|---|---|---|---|---|"]
    DS = RAW / "deepseek_deepseek-v3.2"
    for layer in layers:
        one = {r["id"]: r for r in S.jl(DS / f"{layer}.jsonl") if (r.get("tok") or {}).get("out") and r.get("correct") is not None} if (DS / f"{layer}.jsonl").exists() else {}
        by = collections.defaultdict(dict)
        if (DS / f"samples-{layer}.jsonl").exists():
            for r in S.jl(DS / f"samples-{layer}.jsonl"):
                if r.get("correct") is not None and (r.get("tok") or {}).get("out"): by[r["id"]][r["sample"]] = r
        s16 = {i: v for i, v in by.items() if all(k in v for k in range(16))}
        homo = collections.defaultdict(list); fam = collections.defaultdict(list)
        if (RAW / "debate-homo-plain" / f"{layer}.jsonl").exists():
            for r in S.jl(RAW / "debate-homo-plain" / f"{layer}.jsonl"):
                if (r.get("tok") or {}).get("out"): homo[r["id"]].append(r)
        if (RAW / "debate" / f"{layer}.jsonl").exists():
            for r in S.jl(RAW / "debate" / f"{layer}.jsonl"):
                if (r.get("tok") or {}).get("out"): fam[r["id"]].append(r)
        homo = {i: v for i, v in homo.items() if len(v) == 18}; fam = {i: v for i, v in fam.items() if len(v) == 18}
        ids = sorted(set(one) & set(s16) & set(homo) & set(fam))
        if len(ids) < 10: continue
        n = len(ids)
        out.append(f"| one DeepSeek answer, T = 0 | {layer} | {n} | {sum(one[i]['tok']['out'] for i in ids)/n:.0f} | {sum(bool(one[i]['correct']) for i in ids)/n:.3f} |")
        out.append(f"| 16 DeepSeek draws, plurality | {layer} | {n} | {sum(s16[i][k]['tok']['out'] for i in ids for k in range(16))/n:.0f} | {sum(S.maj_correct([s16[i][k] for k in range(16)]) for i in ids)/n:.3f} |")
        tok = sum(sum(r['tok']['out'] for r in homo[i]) + sum(s16[i][k]['tok']['out'] for k in range(6)) for i in ids) / n
        out.append(f"| 6 DeepSeek copies, 3 debate rounds (incl. 6 seed draws), plurality | {layer} | {n} | {tok:.0f} | {sum(S.maj_correct([r for r in homo[i] if r['round'] == 3]) for i in ids)/n:.3f} |")
        out.append(f"| 6 families, 3 debate rounds (rounds 1–3 only), plurality | {layer} | {n} | {sum(sum(r['tok']['out'] for r in fam[i]) for i in ids)/n:.0f} | {sum(S.maj_correct([r for r in fam[i] if r['round'] == 3]) for i in ids)/n:.3f} |")
    return out


def _lay(i): return {"mmlupro": "mmlu-pro", "gsmhard": "gsm-hard", "gpqa": "gpqa-diamond"}.get(i.split(":")[0], "olympiad")


def section_quoted(layers):
    """N. Numbers the note quotes that live in no other table: McNemar on the matrix, the all-wrong count with cut-off answers
    as missing votes, φ pair coverage, the audit sensitivity, the debate fixes by cut-off status and tie status, the lone-agent
    cases, and the cut-off rows of the homogeneous arms."""
    out = ["## N. Numbers quoted in the note that no other table holds", ""]
    # --- matrix: best vs plurality discordant items, φ pairs defined
    for layer in layers:
        raw = S.raw_matrix(layer)
        if len(raw) < 20: continue
        ids = sorted(raw)
        acc = {m: sum(bool(raw[i][m]["correct"]) for i in ids) / len(ids) for m in MODELS}
        best = max(acc, key=acc.get)
        b_not_p = p_not_b = 0
        for i in ids:
            pc = S.maj_correct([raw[i][m] for m in MODELS]); bc = bool(raw[i][best]["correct"])
            b_not_p += bc and not pc; p_not_b += pc and not bc
        pairs = sum(1 for a, b in itertools.combinations(MODELS, 2) if S.phi([bool(raw[i][a]["correct"]) for i in ids], [bool(raw[i][b]["correct"]) for i in ids]) is not None)
        out.append(f"- {layer}, {len(ids)} items all seven finished: best single ({best}) right and plurality wrong on {b_not_p} items, the reverse on {p_not_b}; exact McNemar p = {S.mcnemar(b_not_p, p_not_b):.3f}. φ defined on {pairs} of 21 pairs.")
    # --- all answering models wrong, items with at least five answers
    out.append("")
    tot_n = tot_w = 0; parts = []
    aud = json.load((HERE / "label-audit.json").open())
    audited = set(aud.get("relabels", {})) | set(aud.get("kept_models_wrong", {})) | set(aud.get("grader_errors_fixed_by_normaliser", {}))
    unaud = []
    for layer in layers:
        rows = collections.defaultdict(dict)
        for m in MODELS:
            f = RAW / m / f"{layer}.jsonl"
            if f.exists():
                for r in S.jl(f): rows[r["id"]][m] = r
        ids = [i for i, d in rows.items() if sum(S.present(r) for r in d.values()) >= 5]
        w = [i for i in ids if not any(r.get("correct") for r in rows[i].values() if S.present(r))]
        unaud += [i for i in w if i not in audited]
        parts.append(f"{layer} {len(w)} of {len(ids)}"); tot_n += len(ids); tot_w += len(w)
    out.append(f"- Every answering model wrong, a cut-off answer a missing vote, items with at least five answers: " + ", ".join(parts) + f"; {tot_w} of {tot_n} ({tot_w/tot_n:.1%}), {len(unaud)} of them outside the audit.")
    # --- audit sensitivity: all-wrong on the A0 items if the judgment-call relabels are reverted / exclusions restored
    out += ["", "### Audit sensitivity: the all-wrong rate on the items all seven finished, under weaker readings of the audit", "",
            "| layer | as audited | MMLU-Pro relabels reverted (gold as published) | plus the excluded items restored with their published gold |", "|---|---|---|---|"]
    rel_mm = {i for i in aud.get("relabels", {}) if _lay(i) == "mmlu-pro"}
    for layer in layers:
        raw = S.raw_matrix(layer)
        if len(raw) < 20: continue
        # excluded items are skipped by S.jl; read them back with their stored (published-gold) correctness
        exc = {}
        for m in MODELS:
            for line in (RAW / m / f"{layer}.jsonl").open():
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                if r["id"] in S.EXCLUDED and r.get("correct") is not None and r.get("answer"): exc.setdefault(r["id"], {})[m] = r
        exc = {i: d for i, d in exc.items() if len(d) == len(MODELS)}
        allw = sum(1 for i, d in raw.items() if not any(r["correct"] for r in d.values()))
        allw_rev = allw + sum(1 for i in raw if i in rel_mm)   # a relabelled item was all-wrong under its published gold by selection
        allw_exc = allw_rev + sum(1 for i, d in exc.items() if not any(r["correct"] for r in d.values()))
        out.append(f"| {layer} | {allw} / {len(raw)} | {allw_rev} / {len(raw)} | {allw_exc} / {len(raw) + len(exc)} |")
    out.append("")
    out.append("The GSM-Hard relabels are arithmetic (a program substituted large numbers into some quantities and not others) and the two olympiad corrections are grading; the nine MMLU-Pro relabels and the fifteen MMLU-Pro exclusions are judgments about knowledge questions, so the table shows what the ceiling looks like if a reader rejects them.")
    # --- debate: fixes by cut-off status, all-answered subset, ties
    items = S.load_debate(layers); A = S.DEBATE_AGENTS
    full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    if full:
        fixes = [i for i in full if not S.maj_correct([full[i][(a, 0)] for a in A]) and S.maj_correct([full[i][(a, 3)] for a in A])]
        cut = [i for i in fixes if any(not S.present(full[i][(a, 0)]) for a in A)]
        six = [i for i in full if all(S.present(full[i][(a, 0)]) for a in A)]
        m0 = sum(S.maj_correct([full[i][(a, 0)] for a in A]) for i in six) / len(six); m3 = sum(S.maj_correct([full[i][(a, 3)] for a in A]) for i in six) / len(six)
        tied = []; split = 0.0
        for i in full:
            rs = [full[i][(a, 0)] for a in A if S.present(full[i][(a, 0)])]
            cnt = collections.Counter(S.canon(r.get("answer")) for r in rs)
            if not cnt: continue
            top = max(cnt.values()); tops = [k for k, v in cnt.items() if v == top]
            corr = {S.canon(r.get("answer")) for r in rs if r.get("correct")}
            split += len([k for k in tops if k in corr]) / len(tops)
            if i in fixes and len(tops) > 1 and any(k in corr for k in tops): tied.append(i)
        out += ["", f"- Mixed debate, {len(full)} items: {len(fixes)} fixes, {len(cut)} of them on items with at least one cut-off round-0 row. On the {len(six)} items all six answered at round 0 the majority goes {m0:.3f} → {m3:.3f}. {len(tied)} of the fixes are items on which the correct answer was tied for the round-0 plurality and lost the tie-break; with ties split as fractional credit the round-0 vote is {split/len(full):.3f}."]
        uw = []
        for i in full:
            rs = [full[i][(a, 3)] for a in A if S.present(full[i][(a, 3)])]
            top, cnt = S.majority([r.get("answer") for r in rs])
            if len(rs) >= 2 and cnt == len(rs) and not any(r.get("correct") for r in rs): uw.append(i)
        out.append(f"- Unanimously wrong among the answering agents at round 3: {len(uw)} items ({', '.join(sorted(uw))}).")
    # --- homogeneous arms: cut-off rows
    parts = []
    for arm in ("plain", "selfonly", "pressure"):
        d = RAW / f"debate-homo-{arm}"
        if not d.exists(): continue
        n = t_ = 0
        for layer in layers:
            f = d / f"{layer}.jsonl"
            if f.exists():
                for r in S.jl(f): n += 1; t_ += r.get("correct") is None
        parts.append(f"{arm} {t_} of {n} rows (rounds 1–3)")
    if parts: out.append("- Homogeneous debate, cut-off rows: " + "; ".join(parts) + ".")
    out += section_fifth(layers)
    return out


def section_fifth(layers):
    """Numbers added in the fifth reading: the cut-off denominator, the seven-model tie order, the cut-off agent's share of the
    debate fixes, the synthesiser's all-wrong items, and the judge panel's wins split by solver ties."""
    out = ["", "### Fifth reading: cut-off denominators, the seven-model tie order, what the fixes and the panel wins are made of", ""]
    # --- cut-off vs missing denominators among the items with at least five answers
    n5 = ncut = nlt7 = aw_cut = 0
    for layer in layers:
        rows = collections.defaultdict(dict)
        for m in MODELS:
            f = RAW / m / f"{layer}.jsonl"
            if f.exists():
                for r in S.jl(f): rows[r["id"]][m] = r
        for i, d in rows.items():
            pr = [r for r in d.values() if S.present(r)]
            if len(pr) < 5: continue
            n5 += 1; cut = any(not S.present(r) for r in d.values()); ncut += cut; nlt7 += len(pr) < 7
            aw_cut += cut and not any(r.get("correct") for r in pr)
    out.append(f"- Of the {n5} items with at least five answers, {ncut} have a cut-off row and {nlt7} have fewer than seven answers (the difference is items with no Mistral row and no cut-off). Every answering model is wrong on {aw_cut} of the {ncut} cut-off items ({aw_cut/ncut:.1%}; {aw_cut} of {nlt7} is {aw_cut/nlt7:.1%}).")
    # --- seven-model plurality: ties and tie orders (the file order is what A0 and A2 use)
    six = ["deepseek_deepseek-v3.2", "z-ai_glm-5.3-flash", "mistralai_mistral-large-2512", "qwen_qwen3.7-plus", "claude_haiku", "claude_sonnet55"]
    orders = [("file order (haiku, opus, sonnet, deepseek, mistral, qwen, glm; the order tables A0 and A2 use)", MODELS), ("six-agent order, Opus last", six + ["claude_opus"]), ("six-agent order, Opus first", ["claude_opus"] + six)]
    def split(rs):
        cnt = collections.Counter(S.canon(r["answer"]) for r in rs); top = max(cnt.values()); tied = [k for k, v in cnt.items() if v == top]
        corr = {S.canon(r["answer"]): bool(r["correct"]) for r in rs}
        return sum(corr[k] for k in tied) / len(tied)
    for layer in layers:
        raw = S.raw_matrix(layer)
        if len(raw) < 20: continue
        ids = sorted(raw)
        acc = {m: sum(bool(raw[i][m]["correct"]) for i in ids) / len(ids) for m in MODELS}; best = max(acc, key=acc.get)
        ties = sum(1 for i in ids if (lambda c: sum(1 for v in c.values() if v == max(c.values())) > 1)(collections.Counter(S.canon(raw[i][m]["answer"]) for m in MODELS)))
        parts = []
        for name, o in orders:
            plu = sum(S.maj_correct([raw[i][m] for m in o]) for i in ids) / len(ids)
            bp = sum(1 for i in ids if raw[i][best]["correct"] and not S.maj_correct([raw[i][m] for m in o])); pb = sum(1 for i in ids if not raw[i][best]["correct"] and S.maj_correct([raw[i][m] for m in o]))
            parts.append(f"{plu:.3f} ({bp} vs {pb}, p = {S.mcnemar(bp, pb):.3f}) under the {name}")
        parts.append(f"{sum(split([raw[i][m] for m in MODELS]) for i in ids)/len(ids):.3f} with ties split")
        # triples under the same orders
        cols = {m: [bool(raw[i][m]["correct"]) for i in ids] for m in MODELS}
        tr = []
        for name, o in orders + [("ties split", None)]:
            gains = []; phis = []
            for sub in itertools.combinations(MODELS, 3):
                ph = [p for a, b in itertools.combinations(sub, 2) if (p := S.phi(cols[a], cols[b])) is not None]
                if not ph: continue
                mv = (sum(split([raw[i][m] for m in sub]) for i in ids) if o is None else sum(S.maj_correct([raw[i][m] for m in o if m in sub]) for i in ids)) / len(ids)
                gains.append(mv - max(sum(cols[m]) / len(ids) for m in sub)); phis.append(sum(ph) / len(ph))
            tr.append(f"{100*sum(gains)/len(gains):+.1f} pts, Spearman {S.spearman(gains, phis):+.2f}")
        out.append(f"- {layer}, {len(ids)} items: {ties} seven-model ties; plurality (best right vs plurality wrong, McNemar) " + "; ".join(parts) + ". Triple gain over the best member and Spearman with the triple's φ under the same four rules: " + "; ".join(tr) + ".")
    # --- debate: the cut-off agent's part in the fixes
    items = S.load_debate(layers); A = S.DEBATE_AGENTS
    full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    if full:
        def mj(i, rd): return S.maj_correct([full[i][(a, rd)] for a in A])
        fixes = [i for i in full if not mj(i, 0) and mj(i, 3)]
        cut = [i for i in fixes if any(not S.present(full[i][(a, 0)]) for a in A)]
        joined = 0
        for i in cut:
            top, _ = S.majority([full[i][(a, 3)].get("answer") for a in A if S.present(full[i][(a, 3)])])
            joined += any(not S.present(full[i][(a, 0)]) and S.present(full[i][(a, 3)]) and S.canon(full[i][(a, 3)].get("answer")) == top for a in A)
        w_cut = [i for i in full if not mj(i, 0) and any(not S.present(full[i][(a, 0)]) for a in A)]; w_six = [i for i in full if not mj(i, 0) and all(S.present(full[i][(a, 0)]) for a in A)]
        out.append(f"- Mixed debate: of the {len(cut)} fixes on items with a cut-off round-0 row, the formerly cut-off agent is in the winning round-3 majority on {joined}; on the other {len(cut)-joined} it was still cut off or wrong at round 3. Items with a wrong round-0 majority: {len(w_cut)} with a cut-off row, {sum(mj(i,3) for i in w_cut)} fixed; {len(w_six)} with six answers, {sum(mj(i,3) for i in w_six)} fixed.")
    # --- synthesiser: the items with no correct input
    nc = [r["id"] for layer in layers if (RAW / "aggregate" / f"synth-{layer}.jsonl").exists() for r in S.jl(RAW / "aggregate" / f"synth-{layer}.jsonl") if r.get("correct") is not None and not r["any_input_correct"]]
    if nc: out.append(f"- Synthesiser over the debate answers: the {len(nc)} items with no correct input are {', '.join(sorted(nc))}, the items on which the debate ends unanimously wrong (listed above as unaudited).")
    # --- judge panel: wins by solver tie, phi on named picks, the strict-majority control, the fallback rule
    rs = S.judge_prepare([r for layer in layers if (RAW / "judge" / f"{layer}.jsonl").exists() for r in S.jl(RAW / "judge" / f"{layer}.jsonl")])
    if not rs: return out
    by = collections.defaultdict(dict)
    for r in rs: by[r["id"]][r["judge"]] = r
    judges = sorted({r["judge"] for r in rs})
    cids = [i for i, d in by.items() if all(j in d for j in judges) and d[judges[0]]["_contested"]]
    def tie(i):
        v = []
        for a in A:
            for c in by[i][judges[0]]["_cands"]:
                if a in c["by"]: v.append(S.canon(c["answer"])); break
        mc = collections.Counter(v).most_common(); return len(mc) > 1 and mc[0][1] == mc[1][1]
    won = [i for i in cids if S.jm_item(by, judges, i) and not by[i][judges[0]]["_plur"]]; lost = [i for i in cids if by[i][judges[0]]["_plur"] and not S.jm_item(by, judges, i)]
    tied = [i for i in cids if tie(i)]; nt = [i for i in cids if not tie(i)]
    out.append(f"- Judge panel, {len(cids)} contested items: {len(tied)} have a solver tie; the fixed order picks a wrong answer on {sum(1 for i in tied if not by[i][judges[0]]['_plur'])} of them and the panel is right on {sum(S.jm_item(by, judges, i) for i in tied)}. Of the panel's {len(won)} wins, {sum(tie(i) for i in won)} are solver ties; of its {len(lost)} losses, {sum(tie(i) for i in lost)}. On the {len(nt)} contested items with no solver tie the solvers' plurality is {sum(by[i][judges[0]]['_plur'] for i in nt)/len(nt):.3f} and the panel {sum(S.jm_item(by, judges, i) for i in nt)/len(nt):.3f}.")
    cols = {j: [(bool(by[i][j]["pick_correct"]) if by[i][j]["_mapped"] else None) for i in cids] for j in judges}
    mp, npairs = S.pairwise_phi(cols)
    out.append(f"- Judges' φ of 'pick correct' over the picks they named (pairwise-complete, a cut-off or stray pick absent): {mp:.2f} on {npairs} pairs, n_eff {S.n_eff(5, mp):.2f}, against 0.16 and 3.07 when such a pick counts as wrong.")
    strict = []; fb = 0.0
    for i in cids:
        picks = [S.canon(by[i][j].get("pick_answer")) for j in judges]; cnt = collections.Counter(p for p in picks if p not in (None, ""))
        if not cnt: continue
        mc = cnt.most_common(); top, n = mc[0]; tt = [a for a, m in mc if m == n]
        if n > len(judges) / 2: strict.append(i)
        fb += by[i][judges[0]]["_plur"] if len(tt) > 1 else any(S.canon(by[i][j].get("pick_answer")) == top and by[i][j]["pick_correct"] for j in judges)
    out.append(f"- Strict majority (more than half of the five judges on one candidate, whatever the number that named a pick): {len(strict)} items, panel {sum(S.jm_item(by, judges, i) for i in strict)/len(strict):.3f}, solvers' plurality on the same items {sum(by[i][judges[0]]['_plur'] for i in strict)/len(strict):.3f}. A tied panel falling back to the solvers' plurality (the rule behind the whole-set 0.953): {fb/len(cids):.3f} on the contested items against the solvers' {sum(by[i][judges[0]]['_plur'] for i in cids)/len(cids):.3f}.")
    return out


def section_ties(layers):
    """T. Tie rules and within-item controls, added in the fourth revision. Every plurality in the note breaks ties by a fixed
    agent order; this section shows what the debate gain and the judge-panel lead look like under other tie rules, and gives
    the 'best agent on its own answered items' control on the same items as the majority it is compared with."""
    out = ["## T. Tie rules and within-item controls", ""]
    items = S.load_debate(layers); A = S.DEBATE_AGENTS
    full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    if not full: return out
    ids = sorted(full)
    acc0 = {a: sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in A}
    strong = sorted(A, key=lambda a: -acc0[a])
    def mj(rows, order):
        rows = [r for r in rows if S.present(r)]
        top, _ = S.majority([r.get("answer") for r in rows])
        if top is None: return 0.0
        return float(bool(next((r for r in rows if S.canon(r.get("answer")) == top), {}).get("correct")))
    def mj_split(rows):
        rows = [r for r in rows if S.present(r)]
        cnt = collections.Counter(S.canon(r.get("answer")) for r in rows if r.get("answer") not in (None, ""))
        if not cnt: return 0.0
        top = max(cnt.values()); tied = [k for k, v in cnt.items() if v == top]
        corr = {S.canon(r.get("answer")): bool(r.get("correct")) for r in rows}
        return sum(corr[k] for k in tied) / len(tied)
    rules = [("fixed order (" + ", ".join(a.split("/")[-1].split(":")[-1] for a in A) + ")", lambda rows: mj(rows, A)),
             ("ties split evenly", mj_split),
             ("strongest agent first, by round-0 accuracy with a cut-off scoring 0 (" + ", ".join(a.split("/")[-1].split(":")[-1] for a in strong) + ")", lambda rows: mj([r for a in strong for r in [dict(rows_by_agent(rows)[a])] if r], strong))]
    def rows_by_agent(rows): return {r["agent"]: r for r in rows}
    out += ["### Mixed debate, " + str(len(ids)) + " items: round-0 and round-3 majority under three tie rules", "",
            "| tie rule | r0 majority | r3 majority | gain | " + " | ".join(f"{l} r0 → r3" for l in layers) + " |", "|---|---|---|---|" + "---|" * len(layers)]
    for name, f in rules:
        def M(sub, rd): return sum(f([dict(full[i][(a, rd)], agent=a) for a in A]) for i in sub) / len(sub)
        per = []
        for layer in layers:
            li = [i for i in ids if full[i][(A[0], 1)]["layer"] == layer]
            per.append(f"{M(li,0):.3f} → {M(li,3):.3f}" if li else "–")
        out.append(f"| {name} | {M(ids,0):.3f} | {M(ids,3):.3f} | {M(ids,3)-M(ids,0):+.3f} | " + " | ".join(per) + " |")
    out += ["", "A cut-off row does not vote under any rule. The fixed order is the one every other table uses; it puts the two agents with the lowest accuracy on the hard layers first.", ""]
    # within-item control
    out += ["### Best agent on its own answered items, with the majority on the same items", "",
            "| layer | agent with the highest accuracy on the items it answered | items it answered | its accuracy | r0 majority on those items | r3 majority on those items | r3 − agent | r3 majority on all items (for comparison) |", "|---|---|---|---|---|---|---|---|"]
    for layer in layers:
        li = [i for i in ids if full[i][(A[0], 1)]["layer"] == layer]
        if not li: continue
        best = None
        for a in A:
            ans = [i for i in li if S.present(full[i][(a, 0)])]
            acc = sum(bool(full[i][(a, 0)]["correct"]) for i in ans) / len(ans)
            if best is None or acc > best[0]: best = (acc, a, ans)
        acc, a, ans = best
        m0 = sum(mj([full[i][(b, 0)] for b in A], A) for i in ans) / len(ans); m3 = sum(mj([full[i][(b, 3)] for b in A], A) for i in ans) / len(ans)
        m3all = sum(mj([full[i][(b, 3)] for b in A], A) for i in li) / len(li)
        out.append(f"| {layer} | {a} | {len(ans)} of {len(li)} | {acc:.3f} | {m0:.3f} | {m3:.3f} | {m3-acc:+.3f} | {m3all:.3f} |")
    out.append("")
    # judges
    rs = S.judge_prepare([r for layer in layers if (RAW / "judge" / f"{layer}.jsonl").exists() for r in S.jl(RAW / "judge" / f"{layer}.jsonl")])
    if not rs: return out
    by = collections.defaultdict(dict)
    for r in rs: by[r["id"]][r["judge"]] = r
    judges = sorted({r["judge"] for r in rs})
    cids = [i for i, d in by.items() if all(j in d for j in judges) and d[judges[0]]["_contested"]]
    allids = [i for i, d in by.items() if all(j in d for j in judges)]
    cont = set(cids)
    jacc = {j: sum(bool(by[i][j]["pick_correct"]) for i in cids) / len(cids) for j in judges}
    jstrong = sorted(judges, key=lambda j: -jacc[j])
    def votes(i):
        v = []
        for a in A:
            for c in by[i][judges[0]]["_cands"]:
                if a in c["by"]: v.append((a, S.canon(c["answer"]), bool(c["correct"]))); break
        return v
    def plur(i, order):
        v = votes(i)
        if not v: return 0.0
        d = {a: (ans, ok) for a, ans, ok in v}; cnt = collections.Counter(ans for _, ans, _ in v); top = max(cnt.values())
        if order is None:
            tied = [k for k, n in cnt.items() if n == top]; corr = {ans: ok for _, ans, ok in v}
            return sum(corr[k] for k in tied) / len(tied)
        first = next(d[a][0] for a in order if a in d and cnt[d[a][0]] == top)
        return float(any(ok for _, ans, ok in v if ans == first))
    def panel(i, order):
        picks = [(j, S.canon(by[i][j].get("pick_answer"))) for j in judges if by[i][j].get("pick_answer") not in (None, "")]
        if not picks: return 0.0
        cnt = collections.Counter(p for _, p in picks); top = max(cnt.values()); corr = {}
        for j, p in picks: corr[p] = corr.get(p, False) or bool(by[i][j]["pick_correct"])
        if order is None:
            tied = [k for k, n in cnt.items() if n == top]; return sum(corr[k] for k in tied) / len(tied)
        first = next(p for j in order for jj, p in picks if jj == j and cnt[p] == top)
        return float(corr[first])
    out += [f"### Judge panel against the solvers' plurality on the {len(cids)} contested items, under three tie rules (the same rule for both sides)", "",
            "| tie rule | solvers' plurality | panel | panel − solvers | items won / lost (McNemar p) | whole set ({} items): plurality | plurality + panel |".format(len(allids)), "|---|---|---|---|---|---|---|"]
    for name, so, jo in [("first in the fixed order (solvers: " + A[0].split("/")[-1] + "; judges: " + judges[0].split(":")[-1].split("/")[-1] + ")", A, judges),
                         ("ties split evenly", None, None),
                         ("strongest first (solvers: " + strong[0].split(":")[-1] + "; judges: " + jstrong[0].split(":")[-1] + ")", strong, jstrong)]:
        sp = sum(plur(i, so) for i in cids) / len(cids); pp = sum(panel(i, jo) for i in cids) / len(cids)
        if so is None: wl = "–"
        else:
            won = sum(1 for i in cids if panel(i, jo) and not plur(i, so)); lost = sum(1 for i in cids if plur(i, so) and not panel(i, jo)); wl = f"{won} / {lost} ({S.mcnemar(won, lost):.2f})"
        wp = sum(plur(i, so) for i in allids) / len(allids); wpp = sum((panel(i, jo) if i in cont else plur(i, so)) for i in allids) / len(allids)
        out.append(f"| {name} | {sp:.3f} | {pp:.3f} | {pp-sp:+.3f} | {wl} | {wp:.3f} | {wpp:.3f} |")
    out.append("")
    # whole set restricted to each agent's answered items
    out += ["### Whole set, restricted to the items each solver answered (the within-item form of the 'best solver on its answered items' control)", "",
            "| solver | items it answered | its accuracy | solvers' plurality on those items | plurality + panel (first-judge ties) | plurality + panel (tied panel falls back to the plurality) |", "|---|---|---|---|---|---|"]
    def panel_ok(i, ties):
        picks = [S.canon(by[i][j].get("pick_answer")) for j in judges]
        cnt = collections.Counter(p for p in picks if p not in (None, ""))
        if not cnt: return False
        mc = cnt.most_common(); top, n = mc[0]; tied = [a for a, m in mc if m == n]
        if ties == "abstain" and len(tied) > 1: return by[i][judges[0]]["_plur"]
        return any(S.canon(by[i][j].get("pick_answer")) == top and by[i][j]["pick_correct"] for j in judges)
    for a in A:
        sub = [i for i in allids if by[i][judges[0]]["_agent"][a] is not None]
        acc = sum(bool(by[i][judges[0]]["_agent"][a]) for i in sub) / len(sub)
        p = sum(by[i][judges[0]]["_plur"] for i in sub) / len(sub)
        pf = sum((panel_ok(i, "first") if i in cont else by[i][judges[0]]["_plur"]) for i in sub) / len(sub)
        pa = sum((panel_ok(i, "abstain") if i in cont else by[i][judges[0]]["_plur"]) for i in sub) / len(sub)
        out.append(f"| {a} | {len(sub)} of {len(allids)} | {acc:.3f} | {p:.3f} | {pf:.3f} | {pa:.3f} |")
    return out
