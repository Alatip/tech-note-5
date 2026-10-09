# tech-note-5 — Where the Swarm Pays

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23241005.svg)](https://doi.org/10.5281/zenodo.23241005) — the note. Code and data: archived release v1.0 on Zenodo (DOI added after the release).

Companion data and code for the technical note **"Where the Swarm Pays"**
(alatip.github.io/where-the-swarm-pays/): one corpus of answer traces over seven
reasoning-model families and 240 hard problems, built so that most questions about
multiplying LLM agents (voting, resampling, debate, roles, synthesis, judging) can be
asked of the same data. Revised 8 October 2026 four times: after a label audit (`label-audit.json`: on the
items where all seven models were scored wrong, the gold label or the grader was at fault about four
times in five) and after a truncation audit (an answer cut off by the 8,192-token completion cap of the
OpenRouter models had been counted as a wrong vote in the debate and judge tables; it is now a missing
vote, and the old tables are kept under "legacy" in `results/tables.md`) and after a reading of the code against the note (tie rules, judge coverage, the flip table, the certify committee, the role correlation; every number the note quotes is now in `results/tables.md`, section N included) and after a fourth reading that recomputed the
controls from the rows (section T: the debate gain and the judge-panel lead under three tie rules, 4.9 → 2.2 points and +8 → −1;
the 'best agent on its answered items' control taken on the same items as the majority; Mistral's 40 missing matrix rows) and after a fifth reading that recomputed the quoted counts (the cut-off denominator, the seven-model tie order, the cut-off agent's share of the debate fixes, the panel's wins split by solver ties; section N, last block). On the answer nothing leaves
the box between the best member and the "someone was right" ceiling; most of what debate appeared to
add was a cut-off agent getting to answer; on verification a panel of judges picking among candidates
lands within a point of the best single solver when a cut-off answer scores 0.

Order of reading:

1. `DESIGN.md` — what was collected, how, with which models, prompts and graders;
   what is deliberately missing.
2. `label-audit.json` — the item-by-item audit (relabels, exclusions, reasons) and `regrade_all.py`,
   the script that applied it to every row (no model calls).
3. `run.py` — rebuilds every table from `raw/` (standard library only, under a minute).
4. `RESULTS.md` — the findings, one paragraph per table, with the caveats.
5. `results/tables.md` — every number the note quotes, and about twenty tables it
   does not.

## Run

```
python3 run.py          # writes results/tables.md; no network, no dependencies
```

## Data

`raw/<model>/<layer>.jsonl` — matrix rows (one answer per model, T = 0);
`raw/<model>/samples-<layer>.jsonl` — 16 draws at T = 1 (Haiku: 8 on olympiad and GPQA);
`raw/<model>/budget-<layer>.jsonl` — budget-forced answers (visible reasoning cut at B tokens);
`raw/<model>__persona-<role>/samples-<layer>.jsonl` — DeepSeek under role prompts;
`raw/debate/<layer>.jsonl` — mixed-family debate, rounds 1–3 (round 0 = matrix rows);
`raw/debate-homo-<arm>/<layer>.jsonl` — six DeepSeek copies, arms plain / selfonly / pressure
(round 0 = that model's samples 0–5);
`raw/judge/<layer>.jsonl` — five judges pick among the matrix candidates;
`raw/endorse/<layer>.jsonl` — five judges certify each candidate separately, plus one decoy;
`raw/aggregate/{synth,moa}-<layer>.jsonl` — DeepSeek synthesises one answer from six debate
transcripts or five of its own draws; `matrix-v2.csv` — the 0/1 correctness matrix.

Each row keeps the parsed answer, its correctness, the provider that served the call and
the token triple (prompt / completion / reasoning). **Model texts are not included.** On 102 rows
(GLM answers cut off mid-reasoning, a few refusals, and the judge/endorse rows that quote them) the
parsed "answer" was a fragment of model text; it is stored as `<text:…>`, a digest that keeps answer
identity (two rows with the same fragment share a token) and nothing else. MATH-500 answers, which
are LaTeX expressions, are kept as written.
GPQA Diamond answers are replaced by random tokens drawn per item and per option and no question text or gold
label is stored anywhere (the GPQA gold column of `matrix-v2.csv` is blank, and the GPQA entries of
`label-audit.json` carry tokens and no solution content), per the dataset's terms; every statistic in
the note needs only correctness and answer identity, which the tokens preserve. (The first release used
an unsalted digest of the option letter, which was invertible; the second drew one token per letter across items, which kept the letter up to a permutation; both were replaced.)

Truncation policy: a row with `correct: null` is an answer cut off by the completion cap. `stats.py`
treats it as missing (`TRUNC_AS_WRONG = False`): not a vote, not a dissenter, not in φ / ICC /
unanimity; an agent's accuracy is over the rows it answered; a single-agent control scores it 0.
`run.py` also prints every debate and judge table under the old policy (`TRUNC_AS_WRONG = True`) in
a "legacy" section. `regrade_all.py` is the script that applied the audit to every row (no model
calls); it is included for the record and does not run from this repository alone: it needs the
collection repository's grader and its `items-gpqa.jsonl` with the GPQA gold tokens, which are withheld.

Layers: MMLU-Pro, GPQA Diamond, GSM-Hard, olympiad (AIME 2024/2025 + HMMT 2025), 60 items
each in the debate/samples/judge stages; the matrix also holds the saturated layers
(MATH-500, GSM8K, CommonsenseQA/StrategyQA, Knights & Knaves) that the note excludes.

Models (`raw/` directory names): `claude_opus`, `claude_sonnet55`, `claude_haiku`
(Claude Opus 5.5, Sonnet 5.5, Haiku 4.5 via the Claude Code CLI, provider logged as
`claude-code-cli`, reasoning on, temperature not controllable), `deepseek_deepseek-v3.2`, `z-ai_glm-5.3-flash`, `qwen_qwen3.7-plus`,
`mistralai_mistral-large-2512` (OpenRouter, provider logged per row), plus partial runs of
Codex, Kimi K2.5 and free-tier models that the note does not use.

Collection scripts, graders and the discovery pipeline that proposed the hypotheses live
in a separate repository (to be released with a later note). Inference cost ≈ $50 on
OpenRouter plus subscription calls for the Claude models, 6–7 October 2026.

## Licence and citation

Everything here is released under [CC BY 4.0](LICENSE): use it freely, with attribution.
Please cite the note and this repository:

> Latipov, A. (2026). *Where the Swarm Pays*. Zenodo. https://doi.org/10.5281/zenodo.23241005
> Latipov, A. (2026). *tech-note-5: companion data and code for "Where the Swarm Pays"* (v1.0). Zenodo / GitHub, https://github.com/Alatip/tech-note-5
