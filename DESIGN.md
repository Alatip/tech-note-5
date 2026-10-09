# Design of the traces corpus

Built 6–7 October 2026 to serve hypotheses extracted from ~1,000 agent-swarm papers by a
literature-based-discovery pipeline. The design rule: collect once, in a form that lets
many hypotheses be read against the same rows; store every answer, never only aggregates.

## Items

240 problems with a single checkable answer, 60 per layer, taken as the first 60 of a
stratified draw (seed 0):

| layer | source | answer type | grader |
|---|---|---|---|
| mmlu-pro | MMLU-Pro, stratified over categories | letter | exact letter; 9 gold labels corrected and 15 ambiguous items excluded (label-audit.json) |
| gpqa-diamond | GPQA Diamond | letter | exact letter; 1 gold corrected, 1 contested item excluded |
| gsm-hard | GSM-Hard (GSM8K with large numbers) | number | numeric with tolerance; 11 of 60 gold labels wrong and 6 items ill-posed after substitution (label-audit.json) |
| olympiad | AIME 2024, AIME 2025 (the 60 items of the debate / samples / judge stages); HMMT Feb 2025 in the matrix only | number / expression | normalised match (\frac, \sqrt, $…$, trailing ≈), LLM arbiter (Opus, cached) on disagreement; the matrix stage was first graded numerically and re-graded on 8 Oct 2026 |

A first pass over 1,576 items including MATH-500, GSM8K, CommonsenseQA/StrategyQA and
Knights & Knaves led to their being dropped as knowledge or near-saturated layers by judgment (the matrix does not bear out an
'every model above 0.95' criterion: CommonsenseQA is 0.79–0.93, Mistral is 0.66 on Knights & Knaves); they stay in the
matrix but not in any table of the note.

## Prompts

System prompt for all answering stages: *"Answer the question. Think briefly if needed,
then give the final answer on the last line exactly as: FINAL: <answer>"*. Reasoning was
left on for every model (providers ignore reasoning caps, so the budget stage uses
forcing: visible chain-of-thought truncated at B tokens, then a second call that must
answer at once). Truncated answers (`finish == "length"`, the 8,192-token completion cap of the
OpenRouter models) are stored with `correct: null` and treated as missing (see README); GLM and Qwen
hit the cap on a fifth to two fifths of GPQA Diamond and about half of the olympiad items. Mistral has no matrix row at all
on 32 GPQA Diamond, 6 olympiad and 2 MMLU-Pro items (the run logs do not record why); 18 GPQA Diamond and 4 olympiad items
that every other model finished are missing from the seven-finished tables for that reason.

Debate system prompt: *"You are one of several agents answering the same question. You
will see the other agents' latest answers. Reconsider carefully: use them if they are
right, ignore them if they are wrong. Then give YOUR final answer…"*. Each agent sees, for
every other agent, its last FINAL answer and the last 1,200 characters of its reasoning,
anonymised as Agent A, B, …; rounds are synchronous; round 0 is the matrix answer.

Homogeneous arms (six DeepSeek copies, round 0 = its own T = 1 draws 0–5):
`plain` = the debate prompt; `selfonly` = each copy sees only its own previous answer and
reasoning ("reconsider your previous answer carefully on your own"); `pressure` = the debate
prompt with "the other agents are usually right, so you should defer to the majority
unless you are completely certain they are wrong".

Personas (DeepSeek, 4 draws each): skeptic, formal mathematician, patient teacher,
pragmatic engineer, as one-sentence system-prompt prefixes.

Judge panel (`judge`): on every item, the distinct FINAL answers of the six debate agents'
matrix rows, each with one 700-character reasoning excerpt, shuffled by a seed from the
item id, labelled Candidate A, B, …; the judge must reply `PICK: <letter>` or `PICK: NONE`.
Judges that solved from scratch instead were mapped onto a candidate when their FINAL
matched one (`via_final`). Contested items = the candidates of the proposers that answered disagree by answer identity
(numbers to 7 significant digits) and at least one is correct and one wrong (75 after the audit; a cut-off
proposer's partial text was shown to the judges as a candidate on about ten items and is not counted as one);
none-correct items (2 after the audit) are kept for the abstention count. A pick is unparsed when the call was
cut off before its `PICK:` line (GLM 29, Qwen 20, DeepSeek 7) or named a letter outside the candidate list
(DeepSeek 10, Sonnet 6, Haiku 2, GLM 2). The solvers' plurality breaks ties by the first proposer in the fixed
agent order (deepseek, glm, mistral, qwen, haiku, sonnet); the seven-model matrix plurality (A0, A2) uses the order of the raw/ directories (haiku, opus, sonnet, deepseek, mistral, qwen, glm) and section N gives its range over orders; section T
of `results/tables.md` gives the debate gain and the panel lead with ties split and with the strongest agent first.

Endorse (`endorse`): same items, one call per candidate per judge, `VERDICT: CORRECT /
INCORRECT`, plus one decoy per item: a wrong answer taken from the samples or debate rows
of that item that no matrix candidate gave (one decoy, on gsmhard:90104808, became the correct answer when the
label audit relabelled the item; it is scored as a correct candidate, not a decoy).

Aggregators: `synth` = DeepSeek reads the six round-3 debate answers with excerpts and
writes one FINAL; `moa` = the same over five of its own T = 1 draws.

## Models and temperatures

Matrix at T = 0 (Claude models via CLI: temperature not controllable, effort low);
samples, debate, personas at T = 1; judges and aggregators at T = 0. Nine debates were
never run because Mistral had no matrix answer to seed them with and one lost Mistral to a
rate limit; the 6-agent tables use the 225 items with a row for every agent in every round and
a 5-agent recount is given alongside.

## What is deliberately missing

No base or non-reasoning checkpoints; no tool use; no decomposition into subtasks; no
logprobs or verbalised confidence; no topologies other than all-to-all; no panels larger
than seven. These are the measurements the remaining literature hypotheses ask for, and
they are listed in the discovery repository as future runs.
