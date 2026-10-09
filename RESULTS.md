# Results

All numbers from `results/tables.md` (rebuilt by `run.py`), after the label audit of 8 October 2026
(`label-audit.json`, applied by `regrade_all.py`) and with a truncated answer treated as missing (the
"legacy" section of the tables shows the old policy). Section letters refer to the tables there.

## Label audit (0, N)

48 items were read: 41 on which all seven models had finished and been scored wrong, 6 on which every
debate agent that finished was wrong, 1 added on inspection. Gold wrong and relabelled: 21 (9 MMLU-Pro,
11 GSM-Hard, 1 GPQA Diamond); ill-posed or two defensible options, excluded: 22 (15 / 6 / 1); grader
failures on olympiad expressions: 2 (all stages re-graded with the normalising grader, 61 further
matrix rows changed); models wrong, gold kept: 3. Not audited (listed in the file): 18 items on which
every answering model was wrong while another model was cut off, 1 all-seven-wrong MMLU-Pro item in an
older run file, and the 4 items on which the six-agent debate ends unanimously wrong. Sensitivity (N): the
GSM-Hard relabels are arithmetic and the olympiad ones grading; the MMLU-Pro relabels and exclusions are a
model's judgments. With the MMLU-Pro relabels reverted the all-wrong count on the 485 finished items is 10
(2.1%); with the exclusions restored as well, 24 of 499 (4.8%), plus 5 of 133 GSM-Hard and 1 of 92 GPQA.

## Voting across families (A0, A, A2, A3, N)

Plurality by answer identity over the seven models (five families) is below the best single model on three
layers and ties it on the fourth: −1.9 (MMLU-Pro, 485 items; McNemar 13 vs 4 discordant, exact p = 0.049),
−1.1 (GPQA, 91), −0.8 (GSM-Hard, 128), 0.0 (olympiad, 29). Mean pairwise φ 0.34–0.39 (0.20 on the
29 olympiad items, where φ is defined on 10 of 21 pairs), n_eff 2.1–2.3 of 7. All-wrong rate on the items
all seven finished: 1 of 485 on MMLU-Pro, 0 elsewhere; the one-factor copula at the tetrachoric ρ
over-predicts it (0.3–1.1%). The finished-item subsets are the easy ones (GPQA 91 of 197, olympiad 29 of
90): with a truncated answer a missing vote and items kept when ≥ 5 models answered, every answering model
is wrong on 12/566, 2/191, 4/143 and 2/87 items (2.0% pooled; 19 of the 229 items on which some model was cut off, 8.3%; 254 items have fewer than seven answers once the 25 Mistral never answered are counted, 7.5%), 18 of them unaudited. Mistral has no
matrix row on 32 GPQA Diamond, 6 olympiad and 2 MMLU-Pro items, which removes 18 GPQA Diamond and 4 olympiad items that every
other model finished from the seven-finished tables. Among the model triples the
plurality gain (answer identity) over the best member is negative on average on every layer (−1.4 / −3.5 /
−0.1 / −0.8) and positively correlated with the triple's mean φ on the MCQ layers (Spearman +0.49, +0.45,
tie-averaged ranks), which is also the sign the accuracy-spread confound produces.

## Resampling one model (A', A'2, A'3)

Within-model φ at T = 1 is 0.45–0.68 (DeepSeek), 0.46–0.75 (Haiku); n_eff of k draws 1.3–2.1.
MV@k over a single draw: 0–2 points on MMLU-Pro/GSM-Hard, +4–7 on GPQA, +15 on olympiad for DeepSeek, on
the items with k complete draws; on all sixty items with a cut-off draw a missing vote, DeepSeek's MV@16 is
0.833 on the olympiad (pass@16 0.883) and 0.850 on GPQA (0.917). Hard-state mass (0/k correct) 0–9%. The
Kish majority (exact binomial, interpolated between integers) equals the single-draw accuracy for n_eff ≤ 2
and the observed plurality exceeds it on 6 of 8 rows by 1–14 points; the two-state fit over-predicts MV@16 by
3–9 points on every 16-draw row.

## Personas (D)

Four role prompts on DeepSeek: φ across personas over all draw pairs 0.69 / 0.55 / 0.81 (MMLU-Pro /
GSM-Hard / olympiad) against 0.52 / 0.58 / 0.75 across plain draws; differences +0.17 / −0.03 / +0.07 with
bootstrap intervals [−0.39, +0.31], [−0.22, +0.16], [−0.08, +0.23]; n_eff of four personas 1.2–1.5 against
1.2–1.6 for four plain draws; plurality of four roles above plain on MMLU-Pro, below on GSM-Hard, level on
the olympiad. No measurable independence bought. (Earlier versions used one draw per role: +0.06 / +0.22 /
+0.07, and a [−0.82, +0.24] interval that was noise from dropped resamples.)

## Mixed-family debate (B, B2, B3, N)

Six agents, 225 items, a cut-off row absent: majority 0.924 → 0.973 (+4.9), fixed 11 / broken 0; agent
accuracy on answered rows 0.884 → 0.971; cut-off rows 86 → 21 (GLM 50 and Qwen 36 at round 0); φ 0.34 →
0.85, n_eff 2.2 → 1.2; unanimous (among answering agents) 152 → 219; unanimous-and-wrong 0 → 4 (all four
unaudited; per the audit file's note two have suspect GSM-Hard gold and two are MMLU-Pro items with two defensible options); all-answering-wrong 1 → 4. Ten of the 11 fixes are on items with a
cut-off round-0 row; on the 162 items with six answers at round 0 the majority goes 0.975 → 0.981 (1 fix);
7 of the 11 fixes are round-0 ties lost by the tie-break (ties split: 0.934, +3.9). By round-0 correct count
among answering agents: 1 → 56% majority correct at round 3; 2 → 86%; ≥ 3 → 100%, but 21 of the 24 items
in the 0–2 strata had a cut-off agent. A correct agent that answers in the next round and faces 3 / 4 / 5
answering, disagreeing peers flips to wrong 29 / 60 / 83% of the time on 24 / 5 / 6 cases (an agent cut off
in the next round is not counted). Per layer, gain over the best single agent with a cut-off answer scoring
0: GPQA +8.9, olympiad +3.7, MMLU-Pro −1.7, GSM-Hard −1.8; over the best agent on the items it answered, with the
majority taken on the same items (T): GPQA +2.0 (Qwen's 49 items, 0.980 → 1.000), olympiad 0, MMLU-Pro −1.8, GSM-Hard −1.8
(earlier versions set the agent's accuracy beside the majority on all items, which gave +0.3, 0, −5.2, −1.8). Tie rules (T):
the round-0 majority is 0.924 with ties to the first agent in the fixed order, 0.934 with ties split, 0.951 with the strongest
agent (Sonnet) first, so the debate gain is 4.9 / 3.9 / 2.2 points, and on MMLU-Pro and GSM-Hard it is zero under the last rule. Stop-at-first-unanimity (among answering agents): accuracy 0.973 at 0.42 rounds (0.68
under the old policy). Three rounds cost ≈ 21,900 completion tokens per item on top of round 0.

## Homogeneous debate (B5)

Six DeepSeek copies. On the 230 items all arms completed: round-0 majority 0.865; plain 0.900 (+3.5,
11 fixed / 3 broken), selfonly 0.883 (+1.7, 8 / 4), pressure 0.909 (+4.3, 12 / 2) at round 3.
Unanimous-and-wrong 6 → 22 (plain), 6 → 2 (selfonly), 6 → 18 (pressure); all-wrong at round 0 fixed
0/15, 1/13, 1/15 (the one escape is gsmhard:36264882, whose gold is suspect); φ 0.58 → 0.97 / 0.58 /
0.95. About half of the gain is self-revision; the information half takes wrong unanimity from 6 to 22.

## Aggregators (B4/C4)

Synthesis over six round-3 debate answers with 900-character reasoning excerpts: 0.969 vs plurality
0.973, 0/4 all-wrong recovered, 1 novel answer (correct). MoA over five own draws: 0.873 vs 0.860,
0/16 recovered, no novel answers.

## Judge panel, pick-one (C, C2)

75 contested items (the answering proposers' candidates disagree by answer identity, at least one correct and
one wrong; 31 olympiad, 24 GPQA, 12 GSM-Hard, 8 MMLU-Pro). Pick accuracy with an unparsed pick (cut-off
call or a letter outside the list) scored wrong: Sonnet .80, Haiku .73, Qwen .65, DeepSeek .52, GLM .48; on
named picks .87 (68), .75 (73), .89 (55), .67 (58), .82 (44). Own solve accuracy on answered items .85,
.56, .89 (45), .45, .82 (34). Solvers' plurality .747 (ties to the first proposer in the fixed agent order).
Judges' pick errors on the 74 items all five judged: φ 0.16, tetrachoric 0.27, n_eff 3.07 of 5. Panel
plurality: 0.838 with ties to the first judge (solvers 0.757; 11 vs 5 discordant, p = 0.21), 0.757 with
ties counted wrong (solvers 0.743); strict majority 0.944 on the 54 items where more than half agree; 13 of
74 items tie; 49 of 74 are judged by fewer than five named picks. Whole set (235 items): solvers' plurality
0.915; plurality + panel 0.940 (first-judge ties) / 0.953 (ties fall back to plurality); best single solver
with a cut-off answer scoring 0: Sonnet 0.945; on the items it answered: Qwen 0.980 (196 items). "NONE" on
the 2 none-correct items: 1 of 10. (v3 broke the solvers' ties by the shuffled presentation order, which
gave 0.803 and an 8 vs 5 lead.) Tie rules (T): with ties split evenly on both sides the panel is 0.842 against the solvers'
0.802 (three items); with the strongest on each side first (Sonnet for both) the solvers' plurality is 0.865 and the panel 0.851,
4 won against 5 lost; on the whole set the strongest-first plurality alone is 0.949 against 0.945 with the panel. On the 196 items
Qwen answered the plurality is 0.959 and plurality + panel 0.954 / 0.974 against Qwen's 0.980 (the earlier "three to four points"
set Qwen's figure beside the pool's whole-set figure).

## Error detection (B3, last block)

DeepSeek's matrix error (rate 0.17) predicted by the number of disagreeing answering peer families (by
answer identity): AUROC 0.98; by the share of disagreeing draws among its first 3 / 5 / 15 own T = 1 draws
that answered: 0.89 / 0.90 / 0.93; Sonnet alone 0.93. Five peers − fifteen own draws: +0.05, bootstrap 95%
[+0.01, +0.10]; Sonnet alone − fifteen own draws: 0.00 [−0.07, +0.07].

## Judge panel, certify-one (C3)

FPR on wrong candidates: Sonnet .06, GLM .23, Qwen .26, Haiku .39, DeepSeek .43. Wrong decoy accepted:
.00 / .19 / .09 / .23 / .24 (one planted decoy became the correct answer under the label audit and is not a
decoy). Own wrong answer accepted: Sonnet .36, GLM .62, Qwen 1.00, Haiku .86, DeepSeek .56, vs others' wrong
.05 / .19 / .29 / .31 / .49. The OpenRouter judges ruled on 50–70% of Sonnet's candidates (spend cap plus
cut-off verdicts). False-accept φ across judges 0.38, n_eff 2.0. Committee over 94 candidates (51%
GSM-Hard), a judge counted as accepting when it accepted any format of an answer (73 candidate–judge pairs
were shown in two formats, 17 with conflicting verdicts): "all five accept" precision 0.97, recall 0.85, 1 of
54 wrong candidates through (0 through at recall 0.68 if a judge must accept every format); "four of five"
0.86 / 0.95 / 6.

## Fifth reading (N, last block)

The seven-model plurality breaks ties by the raw/ directory order (Haiku, Opus, Sonnet, DeepSeek, Mistral, Qwen, GLM): 5 MMLU-Pro and 1 GPQA ties; MMLU-Pro plurality 0.953–0.961 and McNemar 14 vs 3 (p 0.013) to 10 vs 3 (p 0.09) across orders, GPQA 0.967–0.978. Of the 10 debate fixes on items with a cut-off round-0 row, the formerly cut-off agent is in the winning round-3 majority on 4. The synthesiser's 4 no-correct-input items are the 4 unanimously wrong debate items. Judge panel: 11 of 74 contested items are solver ties, 9 of the panel's 11 wins are among them, and on the 63 non-tied items the solvers' plurality is 0.873 against the panel's 0.825; judges' φ on named picks 0.29 (n_eff 2.3); solvers' plurality on the 54 strict-majority items 0.796; tied panel falling back to the plurality gives 0.878 on the contested items.

## Caveats

60 items per layer; reasoning models only (Claude at low effort); an 8,192-token cap on the OpenRouter
models that two of them hit on the hard layers; one family for the homogeneous and persona arms; cos-π
tetrachoric averaged over defined pairs only; the certify committee depends on how conflicting verdicts across formats are merged; the audit covers only the items that were selected and
lists what was not; judges may recognise their own answer in excerpts; the panel result is within noise
and changes sign with the tie-break rule; the judges' pick φ counts a cut-off call as a wrong pick, unlike the solvers' rule; the
certify committee is 51% GSM-Hard where the contested pick items are 16%; nothing pre-registered.
