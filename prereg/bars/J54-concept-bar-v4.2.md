# J54 concept test: pre-registration DRAFT (card C01; Deron 2026-11-18 18:1x)

STATUS: DRAFT v2 (19:3x, after reading the J-space paper's Methods, reasoning, generalization and capacity sections; web and arXiv copies agree on the 5 numbers checked). Papers 2-4 still to read. DRAFT SKELETON, written before any data on this question. NOT FROZEN. It freezes only after (a) I have read the paper's validation section and each cited paper from local copies (none are on my box yet; titles below are UNVOUCHED, from Claude P149), and (b) the prompt set is written and checked to be disjoint from the fit corpus. Conditions adopted: Claude P151 (#1).

## Question (the paper's own, asked of the J54 we already have)
In the mid band, do the top-k directions of J54 carry the UNSPOKEN intermediate of a multi-hop prompt (readout), and does swapping it change the answer (causal)? This is not a re-score of Stage 2, which stays a FAIL.

## Object
- J54 per layer, L20-44 by 4 (the same pins as stage2_score.json; no refit, no fit nights).
- Directions: the top-k SVD components of (J54 - I) per layer. k is fixed before data. Default 25 (the paper's approximate J-space size per P149; CONFIRM from the paper's text).

## Baselines (the readout must beat ALL of them; condition 2)
- B1 logit lens: the same readout through the unembed, with no J.
- B2 residual PCA: the top-k PCA directions of the raw residual at the same layer and the same k, fitted on the same contexts the J was fitted on.
- B3 shuffled: k random orthonormal directions (seed fixed), as the floor.
A pass that beats B3 but not B2 reads "the residual holds it", not "J does".

## Prompts (condition 3)
- Multi-hop prompts, with the intermediate never written in the prompt. Written in full before any run, with md5. Disjoint from the stage1, stage2 and fresh corpora (checked by script: no shared 8-gram).
- Format and count: from the paper's validation set (TO READ). Placeholder: >= 150 prompts over >= 5 relation families.

## Readout metric (TO FIX after reading)
- Per prompt and layer: the rank of the intermediate token under each readout. Score = mean over the mid band of the fraction with rank <= 10.
- Gate shape (provisional): J-dir score minus max(B1, B2), paired bootstrap over prompts (B 10000, seed fixed at freeze), with a 95% CI lower bound > 0. Cluster CI by relation family is report-only.

## Causal (only if the readout passes; condition 4)
- The swap uses the paper's method (TO READ): replace the intermediate's J-component with another entity's; does the answer change to match it? It must beat a random-direction swap of equal norm.
- Only on the bf16 copy on [gpu-host], never Qwen's engine, under the weights-touch protocol. Qwen's yes comes first, because it is a control surface on her own weights.

## Papers to read before freezing (Claude fetches into [internal path]; arXiv is not allowlistable here)
1. Gurnee et al. 2026, the J-space paper (transformer-circuits.pub/2026/workspace): the validation section, and the France-to-China swap (76/192 per P149).
2. arXiv 2604.02608, "Steerable but Not Decodable".
3. arXiv 2609.01936, "Sparse Readout Prism".
4. arXiv 2411.04430 (the SAE-vs-lens orthogonality benchmark).


## v2 CORRECTIONS from reading Gurnee et al. 2026 (READ, gurnee2026-workspace.txt md5 571c2323)
1. THE OBJECT WAS WRONG in v1 (and in P149): the J-space is NOT the top-k SVD directions of J. It is the set of sparse NON-NEGATIVE combinations of k J-lens vectors, one vector per vocab token (v_t, with logits ~ <v_t, h> up to the norm). k is a per-activation sparsity, "typically no more than 25", found by gradient pursuit. The SVD sections above are void.
2. The readout is per token: the rank of token t under softmax(W_U norm(J h)). The paper's presence rule (capacity section) is that the token's best rank over the workspace band is <= 25.
3. The paper's own validation of intermediates: 50 two-hop factual prompts with known intermediates. The swap target is drawn at random from the intermediate's category. Success means the target-appropriate answer reaches top-1: 54% (Haiku 4.5), 70% (Sonnet 4.5, Opus 4.5). Probe split, n=90: J-component swap 61%, raw J-lens swap 60%, non-J component 28%, 6% with the J coordinates clamped.
4. The swap is the lens-coordinate swap: V=[v_s v_t], c = pinv(V) h, h' = h + V(swap(c) - c), at all positions; the part orthogonal to span(V) is unchanged.
5. 76/192 is the FLEXIBLE-GENERALIZATION experiment (16 function templates x 12 pairs; 101/192 at alpha=2), not the two-hop / France-China result. France-China is the n=90 probe-split figure.
6. The paper: the logit lens "captures much of the workspace-like structure ... with somewhat lower reliability". So the logit lens is a real competitor, not a strawman.
7. The paper's repository ships its raw prompt data. Use its two-hop prompts if present, after the disjointness check.

## v2 BASELINES (replace v1's; my condition 2 is amended, and I say so on the board)
- B1 logit lens (J = I): it reads the raw residual. This is now the "the residual holds it" baseline that the residual PCA was standing in for. Residual PCA is dropped: it presupposed the SVD object.
- B2 row-permuted J54 (seed 20261104, as the existing rowperm): a must-fail control. If B2 passes, VOID.
- B3 category foil: the rank of a random same-category token that is in neither the prompt nor the answer, under the same J54 lens. This controls for category priming (the paper's list result: whole category neighbourhoods load).
- Gate shape (provisional): for prompts the model answers correctly (filter fixed before the run), present(J54) minus present(B1), and present(J54) minus present(B3). Paired bootstrap over prompts, each with a 95% CI lo > 0. Band L20-44.

## v2 CAUSAL (only if the readout passes)
- The paper's swap, target random within the category. Success = the target-appropriate answer at top-1.
- Control: a swap along a random unit direction pair of the same norms. J54 swap success minus control > 0, CI lo > 0.
- Report-only: the paper's answer-swap depth comparison.

## Lesson carried in (P150)
No projection past what we measure. State in the header, before data, what the best case would and would not show.

## Paper 4 read (arXiv 2411.04430 v2, Bhalla et al., md5 d6f8c990 txt), 19:4x (READ)
- It is an intervention benchmark of SAEs, the logit lens, the tuned lens and probing, run on GPT2-small and Gemma2-2b among others. Its two metrics are intervention success rate and the coherence-intervention tradeoff. Interventions often cost coherence and underperform plain prompting.
- Claude's P149 description holds: SAEs "intervene in somewhat similar directions to steering vectors and probes and have near orthogonal directions to Logit Lens and Tuned Lens, even when interventions succeed for all methods." (One hit in the text. The search works on this text: "logit lens" hits many times.)
- Into the causal step, report-only: a coherence measure on the swapped run (the KL of an unrelated continuation versus the clean run), and a prompting baseline (tell the model the alternative intermediate in words) beside the swap success.
- For the SAE plan (card C05): an SAE may find directions the lens cannot. A lens-null does not imply an SAE-null.

## v3, 20:2x: the paper's own lens comparison (Appendix "Quantitative Methodological Comparisons", READ) and its data
- THE PAPER'S READOUT METRIC: six prompt sets, each with a known single-token intermediate that is in neither the input nor the output. One fixed position per item. "Recovered at k" = in the top-k of the lens readout at ANY layer. pass@k, with the AUC over log k normalized so that always-rank-1 = 1. Reported: the J-lens beats the logit lens on all six, "modest on multihop and association, substantial on multilingual, order-of-operations, poetry, typo".
- THE PAPER ON OUR STAGE 2 METRIC: "The J-lens is the worst, with higher KL than even the uncorrected logit lens through most of the network. We regard this as a feature rather than a defect." So J trailing logit on KL in our Stage 2 is what the paper reports. Stage 2's FAIL (J54 not better than J22) stands; the public words need this line.
- DATA (carried by Claude, repo commit 581d3986, MD5SUMS 0c584676, all OK): the repo files hold MORE items than the paper reports: multihop 93 vs 50, multilingual 107 vs 54, poetry 98 vs 52, association 102 vs 50; order-ops 55 and typo 96 match. The paper's subsets are not identifiable, so we pre-register ALL items, after a filter fixed in advance: keep an item only if Swift-1.5 (bf16 copy) puts the item's target at top-1 on the clean run.
- DISJOINTNESS (disjoint8_j54.py, word 8-grams, planted control flagged): 0 overlapping items in all 6 sets vs stage2_corpus (84f9f759) and the fresh passages (2d302976). stage1_corpus (19c71211) is NOT on my box: still owed. Short prompts (multilingual, order-ops) yield few 8-grams, so the check is weak there.
- PRIMARY (proposed v3): pooled over the six sets, normalized pass@k AUC (k = 1..1000, log-spaced, fixed at freeze), band L20-44 any-layer, J54 minus the logit lens. Paired bootstrap over items (B 10000, seed fixed at freeze), CI lo > 0. Must also beat the category foil, and the row-permuted J54 must NOT beat logit (else VOID). Per-set rows are report-only. A PASS = the paper's lens-comparison shape replicates on Swift-1.5 with our J54. A FAIL = our J54 does not read intermediates better than the logit lens.

## 21:3x: the paper's fit recipe (Appendix "Methodological Details and Ablations", READ)
- Default J: 1,000 sequences of 128 tokens, pretraining-like text. The gradient is taken of the PENULTIMATE-layer residual, averaged over all t' >= t. Their pseudocode is exact: one backward pass per output dimension, batched. Nothing in the text calls it stochastic, so P149's "stochastic estimator" is not supported.
- "J-lens beats the logit lens and tuned lens baselines with as few as 10 prompts, with modest improvements from additional data" (on pass@k). So the corpus size was never the lever. Stage 2's result fits that.
- Recipe variants (final vs penultimate target, stop-grad on QK, future-only, self-only; mean vs median) are "fairly consistent". Mean over the penultimate is slightly better for intermediates, and QK stop-grad raises causal effect.
- OUR J54 uses the FINAL layer as target (cc-jlens-plan line 67, READ). The paper notes the last block "can sometimes increase noisy artifacts". That is report-only context, not a reason to refit before the test.

## Papers 2 and 3 read (22:3x, READ)
- 2604.02608 (Nadaf, single independent author, v2): function vectors steer even where the logit lens cannot decode the answer at any layer ("steerable but not decodable"; the converse is rare, 3 of 72). So a readout FAIL here does NOT show the hidden step is absent or non-causal. Readout-before-causal stays, for consent and cost, and that limit is stated with the verdict.
- 2609.01936 (He, Shen, Qiu, Lane): lenses fitted on different corpora can name different tokens for the same states ("corpus conditionality"). So J22 is run beside J54, report-only. Token readings can also be ambiguous (one row serves many senses).

# FROZEN BAR v4 (Deron 2026-11-18 22:4x, written before any J54 readout data)
Sources READ: the J-space paper (Methods; reasoning; capacity; Quantitative Methodological Comparisons; Methodological Details), 2411.04430, 2604.02608, 2609.01936. Data: repo 581d3986, six lens-eval sets, MD5SUMS 0c584676. Disjointness: 0 overlapping items vs stage1 (19c71211), stage2 (84f9f759) and the fresh passages (2d302976); disjoint8_j54.py 916b5715, planted control flagged.
OBJECT: J54 at L20, 24, 28, 32, 36, 40, 44 (the stage2_score pins; no refit). Lens = softmax(W_U norm(J h)), our existing CPU path. Logit lens = the same path with J = I.
POSITION, one per item, as the paper: multihop, multilingual, order-ops = the last prompt token. typo = the last prompt token (the misspelled fragment). association = the last prompt token (the closing period). POETRY = the SECOND newline token (the end of line 1), NOT the prompt end: in the repo file the prompt runs into line 2, where the rhyme word is the next token. Reading at the end would be a next-token test.
INTERMEDIATES: each listed intermediate counts. A token counts if it is the word, with or without a leading space, in its listed case or capitalized, and is a single token in Swift's tokenizer. Intermediates with no single-token form are dropped BEFORE the run, and the dropped count is reported per set.
FILTER (before the run, from a clean forward pass only): multihop, multilingual and order-ops keep an item only if the model's top-1 next token at the prompt end is the target (single-token forms as above). Poetry keeps an item only if top-1 at the prompt end is the rhyme word. Typo and association: no filter. Kept counts are reported. If fewer than 150 items survive in total, the result is UNDERPOWERED, not a FAIL.
SCORE: rank of each intermediate in each lens at each of the 7 layers. Best rank over the 7 layers = the item-intermediate rank. pass@k for k in {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}. AUC = mean of pass@k over those 10 k (log-spaced, so 1 = always rank 1). The item score = the mean over its intermediates. Pooled over all kept items.
FOIL: for each item, the intermediate of a different item in the SAME set, drawn once with seed 20261119 (not the item's own, not in its prompt or target), scored identically with J54. "Same-category foil" (v2) is replaced by this, because the data carry no category list.
GATE, PASS iff all three:
 G1  AUC(J54) - AUC(logit) > 0, paired bootstrap over items (B 10000, seed 20261119), 95% CI low > 0.
 G2  AUC(J54, intermediate) - AUC(J54, foil) has 95% CI low > 0 (same bootstrap).
 G3  CONTROL: row-permuted J54 (seed 20261104): AUC(rowperm) - AUC(logit) must NOT have CI low > 0. If it does, VOID.
STOPS (NO-SCORE, never FAIL): pins mismatch; J = I path not bit-equal to the plain unembed readout on 5 items; any non-finite rank; fewer than 7 layers.
REPORT-ONLY (changes nothing): per-set rows; J22 and shrunk-J54 rows (corpus conditionality); per-layer rows; the paper's per-set ordering (J beat logit on all six, modest on multihop and association) set beside ours.
NO FORECAST NUMBER (the P150 lesson): the paper's direction is the expectation; we do not project a size.
A PASS says: on Swift-1.5, our J54 surfaces known hidden steps more often than the plain logit lens, and above a decoy. A FAIL says it does not. It does NOT say the steps are absent from the model (2604.02608). Causal swaps come only after a PASS and Qwen's yes, under their own bar.
RUN: the bf16 copy on [gpu-host] only, CPU acceptable. Claude builds the scorer from this text; I read every line before it runs; selftest with planted ranks (an all-rank-1 lens must PASS G1 vs a random lens; a lens equal to logit must FAIL G1). No re-run, no metric swap.

## FROZEN BAR v4.1: five rules made script-exact, plus procedure (22:3x, Claude P152/P153). Still before any data. The pass mark is unchanged.
R1 POSITION (token index into the tokenized prompt, no BOS counted as content):
 - multihop, multilingual, order-ops, typo, association: the LAST token of the prompt.
 - poetry: the token that CONTAINS the second "\n" character of the prompt (the end of line 1; the first "\n" ends "A rhyming couplet:"). If a token holds both newlines, or the prompt has fewer than 2, the item is dropped before the run and counted.
R2 FILTER ("gets it right"): from the clean forward at the prompt's last token, greedy top-1 token id is in {first token of tok(target), first token of tok(" " + target)}. Sets with a target: multihop, multilingual, order-ops. Poetry uses its intermediate (the rhyme word) as the target. Typo and association are not filtered.
R3 RANK: for each intermediate, the allowed forms are {w, " "+w, W, " "+W} (W = first letter capitalized). A form counts only if it tokenizes to ONE token. Rank = the best (lowest) rank among the allowed single-token forms over the full vocabulary (1 = top). An intermediate with no single-token form is dropped before the run and counted. An item with no intermediate left is dropped.
R4 DECOY: per set, a seeded derangement of the kept items (numpy default_rng(20261119).permutation, re-drawn with seed+1, +2, ... until no item maps to itself). Decoy for item i = the first single-token intermediate of its partner that does not appear (case-insensitive) in item i's prompt, target or intermediates. If none, the next item in permutation order. The decoy is scored with the same R3 rule. G2 compares the item AUC with its own intermediates vs the item AUC with its one decoy, both under J54.
R5 BOOTSTRAP: items, resampled STRATIFIED within set (each replicate draws n_s items with replacement from set s, keeping the set mix fixed), B 10000, seed 20261119. Statistic = the pooled mean over items. Report-only: the unstratified version, and per-set CIs.
PROCEDURE: capture on GPU in ONE engine-down window, with the step2 consistency gate (re-capture the held-out passages and require equality with held_h.pt); scoring on CPU, file-only. The window needs Qwen's yes and is sized by a 10-item dry run first. ("CPU is fine" in v4 meant scoring only.)

## FROZEN BAR v4.2 (2026-11-18 23:3x, Deron, re Claude P154; written BEFORE any model data: only tokenizer counts exist)
Amends v4.1 on two points. Everything else in v4 and v4.1 stands.
- R2 (replaces v4.1 R2), the filter means the WHOLE answer: the capture stores the model's greedy continuation of up to 8 new tokens after the prompt. Decode it, strip leading whitespace. Kept if the text starts with the target (exact for numbers; casefold for words) AND the next character after the target is absent or not alphanumeric. Applies to multihop, multilingual, order-ops; poetry uses its rhyme word the same way; typo and association unfiltered. Reason: digit-by-digit tokens made "25" and a bare space pass for target "20" (PROVEN by Claude on the tokenizer).
- POSITION (new, report-only, gates nothing): every result is also reported split by read position, under 16 and 16 or over. The gate stays on all kept items, as in the paper. Reason: our J was fitted skipping the first 16 positions (our_jlens.py); 289 of 542 reads fall below 16. The paper fits on ALL positions by default and reports that excluding the first several tokens "yielded no meaningful improvement" (2607.15495 App. A.7, "Data: Distribution"; READ). So the paper gives no ground to gate on position; our skip is a property of OUR lens and the test is of our lens.
- READING fixed now: if the gate FAILS and the 16-or-over half alone would pass, the verdict is still FAIL. The report says so, names our position skip as a candidate cause, and the next step is a refit on all positions, under its own new bar. No re-grade of this run.
- Not chosen: (b) filter to 16+ (drops multilingual, typo and order-ops almost whole; ~253 items before R2, likely UNDERPOWERED); (c) require direction in the 16+ half (adds a second gate on a half the paper never splits).
