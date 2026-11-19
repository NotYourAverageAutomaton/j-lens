# j-lens

Evidence repository for the NYAA (Not Your Average Automaton) home lab's "J-Lens" experiment: an attempt to replicate the Jacobian lens from "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., 2026, arXiv 2607.15495; Anthropic's summary is titled "A global workspace in language models") on a different model.

- Model: Swift 1.5 27B, a fine-tune of Qwen 3.8 27B (64 layers, hidden size 5120). The lab runs the model; it did not make the base model.
- Purpose of this repo: the lab posted six public updates (2026-11-03 to 2026-11-10). This repo holds the code, result files, logs and figure sources behind every number in those posts, including the failures and the corrections, so an outsider can check them.

## Status (2026-11-10)

1. The 6-document map failed its pre-written bar (post #2). That result stands.
2. Two instrument tests cited in post #1 were withdrawn as proof in post #4: they ran on a reference copy of the model that lacked the causal mask.
3. A 22-document map improved on the 6-document map and cleared its bar (post #5), but is still worse than the logit lens in the middle layers. The score used for that bar was later judged too easy (post #6).
4. No method tested, including a trained (tuned) lens, predicts tokens beyond the very next one better than a word-frequency baseline (post #6).
5. The 54-document build finished on 2026-11-17 and was scored on 2026-11-18. It failed the pass mark signed before the run (average gain -0.041; `CLAIMS.md` rows 7.1 to 7.9). The pass mark itself measured the wrong thing, and the forecast behind the run was stretched past its data: both are in the corrections table at the end of `CLAIMS.md`.
6. The next test is the paper's own: does the 54-document lens read the unspoken middle step of a prompt (six kinds: two-hop facts, translation, order of operations, rhymes, typos, association)? Its pass mark and controls are in `prereg/bars/J54-concept-bar-v4.2.md`, committed here before the capture run. No result exists yet.

## Layout

| path | contents |
|---|---|
| `CLAIMS.md` | Start here. One row per public claim, with the file and key that holds the number, and its status. |
| `code/` | The scripts as they ran, with machine-specific paths replaced by placeholders (see `SCRUB.md`). The directory layout mirrors the experiment directory (`$JLENS_HOME`). |
| `results/` | Small JSON result files and run logs, same layout as `code/`. |
| `figures/` | For each post: the figure script, its data file, and the PNG as posted. |
| `SCRUB.md` | Every replacement made to the copied files, with counts per file. |
| `prereg/` | The pass marks written before each run and the verdicts, copied whole from the lab's internal board. |
| `MD5SUMS` | md5 of every file: original and repo copy side by side. |

## Running the code

The scripts are published as evidence, not as a package. Paths were replaced with placeholders (`$JLENS_HOME`, `$MODELS_DIR`, `$VENV`, `$HF_HOME`, `$AUX_DIR`), which Python does not expand on its own; restore real paths before running. Several result files pin the md5 of the script that produced them. Those pins refer to the original files; `MD5SUMS` lists original and scrubbed md5 for each.

## Who the names are

Comments, logs and the pre-registration posts name reviewers and authors: Claude, Deron, Qwen, Hans, Plex, Plumb, Lens, Simon. These are AI agent seats in the lab, not people. Claude wrote most of the code here; Deron set the pass marks and graded the results. Ids of the form `P017` stand for posts on the lab's internal review board and `C03` for internal task cards; the original numbers were replaced before publishing; the same id always means the same post, and P-ids are in time order. Niz is the human who runs the lab and publishes the posts.

## A note on md5s

Post #6's pictures print `dose_kl.json 6be964a0`. The copy here has a different md5 because host names and paths were replaced (see `SCRUB.md`). `MD5SUMS` lists the original and the repo copy side by side; the result keys are identical.

## What is not here

- Model weights. Swift 1.5 and Qwen 3.8 are distributed by their own publishers.
- The text corpora (Project Gutenberg excerpts, Wikipedia-derived excerpts, WikiText-103). Scripts and metadata name the sources; the text itself is left out pending a licence check, so the maps cannot be rebuilt from this repo alone.
- Jacobian matrices, activation tensors and caches (`.pt`, about 46 GB). Too large for a git repository. Result files record their md5s.
- Nothing else is left out on purpose. The scripts for the causal-mask diagnosis and the re-checks in posts #4 and #5 are in `code/recheck/`.
- An outside clock for the pre-registrations. The pass marks written before each run are in `prereg/`, copied whole from the lab's internal board with the board's own timestamps, so their ordering is self-reported.
- Shell wrappers that schedule GPU time on the lab machine.
- Results of the 54-document build in progress.

## Licence

This repository is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The full text is in `LICENSE`; a summary is at https://creativecommons.org/licenses/by-nc-sa/4.0/.

In short: you may copy, share and build on this material for any non-commercial purpose, as long as you credit NYAA (Not Your Average Automaton) and release what you build under the same licence.

Commercial use: ask us for terms. Contact is not open yet; when it opens it will be listed on the NYAA GitHub organisation page.

The licence covers the lab's own work in this repository. It does not cover the model weights or the text corpora, which are not included and belong to their own publishers.
