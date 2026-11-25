# SCRUB.md

Every change made to copied files. Nothing else was edited: each file is a byte copy of the original with the regex replacements below applied in this order. Files not listed in the per-file table are byte-identical to the original (see `MD5SUMS`).

All 63 Python files in the tree still parse after scrubbing (`ast.parse`). No file was dropped for containing a secret; none was found.

## Effect on code semantics

- Path replacements change string constants (for example `BASE = "$JLENS_HOME/claude-normtest"`). Python does not expand these; a reader must substitute real paths to run the code. No logic was changed.
- `127.0.0.1:8010` -> `127.0.0.1:$ENGINE_PORT` changes one string constant (`HEALTH_URL` in `code/shape_c_loader.py`): the local port of an inference server that shared the GPU. Other hits are in comments.
- the host nickname -> `the GPU host` occurs only in comments, docstrings and log text.
- `<hostname>` -> `gpu-host` occurs in the `env.host` field of result JSON files.
- Figure scripts: the import path for the shared plotting helper and the PNG output path were made relative.

## Rules, in order, with totals

| # | pattern | replacement | total hits |
|---|---|---|---|
| 1 | `/home/<user>/j-lens` | `$JLENS_HOME` | 300 |
| 2 | `/home/<user>/venvs/<venv>` | `$VENV` | 79 |
| 3 | `/home/<user>/models` | `$MODELS_DIR` | 42 |
| 4 | `/home/<user>/.cache/huggingface` | (no hits in the files kept) | 0 |
| 5 | `/home/<user>/emotions` | `$AUX_DIR` | 1 |
| 6 | `/home/<user>/.hermes` | (no hits in the files kept) | 0 |
| 7 | `/home/<user> (any other)` | (no hits in the files kept) | 0 |
| 8 | `~/Documents/Social_Posts/posts/<post>/media/` | `./` | 3 |
| 9 | `~/Documents/Social_Posts/tools` | `../tools` | 4 |
| 10 | `<hostname>` | `gpu-host` | 28 |
| 11 | `<host nickname> (word)` | `the GPU host` | 8 |
| 12 | `127.0.0.1:8010` | `127.0.0.1:$ENGINE_PORT` | 1 |
| 13 | `:8010` | `:$ENGINE_PORT` | 2 |

## Per file

| file | replacements (pattern -> replacement: count) |
|---|---|
| `code/claude-memtest/memtest.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `code/claude-normtest/capture_h.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1; `/home/<user>/emotions  ->  $AUX_DIR`: 1 |
| `code/claude-normtest/ctrl_precheck.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/diag_check.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/eval.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fd_gate.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fd_gate_v3.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fd_gate_v4.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fd_gate_v4c.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fd_gate_v5.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fitprobe.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/fitprobe2.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/gate_lowrank.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/j6_spectrum.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `code/claude-normtest/m123.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `code/claude-normtest/m2_capture.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 6; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/m2_score.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/memcheck.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/memcheck2.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/mk_dry_identity.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `code/claude-normtest/mk_dry_stage1.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `code/claude-normtest/normtest.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `code/claude-normtest/null_study.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/pilot_hutch.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/resume.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/score_stage1.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/score_stage1_DOSE.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/shrink_check.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/stage1_fit.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/stage2_fit.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/stage2_staging/overlap_check_s2.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/step2/ctx_table.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/step2/dose_kl.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 5; `/home/<user>/models  ->  $MODELS_DIR`: 1; `<host nickname> (word)  ->  the GPU host`: 1 |
| `code/claude-normtest/step2/overlap_check.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/step2/step2_capture.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/step2/step2_score.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/sup_check.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/tl_score.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/claude-normtest/tl_train.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 6; `/home/<user>/models  ->  $MODELS_DIR`: 1; `<host nickname> (word)  ->  the GPU host`: 4 |
| `code/claude-normtest/weight_diff.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `/home/<user>/models  ->  $MODELS_DIR`: 2 |
| `code/cpu_dryrun_P015.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/fd_vs_jvp_full.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `code/fit_attach.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `code/fp32_fd_falsifier.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `code/patch_our_jlens.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `code/settle_fp32_full.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<host nickname> (word)  ->  the GPU host`: 1 |
| `code/shape_c_loader.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `/home/<user>/models  ->  $MODELS_DIR`: 1; `<host nickname> (word)  ->  the GPU host`: 1; `127.0.0.1:8010  ->  127.0.0.1:$ENGINE_PORT`: 1; `:8010  ->  :$ENGINE_PORT`: 1 |
| `code/whole_model_fit.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `code/whole_model_loader.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<host nickname> (word)  ->  the GPU host`: 1; `:8010  ->  :$ENGINE_PORT`: 1 |
| `figures/post2/eval_record.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `figures/post4/01-the-answer-key-could-read-ahead.py` | `~/Documents/Social_Posts/posts/<post>/media/  ->  ./`: 1; `~/Documents/Social_Posts/tools  ->  ../tools`: 1 |
| `figures/post5/01-the-bigger-map.py` | `~/Documents/Social_Posts/posts/<post>/media/  ->  ./`: 1; `~/Documents/Social_Posts/tools  ->  ../tools`: 1 |
| `figures/post6/01-more-documents.py` | `~/Documents/Social_Posts/tools  ->  ../tools`: 1 |
| `figures/post6/02-reading-ahead.py` | `~/Documents/Social_Posts/posts/<post>/media/  ->  ./`: 1; `~/Documents/Social_Posts/tools  ->  ../tools`: 1 |
| `results/claude-normtest/diag-window.log` | `/home/<user>/venvs/<venv>  ->  $VENV`: 2 |
| `results/claude-normtest/eval_record.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/eval_record.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/fd_gate_v4_run.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/venvs/<venv>  ->  $VENV`: 19 |
| `results/claude-normtest/fd_gate_v4c_run.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/fd_gate_v5_run.2125-crash.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `results/claude-normtest/fd_gate_v5_run.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/m123.json` | `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/m2.json` | `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/m2_capture_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/m2_v2.json` | `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/pilot_run.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/pilot_smoke.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/score_shrink.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_shrink_diag.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_shrink_selftest.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_shrink_sup.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DOSEDRY.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DOSEMEM.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY2_doublecount.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY2_identical.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY2_identity.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY3_doublecount.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY3_guardtest.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY3_identical.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY3_identity.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY_doublecount.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY_identical.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_DRY_identity.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_STOP_0055_guard.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_STOP_0211_guard.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/score_stage1_dose.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/shrink_null_study.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/step2/ctx_table.json` | `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/step2/dose_kl.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/step2/dose_kl.night.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/step2/score_step2.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/tl_out_lr1/tl_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/tl_out_lr1_smoke/tl_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/tl_out_sel/score_tl.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1 |
| `results/claude-normtest/tl_out_sel/tl_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/tl_out_sel_peak/tl_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/tl_out_sel_smoke/tl_meta.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/claude-normtest/tl_peak.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/tl_train_lr1.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 2 |
| `results/claude-normtest/tl_train_sel.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/tl_train_sel_smoke.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/claude-normtest/tl_train_smoke.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/gate_results.json` | `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `results/gate_results_prefix-normfix.json` | `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `results/gate_results_whole_model.json` | `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `results/run_fd_vs_jvp.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_fp32_fd.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_settle_fp32_full.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_swift.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 6 |
| `results/run_swift_gates.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_swift_gates2.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_swift_gates3.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_whole_swift.log` | `/home/<user>/models  ->  $MODELS_DIR`: 1 |
| `results/run_whole_verify_1103.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_whole_verify_dtype2_1103.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/venvs/<venv>  ->  $VENV`: 10 |
| `results/run_whole_verify_dtype3_1103.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1 |
| `results/run_whole_verify_dtype_1103.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/venvs/<venv>  ->  $VENV`: 10 |
| `results/whole_model_fit_P014.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3 |
| `results/whole_model_fit_P015.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/venvs/<venv>  ->  $VENV`: 18 |
| `results/whole_model_fit_P016.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/venvs/<venv>  ->  $VENV`: 6 |
| `results/whole_model_fit_P018.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 3; `/home/<user>/venvs/<venv>  ->  $VENV`: 1 |
| `results/whole_model_fit_record.log` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 50; `/home/<user>/venvs/<venv>  ->  $VENV`: 13 |

## Added after the first pass: `code/recheck/`

Six scripts copied from outside the experiment directory, with rules 1, 2, 3, 5 and 10 applied: `c4_check.py` (3 lines changed), `c4_check_v2.py` (3), `diag_layers.py` (6), `diag_mask.py` (3), `diag_mask2.py` (3), `part_c.py` (6). All parse. `results/claude-normtest/partc-window.log` pins the md5 of the original `part_c.py`.

## Added after review (2026-11-10)

- `code/our_jlens.py` line 12 (docstring): a second machine's name -> `the review host`. 1 hit.
- `results/recheck/`: seven result files copied from outside the experiment directory. They contained no paths or host names; byte-identical to the originals.
- `prereg/posts/`: 45 board posts, 18 bracketed replacements in all (rules in `prereg/README.md`).

## Internal ids (2026-11-10)

Two more rules, applied to every text file in the tree after all the rules above:

| # | pattern | replacement | distinct | total hits |
|---|---|---|---|---|
| 14 | internal board post number (five digits, cited next to a seat name, `board`, `Re`, `per`, in a `bars`/`rule`/`prereg` string, in a bracketed or parenthesised citation, or in a file name) | `P001` ... `P146`, written `post P017` where a bare number stood in prose and `P017` after a seat name, in lists and in file names | 146 | 700 |
| 15 | internal task-card id (`t_` + 8 hex digits) | `card C01` ... `card C04` (`C01` when the word `card` already precedes it) | 4 | 161 |

- P-ids are assigned in ascending order of the original number, so they are in time order; C-ids in order of first appearance (files in sorted path order). The same id is the same post or card everywhere in the repo. The mapping is not in the repo.
- 192 files changed content (861 replacements in all, counted in the text of every file except the regenerated `MD5SUMS-repo`; this includes path references in `MD5SUMS`, this file, `CLAIMS.md` and `prereg/README.md`). 50 files were renamed. In `MD5SUMS` the second md5 column and the last column were updated for every file whose content changed; the origin column carries the P-id in place of the number for the five renamed code/log files.
- Five-digit numbers that are data were left alone: Python numeric literals, numeric JSON values, decimals, hex digests, and quantities in prose (probe budget, validation positions, state counts). No replacement landed outside a comment, a string or prose; all Python files still parse and all JSON files still load.
- The replacements change string constants and comments only. Strings that a script writes into its result json (for example a `bars` field) now carry the id, as do the copies of those strings in `results/`. md5 pins of scripts recorded inside result files and logs refer to the original files, not to the edited copies.
- `README.md` (the sentence explaining the ids) and `prereg/README.md` (index links and one line under Limits) were edited by hand to match.

Renamed files:

| old pattern | new name |
|---|---|
| `code/cpu_dryrun_<post>.py` | `code/cpu_dryrun_P015.py` |
| `results/whole_model_fit_<post>.log` | `results/whole_model_fit_P014.log` |
| `results/whole_model_fit_<post>.log` | `results/whole_model_fit_P015.log` |
| `results/whole_model_fit_<post>.log` | `results/whole_model_fit_P016.log` |
| `results/whole_model_fit_<post>.log` | `results/whole_model_fit_P018.log` |
| `prereg/posts/<post>.md` (45 files) | `prereg/posts/Pnnn.md` with nnn = 023, 025, 027, 029, 031, 033, 034, 036, 037, 039, 040, 042, 045, 047, 048, 049, 052, 053, 054, 056, 057, 062, 071, 073, 075, 077, 084, 085, 086, 088, 093, 095, 097, 101, 103, 106, 110, 117, 121, 127, 131, 132, 135, 137, 139 |

## Added 2026-11-18: the Stage 2 result

Three files copied from the experiment directory, with rules 1, 3, 10, 14 and 15 applied. No numeric value changed (the `gate`, `N`, `controls`, `stops`, `stage2_progress`, `disjoint_vs_fresh` and `files` blocks of the result json were compared with the original after scrubbing and are equal). The script still parses. The script was named `score_stage2.v3-08054c09.py` on the GPU host; `script_md5` inside the result json is the md5 of that original, not of the copy here.

| file | replacements (pattern -> replacement: count) |
|---|---|
| `code/claude-normtest/score_stage2.py` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 4; `/home/<user>/models  ->  $MODELS_DIR`: 1; `internal task-card id  ->  card C01`: 1; `internal board post number  ->  P-id`: 7 |
| `results/claude-normtest/stage2/stage2_score.json` | `/home/<user>/j-lens  ->  $JLENS_HOME`: 1; `<hostname>  ->  gpu-host`: 1; `internal board post number  ->  P-id`: 2 |
| `results/claude-normtest/stage2/stage2_score.stdout` | none: byte-identical |

Two post ids are new in this update: `P147` and `P148`. Both posts are later than every post already mapped, so the ids stay in time order. Their text is not in `prereg/posts/`.

## Added 2026-11-19: the J54 concept-test pass mark

One file copied from the experiment directory: `prereg/bars/J54-concept-bar-v4.2.md` (original md5 `2217480f0e0321ef927fdde9b7f921d3`, repo copy `4a62b0b3ce9773779e6b22ab09e119ae`). 15 of its 117 lines differ from the original, and only by these replacements: `internal task-card id -> card C01`: 1; `internal task-card id -> card C05`: 1; `internal board post number -> P-id`: 12; `the GPU machine's name -> [gpu-host]`: 2; `internal file path -> [internal path]`: 1.

Six post ids are new in this update: `P149` to `P154`. All six posts are later than every post already mapped, so the ids stay in time order. Their text is not in `prereg/posts/`. One card id is new: `card C05`.

Seat names in the file (Claude, Deron, Lens, Qwen) are left as written; the top-level README says who they are.

## Post #8 evidence (added 2026-11-24): the 21 Nov hidden-step readout

Byte copies of the originals with these replacements; Python files still parse, JSON still loads.

| file | original md5 | published md5 | replacements |
|---|---|---|---|
| `code/concept/concept_score.py` | `c56b083a11fb79511aabfe4d0962df53` | `22763366d5ae946f6b3fc468ca5be929` | `/home/<user>/j-lens` -> `$JLENS_HOME`: 2; `/home/<user> (other)` -> `/home/<user>`: 1; `card <id>` -> `the pre-registration card`: 1; `-t_[0-9a-f]{8}\.md` -> `.md`: 1; `(<seat> <date>` -> `(<date>`: 1; `<seat> <post>` -> `review note`: 2; `<seat name>` -> `a reviewer`: 2 |
| `code/concept/concept_common.py` | `6036073662606dcf96849beccb4cc333` | `06edbf200e6293e0109b98fdac5a46d1` | `card <id>` -> `the pre-registration card`: 1; `(<seat> <date>` -> `(<date>`: 1; `<seat> <post>` -> `review note`: 1; `<seat name>` -> `a reviewer`: 1 |
| `results/concept/concept_score.json` | `4cf8834313d91cd47609d0cbfa29b38f` | `2ccae811e439c3da160d984b1b30f864` | `/home/<user>/j-lens` -> `$JLENS_HOME`: 6; `<hostname>` -> `gpu-host`: 1; `-t_[0-9a-f]{8}\.md` -> `.md`: 1 |
| `results/concept/concept_score_standin_random_states.json` | `682a5bc40fa1e8f155c37f691974c480` | `a32b2eb0b9c0a3daf4a777657c67876d` | `/home/<user>/j-lens` -> `$JLENS_HOME`: 5; `<hostname>` -> `gpu-host`: 1; `-t_[0-9a-f]{8}\.md` -> `.md`: 1 |
| `results/concept/decoys.json` | `7ec0280becdebc09e4dd9e1ee41c6bcd` | `7ec0280becdebc09e4dd9e1ee41c6bcd` | none |

Not published: the captured activations (`concept_h.pt`, large, the model's internal states: the owner's call) and the item files (they come from the paper's public data repository at commit 581d3986; cite it there).
