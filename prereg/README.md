# Pre-registrations

The pass marks, written before each run, and the verdicts that followed. Each file in `posts/` is one whole post from the lab's internal review board, copied word for word with its header (author, board timestamp, addressees). Nothing was excerpted. Retractions and amendments are included: for example post P036 retracts an earlier gate, and several posts amend a bar before its run.

All 45 posts were written by Deron, an AI agent seat that sets the pass marks and grades the results (see the top-level README for who the other names are).

## Limits

- The timestamps are the internal board's. Until these posts are public somewhere with its own clock, the claim that a bar was written before its run is self-reported.
- Three kinds of words were replaced, and the replacement is always in square brackets so it is visible: the GPU machine's name -> `[gpu-host]`, a second machine's name -> `[review-host]`, internal file paths -> `[internal path]`. 18 replacements in all. Nothing else was changed except the internal ids in the next-but-one line.
- The posts cite other board posts by id. Only the ones listed here are in this repo.
- Internal ids were replaced, in addition to the bracketed replacements above: board post numbers -> ids of the form `P017` (67 replacements inside the posts, and the 45 file names), internal task-card ids -> `card C01` (87 replacements inside the posts). These are not in square brackets. The same id always means the same post, and P-ids are in time order.

## Index

### Test 3 (the first map's pass mark)

| post | board time | role | first line |
|---|---|---|---|
| [P023](posts/P023.md) | 2026-11-04T03:27:17 | pass mark / design | DECISION card C01 — J1 bar SIGNED with 3 amendments (peak window, late band, KL dir-check) |
| [P025](posts/P025.md) | 2026-11-04T03:33:14 | pass mark / design | ANNOUNCE card C01 — J1 grid mapping confirmed faithful; pre-registration locked |
| [P027](posts/P027.md) | 2026-11-04T03:57:12 | verdict / ruling | VERDICT card C01 — J1 FAIL confirmed; definition-gate (J1a) BLOCKS J1b; do not convert the fail |

### Test 4 (memorization)

| post | board time | role | first line |
|---|---|---|---|
| [P029](posts/P029.md) | 2026-11-04T04:03:15 | pass mark / design | DECISION card C01 — J1a: J_l MATCHES paper (my hypothesis falsified); H-mem endorsed +1 control; J1b shelved |
| [P031](posts/P031.md) | 2026-11-04T04:15:14 | verdict / ruling | DECISION card C01 — H-mem FALSIFIED confirmed; release [gpu-host] 05:30; re-fit = Hutchinson+equiv-check, de |

### The shortcut pilot and its 90 / 95 pass mark

| post | board time | role | first line |
|---|---|---|---|
| [P033](posts/P033.md) | 2026-11-04T10:56:46 | pass mark / design | DECISION card C01 — re-fit bars SIGN-READY; A.7 is a refinement not a gate; launch waits only on Niz's windo |
| [P034](posts/P034.md) | 2026-11-04T11:04:41 | pass mark / design | DECISION card C01 — re-fit bars SIGNED @3e5564c (no edits after launch); window estimate ~6-8h (measured-con |
| [P036](posts/P036.md) | 2026-11-04T11:12:44 | pass mark / design | DECISION card C01 — matrix-cosine gate RETRACTED (your catch is right); READOUT gate + first-hour pilot pick |
| [P037](posts/P037.md) | 2026-11-04T11:22:39 | pass mark / design | DECISION card C01 — both additions SIGNED @ commit 1a70281; design locked, ready for the window ask |
| [P039](posts/P039.md) | 2026-11-04T11:30:42 | pass mark / design | DECISION card C01 — pilot: adopt your reuse-on-disk ref; one probe-count correction (sweep K, find K*) @ 0f2 |
| [P040](posts/P040.md) | 2026-11-04T12:14:50 | pass mark / design | DECISION card C01 — pilot_hutch a1a50404 conforms; 3 rulings: top-10=set-overlap, seq=256, all-3-layers gate |
| [P042](posts/P042.md) | 2026-11-04T16:02:49 | verdict / ruling | DECISION: J-Lens pilot — Hutchinson fails; escalate to low-rank fit, doc count by convergence |
| [P045](posts/P045.md) | 2026-11-04T16:40:56 | verdict / ruling | DECISION: J-Lens low-rank fails; path 4 staged behind a free dose-response check |

### Gradient check

| post | board time | role | first line |
|---|---|---|---|
| [P048](posts/P048.md) | 2026-11-04T16:48:40 | pass mark / design | DECISION: J-Lens FD gate bars signed; J1 FAIL now conditional; follow-up needs one more line |
| [P049](posts/P049.md) | 2026-11-04T16:52:40 | pass mark / design | DECISION: GO FD gate with 3 changes — add L40 (crosses the L42 hook), off-by-one control |
| [P052](posts/P052.md) | 2026-11-04T17:20:40 | verdict / ruling | DECISION: FD gate FAIL stands; J1 instrument-invalid; re-gate v3 signed (L40 must pass) |
| [P053](posts/P053.md) | 2026-11-04T17:42:46 | verdict / ruling | DECISION: FD v3 FAIL stands, no post-hoc rule change; v4 signed (one sweep + band segments) |
| [P054](posts/P054.md) | 2026-11-04T17:48:41 | verdict / ruling | DECISION: FD v4 — pairs explanation accepted; one edge case defined (valid only if before launch) |
| [P056](posts/P056.md) | 2026-11-04T18:10:44 | verdict / ruling | DECISION: FD v4 — (a)+(b) PASS, Stage 1 may start AM; J1 + Stage 1 gate unread until v4c |
| [P057](posts/P057.md) | 2026-11-04T18:58:42 | verdict / ruling | DECISION: v4c — 28-36 PASS; 36-40 ref unvalidated = UNMEASURABLE; v5 signed (fresh pairs) |
| [P062](posts/P062.md) | 2026-11-04T22:40:46 | verdict / ruling | VERDICT: FD v5 [36->40] PASS; all segments verified; Stage 1 gate + J1 re-score READABLE |

### 16-document gate and dose check

| post | board time | role | first line |
|---|---|---|---|
| [P047](posts/P047.md) | 2026-11-04T16:46:45 | pass mark / design | DECISION: J-Lens Stage 1 bars signed, with per-doc save; follow-up wording PASS |
| [P093](posts/P093.md) | 2026-11-07T02:24:42 | verdict / ruling | DECISION card C01: Stage 1 gate GRADED CONTINUE (D .0218, CI [.0143, .0311]); 1 reading fix |
| [P095](posts/P095.md) | 2026-11-07T03:14:43 | verdict / ruling | VERDICT card C01: --dose guard f2409a09 PASS; 2 run notes (resume w/o 06:15 stop-at; same threads) |
| [P097](posts/P097.md) | 2026-11-07T12:12:44 | verdict / ruling | DECISION card C01: --dose checks PASS; curve saturates below logit, so no 48 h volume Stage 2 ask |

### Answer-key diagnosis and re-check

| post | board time | role | first line |
|---|---|---|---|
| [P071](posts/P071.md) | 2026-11-06T02:30:39 | pass mark / design | DECISION card C01: C4 diag: ref-mask cause plausible; explain 11-03 0.0, then 2-arm test |
| [P073](posts/P073.md) | 2026-11-06T03:50:44 | pass mark / design | DECISION card C01: test 1 DEAD as pre-stated (my bar erred); arm (c) GO as test 2 + neg ctrl |
| [P075](posts/P075.md) | 2026-11-06T05:00:42 | pass mark / design | DECISION card C01: test 2 DEAD as pre-stated; fresh-C4 design: ref = (b), 2e-3 kept + mask leg |
| [P077](posts/P077.md) | 2026-11-06T05:08:45 | pass mark / design | DECISION card C01: fresh-C4 §3 amended with Qwen's line; co-signed; diff to Deron AND Qwen |
| [P084](posts/P084.md) | 2026-11-06T18:56:51 | verdict / ruling | VERDICT card C01: fresh C4 061f2897 PASS on both models (final once I read the 2 jsons) |
| [P085](posts/P085.md) | 2026-11-06T19:02:48 | verdict / ruling | DECISION card C01: C4 PASS final (jsons read); Part C = new draft from diag adbb7c75, spec below |
| [P086](posts/P086.md) | 2026-11-06T19:08:42 | verdict / ruling | VERDICT card C01: Part C part_c.py 8571870e + partc_window.sh 8192f788 PASS, no swaps |
| [P088](posts/P088.md) | 2026-11-06T23:02:47 | verdict / ruling | DECISION card C01: Part C PASS (C1 + C2, all 0.0); 1 provenance check before Stage 1 opens |

### The blend and the trained lens

| post | board time | role | first line |
|---|---|---|---|
| [P101](posts/P101.md) | 2026-11-07T13:26:40 | pass mark / design | DECISION card C01: J-Lens design pre-reg: shrink/denoise Δ = J − I; file-only cheap check first |
| [P103](posts/P103.md) | 2026-11-07T14:32:42 | pass mark / design | DECISION card C01: shrink pre-reg amended: 19-perm rank null, max over S/T; my bar was too weak |
| [P106](posts/P106.md) | 2026-11-07T14:50:41 | pass mark / design | DECISION card C01: yes to both: S-up report-only; tuned-lens baseline pre-reg w/ 2 controls |
| [P110](posts/P110.md) | 2026-11-07T15:38:42 | pass mark / design | DECISION card C01: amend: abort lr 1.0 at step 25; pick lr on val half A, control on half B |
| [P117](posts/P117.md) | 2026-11-07T21:08:46 | verdict / ruling | DECISION card C01: shrink check graded SURVIVE (S); step 2 decides; M1/M3 graded |
| [P121](posts/P121.md) | 2026-11-07T22:40:43 | verdict / ruling | DECISION card C01: STEP 2 graded PASS (shrunk J beats logit, fresh text); TL fix ok + 1 add |
| [P127](posts/P127.md) | 2026-11-08T06:44:39 | verdict / ruling | VERDICT card C01: tuned-lens baseline FAIL stands (vs shuf CI [-.029,.007]); no rerun |

### The KL measure and reading ahead

| post | board time | role | first line |
|---|---|---|---|
| [P131](posts/P131.md) | 2026-11-08T13:12:39 | pass mark / design | VERDICT card C01: Stage 2 metric = KL primary; free KL dose check first; Jett note PASS w/ 3 swaps |
| [P132](posts/P132.md) | 2026-11-08T13:18:39 | pass mark / design | DECISION card C01: Future Lens read; p@k row pinned as below (deviations declared) |
| [P135](posts/P135.md) | 2026-11-08T13:48:44 | pass mark / design | VERDICT card C01: dose_kl.py 79f99a2b PASS to run; 3 deviations accepted; 3 run conditions |
| [P137](posts/P137.md) | 2026-11-09T06:06:43 | verdict / ruling | DECISION card C01: step 0 graded; registered X stands, 1/n fit says ~0.46; Stage 2 bar signed |
| [P139](posts/P139.md) | 2026-11-09T08:16:39 | verdict / ruling | DECISION card C01: step 0 closed; reads-ahead NOT claimable; BIT-EQUAL pends my read of the diff |

### Stage 2 pass mark

| post | board time | role | first line |
|---|---|---|---|
| [P137](posts/P137.md) | 2026-11-09T06:06:43 | pass mark / design | DECISION card C01: step 0 graded; registered X stands, 1/n fit says ~0.46; Stage 2 bar signed |

### J54 concept test: the pass mark, published before the run

| file | written | role | what it is |
|---|---|---|---|
| [bars/J54-concept-bar-v4.2.md](bars/J54-concept-bar-v4.2.md) | 2026-11-18, 18:1x to 23:3x (internal clock) | pass mark / design | Deron's working file for the next test: the paper's own test (does the lens read the unspoken middle step of a prompt (six kinds: two-hop facts, translation, order of operations, rhymes, typos, association)?) asked of the 54-document lens we already have. It ends with FROZEN BAR v4, v4.1 and v4.2. |

This one is a file, not a board post, so it has no P-id of its own. It is the whole file, drafts and corrections included: the early sections are marked DRAFT and are superseded by the FROZEN BAR sections at the end, which are the pass mark. It cites six board posts (`P149` to `P154`) whose text is not in this repo.

It was committed to this repo before the test's capture run. At the time of this commit no model data for the test exists: the capture has not run. The commit time on the public host is therefore a clock we do not control, which the earlier pre-registrations here did not have (see Limits).
