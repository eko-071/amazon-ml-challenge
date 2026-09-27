# Business Entity Resolution Pipeline — Fix & Improvement Spec

This document describes every known problem in the current pipeline
(`bulletproof_multipass.py`, `run_inference.py`, `train_better_xgb.py`,
`merge_final.py`), why it matters, and exactly how to fix it. Ordered by
priority — fix Tier 1 before touching anything else, since those bugs make
score improvements from later tiers unmeasurable or moot.

Each item includes: the problem, root cause, why it matters for the F_0.5
metric or pipeline correctness, the concrete fix, and expected performance
impact (compute time, memory) so implementation can be scoped and sequenced
sensibly.

---

## Tier 0 — Target project structure (applied incrementally, one issue at a time)

This is **not a separate upfront restructuring commit.** Each file below
moves to its new location as part of the same commit that fixes the issue
associated with it — e.g. the `merge_final.py` glob-path fix (item 1.1)
*is* the commit that also renames it to `src/submission/merge_submission.py`.
Treat the mapping table below as a routing reference for "where does this
file end up" as you work through Tiers 1-3, not as something to execute in
one pass. No stub files, no empty `__init__.py` placeholders, no directory
skeleton created ahead of need — a directory and its `__init__.py` get
created the first time a real module actually lands in it, as part of that
module's own commit.

### Why the current layout is a problem

The four scripts (`bulletproof_multipass.py`, `run_inference.py`,
`train_better_xgb.py`, `merge_final.py`) are siblings with no shared
module — this is *why* `run_inference.py` and `train_better_xgb.py`
currently duplicate the entire rapidfuzz feature-extraction block instead
of importing a common function, which is exactly the train/inference skew
risk flagged in item 2.6. There's also nowhere for cached artifacts
(fitted vectorizers, embedding indices, transliterated text, numeric
inverted indices) to live in a way that's clearly separated from source
code and from the `dataset/`/`output/` directories that are already
`.gitignore`d.

### Target structure

This fits inside the existing top-level layout
(`code/business_entity_resolution/{pyproject.toml,uv.lock,README.md,src/}`)
established earlier — only the *inside* of `src/` changes, plus one new
top-level `cache/` directory alongside `dataset/`/`output/`. Shown fully
assembled below for reference; it's reached gradually as each numbered
issue's commit lands, not created all at once.

```
code/business_entity_resolution/
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
    ├── common/
    │   ├── __init__.py
    │   ├── config.py           # NEW — central path constants, replaces hardcoded
    │   │                       #   "dataset/test/..." strings scattered across scripts
    │   ├── normalize.py        # normalize_text() + transliterate_to_latin() extracted
    │   │                       #   from bulletproof_multipass.py (items 2.1, 2.3)
    │   ├── features.py         # NEW — compute_features() shared by training and
    │   │                       #   inference (fixes the duplication behind item 2.6)
    │   └── metrics.py          # NEW — macro_f05() / compute_entity_f05() (item 1.3)
    │
    ├── blocking/
    │   ├── __init__.py
    │   ├── fit_target_index.py     # NEW — one-time fit+cache phase per target source:
    │   │                           #   TF-IDF vectorizers, embedding index, numeric
    │   │                           #   inverted index (fixes item 3.1's per-chunk refit)
    │   ├── generate_candidates.py  # ← renamed from bulletproof_multipass.py; becomes
    │   │                           #   transform/query-only once fit_target_index.py exists
    │   ├── embedding_index.py      # NEW — multilingual embedding blocking net (item 2.2)
    │   └── numeric_index.py        # NEW — numeric-token blocking net (item 2.4)
    │
    ├── training/
    │   ├── __init__.py
    │   ├── build_train_candidates.py  # NEW — generates candidates for all
    │   │                               #   source × country combinations (item 1.2)
    │   ├── train_model.py             # ← renamed from train_better_xgb.py; uses
    │   │                               #   common/features.py, adds val split (item 1.3)
    │   └── calibrate_threshold.py     # NEW — calibration + F_0.5 threshold sweep (item 1.3)
    │
    ├── inference/
    │   ├── __init__.py
    │   └── predict.py           # ← renamed from run_inference.py; uses
    │                            #   common/features.py + saved calibrator/threshold
    │
    ├── submission/
    │   ├── __init__.py
    │   └── merge_submission.py  # ← renamed from merge_final.py; glob path + sanity
    │                            #   checks fixed (item 1.1)
    │
    └── data.ipynb               # existing EDA notebook, unchanged location

cache/                           # NEW, top-level, sibling to dataset/ and output/
├── target_indices/              # fitted vectorizers, embedding indices, numeric
│                                 #   indices — one set per target source (item 3.1)
└── translit/                    # cached transliterated text columns (item 2.1)
```

`cache/` is generated, not source — add it to `.gitignore` alongside
`dataset/` and `output/` (see the `.gitignore` from earlier in this
project). It doesn't need to ship in the final submission zip, since
anything in it is fully reproducible by rerunning `fit_target_index.py`.

### File mapping — which commit each file belongs to

Each row is one atomic commit: fix the issue *and* write the file directly
at its new location in the same commit (just write the new file fresh at
the new path and delete the old one — no need to preserve rename history
with `git mv` if you're rewriting the content anyway).

| Old file (deleted in its commit) | New file (written fresh in the same commit) | Belongs to issue |
|---|---|---|
| `merge_final.py` | `src/submission/merge_submission.py` | 1.1 (glob path + sanity checks) |
| *(none — new file)* | `src/common/metrics.py` | 1.3-harness (`macro_f05()` + entity-level split only — calibration deferred, see note below) — first commit to introduce `src/common/` |
| `train_better_xgb.py` | `src/training/train_model.py` | 1.2 (multi-source training) — this commit is also the first one to introduce `src/training/` |
| *(edit in place)* | `src/training/train_model.py` | 1.3-calibrate (calibration + threshold sweep, now run against the 1.2 model, not the old US/S2-only one) |
| *(none — new file)* | `src/blocking/fit_target_index.py` | 3.1 (one-time fit+cache phase, keyed by `{split}/{source}` — see note below) |
| `bulletproof_multipass.py` | `src/blocking/generate_candidates.py` + `src/common/normalize.py` | 2.1 (transliteration) — this is where `normalize_text()` gets extracted out to `common/normalize.py`, since 2.1 is the first change that touches it |
| *(none — new file)* | `src/blocking/numeric_index.py` | 2.4 (numeric-token blocking) |
| `run_inference.py` | `src/inference/predict.py` + `src/common/features.py` | 2.6 (shared feature module *only* — extracts the duplicated rapidfuzz block from `train_model.py`/`predict.py` into one place; does **not** add an embedding feature yet, since 2.2 hasn't landed at this point in the sequence) |
| *(edit in place)* | `src/blocking/generate_candidates.py` | 2.3 (separate name/address vectorizers + total-K capping — see note below) |
| *(none — new file)* | `src/blocking/embedding_index.py` | 2.2 (multilingual embedding blocking net; this commit also extends `common/features.py` with `name_embedding_cosine`/`addr_embedding_cosine`, since that's the first point the score actually exists) |
| *(unchanged)* | `utils/validate_submission.py` | Stays where it is — organizer-provided validator, not part of your own source tree |

`src/common/config.py` isn't tied to one specific issue — introduce it
whenever the first commit that would otherwise hardcode a new path string
happens (likely 1.2 or 2.1), and add to it incrementally rather than
writing it fully upfront.

**Two ordering wrinkles worth naming explicitly, since neither is a bug —
just a dependency direction worth being aware of while implementing:**

- **1.2 runs before 2.1/3.1 in the sequence, but 1.2 needs train-time
  blocking to generate its candidate files.** Since `bulletproof_multipass.py`
  hasn't been split into `fit_target_index.py` / `generate_candidates.py`
  yet at that point, 1.2's `build_train_candidates.py` will necessarily
  duplicate blocking logic from the *old*, still-flat script (refit
  per-chunk, no transliteration) rather than call the eventual clean
  version. That's fine — it unblocks training immediately — but treat that
  duplicated logic as disposable: once 2.1 and 3.1 land, revisit whether
  `build_train_candidates.py` should be rewritten to call the new
  `fit_target_index.py`/`generate_candidates.py` instead of keeping its own
  copy, rather than letting both versions of the same blocking logic
  persist side by side indefinitely.
- **`train_model.py` is touched in three separate commits** — created in
  1.2, edited again in 1.3-calibrate, and edited again in 2.6 (once
  `common/features.py` exists, `train_model.py` should be updated to import
  from it instead of keeping its own inline feature block). None of these
  are the "final" version — don't treat the 1.2 commit's `train_model.py`
  as done; it's expected to keep changing through 1.3 and 2.6.

### Notes for whichever commit does the moving

- **`common/features.py` (issue 2.6) is where the train/inference
  duplication actually gets resolved** — until that commit,
  `train_model.py` and `predict.py` will still each have their own copy of
  the rapidfuzz feature block (inherited from `train_better_xgb.py` /
  `run_inference.py`). That's fine; don't extract it early "just in case."
  Extracting a shared module before there's a second real caller is
  premature — wait for the 2.6 commit, where the extraction *is* the fix.
- **`.gitignore`**: add `cache/` (for issue 3.1's fitted vectorizers,
  embedding indices, numeric indices) in whichever commit first writes
  something under `cache/` — no need to add it speculatively earlier.
- **Relative paths**: all four current scripts hardcode paths like
  `"dataset/test/test_source1.tsv"`, relying on being invoked from the
  project root. Moving a script into a subdirectory doesn't break this as
  long as invocation still happens from the project root (e.g.
  `python src/blocking/generate_candidates.py`) — so this isn't something
  that needs fixing purely because of the move. Still worth centralizing
  into `common/config.py` once that file exists, but treat it as part of
  whichever issue's commit first duplicates a path string a third time,
  not as a mandatory step of the restructuring itself.
- **Imports**: each file that moves into a subdirectory and starts
  importing from `common/` needs `src/` on the Python path — either
  `PYTHONPATH=src` in run commands, or installing the project in editable
  mode via `uv` (consistent with the existing `pyproject.toml`/`uv.lock`).
  Update the run command in `README.md` in the same commit that moves the
  file, so the README never describes an invocation that doesn't work.
- **`scripts/run_pipeline.sh` / `Makefile`**: optional, and only makes
  sense once most of the individual pieces exist — treat it as its own
  commit near the end, not something to scaffold early.

**Performance impact:** none — this only changes where code lives, not
what runs. No cost to training time, blocking time, or memory.

---

## Tier 1 — Correctness bugs (blocking issues, fix first)

### 1.1 `merge_final.py` reads from the wrong directory

**Verdict: objective — do for sure.**

**File:** `merge_final.py`

**Problem:** `bulletproof_multipass.py` writes candidate files to
`output/multipass_cands_s{target_source}_{chunk}.tsv`, and `run_inference.py`
writes match files to `output/matching_results_s{target_source}_{chunk}.tsv`.
But `merge_final.py` globs:
```python
process_and_merge('final_output/matching_results_*.tsv', 'matched_entity_ids', 'matching_results.tsv')
process_and_merge('final_output/multipass_cands_*.tsv', 'candidate_entity_ids', 'candidate_pairs.tsv')
```
`final_output/` is a directory nothing else in the pipeline writes to.
`glob.glob()` against a pattern that matches zero files returns `[]` —
`process_and_merge` detects this (`if not files: print(...); return`) and
exits early **without raising an error**. The function returns silently,
`base_submission` (all S1 entities, no matches) is never written to for
that call, and whatever `output/matching_results.tsv` /
`output/candidate_pairs.tsv` already exists on disk from a previous run (or
doesn't exist at all) is what actually gets submitted.

**Why it matters:** This is not a minor bug — it can produce a submission
that is silently wrong (either stale from an old run, or missing entirely)
with a console message (`"No files found for final_output/..."`) that is
easy to miss in a longer log. If `output/matching_results.tsv` doesn't
exist at all, the final zip/leaderboard upload step fails obviously; if it
exists from a stale earlier run, you submit outdated results without
realizing it. Given the F_0.5 macro-average scoring, a submission that's
accidentally all-singletons (or just stale) scores far below what the
actual pipeline is capable of.

**Fix:**
1. Change the glob patterns in `merge_final.py` from `final_output/...` to
   `output/...`, matching where `bulletproof_multipass.py` and
   `run_inference.py` actually write:
   ```python
   process_and_merge('output/matching_results_s*_*.tsv', 'matched_entity_ids', 'matching_results.tsv')
   process_and_merge('output/multipass_cands_s*_*.tsv', 'candidate_entity_ids', 'candidate_pairs.tsv')
   ```
   Note the pattern is `matching_results_s*_*.tsv` (source and chunk both
   wildcarded), not just `matching_results_*.tsv`, since filenames are
   `matching_results_s{target_source}_{chunk}.tsv` — confirm this against
   the actual filenames on disk before finalizing the pattern (a too-loose
   glob could double-count, e.g. if `matching_results.tsv` itself already
   exists in `output/` from a prior merge and matches the wildcard).
2. **Change `process_and_merge` to raise/exit loudly instead of silently
   returning** when no files are found:
   ```python
   if not files:
       raise RuntimeError(f"No files found for pattern {file_pattern} — aborting merge.")
   ```
   This converts a silent bad-submission risk into a build failure you
   can't miss.
3. After merging, add an explicit row-count and non-singleton-fraction
   sanity check before writing the final files, e.g.:
   ```python
   n_total = len(submission)
   n_nonempty = (submission[col_name] != "").sum()
   print(f"{out_name}: {n_total} rows, {n_nonempty} with non-empty {col_name} ({n_nonempty/n_total:.1%})")
   assert n_nonempty > 0, f"{out_name} has zero non-empty predictions — pipeline likely broken upstream."
   ```
   Given the EDA finding that only ~5.6% of training entities are true
   singletons, a `matching_results.tsv` where the non-empty fraction is
   anywhere near 0% (or near 100% — also suspicious) should trip this check.
4. Run `utils/validate_submission.py` immediately after `merge_final.py`
   as a hard gate in whatever script/Makefile orchestrates the pipeline —
   don't treat it as a manual pre-submission step; wire it into the run
   command so a bad merge fails the build.

**Performance impact:** None — this is a path/string and control-flow fix,
zero compute cost.

---

### 1.2 XGBoost model is trained on a single, narrow slice of the data

**Verdict: objective — do for sure.**

**File:** `train_better_xgb.py`

**Problem:** The training script reads exactly one candidate file:
```python
cands = pd.read_csv("final_results/train_candidates_US_S2.tsv", sep='\t')
```
This is candidates generated for **Source-1 ↔ Source-2 pairs, US country
only**. The resulting model (`xgb_model_v2.json`) has never seen a single
training example involving Source-3, and never seen a single training
example from India. `run_inference.py` then applies this model to **both**
target sources (`--target_source` arg takes value 2 or 3) and to **all**
countries in the test set (US, India, and France, which has zero training
data in any source).

**Why it matters:** The model's learned decision boundary — which
combination of `name_ratio`, `addr_token_set`, `addr_num_match`, etc.
indicates a true match — is fit entirely to US/Source-2 noise patterns. If
Source-3's noise characteristics differ from Source-2's (different address
formats, different name-abbreviation conventions, different missing-field
rates), or if India's naming/address conventions differ from US's (which
the EDA confirms — India data includes non-Latin script content that US
data does not), the model is extrapolating on out-of-distribution data for
the majority of actual test-time candidate pairs. This is very likely
costing more F_0.5 than any blocking refinement, because it affects every
single Source-3 prediction and every India prediction, not a subset.

**Fix:**
1. Generate training candidate files for **both target sources**:
   `train_candidates_S2.tsv` and `train_candidates_S3.tsv` — using the same
   blocking logic as `bulletproof_multipass.py` but pointed at
   `dataset/train/` instead of `dataset/test/`. Do **not** partition these
   files by country (no `train_candidates_US_S2.tsv` /
   `train_candidates_India_S2.tsv` split). Country stays as a *column* in
   the candidate/feature frame, not a filename partition — this is
   required, not optional: the problem statement's "country is an open
   set" rule (item 2.5 covers the same principle for test-time blocking)
   means any pipeline stage that hardcodes a fixed country list — even
   implicitly, by generating one file per known training country — will
   need special-casing the moment France-like unseen-country data needs to
   flow through the same code path. Keeping country as a feature column
   throughout, generated by whatever code already reads it per-record,
   sidesteps that entirely: the training pipeline doesn't need to know in
   advance which countries exist.
2. In `train_better_xgb.py`, load and concatenate both into one training
   frame:
   ```python
   import glob
   cand_files = glob.glob("final_results/train_candidates_S*.tsv")
   assert len(cand_files) >= 2, f"Expected candidate files for both target sources, found: {cand_files}"
   cands = pd.concat([pd.read_csv(f, sep='\t') for f in cand_files], ignore_index=True)
   ```
3. When merging candidates against `train_source2`/`train_source3` to pull
   in name/address text, merge against **whichever source file matches
   each row's actual target source** — don't hardcode `train_source2` the
   way the current script does. This requires tracking which source each
   candidate file's `candidate_entity_id`s came from (the `S2-`/`S3-`
   prefix on `entity_id` already encodes this — use it to route the merge,
   e.g. split `cands` into an S2-subset and S3-subset before merging each
   against its respective source file, then concatenate the feature frames
   back together).
4. Retrain and save as a new model file (e.g. `xgb_model_v3.json`) so you
   can A/B the old vs. new model on your validation split (see item 1.3)
   before committing to the replacement.
5. Sanity-check the resulting training set's composition before fitting —
   print the count and positive-rate breakdown by source and the *observed*
   country values (whatever they happen to be — don't assume `{US, India}`
   anywhere in this check either):
   ```python
   print(cands.groupby(['source', 'country'])['is_match'].agg(['count', 'mean']))
   ```
   (requires carrying a `source` and `country` column through the
   candidate-generation and merge steps — add these if not already
   present). If one combination is drastically smaller or has a wildly
   different positive rate than the others, that's worth understanding
   before training on the pooled set. This groupby is a diagnostic printout,
   not a filter or a branch in the code — it never becomes a hardcoded
   country list.
6. **Dependency note:** this commit's candidate-generation step duplicates
   blocking logic from the still-flat `bulletproof_multipass.py`, since
   2.1/3.1 (which split and clean up that script) haven't landed yet at
   this point in the sequence. See the "ordering wrinkles" note in Tier 0 —
   this is expected, not a mistake, but the duplicated logic should be
   revisited once 2.1/3.1 exist.

**Performance impact:**
- Candidate generation: running `bulletproof_multipass.py`-equivalent logic
  against training data adds roughly the same wall-clock cost as one full
  run of the existing test-time blocking script (since train_source1 is
  comparable in scale to test_source1) — call it ~1x the current blocking
  runtime, done once, not per-experiment.
- Training set size grows roughly ~2x (both target sources instead of one). XGBoost training time scales close to linearly in row
  count at fixed `n_estimators`/`max_depth`, so expect training time to
  grow proportionally — likely still well under an hour on CPU for a
  candidate set in the low-single-digit millions of rows, but budget for
  it and consider `tree_method='hist'` (already set) and reducing
  `n_estimators` during iteration/debugging, restoring it for the final
  run.
- Memory: the concatenated candidate+feature frame is proportionally
  larger; if this becomes a problem, process each target source's
  features separately and `xgb.DMatrix`-append or use
  XGBoost's external-memory training rather than holding everything as one
  pandas frame.

---

### 1.3 No held-out validation split, no probability calibration, no measured threshold

**Files:** `train_better_xgb.py` / `src/training/train_model.py`,
`run_inference.py` / `src/inference/predict.py`

**Split into two commits, not one — see why under "Sequencing note"
below.**

**Problem:** `train_better_xgb.py` fits on 100% of its training data with
no held-out evaluation set. `run_inference.py` calls
`model.predict(cands[features])`, which applies XGBoost's default 0.5
probability threshold. The only precision-leaning knob in use is
`scale_pos_weight=0.8` in the classifier constructor — a value that appears
to be a guess ("Penalizes false positives slightly to protect F_0.5" per
the code comment) rather than something measured against the actual F_0.5
formula.

**Why it matters:** Two independent problems compound here:
1. **Calibration:** candidate sets generated by blocking are heavily
   imbalanced (far more true negatives than true positives per
   Source-1 entity, since blocking retrieves top-K candidates and only 1-2
   are usually correct). Under this imbalance, raw GBDT output scores are
   not well-calibrated probabilities — the model can be systematically
   over- or under-confident in ways that don't matter for raw
   accuracy/AUC but matter a great deal once you threshold at a fixed
   value.
2. **Threshold selection:** F_0.5 weights precision 2x over recall. The
   threshold that maximizes accuracy or even F1 is generally *not* the
   threshold that maximizes F_0.5 — the optimal cutoff for a
   precision-heavy metric is typically higher than 0.5. Without measuring
   F_0.5 directly on held-out data across a range of thresholds, there's
   no way to know whether 0.5 (implicit) is anywhere near optimal, and no
   way to know whether `scale_pos_weight=0.8` is helping, hurting, or
   doing approximately nothing.

Without this, every change to features, blocking, or model
hyperparameters is unmeasurable — there's no local signal to tell whether
a change helped, only leaderboard submissions (capped at 5/day).

**Sequencing note:** don't calibrate yet in this commit if item 1.2 hasn't
landed. Calibrating against the current US/Source-2-only model, then
retraining on the full multi-source set in 1.2, means the calibration work
is immediately thrown away and has to be redone against the new model —
two calibration passes for one useful result. Split this into:
- **1.3-harness** (this commit, can land before or right alongside 1.2):
  entity-level train/val split logic and the `macro_f05` implementation —
  useful on its own even before calibration, since it already gives a real
  local metric to check blocking recall and raw model discrimination
  against.
- **1.3-calibrate** (a later commit, after 1.2's retrain exists): fit the
  calibrator and sweep the threshold against the *actual* model that will
  ship, not an intermediate one.

**Fix — 1.3-harness:**

**Verdict: objective — do for sure (measurement harness; calibration itself is measure-first, see 1.3-calibrate).**
1. **Split at the entity level, not the row level.** Hold out a random
   ~15-20% of `train_source1` entity IDs *before* generating candidates
   for them. If you split candidate rows randomly instead, candidate pairs
   for the same Source-1 entity can end up split across train and
   validation, which leaks information (the model implicitly learns
   entity-specific patterns from training rows that reappear, in effect,
   in validation rows for the same entity) and gives an optimistic bias to
   your validation score.
   ```python
   import numpy as np
   rng = np.random.RandomState(42)
   s1_ids = truth_df['source1_entity_id'].unique()  # or train_source1['entity_id']
   val_ids = set(rng.choice(s1_ids, size=int(0.18 * len(s1_ids)), replace=False))
   train_mask = ~cands['source1_entity_id'].isin(val_ids)
   val_mask = cands['source1_entity_id'].isin(val_ids)
   ```
2. **Implement the real F_0.5 macro-average** in `src/common/metrics.py` —
   per Source-1 entity, not per pair — with the per-entity true-match sets
   precomputed **once**, outside any threshold loop, since they don't
   depend on the threshold at all:
   ```python
   def macro_f05_sweep(df_val, thresholds, prob_col='prob'):
       \"\"\"Returns {threshold: macro_f05} for every threshold in one pass,
       computing true_by_entity and all_entities exactly once regardless
       of how many thresholds are swept.\"\"\"
       true_by_entity = (
           df_val[df_val['is_match'] == 1]
           .groupby('source1_entity_id')['candidate_entity_id']
           .apply(set)
       )
       all_entities = df_val['source1_entity_id'].unique()
       results = {}
       for t in thresholds:
           preds = df_val[df_val[prob_col] >= t]
           pred_by_entity = preds.groupby('source1_entity_id')['candidate_entity_id'].apply(set)
           scores = []
           for eid in all_entities:
               pred_set = pred_by_entity.get(eid, set())
               true_set = true_by_entity.get(eid, set())
               if not pred_set and not true_set:
                   scores.append(1.0)
               elif not pred_set or not true_set:
                   scores.append(0.0)
               else:
                   tp = len(pred_set & true_set)
                   p = tp / len(pred_set)
                   r = tp / len(true_set)
                   scores.append(0.0 if (p == 0 and r == 0) else (1.25*p*r)/(0.25*p+r))
           results[t] = np.mean(scores)
       return results
   ```
   Note `true_by_entity` and `all_entities` are computed exactly once
   before the `for t in thresholds` loop — the earlier draft of this fix
   recomputed `true_by_entity` inside a per-threshold function call, which
   is redundant work repeated once per swept threshold value (~30x with a
   typical sweep granularity) for a quantity that never changes across the
   sweep. `pred_by_entity` is the only thing that legitimately needs
   recomputation per threshold, since it's the only part that depends on
   `t`.
3. This must account for Source-1 entities with **zero candidates** at
   all (complete blocking misses) — those need to be included in
   `all_entities` with an automatic score of 0.0 if they have true
   matches, or 1.0 if they're true singletons, or the val F_0.5 will be
   systematically optimistic relative to the real leaderboard score, which
   includes every test entity regardless of blocking success.
4. Re-run this val-F_0.5 measurement after every meaningful pipeline
   change (new features, new blocking net, retrained model) so you have a
   fast local signal instead of spending one of your 5 daily leaderboard
   submissions to find out if a change helped.

**Fix — 1.3-calibrate (after 1.2's model exists):**

**Verdict: subjective — measure-first (isotonic vs sigmoid, threshold value, and `scale_pos_weight` all decided on val, never assumed).**
1. **Fit the model on the train split only**, then get raw predicted
   probabilities on the val split:
   ```python
   model.fit(cands.loc[train_mask, features], cands.loc[train_mask, 'is_match'])
   val_probs = model.predict_proba(cands.loc[val_mask, features])[:, 1]
   ```
2. **Calibrate using an already-fit model, not a refit one.**
   `sklearn.calibration.CalibratedClassifierCV` by default performs its own
   internal cross-validation and **refits the base estimator on each
   fold** — passing your already-trained `model` in without telling it not
   to silently discards the train/val separation from step 1, since the
   "calibrated" model ends up retrained on data that includes rows outside
   your intended train split. Pass `cv='prefit'` (sklearn <1.6) or wrap the
   model with `sklearn.frozen.FrozenEstimator` first (sklearn ≥1.6, where
   `cv='prefit'` is deprecated) so calibration only fits the calibration
   mapping itself, not the underlying model:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   # sklearn < 1.6:
   calibrator = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
   # sklearn >= 1.6:
   # from sklearn.frozen import FrozenEstimator
   # calibrator = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
   calibrator.fit(cands.loc[val_mask, features], cands.loc[val_mask, 'is_match'])
   ```
   Check the installed sklearn version before picking the syntax — the two
   are not interchangeable and the deprecated path will warn or fail
   depending on version. Use `method='isotonic'` if the validation set is
   large (more flexible, fine when there's enough data), or
   `method='sigmoid'` (Platt scaling) if it's smaller — isotonic can
   overfit with too few points. Ideally use a further internal split of
   validation into separate calibration/threshold-selection subsets, to
   avoid tuning both the calibration mapping and the decision threshold on
   the exact same rows.
3. Sweep thresholds using the `macro_f05_sweep` function from
   1.3-harness, applied to the *calibrated* probabilities:
   ```python
   calibrated_probs = calibrator.predict_proba(cands.loc[val_mask, features])[:, 1]
   val_df = cands.loc[val_mask].copy()
   val_df['prob'] = calibrated_probs
   results = macro_f05_sweep(val_df, np.arange(0.1, 0.95, 0.025))
   best_thresh = max(results, key=results.get)
   print(f"Best threshold: {best_thresh:.3f}, val macro F_0.5: {results[best_thresh]:.4f}")
   ```
4. Persist the calibrator and chosen threshold alongside the model
   (e.g. `xgb_model_v3.json` + `calibrator_v3.pkl` + a small JSON/text file
   recording the threshold), and have `predict.py` load and apply both
   rather than calling raw `model.predict()`.

**Performance impact:**
- Calibration fitting: negligible, operates on already-computed
  probabilities for the validation set only (thousands to low-millions of
  rows depending on validation size), sub-second to low-seconds.
- Threshold sweep: trivial with the hoisted version above —
  `true_by_entity`/`all_entities` computed once, `pred_by_entity` is the
  only per-threshold cost, so total cost is close to one pass over the
  validation set rather than ~30 passes. Should run in seconds to
  low-tens-of-seconds depending on validation set size.
- Net effect on total pipeline time: small addition (a validation-set
  scoring pass), but replaces "guess and submit" cycles that cost far more
  wall-clock time (and burn scarce leaderboard submissions) than local
  evaluation does.

---

## Tier 2 — Preprocessing & blocking (the multilingual / recall fix)

### 2.1 No transliteration pass — cross-script matches are likely being dropped by blocking

**Verdict: split — transliteration direction is objective (do for sure); library/scheme choice is measure-first (audit scripts on val first).**

**File:** `bulletproof_multipass.py` (`normalize_text`, vectorizer setup)

**Problem:** `normalize_text` lowercases, strips punctuation, and expands a
handful of Latin-script abbreviations (`corp`, `ltd`, `pvt`, `co`, `rd`,
`st`). It does nothing for non-Latin scripts. The two TF-IDF nets
(`word_vec`: word n-grams 1-2; `char_vec`: character 4-grams) are then
fit and applied directly on this normalized-but-still-multiscript text.

**Why it matters:** Character n-gram and word n-gram TF-IDF vectors for a
Latin-script string and a Devanagari-script string referring to the same
business share **zero or near-zero features** — they don't tokenize into
overlapping n-grams regardless of semantic similarity, because the
underlying Unicode code points are entirely different ranges. Since
Source-1 is confirmed 100% Latin script (from prior EDA) and a meaningful
fraction of Source-2/Source-3 names/addresses are non-Latin, every true
match where the Source-2/3 side is non-Latin is likely scoring ~0 cosine
similarity in both current blocking nets and is at serious risk of falling
out of the top-K candidate list entirely. Since blocking recall is a hard
ceiling on final pipeline recall (nothing downstream can recover a match
that never became a candidate), this is very likely the single largest
recall leak in the current pipeline for the India-country subset of the
data (and needs to be re-verified from scratch for France in the test
set — see item 2.5).

**Fix:**
1. Add a deterministic script-to-Latin transliteration step, applied to
   **all** text (Latin text should pass through unchanged or be trivially
   affected):
   ```python
   from indic_transliteration import sanscript
   from indic_transliteration.sanscript import transliterate

   def transliterate_to_latin(text):
       if not isinstance(text, str) or not text:
           return ""
       # Devanagari is the primary non-Latin script observed in EDA;
       # extend the scheme list if other scripts are found in the real data.
       try:
           return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
       except Exception:
           return text  # fall back to original text if transliteration fails
   ```
   Before committing to `indic_transliteration` specifically, re-run (or
   extend) the script-audit section of the EDA notebook against the full
   real dataset to confirm which non-Latin Unicode blocks actually appear
   (Devanagari, Tamil, Bengali, etc.) — the library and transliteration
   scheme need to cover whatever's actually present, not just the
   Devanagari example from the synthetic smoke test.
2. Apply this **before** `normalize_text`, so the abbreviation-expansion
   regexes in `normalize_text` (which are Latin-only, e.g. `\b(road|rd)\b`)
   operate on already-transliterated text and have a chance to fire on
   romanized versions of previously non-Latin content:
   ```python
   def full_normalize(text):
       return normalize_text(transliterate_to_latin(text))
   ```
   and use `full_normalize` everywhere `normalize_text` is currently
   called in `bulletproof_multipass.py`.
3. State this explicitly in the methodology document as a deterministic,
   rule-based script conversion — not an external business-identity
   lookup or data augmentation — to keep it unambiguously inside the
   challenge's fair-play rules (no external database/API lookups).
4. Re-run the blocking-recall measurement (recall@K against ground truth,
   as discussed in the original pipeline design) split out by whether the
   true match's original text was Latin or non-Latin, to directly confirm
   this fix closes the gap it's meant to close, rather than assuming it
   does.

**Performance impact:**
- Transliteration itself is a pure-Python, per-string operation with no
  network/external calls — cost scales linearly with total character
  count across all name/address fields. For a library like
  `indic_transliteration`, expect roughly low-microseconds-to-low-tens-of-microseconds
  per short string; across the low-millions of records in Source-2/3, this
  is a one-time preprocessing pass on the order of tens of seconds to a
  few minutes, not a per-chunk or per-query cost — **compute it once per
  source file and cache/persist the transliterated column** (e.g. write an
  augmented `business_name_translit` / `business_address_translit` column
  back to a preprocessed parquet/TSV) rather than recomputing inside every
  chunked invocation of `bulletproof_multipass.py`. Given the script
  currently reloads and reprocesses text per chunk/target-source
  invocation, failing to cache this would multiply the transliteration
  cost by `total_chunks × number of target sources` for no reason.
- No meaningful memory impact — output is a same-order-of-magnitude string
  column alongside the existing text columns.

---

### 2.2 Add a multilingual embedding retrieval net as a third blocking signal

**Verdict: subjective — measure-first (largest compute cost; do matcher-only cosine before full retrieval).**

**File:** `bulletproof_multipass.py`

**Problem:** Current blocking relies exclusively on two sparse lexical
signals (word n-gram TF-IDF, character n-gram TF-IDF). Even after adding
transliteration (item 2.1), romanization is imperfect and inconsistent —
the same Devanagari string can transliterate to slightly different Latin
spellings depending on the scheme or the original source's own
romanization choices, which lexical n-gram overlap is still somewhat
brittle against. There's currently no semantic/paraphrase-level signal in
blocking at all.

**Why it matters:** A multilingual sentence embedding model is trained
specifically so that semantically equivalent strings — including across
scripts and languages, and including paraphrase-level variation that exact
transliteration won't normalize away — land close together in embedding
space. This is a genuinely different, complementary signal to lexical
n-gram overlap, not a redundant one: it should recover some fraction of
true matches that both TF-IDF nets (even with transliteration) still miss,
particularly where transliteration + noise compounds (e.g. a
transliterated name that also has a typo, or a translated rather than
transliterated name).

**Fix:**
1. Choose a small, permissively-licensed multilingual sentence embedding
   model. `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M
   params) is a reasonable default; if broader script/language coverage is
   needed after confirming the real script-audit results, consider
   `bge-m3` (MIT, ~568M params) — verify current license terms and param
   count directly before committing, since these details can change
   between model card revisions. Both are comfortably under the
   challenge's 8B-parameter ceiling.
2. Encode the (transliterated or original — test empirically which
   performs better, or concatenate both as separate embedding inputs)
   name text for every record in each target source, once, and build an
   approximate nearest-neighbor index (FAISS `IndexHNSWFlat` or
   `IndexIVFFlat` depending on corpus size) over the resulting embeddings.
3. For each Source-1 chunk, encode its (transliterated) name text and
   query the ANN index for top-K nearest neighbors, analogous to how
   `get_top_k` currently extracts top-K from the sparse TF-IDF score
   matrices.
4. **Union** this embedding-based candidate set into the existing
   `union_idx = set(w_top).union(set(c_top))` line, e.g.
   `union_idx = set(w_top).union(set(c_top)).union(set(e_top))`.
5. Persist the per-record embedding cosine similarity score for whichever
   candidates make it through blocking — this becomes a genuinely useful
   **matcher feature** too (see item 2.6), not just a retrieval signal, so
   don't discard it after blocking.

**Performance impact — this is the most compute-intensive addition in this
spec, plan for it explicitly:**
- **Encoding cost:** embedding a few million short strings with a ~118M
  parameter transformer is meaningfully more expensive than TF-IDF
  vectorization. On CPU, expect on the order of hundreds of strings/sec to
  low-thousands/sec depending on hardware and batch size — for a
  multi-million-row source file, this is plausibly tens of minutes to a
  few hours on CPU. **Strongly prefer GPU if available at all** — the same
  workload is typically one to two orders of magnitude faster on even a
  modest GPU. Batch aggressively (e.g. batch size 256-1024 depending on
  memory) rather than encoding row-by-row.
- **Caching is essential, same principle as item 2.1:** encode each target
  source's records exactly once, persist the resulting embedding matrix
  (e.g. as a `.npy` file or FAISS index serialized to disk) keyed by
  entity_id, and load it in every chunked invocation rather than
  re-encoding per chunk/per target-source run. Given the current script
  structure re-fits everything per invocation, this is the single biggest
  place a naive implementation could balloon total runtime — re-encoding
  the same ~5M-row target source 4 times (once per chunk) for no reason
  would multiply an already-expensive step by 4x.
- **Index build cost:** building an HNSW or IVF index over millions of
  embedding vectors is itself non-trivial (minutes, not seconds, depending
  on parameters) — build once, persist to disk (FAISS supports
  `write_index`/`read_index`), and load rather than rebuild per chunk.
- **Query cost:** ANN top-K queries are fast once the index is built
  (typically sub-millisecond to low-milliseconds per query), so the
  per-chunk query step itself is cheap relative to the one-time
  encode+index cost above.
- **Memory:** a 118M-param model in fp32 is roughly 470MB; embeddings
  themselves (e.g. 384-dim float32 per record × ~5M records ≈ 7.7GB) are
  the more significant memory cost — consider float16 storage for the
  embedding matrix/index if memory-constrained, and confirm FAISS index
  type memory overhead before choosing IVF vs. HNSW at this scale.

---

### 2.3 Separate name and address vectorization instead of concatenating fields

**Verdict: subjective — measure-first (A/B recall@K on val; total-K cap mandatory, 0.7/0.3 weight is a guess).**

**File:** `bulletproof_multipass.py`

**Problem:** Both TF-IDF nets are built on
`(business_name + " " + business_address)` concatenated into a single
string per record, then vectorized as one blob.

**Why it matters:** Addresses are typically much longer than names and
contain a lot of shared, low-discriminative vocabulary (street/road
tokens, city names shared across many businesses in the same area). When
concatenated and TF-IDF'd together, a long address can dilute the
contribution of a short, highly distinctive business name — two records
with very similar addresses (e.g. same street) but completely different
business names could end up with inflated combined similarity, while two
records with a strong name match but a slightly different address format
could see that name signal diluted. Separating the fields lets each be
weighted and evaluated on its own terms.

**Fix:**
1. Fit **four** vectorizers instead of two: `name_word_vec`,
   `name_char_vec`, `addr_word_vec`, `addr_char_vec`, each fit only on the
   respective field's (transliterated, normalized) text.
2. Retrieve top-K per field per net (so up to 4 candidate sets instead of
   2) with a **mandatory total-K cap** (pure 4-way union can reach ~80/S1,
   i.e. ~136M matcher pairs at test scale — infeasible for rapidfuzz
   scoring). Take a smaller K per net and union, or combine name and
   address scores with a weighted sum before taking top-K (e.g.
   `combined = 0.7 * name_score + 0.3 * addr_score`) rather than pure
   union, and tune the weighting empirically against blocking recall on
   the validation split.
3. Compare blocking recall@K before/after this change on the validation
   split (item 1.3) to confirm it's a net improvement before committing —
   it's a reasonable hypothesis but should be measured, not assumed.

**Performance impact:**
- Roughly doubles the vectorization and scoring cost of the current
  TF-IDF blocking step (4 vectorizer fits/transforms and 4 score matrix
  computations instead of 2), since name and address fields are now
  processed independently rather than as one combined string. This is a
  linear, predictable increase, not a complexity-class change — expect
  blocking wall-clock time for the TF-IDF portion specifically to
  increase by roughly 1.5-2x, which should still be small relative to the
  embedding-encoding cost in item 2.2 if that's implemented.
- Memory: doubles the number of sparse matrices held at once (name and
  address separately vs. one combined) — manageable given TF-IDF matrices
  are sparse and the vocabulary sizes here are modest (`min_df=2`,
  `max_df=0.05`/`0.01` already bound vocabulary growth).

---

### 2.4 Numeric-token blocking net (street numbers, postal codes)

**Verdict: objective — do for sure (frequency cutoff for common numbers is measure-first: pick one number on val).**

**File:** `bulletproof_multipass.py`

**Problem:** No blocking signal currently uses numeric tokens from
addresses at all — both TF-IDF nets operate on full text including
numbers, but numbers aren't treated specially, and dilute into the general
n-gram vocabulary rather than being used as a targeted signal.

**Why it matters:** Prior EDA established that street numbers and
postal/PIN codes tend to be near-invariant across language, script, and
formatting differences in ways that name/address text overall is not — a
numeric token like a building number or postal code is one of the few
signals that transfers cleanly to unseen conventions (relevant given
France appears only in test — see item 2.5). A record pair with a
completely different-looking address in every other respect but a shared,
distinctive numeric token (e.g. a 6-digit PIN code) is a strong,
language-agnostic match signal that the current TF-IDF-only blocking
doesn't specifically exploit — it may partially benefit by accident (since
numbers appear in the address text fed to TF-IDF) but isn't guaranteed to
surface such pairs into the top-K if the surrounding text differs a lot.

**Fix:**
1. Extract numeric tokens (2+ digit sequences, consistent with the
   `NUMERIC_TOKEN_RE = r"\b\d{2,}\b"` pattern already used in the EDA
   notebook) from each record's address field.
2. Build an inverted index: numeric token → list of entity_ids containing
   that token, for each target source.
3. For each Source-1 record, look up its own numeric tokens in this index
   and add any entity_ids sharing at least one token to the candidate
   union — this is a straightforward hash-map lookup, not a
   similarity-score computation, so it's cheap.
4. Union this candidate set into `union_idx` alongside the TF-IDF (and
   embedding, if implemented) candidates.
5. Consider excluding very common numeric tokens (e.g. tokens appearing in
   a large fraction of records, similar in spirit to `max_df` in the
   TF-IDF vectorizers) to avoid this net contributing near-useless
   high-frequency numbers (e.g. a common building-number digit) as
   candidates for a large fraction of records, which would bloat the
   candidate pool without adding recall.

**Performance impact:**
- Building the inverted index is a single linear pass over each target
  source's address field (regex extraction + dict insertion) — cost is
  small relative to TF-IDF fitting or embedding encoding, expect this to
  run in low seconds to tens of seconds even at millions of rows,
  dominated by regex extraction throughput.
- Lookup cost per Source-1 record is O(number of numeric tokens in that
  record's address), i.e. essentially constant per record — negligible
  addition to per-chunk processing time.
- Same caching principle as items 2.1/2.2 applies: build the inverted
  index once per target source, persist it (e.g. pickle the dict), and
  load it in each chunk invocation rather than rebuilding per chunk.

---

### 2.5 Soft, non-hard-filtering country-based blocking

**Verdict: subjective — measure-first (hard filter is forbidden, objective; soft partitioning needs the same-country rate number first — skip if cross-country >1-2%).**

**File:** `bulletproof_multipass.py`

**Problem:** No country-aware blocking currently exists — every Source-1
chunk is scored against the *entire* target source corpus regardless of
country.

**Why it matters:** Two competing considerations, both important:
1. **Compute:** scoring every Source-1 record against every target-source
   record (even within the current TF-IDF top-K retrieval scheme, which
   is much cheaper than true all-pairs) is more work than necessary if
   most true matches are same-country, which is plausible for local
   businesses (unconfirmed — should be checked against the training data
   first, e.g. what fraction of ground-truth matches are same-country vs.
   cross-country).
2. **Correctness, especially for France:** the problem statement is
   explicit that country must be treated as an open set and the pipeline
   must not hardcode or filter to `{US, India}` — France appears only in
   the test set. A **hard** country filter is a real correctness risk:
   the challenge design seems to specifically test whether solutions
   generalize to an unseen country label, and if country-filtering logic
   assumes only two known countries, France-labeled test records could be
   mishandled or dropped even if they have valid matches with e.g. a
   slightly different or missing country label on the matching side.

**Fix:**
1. First, measure whether restricting to same-country actually costs
   meaningful recall — using the training ground truth, compute what
   fraction of true matches are same-country vs. cross-country vs.
   missing/inconsistent country labels on one side. This single number
   should drive the design choice below.
2. If same-country matches dominate (likely, but confirm): use country as
   a **soft prioritization**, not a hard filter — e.g. retrieve top-K
   same-country candidates first via a country-partitioned index, and
   only fall back to a full-corpus search for a given Source-1 record if
   the same-country candidate pool is empty, too small, or the country
   label itself is missing/unreliable. This preserves the compute benefit
   for the common case without creating a hard failure mode for
   mismatched or unseen country labels.
3. Never write logic that assumes a fixed, enumerated country set (e.g.
   `if country in ['US', 'India']: ...`) anywhere in blocking or feature
   engineering — treat `country` as an arbitrary string throughout, and
   test explicitly against a France-labeled synthetic record (or the real
   test data once available) to confirm the pipeline doesn't silently
   misbehave on it.

**Performance impact:**
- If implemented as intended (partition candidates by country before
  running TF-IDF/embedding retrieval within each partition), this
  **reduces** compute for the common case: each Source-1 record is scored
  against a same-country subset rather than the full corpus, which for a
  corpus split roughly evenly across countries could cut per-chunk
  scoring cost by a factor close to the number of countries (roughly
  2-3x, given US/India/France), though this depends on how skewed the
  actual country distribution is (confirm via the EDA country-distribution
  section before assuming an even split).
- The fallback path (full-corpus search for missing/rare country labels)
  adds no cost in the common case and only pays the original full-corpus
  cost for the presumably small subset of records that need it.
- Partitioning itself (splitting a target source into per-country subsets
  and building separate indexes/vectorizers) is a bookkeeping overhead,
  not a complexity-class cost — should be a small addition to the caching
  step already described in items 2.1/2.2/2.4.

---

### 2.6 Matcher features don't include the multilingual embedding signal — the model will reject cross-script matches even if blocking finds them

**Verdict: split — shared `features.py` extraction is objective (do for sure); the embedding-signal feature itself is measure-first (matcher-only cosine first, full retrieval last).**

**Files:** `run_inference.py`, `train_better_xgb.py`

**Problem:** All current matcher features
(`name_exact`, `name_ratio`, `addr_ratio`, `name_token_set`,
`addr_token_set`, `name_partial`, `name_len_diff`, `addr_num_match`) are
computed via `rapidfuzz`, which operates on literal character sequences —
edit distance and token overlap over the raw (lowercased) strings. None of
these are script-aware or semantic in any sense.

**Why it matters:** This creates a specific and easy-to-miss failure mode:
even after fixing blocking (items 2.1-2.2) to *successfully retrieve*
cross-script true-match candidates, the matcher itself has no feature that
would give a Latin↔transliterated-Devanagari (or Latin↔raw-Devanagari, if
transliteration is imperfect) pair a high similarity score — rapidfuzz
edit-distance-style metrics between, say, a raw Devanagari string and its
Latin transliteration are still going to look quite different character-by
-character even when they're the correct match, especially with any
transliteration noise. Blocking recall and matcher precision/recall are
separate stages with separate failure modes; fixing one without the other
leaves the overall pipeline recall bottlenecked at whichever stage is
still weak. If blocking finds the right candidates but the matcher's
feature set can't recognize them as matches, item 2.1/2.2's blocking work
doesn't translate into an actual score improvement.

**Fix:**
1. If item 2.2 (multilingual embedding blocking) is implemented, the
   embedding cosine similarity between each Source-1/candidate pair is
   already being computed during blocking — **persist it** in a parallel
   sidecar file keyed by `(source1_entity_id, candidate_entity_id)`
   (do **not** add a column to `candidate_pairs.tsv` — the validator
   requires exactly its two columns and rejects anything else) rather
   than discarding it once blocking is done.
2. In both `run_inference.py` and `train_better_xgb.py`, join this
   persisted embedding similarity into the feature frame and add it to
   the `features` list used by XGBoost:
   ```python
   features = [
       'name_exact', 'name_ratio', 'addr_ratio', 'name_token_set',
       'addr_token_set', 'name_partial', 'name_len_diff', 'addr_num_match',
       'name_embedding_cosine',  # new
   ]
   ```
   Add `addr_embedding_cosine` too if address-field embeddings are
   computed (consistent with the separate name/address vectorization in
   item 2.3).
3. If embedding blocking (item 2.2) is *not* implemented for some reason,
   at minimum compute embedding similarity as a **matcher-only** feature
   (i.e., run it only on the much smaller post-blocking candidate set,
   not as a retrieval mechanism) — this is far cheaper than using it for
   blocking (since the candidate set is orders of magnitude smaller than
   the full corpus) and still gives the matcher the semantic signal it's
   currently missing, even if blocking recall for cross-script matches
   remains limited to whatever the transliteration-enhanced TF-IDF nets
   achieve.
4. Ensure feature computation is **identical** between
   `train_better_xgb.py` and `run_inference.py` — both already largely
   duplicate the same rapidfuzz feature block; when adding the embedding
   feature, factor this duplicated logic into a shared function/module
   (e.g. `features.py` with a `compute_features(cands_df) -> pd.DataFrame`
   function imported by both scripts) rather than copy-pasting the
   addition into both files separately, to prevent train/inference
   feature skew going forward.

**Performance impact:**
- If embedding similarity is already computed during blocking (item 2.2),
  this fix has effectively **zero additional cost** — it's a persistence
  and join operation on numbers already computed, not new computation.
- If computed only at the matcher stage (fallback in step 3 above), the
  cost is embedding **only the post-blocking candidate set** — orders of
  magnitude smaller than the full corpus (candidate set size is bounded
  by `TOP_K_PER_NET` per net × number of Source-1 records, vs. the full
  multi-million-row target source), so even without caching this is a
  comparatively cheap addition — likely seconds to low-minutes depending
  on final candidate set size, not the tens-of-minutes-to-hours range
  relevant to item 2.2's full-corpus encoding.

---

## Tier 3 — Efficiency cleanups (do after Tier 1/2, but don't skip — they affect whether you can iterate fast enough to use the fixes above)

### 3.1 TF-IDF vectorizers are refit from scratch on every chunk invocation

**Verdict: objective — do for sure (identical output, ~4x less fitting).**

**File:** `bulletproof_multipass.py`

**Problem:** `TfidfVectorizer().fit_transform(st_text)` is called fresh
inside every invocation of the script — once per `(target_source, chunk)`
combination. With the default `--total_chunks 4` and two target sources
(S2, S3), that's **8 separate full-corpus `fit_transform` calls** (4 for
word, 4 for char, per target source — actually 8 total invocations × 2
vectorizers = 16 fits) across a single end-to-end blocking run, each
refitting on the same, unchanged target-source corpus (millions of rows)
every time.

**Why it matters:** This is pure wasted computation with no correctness
benefit — the target-source corpus doesn't change between chunks (only
which slice of Source-1 is being processed changes), so the vectorizer
vocabulary and the target-side TF-IDF matrix should be identical across
all chunk invocations for a given target source. Refitting 4x per target
source multiplies the (already substantial, at millions-of-rows scale)
vectorization cost by 4 for zero benefit, and this cost compounds with
every other Tier 2 addition (transliteration, embeddings, numeric
indexing) if those aren't also cached — see the repeated caching notes in
items 2.1/2.2/2.4.

**Fix:**
1. Split the script into two phases: a **fit phase** (run once per target
   source) that fits `word_vec`/`char_vec` (and, per item 2.3, the
   separate name/address variants) on the full target-source corpus,
   transforms the target corpus itself, and persists both the fitted
   vectorizer objects (`pickle` or `joblib.dump`) and the resulting sparse
   matrices (`scipy.sparse.save_npz`) to disk.
2. A separate **chunk/query phase** (what `bulletproof_multipass.py`
   currently does per invocation) loads the persisted vectorizers and
   target matrices, transforms only the current Source-1 chunk, and scores
   against the already-computed target matrices.
3. Concretely, restructure roughly as:
   ```python
   # fit_target_vectorizers.py — run once per {split, source}
   word_vec = TfidfVectorizer(...).fit(st_text)
   st_word_mat = word_vec.transform(st_text)
   joblib.dump(word_vec, f"cache/{split}/word_vec_s{target_source}.joblib")
   sp.save_npz(f"cache/{split}/st_word_mat_s{target_source}.npz", st_word_mat)
   # ... same for char_vec

   # bulletproof_multipass.py — per chunk, loads instead of fits
   word_vec = joblib.load(f"cache/{split}/word_vec_s{target_source}.joblib")
   st_word_mat = sp.load_npz(f"cache/{split}/st_word_mat_s{target_source}.npz")
   s1_word_mat = word_vec.transform(s1_chunk_text)  # transform only, no fit
   ```
4. Apply the same fit-once/cache pattern to the transliteration output
   (item 2.1), embedding index (item 2.2), and numeric-token inverted
   index (item 2.4) — all of these only need to be computed once per
   target source and reused across chunks, and all currently risk the
   same multiplication-by-chunk-count problem if implemented naively
   inside the existing per-chunk script structure.

**Performance impact:**
- This is a direct, close-to-linear reduction in blocking runtime: if
  vectorizer fitting currently accounts for a significant fraction of
  each chunk invocation's wall-clock time (plausible, since `fit_transform`
  on a multi-million-row corpus is not cheap), caching should cut total
  blocking wall-clock time by roughly a factor close to `total_chunks`
  (e.g. close to 4x with the current default) for the fitting portion
  specifically, since it's now done once instead of once-per-chunk.
  Transform-only calls on each Source-1 chunk remain necessary and
  unchanged in cost.
- Disk cache size: sparse TF-IDF matrices for a few-million-row corpus
  with bounded vocabulary (`min_df=2`, capped `max_df`) are typically
  modest relative to dense storage — expect low hundreds of MB to
  low-single-digit GB per cached matrix depending on final vocabulary
  size and average non-zero entries per row; confirm actual size after
  first fit rather than assuming, and adjust `min_df`/`max_df` if the
  cached artifacts turn out to be unexpectedly large.
- This fix should be implemented **before or alongside** items 2.1/2.2/2.4
  rather than strictly after, since those additions are the ones most at
  risk of multiplying an already-expensive step by the chunk count if the
  caching pattern isn't in place first — sequencing this fix late means
  temporarily re-deriving embeddings/transliteration/numeric-index 4x
  more than necessary while other Tier 2 work is being built and tested.

---

## Summary table

| # | Problem | Tier | Verdict | Primary impact | Compute cost of fix |
|---|---------|------|---------|-----------------|----------------------|
| 1.1 | `merge_final.py` wrong glob path | 1 | do-for-sure | Correctness (silent bad submission); commit also moves file to `src/submission/merge_submission.py` | None |
| 1.2 | Model trained on US/S2 only | 1 | do-for-sure | Score (extrapolation on majority of test data) | ~2x training set size |
| 1.3-harness | Entity-level val split + `macro_f05` | 1 | do-for-sure | Score + ability to measure future changes | Small (val-set scoring only) |
| 1.3-calibrate | Calibration + F_0.5 threshold sweep | 1 | measure-first | Score (threshold/method decided on val) | Small (val-set scoring only) |
| 2.1-direction | No transliteration | 2 | do-for-sure | Blocking recall on non-Latin matches | One-time pass, cache required |
| 2.1-library | Transliteration library/scheme choice | 2 | measure-first | Correctness of 2.1-direction | Audit cost only |
| 2.2 | No multilingual embedding blocking | 2 | measure-first | Blocking recall (semantic/paraphrase signal) | Largest cost — GPU recommended, cache required |
| 2.3 | Name+address concatenated in TF-IDF | 2 | measure-first | Blocking precision/recall balance | ~1.5-2x TF-IDF blocking cost |
| 2.4 | No numeric-token blocking net | 2 | do-for-sure (cutoff measure-first) | Blocking recall (language-agnostic signal) | Small — inverted index build |
| 2.5-partition | Soft country-based blocking | 2 | measure-first | Compute (and correctness for France) | Net reduction if implemented, confirm same-country match rate first |
| 2.6-extract | Shared `features.py` extraction | 2 | do-for-sure | Correctness (prevents train/inference skew) | None |
| 2.6-embed | Embedding feature in matcher | 2 | measure-first | Score (blocking gains don't reach the matcher otherwise) | Near-zero if 2.2 done, small otherwise |
| 3.1 | TF-IDF refit every chunk | 3 | do-for-sure | Wasted compute, blocks fast iteration on 2.x work | Implement early — enables everything else to iterate faster |

**Recommended sequencing:** 1.1 → 1.3-harness → 1.2 → 1.3-calibrate → 3.1 (caching infrastructure)
→ 2.1 → 2.4 → 2.3 → 2.2 → 2.6, re-measuring validation F_0.5 after each
step. Each commit both fixes its issue and (where the Tier 0 mapping table
calls for it) writes the affected file at its new location — there's no
separate restructuring pass. Doing 1.3 early, even before the
score-improving fixes, means every subsequent change is measured against a
real local metric instead of guessed at.
