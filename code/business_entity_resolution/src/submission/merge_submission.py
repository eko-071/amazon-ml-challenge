import pandas as pd
import glob
import os

print("Merging Inference Results and Candidates...")
os.makedirs("output", exist_ok=True)

# 1. Load test source 1 to ensure ALL entities are present
s1_test = pd.read_csv("dataset/test/test_source1.tsv", sep='\t', dtype=str)
base_submission = s1_test[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})

def process_and_merge(file_pattern, col_name, out_name):
    files = glob.glob(file_pattern)
    if not files:
        raise RuntimeError(f"No files found for pattern {file_pattern} — aborting merge.")

    dfs = [pd.read_csv(f, sep='\t', dtype=str).fillna("") for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Split, strip, and remove empty strings
    df[col_name] = df[col_name].apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])

    # Group by source1_entity_id and combine lists
    final = df.groupby('source1_entity_id')[col_name].sum().reset_index()

    # Deduplicate and sort, then join by comma
    final[col_name] = final[col_name].apply(lambda x: ','.join(sorted(list(set(x)))))

    # Merge with base_submission to inject singletons (empty matches)
    submission = base_submission.merge(final, on='source1_entity_id', how='left').fillna("")

    # Sanity check before writing: near-0% or near-100% non-empty is suspicious
    # (train singletons are ~5.6%, so most entities should have matches).
    n_total = len(submission)
    n_nonempty = int((submission[col_name] != "").sum())
    print(f"{out_name}: {n_total} rows, {n_nonempty} with non-empty {col_name} ({n_nonempty / n_total:.1%})")
    assert n_nonempty > 0, f"{out_name} has zero non-empty predictions — pipeline likely broken upstream."

    # Save
    submission.to_csv(f"output/{out_name}", sep='\t', index=False)
    print(f"SUCCESS: output/{out_name} generated.")

# Process both required files for the submission package.
# Patterns use the s*_ prefix so the merged output files themselves
# (output/matching_results.tsv, output/candidate_pairs.tsv) never match
# the glob on repeat runs — chunk files are the only inputs.
# Run from the project root: python code/business_entity_resolution/src/submission/merge_submission.py
process_and_merge('output/matching_results_s*_*.tsv', 'matched_entity_ids', 'matching_results.tsv')
process_and_merge('output/multipass_cands_s*_*.tsv', 'candidate_entity_ids', 'candidate_pairs.tsv')

print("Final package generation complete! Files are ready.")
