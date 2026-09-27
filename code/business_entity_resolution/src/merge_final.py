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
        print(f"No files found for {file_pattern}")
        return
    
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
    
    # Save
    submission.to_csv(f"output/{out_name}", sep='\t', index=False)
    print(f"SUCCESS: output/{out_name} generated.")

# Process both required files for the submission package
process_and_merge('final_output/matching_results_*.tsv', 'matched_entity_ids', 'matching_results.tsv')
process_and_merge('final_output/multipass_cands_*.tsv', 'candidate_entity_ids', 'candidate_pairs.tsv')

print("Final package generation complete! Files are ready.")
