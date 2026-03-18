from datasets import Dataset, DatasetDict
import os
import pandas as pd

def clean_bt_data(df):
    """
    Applies heuristics to remove noisy/broken translations from the BT corpus.
    """
    initial_len = len(df)
    
    # 1. Remove exact matches (failed translation/copying)
    df = df[df['bengali'] != df['english']]
    
    # 2. Filter by length
    # Minimum 15 chars and 4 words
    df = df[df['bengali'].str.len() > 15]
    df = df[df['english'].str.len() > 15]
    df = df[df['bengali'].str.split().str.len() > 3]
    df = df[df['english'].str.split().str.len() > 3]
    
    # 3. Filter Repetitive Patterns (indicates model loop/hallucination)
    import re
    repetition_pattern = re.compile(r'(.)\1{4,}') # 5 or more identical consecutive chars
    def has_repetition(text):
        if not isinstance(text, str): return False
        return bool(repetition_pattern.search(text))
        
    df = df[~df['english'].apply(has_repetition)]
    df = df[~df['bengali'].apply(has_repetition)]
    
    # 4. Filter excessive non-alphanumeric (garbage symbols)
    def is_mostly_garbage(text):
        if not text: return True
        # Count symbols/punctuation vs total chars
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        if alnum_count / len(text) < 0.7: # More than 30% "weird" chars
            return True
        return False
        
    df = df[~df['english'].apply(is_mostly_garbage)]
    df = df[~df['bengali'].apply(is_mostly_garbage)]
    
    print(f"BT Cleaning: Removed {initial_len - len(df)} noisy pairs. Remaining: {len(df)}")
    return df

def prepare_dataset(parallel_csv, bt_csv=None, output_dir=None):
    """
    Combines parallel data with back-translated data.
    Tags BT data with <BT> as per paper.
    """
    print(f"Loading parallel data from {parallel_csv}...")
    df_parallel = pd.read_csv(parallel_csv).dropna(subset=['bengali', 'english'])
    df_parallel['source'] = 'gold'
    
    if bt_csv and os.path.exists(bt_csv):
        print(f"Loading back-translated data from {bt_csv}...")
        df_bt = pd.read_csv(bt_csv).dropna(subset=['bengali', 'english'])
        
        # --- NEW: Noise Filtering ---
        df_bt = clean_bt_data(df_bt)
        
        # Prepend <BT> tag to source text for synthetic data
        df_bt['bengali'] = df_bt['bengali'].apply(lambda x: f"<BT> {x}")
        df_bt['source'] = 'silver'
        df = pd.concat([df_parallel, df_bt], ignore_index=True)
    else:
        df = df_parallel

    print(f"Total combined dataset size: {len(df)}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # --- Create Stage 1 Dataset (Combined) ---
    dataset = Dataset.from_pandas(df[['bengali', 'english']])
    ds_split = dataset.train_test_split(test_size=0.01, seed=42)
    final_ds = DatasetDict({
        "train": ds_split["train"],
        "validation": ds_split["test"]
    })
    
    os.makedirs(output_dir, exist_ok=True)
    final_ds.save_to_disk(output_dir)
    print(f"Stage 1 Dataset saved to {output_dir}")

    # --- Create Stage 2 Dataset (Gold/Supreme Court Only) ---
    # Filter for the ~scr prefix which denotes high-quality SC judgments
    if 'source' in df.columns:
        # We only want 'gold' data, and specifically Supreme Court judgments if possible
        # However, our 'bengali' or 'english' strings don't necessarily have the prefix anymore
        # but we can reload the parallel corpus which has 'source_file' if we kept it.
        # For now, let's use the 'source' column we added.
        df_gold = df[df['source'] == 'gold'].copy()
        
        # If we want to be even stricter, we could filter by content or doc_id
        # For our purposes, 'gold' represents the 155k aligned pairs.
        gold_dataset = Dataset.from_pandas(df_gold[['bengali', 'english']])
        gold_split = gold_dataset.train_test_split(test_size=0.05, seed=42) # More valuation for final
        
        stage2_dir = output_dir + "_gold"
        DatasetDict({
            "train": gold_split["train"],
            "validation": gold_split["test"]
        }).save_to_disk(stage2_dir)
        print(f"Stage 2 (Gold) Dataset saved to {stage2_dir}")

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "final")
    PARALLEL_CSV = os.path.join(DATA_DIR, "parallel_corpus.csv")
    BT_CSV = os.path.join(DATA_DIR, "backtranslated_corpus.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "hf_dataset")
    
    prepare_dataset(PARALLEL_CSV, BT_CSV, OUTPUT_DIR)
