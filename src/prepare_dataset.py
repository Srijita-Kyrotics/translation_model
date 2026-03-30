import pandas as pd
import json
import uuid
import os
from datasets import Dataset, DatasetDict

def prepare_hf_dataset():
    input_file = "data/final/parallel_corpus_v6_combined_clean.csv"
    hf_dataset_path = "data/final/hf_dataset"
    train_out = "data/final/train.jsonl"
    val_out = "data/final/val.jsonl"

    print(f"Loading merged corpus from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run merge_datasets.py first.")
        return

    df = pd.read_csv(input_file)
    
    print(f"Original length: {len(df)}")
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['english', 'bengali_original'])
    print(f"Length after dropping duplicates: {len(df)}")
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Rename columns to match finetune.py expectations (if needed)
    # finetune.py expects 'english' and 'bengali'
    df = df.rename(columns={'bengali_original': 'bengali'})
    
    # 95% Train, 5% Val Split
    train_size = int(len(df) * 0.95)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"Train Split: {len(train_df)} examples.")
    print(f"Val Split: {len(val_df)} examples.")
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Saving Hugging Face dataset to {hf_dataset_path}...")
    dataset_dict.save_to_disk(hf_dataset_path)
    
    # Also save as JSONL for compatibility with other scripts
    def df_to_jsonl(dataframe, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                record = {
                    "id": str(uuid.uuid4()),
                    "translation": {
                        "en": str(row['english']),
                        "bn": str(row['bengali'])
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
    print(f"Writing {train_out} ...")
    df_to_jsonl(train_df, train_out)
    
    print(f"Writing {val_out} ...")
    df_to_jsonl(val_df, val_out)
    
    print("✅ Done! Dataset is formatted and saved to disk.")

if __name__ == "__main__":
    prepare_hf_dataset()
