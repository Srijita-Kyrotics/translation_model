from datasets import Dataset, DatasetDict
import os
import pandas as pd

def prepare_dataset(csv_path, output_dir):
    """
    Reads the aligned parallel_corpus.csv and creates a Hugging Face Dataset.
    Splits it into train and validation sets.
    """
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    # Ensure there are no null rows and drop them
    df = df.dropna(subset=['bengali', 'english'])
    
    # Randomly shuffle data, very important since it's ordered by doc
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Optional: ensure we only pass standard strings as lists
    bn_lines = df['bengali'].astype(str).tolist()
    en_lines = df['english'].astype(str).tolist()
        
    if len(bn_lines) != len(en_lines):
        raise ValueError("Bengali and English columns must have the same number of lines!")
    
    print(f"Total valid aligned pairs to prepare: {len(bn_lines)}")
    
    # Create dataset
    raw_data = {
        "bengali": bn_lines,
        "english": en_lines
    }
    dataset = Dataset.from_dict(raw_data)
    
    # Split: 99% Train, 1% Val
    # 1% of 150K is ~1.5K
    ds_split = dataset.train_test_split(test_size=0.01, seed=42)
    
    final_ds = DatasetDict({
        "train": ds_split["train"],
        "validation": ds_split["test"]
    })
    
    print(f"Split results: {final_ds}")
    
    os.makedirs(output_dir, exist_ok=True)
    final_ds.save_to_disk(output_dir)
    print(f"Dataset successfully packaged and saved to {output_dir}")

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "final")
    CSV_FILE = os.path.join(DATA_DIR, "parallel_corpus.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "hf_dataset")
    
    if os.path.exists(CSV_FILE):
        prepare_dataset(CSV_FILE, OUTPUT_DIR)
    else:
        print(f"Error: Alignment file {CSV_FILE} not found!")
