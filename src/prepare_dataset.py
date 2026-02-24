from datasets import Dataset, DatasetDict
import os

def prepare_dataset(bn_path, en_path, output_dir):
    """
    Reads aligned .bn and .en files and creates a Hugging Face Dataset.
    Splits it into train and validation sets.
    """
    print(f"Loading data from {bn_path} and {en_path}...")
    
    with open(bn_path, "r", encoding="utf-8") as f_bn:
        bn_lines = [line.strip() for line in f_bn]
        
    with open(en_path, "r", encoding="utf-8") as f_en:
        en_lines = [line.strip() for line in f_en]
        
    if len(bn_lines) != len(en_lines):
        raise ValueError("Bengali and English files must have the same number of lines!")
    
    print(f"Total pairs: {len(bn_lines)}")
    
    # Create dataset
    raw_data = {
        "bengali": bn_lines,
        "english": en_lines
    }
    dataset = Dataset.from_dict(raw_data)
    
    # Split: 98% Train, 2% Val
    # 2% of 600K is ~12K, which is plenty for evaluation
    ds_split = dataset.train_test_split(test_size=0.02, seed=42)
    
    final_ds = DatasetDict({
        "train": ds_split["train"],
        "validation": ds_split["test"]
    })
    
    print(f"Split results: {final_ds}")
    
    os.makedirs(output_dir, exist_ok=True)
    final_ds.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "final")
    BN_FILE = os.path.join(DATA_DIR, "corpus.bn")
    EN_FILE = os.path.join(DATA_DIR, "corpus.en")
    OUTPUT_DIR = os.path.join(DATA_DIR, "hf_dataset")
    
    if os.path.exists(BN_FILE) and os.path.exists(EN_FILE):
        prepare_dataset(BN_FILE, EN_FILE, OUTPUT_DIR)
    else:
        print("Error: Alignment files not found in data/final/")
