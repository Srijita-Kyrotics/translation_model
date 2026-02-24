import pandas as pd
import os

def export_parallel_data(csv_path, output_dir):
    """
    Reads the parallel corpus CSV and exports it in multiple formats:
    - Side-by-side TSV (Bengali \t English)
    - Line-aligned .bn and .en files
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Basic Cleaning/Filtering
    # Remove rows where either side is null or just whitespace
    initial_count = len(df)
    df = df.dropna(subset=['bengali', 'english'])
    df = df[df['bengali'].str.strip() != ""]
    df = df[df['english'].str.strip() != ""]
    
    # Optional: Filter extremely short fragments (less than 3 characters)
    # This helps avoid noise like single punctuation marks
    df = df[df['bengali'].str.len() > 3]
    df = df[df['english'].str.len() > 3]
    
    final_count = len(df)
    print(f"Filtered {initial_count - final_count} rows. Remaining: {final_count} pairs.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Export Side-by-Side TSV
    tsv_path = os.path.join(output_dir, "parallel_side_by_side.txt")
    df[['bengali', 'english']].to_csv(tsv_path, sep='\t', index=False, header=False, encoding='utf-8')
    print(f"Exported side-by-side TSV to {tsv_path}")
    
    # 3. Export Line-Aligned .bn and .en
    bn_path = os.path.join(output_dir, "corpus.bn")
    en_path = os.path.join(output_dir, "corpus.en")
    
    with open(bn_path, "w", encoding="utf-8") as f_bn, open(en_path, "w", encoding="utf-8") as f_en:
        for _, row in df.iterrows():
            f_bn.write(row['bengali'].strip() + "\n")
            f_en.write(row['english'].strip() + "\n")
            
    print(f"Exported line-aligned files: {bn_path}, {en_path}")

if __name__ == "__main__":
    CSV_PATH = os.path.join("data", "final", "parallel_corpus.csv")
    OUTPUT_DIR = os.path.join("data", "final")
    
    if os.path.exists(CSV_PATH):
        export_parallel_data(CSV_PATH, OUTPUT_DIR)
    else:
        print(f"Error: {CSV_PATH} not found.")
