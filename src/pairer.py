import os
import pandas as pd
import glob

def pair_corpora(src_dir, tgt_dir, output_file):
    """
    Pairs source (Bengali) and target (English) files by filename and index.
    Saves the final parallel corpus as a CSV.
    """
    src_files = glob.glob(os.path.join(src_dir, "*.txt"))
    all_pairs = []
    
    for src_path in src_files:
        file_name = os.path.basename(src_path)
        tgt_path = os.path.join(tgt_dir, file_name)
        
        if not os.path.exists(tgt_path):
            print(f"Warning: No matching target file for {file_name}")
            continue
            
        print(f"Pairing {file_name}...")
        
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_lines = [l.strip() for l in f_src.readlines() if l.strip()]
            
        with open(tgt_path, "r", encoding="utf-8") as f_tgt:
            tgt_lines = [l.strip() for l in f_tgt.readlines() if l.strip()]
            
        # Deterministic index-based pairing
        # We align up to the shorter one if there's a mismatch
        min_len = min(len(src_lines), len(tgt_lines))
        for i in range(min_len):
            all_pairs.append({
                "source_file": file_name,
                "line_index": i,
                "bengali": src_lines[i],
                "english": tgt_lines[i]
            })
            
    if all_pairs:
        df = pd.DataFrame(all_pairs)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Saved {len(all_pairs)} pairs to {output_file}")
    else:
        print("No pairs generated.")

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join("data", "processed")
    TRANSLATED_DIR = os.path.join("data", "translated")
    FINAL_CSV = os.path.join("data", "final", "parallel_corpus.csv")
    
    pair_corpora(PROCESSED_DIR, TRANSLATED_DIR, FINAL_CSV)
