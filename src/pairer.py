import os
import pandas as pd
import glob

def pair_directory(src_dir, tgt_dir, dataset_type):
    """
    Pairs source (Bengali) and target (English) files by filename and index.
    Returns a list of pairs.
    """
    src_files = glob.glob(os.path.join(src_dir, "*.txt"))
    all_pairs = []
    
    for src_path in src_files:
        file_name = os.path.basename(src_path)
        # Adjust target filename if necessary (for judgments, they have the same base name)
        # Assuming judgments_bn/file_b.txt matches judgments_en/file_e.txt logic was handled in loader
        # Actually loader saves them as e.g. 1950~scr_..._b.txt and 1950~scr_..._e.txt
        # We need to match them.
        
        target_name = file_name
        if dataset_type == "judgments":
            if file_name.endswith("_b.txt"):
                target_name = file_name.replace("_b.txt", "_e.txt")
            else:
                continue # Skip non-Bengali files in this pass
        
        tgt_path = os.path.join(tgt_dir, target_name)
        
        if not os.path.exists(tgt_path):
            # print(f"Warning: No matching target file for {file_name}")
            continue
            
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_lines = [l.strip() for l in f_src.readlines() if l.strip()]
            
        with open(tgt_path, "r", encoding="utf-8") as f_tgt:
            tgt_lines = [l.strip() for l in f_tgt.readlines() if l.strip()]
            
        min_len = min(len(src_lines), len(tgt_lines))
        for i in range(min_len):
            all_pairs.append({
                "source_file": file_name,
                "line_index": i,
                "bengali": src_lines[i],
                "english": tgt_lines[i],
                "dataset": dataset_type
            })
    return all_pairs

def main():
    PROCESSED_BASE = os.path.join("data", "processed")
    TRANSLATED_BASE = os.path.join("data", "translated")
    FINAL_CSV = os.path.join("data", "final", "parallel_corpus.csv")
    
    all_data = []
    
    # 1. Original docx data (Bengali in 'processed', English in 'translated')
    print("Pairing docx data...")
    all_data.extend(pair_directory(PROCESSED_BASE, TRANSLATED_BASE, "docx"))
    
    # 2. Judgment PDF data (Bengali in 'processed/judgments_bn', English in 'processed/judgments_en')
    print("Pairing judgment data...")
    proc_bn_pdf = os.path.join(PROCESSED_BASE, "judgments_bn")
    proc_en_pdf = os.path.join(PROCESSED_BASE, "judgments_en")
    if os.path.exists(proc_bn_pdf) and os.path.exists(proc_en_pdf):
        all_data.extend(pair_directory(proc_bn_pdf, proc_en_pdf, "judgments"))
    
    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
        df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        print(f"Total: Saved {len(all_data)} pairs to {FINAL_CSV}")
    else:
        print("No pairs generated.")

if __name__ == "__main__":
    main()
