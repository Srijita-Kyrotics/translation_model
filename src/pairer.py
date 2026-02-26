import os
import pandas as pd
import glob
import torch
from sentence_transformers import SentenceTransformer, util

def pair_directory_labse(src_dir, tgt_dir, dataset_type, model):
    """
    Pairs source (Bengali) and target (English) files by filename and semantic alignment.
    Returns a list of pairs.
    """
    src_files = glob.glob(os.path.join(src_dir, "*.txt"))
    all_pairs = []
    
    for src_path in src_files:
        file_name = os.path.basename(src_path)
        
        target_name = file_name
        if dataset_type == "judgments":
            if file_name.endswith("_b.txt"):
                target_name = file_name.replace("_b.txt", "_e.txt")
            else:
                continue
        
        tgt_path = os.path.join(tgt_dir, target_name)
        
        if not os.path.exists(tgt_path):
            continue
            
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_lines = [l.strip() for l in f_src.readlines() if l.strip()]
            
        with open(tgt_path, "r", encoding="utf-8") as f_tgt:
            tgt_lines = [l.strip() for l in f_tgt.readlines() if l.strip()]
            
        if not src_lines or not tgt_lines:
            continue
            
        # Semantic Alignment using LaBSE
        print(f"Aligning {file_name} -> {len(src_lines)} BN vs {len(tgt_lines)} EN")
        # Ensure we use CUDA if available
        src_embeddings = model.encode(src_lines, convert_to_tensor=True, batch_size=32)
        tgt_embeddings = model.encode(tgt_lines, convert_to_tensor=True, batch_size=32)
        
        cosine_scores = util.cos_sim(src_embeddings, tgt_embeddings)
        
        # Greedy pairing: for each target, find best source above threshold
        threshold = 0.65
        for tgt_idx, tgt_str in enumerate(tgt_lines):
            scores = cosine_scores[:, tgt_idx]
            best_score, best_src_idx = torch.max(scores, dim=0)
            
            if best_score.item() > threshold:
                all_pairs.append({
                    "source_file": file_name,
                    "line_index": tgt_idx,
                    "bengali": src_lines[best_src_idx.item()],
                    "english": tgt_str,
                    "dataset": dataset_type,
                    "score": round(best_score.item(), 3)
                })
                
    return all_pairs

def pair_directory_exact(src_dir, tgt_dir, dataset_type):
    """Fallback to exact line-by-line matching for docx, as they are parallel translated."""
    src_files = glob.glob(os.path.join(src_dir, "*.txt"))
    all_pairs = []
    
    for src_path in src_files:
        file_name = os.path.basename(src_path)
        tgt_path = os.path.join(tgt_dir, file_name)
        
        if not os.path.exists(tgt_path):
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
                "dataset": dataset_type,
                "score": 1.0 # Synthetic exact matches
            })
    return all_pairs


def main():
    PROCESSED_BASE = os.path.join("data", "processed")
    TRANSLATED_BASE = os.path.join("data", "translated")
    FINAL_CSV = os.path.join("data", "final", "parallel_corpus.csv")
    
    all_data = []
    
    # 1. Original docx data (synthetic translated, exact matching)
    print("Pairing docx data (Line-by-Line)...")
    all_data.extend(pair_directory_exact(PROCESSED_BASE, TRANSLATED_BASE, "docx"))
    
    # 2. Judgment PDF data (LaBSE semantic alignment)
    print("Loading LaBSE model for semantic alignment...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/LaBSE', device=device)
    
    print("Pairing judgment data (Semantic alignment)...")
    proc_bn_pdf = os.path.join(PROCESSED_BASE, "judgments_bn")
    proc_en_pdf = os.path.join(PROCESSED_BASE, "judgments_en")
    if os.path.exists(proc_bn_pdf) and os.path.exists(proc_en_pdf):
        all_data.extend(pair_directory_labse(proc_bn_pdf, proc_en_pdf, "judgments", model))
    
    if all_data:
        df = pd.DataFrame(all_data)
        # Keep only the translation pairs for the final dataset
        df = df[['bengali', 'english']]
        os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
        df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        print(f"Total: Saved {len(all_data)} pairs to {FINAL_CSV}")
    else:
        print("No pairs generated.")

if __name__ == "__main__":
    main()
