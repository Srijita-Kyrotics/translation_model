"""
Process the downloaded 4K+ Calcutta High Court judgment PDFs:
1. Extract text from all Bengali and English PDFs using PyMuPDF
2. Match pairs by filename
3. Align using LaBSE semantic similarity
4. Build the expanded parallel corpus
"""
import os
import glob
import fitz  # PyMuPDF
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Split into lines and clean
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 3]
        return lines
    except Exception as e:
        return []

def process_all_pdfs():
    bn_dir = os.path.join("data", "raw", "judgments", "bengali")
    en_dir = os.path.join("data", "raw", "judgments", "english")
    
    bn_pdfs = sorted(glob.glob(os.path.join(bn_dir, "*.pdf")))
    en_pdfs = sorted(glob.glob(os.path.join(en_dir, "*.pdf")))
    
    print(f"Found {len(bn_pdfs)} Bengali PDFs and {len(en_pdfs)} English PDFs")
    
    # Build filename mapping: base_name -> (bn_path, en_path)
    bn_map = {}
    for p in bn_pdfs:
        name = os.path.basename(p)
        base = name.replace("_b.pdf", "")
        bn_map[base] = p
    
    en_map = {}
    for p in en_pdfs:
        name = os.path.basename(p)
        base = name.replace("_e.pdf", "")
        en_map[base] = p
    
    # Find matching pairs
    common_bases = set(bn_map.keys()) & set(en_map.keys())
    print(f"Found {len(common_bases)} matching Bengali-English pairs")
    
    # Load LaBSE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading LaBSE model on {device}...")
    model = SentenceTransformer('sentence-transformers/LaBSE', device=device)
    
    all_pairs = []
    skipped = 0
    threshold = 0.65
    
    for base in tqdm(sorted(common_bases), desc="Processing pairs"):
        bn_lines = extract_text_from_pdf(bn_map[base])
        en_lines = extract_text_from_pdf(en_map[base])
        
        if not bn_lines or not en_lines:
            skipped += 1
            continue
        
        # Filter out very short lines (page numbers, headers)
        bn_lines = [l for l in bn_lines if len(l) > 10]
        en_lines = [l for l in en_lines if len(l) > 10]
        
        if not bn_lines or not en_lines:
            skipped += 1
            continue
        
        try:
            # Encode in batches
            bn_emb = model.encode(bn_lines, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            en_emb = model.encode(en_lines, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            
            # Compute cosine similarity matrix
            cos_scores = util.cos_sim(bn_emb, en_emb)
            
            # Greedy best match for each Bengali line
            used_en = set()
            for bn_idx in range(len(bn_lines)):
                scores = cos_scores[bn_idx]
                # Find best unused English match
                sorted_indices = torch.argsort(scores, descending=True)
                for en_idx in sorted_indices:
                    en_idx = en_idx.item()
                    if en_idx not in used_en and scores[en_idx].item() > threshold:
                        all_pairs.append({
                            "bengali": bn_lines[bn_idx],
                            "english": en_lines[en_idx],
                        })
                        used_en.add(en_idx)
                        break
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\nProcessed: {len(common_bases)} pairs, Skipped: {skipped}")
    print(f"Total aligned sentence pairs: {len(all_pairs)}")
    
    # Save
    if all_pairs:
        df = pd.DataFrame(all_pairs)
        
        # Also include the existing docx pairs
        existing_csv = os.path.join("data", "final", "parallel_corpus.csv")
        if os.path.exists(existing_csv):
            existing_df = pd.read_csv(existing_csv)
            print(f"Adding {len(existing_df)} existing docx pairs...")
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=['bengali', 'english'])
        
        output_path = os.path.join("data", "final", "parallel_corpus.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"FINAL: Saved {len(df)} pairs to {output_path}")
    
    return len(all_pairs)

if __name__ == "__main__":
    process_all_pdfs()
