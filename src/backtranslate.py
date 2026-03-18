import os
import pandas as pd
import glob
from tqdm import tqdm
from src.translator import Translator
import torch

def collect_monolingual_data(processed_dir, parallel_csv_path, target_count=272000):
    """
    Collects sentences from processed text files that are not already in the parallel corpus.
    """
    print("Loading existing parallel corpus to identify used sentences...")
    df_parallel = pd.read_csv(parallel_csv_path)
    used_bn = set(df_parallel['bengali'].astype(str).unique())
    used_en = set(df_parallel['english'].astype(str).unique())
    
    monolingual_pool = []
    
    print(f"Scanning {processed_dir} for monolingual sentences...")
    txt_files = glob.glob(os.path.join(processed_dir, "**", "*.txt"), recursive=True)
    
    for file_path in tqdm(txt_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Basic filtering for valid sentences (at least 3 words)
                    if len(line.split()) < 3:
                        continue
                        
                    # Check if already used or if it's already in our pool
                    if line not in used_bn and line not in used_en:
                        monolingual_pool.append(line)
        except Exception as e:
            continue
            
    # Deduplicate
    print("Deduplicating pool...")
    monolingual_pool = list(set(monolingual_pool))
    print(f"Total unique monolingual sentences found: {len(monolingual_pool)}")
    
    if len(monolingual_pool) > target_count:
        import random
        random.seed(42)
        monolingual_pool = random.sample(monolingual_pool, target_count)
        print(f"Sampled {target_count} sentences for back-translation.")
        
    return monolingual_pool

def run_backtranslation(sentences, output_path):
    """
    Detects language and back-translates sentences.
    Processes EN->BN and BN->EN sequentially to save VRAM.
    """
    bn_sentences = []
    en_sentences = []
    
    print("Classifying sentences by language...")
    for s in sentences:
        if any('\u0980' <= char <= '\u09FF' for char in s):
            bn_sentences.append(s)
        else:
            en_sentences.append(s)
            
    # --- RESUME LOGIC ---
    start_bn_idx = 0
    start_en_idx = 0
    if os.path.exists(output_path):
        print(f"Existing checkpoint found at {output_path}. Checking progress...")
        try:
            df_existing = pd.read_csv(output_path)
            backtranslated_pairs = df_existing.to_dict('records')
            
            # Count how many of each we have already done
            if 'source_direction' in df_existing.columns:
                done_bn = len(df_existing[(df_existing['type'] == 'BT') & (df_existing['source_direction'] == 'bn-en')])
                done_en = len(df_existing[(df_existing['type'] == 'BT') & (df_existing['source_direction'] == 'en-bn')])
            else:
                # 🛡️ Legacy Checkpoint Logic: If the tag is missing, assume they are all BN->EN 
                # because that is our starting direction.
                done_bn = len(df_existing[df_existing['type'] == 'BT'])
                done_en = 0
                # Retroactively add the tag to the dictionary so future saves are correct
                for p in backtranslated_pairs:
                    if p['type'] == 'BT' and 'source_direction' not in p:
                        p['source_direction'] = 'bn-en'
            
            start_bn_idx = done_bn
            start_en_idx = done_en
            print(f"Resuming: Already completed {start_bn_idx} BN sentences and {start_en_idx} EN sentences.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            backtranslated_pairs = []

    batch_size = 8
    
    # 1. Process Bengali -> English
    if bn_sentences and start_bn_idx < len(bn_sentences):
        print(f"Translating {len(bn_sentences) - start_bn_idx} remaining Bengali sentences to English...")
        translator = Translator(src_lang="ben_Beng", tgt_lang="eng_Latn", use_correction=False)
        translator.load_model()
        
        for i in tqdm(range(start_bn_idx, len(bn_sentences), batch_size), desc="BN->EN"):
            batch = bn_sentences[i:i+batch_size]
            translations = translator.translate_batch(batch)
            for s, t in zip(batch, translations):
                backtranslated_pairs.append({"bengali": s, "english": t, "type": "BT", "source_direction": "bn-en"})
            
            # Periodic saving
            if i % (batch_size * 50) == 0:
                pd.DataFrame(backtranslated_pairs).to_csv(output_path, index=False)
                
        # Clean up
        del translator
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # 2. Process English -> Bengali
    if en_sentences and start_en_idx < len(en_sentences):
        print(f"Translating {len(en_sentences) - start_en_idx} remaining English sentences to Bengali...")
        translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
        translator.load_model()
        
        for i in tqdm(range(start_en_idx, len(en_sentences), batch_size), desc="EN->BN"):
            batch = en_sentences[i:i+batch_size]
            translations = translator.translate_batch(batch)
            for s, t in zip(batch, translations):
                backtranslated_pairs.append({"bengali": t, "english": s, "type": "BT", "source_direction": "en-bn"})
            
            # Periodic saving
            if i % (batch_size * 50) == 0:
                pd.DataFrame(backtranslated_pairs).to_csv(output_path, index=False)
                
        # Clean up
        del translator
        torch.cuda.empty_cache()
        gc.collect()

    # Final save
    df_bt = pd.DataFrame(backtranslated_pairs)
    df_bt.to_csv(output_path, index=False)
    print(f"Back-translation complete. Saved {len(df_bt)} pairs to {output_path}")

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join("data", "processed")
    PARALLEL_CSV = os.path.join("data", "final", "parallel_corpus.csv")
    OUTPUT_BT_CSV = os.path.join("data", "final", "backtranslated_corpus.csv")
    
    pool = collect_monolingual_data(PROCESSED_DIR, PARALLEL_CSV)
    if pool:
        run_backtranslation(pool, OUTPUT_BT_CSV)
    else:
        print("No new monolingual data found for back-translation.")
