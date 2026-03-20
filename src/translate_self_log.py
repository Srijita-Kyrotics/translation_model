import os
import glob
import time
import torch
import sys

# Add current dir and toolkit to path internally to avoid space issues
sys.path.append("/home/kyrotics/ML Projects/translation_model-main")
sys.path.append("/home/kyrotics/ML Projects/Bhasantar_Legal_General/rotary-indictrans2-en-indic-1B/IndicTransToolkit")

from src.translator import Translator

LOG_FILE = "/home/kyrotics/active_translation.log"

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def translate_corpus(input_dir, output_dir, batch_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log(f"--- TRANSLATION PROCESS STARTED ---")
    log(f"Initializing Translator (EN -> BN)...")
    translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
    translator.load_model()
    log("Model loaded successfully.")

    txt_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    total = len(txt_files)
    log(f"Found {total} English files to translate.")

    for i, file_path in enumerate(txt_files):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            continue
            
        # Log every file for high visibility
        log(f"[{i+1}/{total}] Processing: {filename}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()
            if not raw_text: continue

            pages = raw_text.split("--- PAGE BREAK ---")
            translated_pages = []
            for page in pages:
                # Use sub-batches of 16 sentences to avoid VRAM spikes and show more activity
                all_sentences = [s.strip() for s in page.split(".") if s.strip()]
                if not all_sentences:
                    translated_pages.append("")
                    continue
                
                translated_sentences = []
                for j in range(0, len(all_sentences), 8):
                    sub_batch = all_sentences[j:j+8]
                    translations = translator.translate_batch(sub_batch)
                    translated_sentences.extend(translations)
                
                translated_pages.append(" ".join(translated_sentences))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n\n--- PAGE BREAK ---\n\n".join(translated_pages))
                
        except Exception as e:
            log(f"Error translating {filename}: {str(e)}")

if __name__ == "__main__":
    translate_corpus("data/raw/judgments_en/", "data/translated/", batch_size=32)
