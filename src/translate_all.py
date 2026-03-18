import os
import glob
import time
from tqdm import tqdm
import torch
import nltk
from src.translator import Translator

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

def translate_corpus(input_dir, output_dir, batch_size=24):
    """
    Translates all English txt files in input_dir to Bengali.
    Saves results in output_dir with same filename suffixes for pairing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Initialize translator (EN -> BN)
    print("Initializing Translator (EN -> BN)...")
    translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
    translator.load_model()

    # Find all English txt files
    txt_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    print(f"Found {len(txt_files)} English files to translate.")

    # Main translation loop
    for file_path in tqdm(txt_files, desc="Translating files"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        # Resume logic: skip if file exists and has size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            continue

        try:
            start_time = time.time()
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()

            if not raw_text:
                continue

            sentences = nltk.sent_tokenize(raw_text)
            translated_lines = []
            
            # Translate in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                translated_batch = translator.translate_batch(batch)
                translated_lines.extend(translated_batch)

            # Join with newlines
            translated_content = "\n".join(translated_lines)

            # Atomic write
            temp_path = output_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            os.rename(temp_path, output_path)
            
            elapsed = time.time() - start_time
            # Small log for background monitoring
            if len(txt_files) > 0:
                 pass # Tqdm handles progress bar

        except Exception as e:
            print(f"\nError translating {filename}: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    INPUT_DIR = "data/raw/judgments_en/"
    OUTPUT_DIR = "data/raw/judgments_en_translated/"
    
    # Run translation
    # Batch size of 24 is conservative for 24GB VRAM
    translate_corpus(INPUT_DIR, OUTPUT_DIR, batch_size=24)
