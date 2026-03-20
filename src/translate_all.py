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

def translate_corpus(input_dir, output_dir, batch_size=32):
    """
    Translates all English txt files in input_dir to Bengali.
    Saves results in output_dir with same filename suffixes for pairing.
    Handles '--- PAGE BREAK ---' markers by translating page-by-page.
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
    total = len(txt_files)
    for i, file_path in enumerate(txt_files):
        if i % 10 == 0:
            print(f"Progress: {i}/{total} files ({(i/total)*100:.2f}%)")
        
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

            # Page-Aware Logic: Split by the marker
            page_marker = "--- PAGE BREAK ---"
            pages = raw_text.split(page_marker)
            translated_pages = []

            for page in pages:
                page = page.strip()
                if not page:
                    translated_pages.append("")
                    continue
                
                # Sentence tokenize the page
                sentences = nltk.sent_tokenize(page)
                translated_sentences = []
                
                # Translate in batches
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i+batch_size]
                    translated_batch = translator.translate_batch(batch)
                    translated_sentences.extend(translated_batch)
                
                translated_pages.append("\n".join(translated_sentences))

            # Re-join with the marker
            translated_content = f"\n\n{page_marker}\n\n".join(translated_pages)

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
    OUTPUT_DIR = "data/translated/"
    
    # Run translation
    # Batch size of 32 is safe for stable background operation
    translate_corpus(INPUT_DIR, OUTPUT_DIR, batch_size=32)
