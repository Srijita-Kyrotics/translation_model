import os
import glob
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys
from IndicTransToolkit import IndicProcessor
from src.corrector import GrammarCorrector

class Translator:
    def __init__(self, model_name="prajdabre/rotary-indictrans2-indic-en-1B", src_lang="ben_Beng", tgt_lang="eng_Latn", use_correction=True):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_correction = use_correction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.ip = None
        self.corrector = None
        
    def load_model(self):
        """Loads the transformers model, tokenizer, IndicProcessor, and GrammarCorrector."""
        print(f"Loading model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        self.ip = IndicProcessor(inference=True)
        print("Translation model loaded successfully.")

        if self.use_correction:
            print("Initializing Grammar Corrector...")
            self.corrector = GrammarCorrector()
            self.corrector.load_model()

    def translate_batch(self, sentences):
        """Translates a batch of sentences using IndicTrans2 and optionally corrects grammar."""
        if self.model is None or self.tokenizer is None or self.ip is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocessing
        batch = self.ip.preprocess_batch(sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        
        # Tokenization
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs, 
                max_length=1024,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decoding
        decoded_tokens = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Postprocessing
        translated_sentences = self.ip.postprocess_batch(decoded_tokens, lang=self.tgt_lang)
        
        # Grammar Correction
        if self.use_correction and self.corrector:
            translated_sentences = self.corrector.correct_batch(translated_sentences)

        return translated_sentences

def process_translation(input_dir, output_dir, limit_files=None):
    """Translates text files in input_dir and saves them to output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    translator = Translator()
    translator.load_model()
    
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    if limit_files:
        files = files[:limit_files]
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Translating {file_name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if not lines:
            continue
            
        # Process in batches
        batch_size = 4
        translated_lines = []
        for i in tqdm(range(0, len(lines), batch_size)):
            batch = lines[i:i+batch_size]
            translated_lines.extend(translator.translate_batch(batch))
            
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            for line in translated_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    PROCESSED_DIR = os.path.join("data", "processed")
    TRANSLATED_DIR = os.path.join("data", "translated")
    
    # Process all files for the final production dataset
    process_translation(PROCESSED_DIR, TRANSLATED_DIR)
