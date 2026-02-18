import os
import glob
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-bn-en"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Loads the transformers model and tokenizer."""
        print(f"Loading model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        print("Model loaded successfully.")

    def translate_batch(self, sentences):
        """Translates a batch of sentences using OPUS-MT."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs, 
                max_length=256,
                num_beams=4
            )
        
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

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
        batch_size = 8
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
