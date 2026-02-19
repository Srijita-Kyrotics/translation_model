import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

import re

class GrammarCorrector:
    def __init__(self, model_name="pszemraj/flan-t5-large-grammar-synthesis"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Loads the T5 grammar correction model."""
        print(f"Loading grammar corrector: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        print("Corrector loaded successfully.")

    def correct_batch(self, sentences):
        """Corrects grammar for a batch of sentences."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Corrector model not loaded. Call load_model() first.")
            
        # 1. Clean residual non-English characters (like \u09BC)
        # Keep Latin, numbers, punctuation, and common symbols.
        cleaned_sentences = []
        for s in sentences:
            # Remove characters that are NOT (Latin, numbers, punctuation, whitespace)
            # This regex keeps ASCII printable characters.
            s = re.sub(r'[^\x00-\x7F]+', '', s) 
            cleaned_sentences.append(s.strip())

        # 2. Prepare for T5
        # flan-t5 models work well with natural language instructions
        prefixed_sentences = [f"Fix grammar: {s}" for s in cleaned_sentences]
        
        inputs = self.tokenizer(
            prefixed_sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs, 
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        corrected = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # Post-clean: Remove the prompt prefix if echoed by the model
        final_corrected = []
        for c in corrected:
            # Handle various echo patterns: "Fix grammar:", "To fix grammar:", "I fix grammar:", "gec:"
            clean = re.sub(r'^(To\s+)?(I\s+)?(Fix grammar|gec)[.:,]?\s*', '', c, flags=re.IGNORECASE)
            # Apply again in case of double echo
            clean = re.sub(r'^(To\s+)?(I\s+)?(Fix grammar|gec)[.:,]?\s*', '', clean, flags=re.IGNORECASE)
            final_corrected.append(clean.strip())
            
        return final_corrected

if __name__ == "__main__":
    # Simple test
    corrector = GrammarCorrector()
    corrector.load_model()
    broken = ["I goes to school yesterday.", "She don't like apples."]
    corrected = corrector.correct_batch(broken)
    for b, c in zip(broken, corrected):
        print(f"Original: {b}\nCorrected: {c}\n")
