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
        
        # Minimum word count to attempt correction.
        # Short strings (names, numbers, fragments) cause hallucinations.
        MIN_WORDS_FOR_CORRECTION = 5
            
        # 1. Clean residual non-English characters (like \u09BC)
        cleaned_sentences = []
        for s in sentences:
            s = re.sub(r'[^\x00-\x7F]+', '', s) 
            cleaned_sentences.append(s.strip())

        # 2. Separate short vs long sentences
        # Only send long sentences to the corrector
        indices_to_correct = []
        sentences_to_correct = []
        results = list(cleaned_sentences)  # start with cleaned originals

        for i, s in enumerate(cleaned_sentences):
            word_count = len(s.split())
            if word_count >= MIN_WORDS_FOR_CORRECTION:
                indices_to_correct.append(i)
                sentences_to_correct.append(s)
        
        # If nothing to correct, return cleaned originals
        if not sentences_to_correct:
            return results

        # 3. Run T5 correction only on longer sentences
        prefixed = [f"Fix grammar: {s}" for s in sentences_to_correct]
        
        inputs = self.tokenizer(
            prefixed, 
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
        
        # 4. Post-clean and validate each correction
        hallucination_pattern = re.compile(
            r'(fix\s+(the\s+)?grammar|change\s+(the\s+)?grammar|failed\s+grammar|gec)',
            re.IGNORECASE
        )
        
        for idx, original, corrected_text in zip(indices_to_correct, sentences_to_correct, corrected):
            # Strip any echoed prompt prefix
            clean = re.sub(
                r'^(To\s+)?(I\s+)?(Fix\s+(the\s+)?grammar|Change\s+(the\s+)?grammar|Failed\s+grammar|gec)[.:,;]?\s*',
                '', corrected_text, flags=re.IGNORECASE
            )
            # Apply twice for double echoes
            clean = re.sub(
                r'^(To\s+)?(I\s+)?(Fix\s+(the\s+)?grammar|Change\s+(the\s+)?grammar|Failed\s+grammar|gec)[.:,;]?\s*',
                '', clean, flags=re.IGNORECASE
            )
            clean = clean.strip()
            
            # Fallback to original if correction looks bad:
            # - Still contains hallucination markers
            # - Is empty
            # - Is drastically shorter (lost content)
            if (not clean 
                or hallucination_pattern.search(clean)
                or len(clean) < len(original) * 0.3):
                results[idx] = original
            else:
                results[idx] = clean
            
        return results

if __name__ == "__main__":
    # Simple test
    corrector = GrammarCorrector()
    corrector.load_model()
    broken = ["I goes to school yesterday.", "She don't like apples."]
    corrected = corrector.correct_batch(broken)
    for b, c in zip(broken, corrected):
        print(f"Original: {b}\nCorrected: {c}\n")
