from src.translator import Translator
import torch

def test_correction():
    print("Initializing Translator (with Correction)...")
    translator = Translator(use_correction=True)
    translator.load_model()
    
    sentences = [
        "আমি গতকাল স্কুলে গিয়েছিলাম।", 
    ]
    
    print("\n--- Testing Translation Correction ---")
    translator.use_correction = False
    raw_translations = translator.translate_batch(sentences)
    
    corrected_translations = translator.corrector.correct_batch(raw_translations)
    # Debug cleaning in corrector
    # We can't easily access the internal cleaning of corrector from here without modifying it or copying logic.
    # Let's just trust corrector's logic for now and see the final output.
    
    for s, raw, corr in zip(sentences, raw_translations, corrected_translations):
        print(f"Original: {s}")
        print(f"Raw: {raw}")
        print(f"Corrected: {corr}")

    print("\n--- Testing Independent Grammar Correction ---")
    broken_english = [
        "I goes to school yesterday.",
        "She don't like apples.",
        "Him is a good boy.",
        "I was at the school yesterday, 09BC."
    ]
    corrected_broken = translator.corrector.correct_batch(broken_english)
    for b, c in zip(broken_english, corrected_broken):
        print(f"Broken: {b}")
        print(f"Fixed:  {c}")

if __name__ == "__main__":
    test_correction()
