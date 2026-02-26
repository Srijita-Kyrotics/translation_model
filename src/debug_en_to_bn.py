from src.translator import Translator
import torch

def debug_translation():
    # Simple sentence
    sample_en = "This is a legal judgment from the Calcutta High Court."

    # Initialize Translator for EN -> BN
    translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
    translator.load_model()

    print("\nPre-processing...")
    batch = translator.ip.preprocess_batch([sample_en], src_lang="eng_Latn", tgt_lang="ben_Beng")
    print(f"Pre-processed batch: {batch}")

    print("\nTokenizing...")
    inputs = translator.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(translator.device)
    
    print("\nGenerating (Greedy Search)...")
    with torch.no_grad():
        generated_tokens = translator.model.generate(
            **inputs, 
            max_length=128,
            num_beams=1, # Greedy search to avoid beam search issues
            use_cache=False
        )
    
    print("\nDecoding...")
    decoded_tokens = translator.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_sentences = translator.ip.postprocess_batch(decoded_tokens, lang="ben_Beng")
    
    print(f"\nResult: {translated_sentences[0]}")

if __name__ == "__main__":
    debug_translation()
