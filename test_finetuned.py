import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

def test_model():
    model_dir = "./indictrans2-finetuned-en-bn/stage1"
    
    print(f"Loading full fine-tuned model and tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    ip = IndicProcessor(inference=True)
    
    test_sentences = [
        "The judgment is hereby passed by the High Court of Calcutta.",
        "The defendant was found guilty of all charges.",
        "The plaintiff has filed an affidavit in support of their claims.",
        "This is a legally binding contract."
    ]
    
    print("\n--- Translation Test (English to Bengali) ---")
    for sentence in test_sentences:
        batch = ip.preprocess_batch([sentence], src_lang="eng_Latn", tgt_lang="ben_Beng")
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=5,
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_translation = ip.postprocess_batch(decoded, lang="ben_Beng")[0]
        
        print(f"\nEnglish: {sentence}")
        print(f"Bengali: {final_translation}")

if __name__ == "__main__":
    test_model()
