import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import time

def run_stress_test():
    model_dir = "./indictrans2-finetuned-en-bn/stage1"
    
    print("Loading model and tokenizer for stress test...")
    ip = IndicProcessor(inference=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    base_paragraph = "The Supreme Court of India today ruled on the ongoing litigation regarding the fundamental rights under Article 21. The bench observed that the right to privacy is intrinsic to life and liberty. Furthermore, the defendant's appeal was dismissed due to lack of substantive evidence, and they were ordered to pay compensation. The plaintiff's earlier affidavit stands thoroughly verified. "
    
    print("\n--- Model Capacity Test ---")
    
    for i in [1, 2, 4, 8, 16]:
        # Create long paragraph
        input_text = base_paragraph * i
        word_count = len(input_text.split())
        
        batch = ip.preprocess_batch([input_text], src_lang="eng_Latn", tgt_lang="ben_Beng")
        inputs = tokenizer(batch, return_tensors="pt", truncation=False).to(model.device)
        input_token_len = inputs['input_ids'].shape[1]
        
        print(f"\n[Test Iteration {i}]")
        print(f"Input: {word_count} words | {input_token_len} tokens")
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,  # Allows very long generation
                num_beams=5,
            )
        gen_time = time.time() - start_time
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_translation = ip.postprocess_batch(decoded, lang="ben_Beng")[0]
        
        out_word_count = len(final_translation.split())
        
        print(f"Output Word Count: ~{out_word_count} words")
        print(f"Time Taken: {gen_time:.2f} seconds")
        print(f"Sample Output Snapshot (first 150 chars): {final_translation[:150]}...")
        
        # Check hallucination or cutoff
        if out_word_count < (word_count * 0.4):
            print("WARNING: Output seems drastically cut off!")
        elif out_word_count > (word_count * 2.5):
            print("WARNING: Potential hallucination (repetition) detected!")

if __name__ == "__main__":
    run_stress_test()
