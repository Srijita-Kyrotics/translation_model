"""
Complete remaining tasks:
1. Evaluate fine-tuned model (BLEU/chrF)
2. Merge LoRA adapter into base model
3. Qualitative testing on legal sentences
"""
import os
import torch
from datasets import load_from_disk
from tqdm import tqdm
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit import IndicProcessor

def load_finetuned_model(model_name, adapter_path, device):
    """Load the base model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def translate_batch(model, tokenizer, ip, sentences, device, src_lang="ben_Beng", tgt_lang="eng_Latn"):
    """Translate a batch using IndicProcessor + model."""
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=1024, num_beams=5, num_return_sequences=1)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translated = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translated

def task1_evaluate(model, tokenizer, ip, device):
    """Task 1: Run BLEU/chrF on the fine-tuned model."""
    print("\n" + "="*60)
    print("TASK 1: Fine-Tuned Model Evaluation (BLEU/chrF)")
    print("="*60)
    
    dataset_path = os.path.join("data", "final", "hf_dataset")
    dataset = load_from_disk(dataset_path)
    val_data = dataset["validation"]
    print(f"Loaded {len(val_data)} validation pairs.")

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    predictions = []
    references = [[en] for en in val_data["english"]]

    batch_size = 4
    for i in tqdm(range(0, len(val_data["bengali"]), batch_size), desc="Evaluating"):
        batch_bn = val_data["bengali"][i:i+batch_size]
        translated = translate_batch(model, tokenizer, ip, batch_bn, device)
        predictions.extend(translated)

    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    chrf_results = chrf.compute(predictions=predictions, references=references)

    print(f"\nFine-Tuned SacreBLEU: {bleu_results['score']:.2f}")
    print(f"Fine-Tuned chrF:      {chrf_results['score']:.2f}")
    
    # Save results
    output_path = os.path.join("data", "final", "finetuned_evaluation_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FINE-TUNED MODEL EVALUATION\n")
        f.write("="*40 + "\n")
        f.write(f"SacreBLEU Score: {bleu_results['score']:.2f}\n")
        f.write(f"chrF Score: {chrf_results['score']:.2f}\n\n")
        f.write("Sample Translations:\n")
        f.write("-"*50 + "\n")
        for i in range(min(5, len(predictions))):
            f.write(f"Source (BN): {val_data['bengali'][i]}\n")
            f.write(f"Target (EN): {val_data['english'][i]}\n")
            f.write(f"Model  (EN): {predictions[i]}\n")
            f.write("-"*50 + "\n")
    print(f"Results saved to {output_path}")
    return bleu_results['score'], chrf_results['score']

def task2_merge_lora(model_name, adapter_path, device):
    """Task 2: Merge LoRA adapter into base model for faster inference."""
    print("\n" + "="*60)
    print("TASK 2: Merging LoRA Adapter into Base Model")
    print("="*60)
    
    merged_path = "./indictrans2-merged-court"
    
    print("Loading base model + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    print(f"Merged model saved to {merged_path}")
    return merged_path

def task3_qualitative_testing(model, tokenizer, ip, device):
    """Task 3: Qualitative testing on legal-domain sentences."""
    print("\n" + "="*60)
    print("TASK 3: Qualitative Legal Document Testing")
    print("="*60)
    
    legal_sentences = [
        "মাননীয় বিচারপতি পটঞ্জলি শাস্ত্রী, মুখার্জী এবং দাস এই মামলার রায় দেন।",
        "আপিলকারীর বিরুদ্ধে আয়কর কমিশনারের আদেশ বহাল রাখা হয়।",
        "সংবিধানের ১৪ অনুচ্ছেদ অনুসারে সকল নাগরিকের সমান অধিকার রয়েছে।",
        "এই আদালত মহিন্দরগড় জেলার ভূমি অধিগ্রহণ মামলা পর্যালোচনা করেন।",
        "রাষ্ট্রপতির আদেশক্রমে এই বিশেষ আপিলের অনুমতি প্রদান করা হয়।",
        "ভারতের সর্বোচ্চ আদালত দেওয়ানি আপিল নং ১২৩৪/২০২৫ গ্রহণ করেন।",
        "আসামীপক্ষের আইনজীবী যুক্তি প্রদর্শন করেন যে উক্ত চুক্তি বাতিলযোগ্য।",
        "জমি রাজস্ব আইন, ১৮৮৭ এর ধারা ৪৫ অনুযায়ী এই সিদ্ধান্ত গ্রহণ করা হয়েছে।",
    ]
    
    results = []
    print("\nTranslating legal-domain test sentences...\n")
    for i, sentence in enumerate(legal_sentences, 1):
        translated = translate_batch(model, tokenizer, ip, [sentence], device)
        result = translated[0]
        results.append((sentence, result))
        print(f"{i}. BN: {sentence}")
        print(f"   EN: {result}\n")
    
    # Save qualitative results
    output_path = os.path.join("data", "final", "qualitative_test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("QUALITATIVE LEGAL DOCUMENT TESTING\n")
        f.write("="*60 + "\n\n")
        for i, (bn, en) in enumerate(results, 1):
            f.write(f"[{i}] Bengali:  {bn}\n")
            f.write(f"    English:  {en}\n\n")
    print(f"Qualitative results saved to {output_path}")

def main():
    model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
    adapter_path = "./indictrans2-finetuned-court"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(model_name, adapter_path, device)
    ip = IndicProcessor(inference=True)
    
    # Task 1: Evaluate
    bleu, chrf_score = task1_evaluate(model, tokenizer, ip, device)
    
    # Task 3: Qualitative (run before merge since we have the model loaded)
    task3_qualitative_testing(model, tokenizer, ip, device)
    
    # Free GPU memory before merge
    del model
    torch.cuda.empty_cache()
    
    # Task 2: Merge LoRA
    merged_path = task2_merge_lora(model_name, adapter_path, device)
    
    print("\n" + "="*60)
    print("ALL REMAINING TASKS COMPLETED")
    print("="*60)
    print(f"Fine-Tuned BLEU: {bleu:.2f} | chrF: {chrf_score:.2f}")
    print(f"Merged model:    {merged_path}")
    print(f"Eval results:    data/final/finetuned_evaluation_results.txt")
    print(f"Qual results:    data/final/qualitative_test_results.txt")

if __name__ == "__main__":
    main()
