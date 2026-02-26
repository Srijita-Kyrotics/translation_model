import os
import torch
from datasets import load_from_disk
from tqdm import tqdm
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from src.translator import Translator
from IndicTransToolkit import IndicProcessor

def evaluate_finetuned_model():
    dataset_path = os.path.join("data", "final", "hf_dataset")
    print(f"Loading validation dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    val_data = dataset["validation"]
    
    print(f"Loaded {len(val_data)} validation pairs.")

    print("Loading base model and fine-tuned adapter...")
    base_model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
    adapter_path = "./indictrans2-finetuned-court"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    translator = Translator(
        model_name=base_model_name,
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        use_correction=False
    )
    
    # Inject the fine-tuned model into the Translator wrapper
    translator.model = model
    translator.tokenizer = tokenizer
    
    # Needs to be explicitly initialized and assigned to bypass checks
    translator.ip = IndicProcessor(inference=True)
    
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    print("Translating validation set with fine-tuned model...")
    batch_size = 8
    limit_samples = 200  # For quick verification
    val_bn = val_data["bengali"][:limit_samples]
    val_en = val_data["english"][:limit_samples]
    
    predictions = []
    for i in tqdm(range(0, len(val_bn), batch_size)):
        batch_bn = val_bn[i:i+batch_size]
        translated_batch = translator.translate_batch(batch_bn)
        predictions.extend(translated_batch)

    references = [[en] for en in val_en]

    print("\nCalculating metrics...")
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    chrf_results = chrf.compute(predictions=predictions, references=references)

    print("\n" + "="*40)
    print("FINETUNED EVALUATION RESULTS")
    print("="*40)
    print(f"SacreBLEU Score: {bleu_results['score']:.2f}")
    print(f"chrF Score:      {chrf_results['score']:.2f}")
    print("="*40)

    output_log_path = os.path.join("data", "final", "finetuned_evaluation_results.txt")
    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write(f"SacreBLEU Score: {bleu_results['score']:.2f}\n")
        f.write(f"chrF Score:      {chrf_results['score']:.2f}\n\n")
        f.write("Sample Translations (Fine-tuned):\n")
        f.write("-" * 50 + "\n")
        
        for i in range(min(5, len(predictions))):
            f.write(f"Source (BN): {val_data['bengali'][i]}\n")
            f.write(f"Target (EN): {val_data['english'][i]}\n")
            f.write(f"Model  (EN): {predictions[i]}\n")
            f.write("-" * 50 + "\n")
            
    print(f"\nDetailed logs saved to {output_log_path}")

if __name__ == "__main__":
    evaluate_finetuned_model()
