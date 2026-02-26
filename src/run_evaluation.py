import os
import torch
from datasets import load_from_disk
from tqdm import tqdm
import evaluate

from src.translator import Translator

def evaluate_model():
    """
    Evaluates the base (or fine-tuned) model against the Hugging Face validation dataset
    using SacreBLEU and chrF metrics.
    """
    dataset_path = os.path.join("data", "final", "hf_dataset")
    print(f"Loading validation dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    val_data = dataset["validation"]
    
    print(f"Loaded {len(val_data)} validation pairs.")

    # Note: For evaluation, we want to see the raw output of the model 
    # without grammar correction to accurately gauge translation quality.
    print("Initialize translator (No correction)...")
    translator = Translator(
        model_name="prajdabre/rotary-indictrans2-indic-en-1B",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        use_correction=False
    )
    translator.load_model()

    # Metrics
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    print("Translating validation set...")
    batch_size = 8
    limit_samples = 200 # For fair comparison
    val_bn = val_data["bengali"][:limit_samples]
    val_en = val_data["english"][:limit_samples]

    predictions = []
    references = [[en] for en in val_en]

    for i in tqdm(range(0, len(val_bn), batch_size)):
        batch_bn = val_bn[i:i+batch_size]
        translated_batch = translator.translate_batch(batch_bn)
        predictions.extend(translated_batch)

    assert len(predictions) == len(references), "Mismatch between predictions and references length."

    print("\nCalculating metrics...")
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    chrf_results = chrf.compute(predictions=predictions, references=references)

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"SacreBLEU Score: {bleu_results['score']:.2f}")
    print(f"chrF Score:      {chrf_results['score']:.2f}")
    print("="*40)

    # Save detailed output for qualitative review
    output_log_path = os.path.join("data", "final", "evaluation_results.txt")
    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write(f"SacreBLEU Score: {bleu_results['score']:.2f}\n")
        f.write(f"chrF Score: {chrf_results['score']:.2f}\n\n")
        f.write("Sample Translations:\n")
        f.write("-" * 50 + "\n")
        
        # Write out the first few examples for manual review
        for i in range(min(5, len(predictions))):
            f.write(f"Source (BN): {val_data['bengali'][i]}\n")
            f.write(f"Target (EN): {val_data['english'][i]}\n")
            f.write(f"Model  (EN): {predictions[i]}\n")
            f.write("-" * 50 + "\n")
            
    print(f"\nDetailed logs saved to {output_log_path}")


if __name__ == "__main__":
    evaluate_model()
