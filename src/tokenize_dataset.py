"""
Pre-tokenizes the train.jsonl and val.jsonl datasets and saves to disk in Arrow format.
This avoids the `cannot pickle staticmethod` error when using IndicProcessor inside Hugging Face dataset.map().
Run once before training:
    python src/tokenize_dataset.py
"""
import json
from transformers import AutoTokenizer
from IndicTransToolkit import IndicProcessor

MODEL_ID = "prajdabre/rotary-indictrans2-en-indic-1B"
MAX_LEN = 512

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
ip = IndicProcessor(inference=False)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def tokenize_and_save(data, output_path):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    
    batch_size = 256
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        inputs_raw = [ex["translation"]["en"] for ex in batch]
        targets_raw = [ex["translation"]["bn"] for ex in batch]
        
        inputs_processed = ip.preprocess_batch(inputs_raw, src_lang="eng_Latn", tgt_lang="ben_Beng")
        targets_processed = ip.preprocess_batch(targets_raw, src_lang="ben_Beng", tgt_lang="ben_Beng")
        
        model_inputs = tokenizer(inputs_processed, max_length=MAX_LEN, truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            labels_enc = tokenizer(targets_processed, max_length=MAX_LEN, truncation=True, padding="max_length")
        
        for j in range(len(batch)):
            input_ids_list.append(model_inputs["input_ids"][j])
            attention_mask_list.append(model_inputs["attention_mask"][j])
            raw_labels = labels_enc["input_ids"][j]
            labels_list.append([(l if l != tokenizer.pad_token_id else -100) for l in raw_labels])
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(data)} examples...")
    
    import torch
    torch.save({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }, output_path)
    print(f"✅ Saved {len(data)} tokenized examples to {output_path}")

print("Loading train.jsonl...")
train_data = load_jsonl("data/final/train.jsonl")
print(f"Tokenizing {len(train_data)} training examples...")
tokenize_and_save(train_data, "data/final/train_tokenized.pt")

print("Loading val.jsonl...")
val_data = load_jsonl("data/final/val.jsonl")
print(f"Tokenizing {len(val_data)} validation examples...")
tokenize_and_save(val_data, "data/final/val_tokenized.pt")

print("🎉 Pre-tokenization complete!")
