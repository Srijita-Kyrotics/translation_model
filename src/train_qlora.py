import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class PreTokenizedDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, weights_only=False)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]
        # IndicTrans2 specific IDs
        self.pad_token_id = 1
        self.decoder_start_token_id = 2

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Manually create decoder_input_ids by shifting labels right
        # This fixes the "Value Error: specify either decoder_input_ids..." in custom model
        shifted_labels = labels.clone()
        # Replace -100 (ignore index) with pad_token_id for shifting
        shifted_labels[shifted_labels == -100] = self.pad_token_id
        
        decoder_input_ids = torch.full_like(shifted_labels, fill_value=self.pad_token_id)
        decoder_input_ids[1:] = shifted_labels[:-1]
        decoder_input_ids[0] = self.decoder_start_token_id

        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": labels,
            "decoder_input_ids": decoder_input_ids
        }


from torch.utils.data import Dataset, Subset

# ... (PreTokenizedDataset remains the same)

def train():
    DEBUG_MODE = False  # Switched to full dataset training
    model_id = "prajdabre/rotary-indictrans2-en-indic-1B"
    output_dir = "models/indictrans2-v6-scratch"

    print("=========================================", flush=True)
    print("🚀 INITIALIZING SCRATCH QLoRA TRAINING (V6)", flush=True)
    if DEBUG_MODE:
        print("🛠️ DEBUG MODE ENABLED: Using 1,000 samples for smoke test.", flush=True)
    print("=========================================", flush=True)

    print("Loading pre-tokenized (MAX_LEN=512) datasets...", flush=True)
    full_train_dataset = PreTokenizedDataset("data/final/train_tokenized.pt")
    val_dataset = PreTokenizedDataset("data/final/val_tokenized.pt")
    
    if DEBUG_MODE:
        train_dataset = Subset(full_train_dataset, range(min(1000, len(full_train_dataset))))
    else:
        train_dataset = full_train_dataset
        
    print(f"Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples", flush=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Monkey-patch BNB
    import transformers.integrations.bitsandbytes as bnb_integration
    bnb_integration.get_keys_to_not_convert = lambda model: []

    print("Loading Base Model in 4-bit NF4...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    print("Injecting LoRA Adapters (r=64, alpha=16)...", flush=True)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    print("Configuring Trainer (IndicTrans2 Specs)...", flush=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32, 
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        warmup_steps=100 if DEBUG_MODE else 4000,
        lr_scheduler_type="linear",
        label_smoothing_factor=0.1,
        logging_steps=1,                 
        eval_strategy="no",
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        predict_with_generate=False, 
        report_to="none"
    )

    from transformers import default_data_collator

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("=========================================", flush=True)
    print("🔥 TRAINING LAUNCHED! Monitoring GPU... 🔥", flush=True)
    print("=========================================", flush=True)
    trainer.train()

    if not DEBUG_MODE:
        print(f"✅ Training Complete! Saving Final Model to {output_dir}/final", flush=True)
        trainer.save_model(f"{output_dir}/final")


if __name__ == "__main__":
    train()
