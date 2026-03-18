import torch
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

def finetune(direction="bn-en"):
    if direction == "bn-en":
        model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
        src_lang, tgt_lang = "ben_Beng", "eng_Latn"
        output_dir = "./indictrans2-finetuned-bn-en"
    else:
        model_name = "prajdabre/rotary-indictrans2-en-indic-1B"
        src_lang, tgt_lang = "eng_Latn", "ben_Beng"
        output_dir = "./indictrans2-finetuned-en-bn"

    dataset_path = os.path.join("data", "final", "hf_dataset")
    
    print(f"Loading dataset from {dataset_path} for direction {direction}...")
    dataset = load_from_disk(dataset_path)
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. Load Model in fp16
    print(f"Loading model {model_name} in fp16...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 3. Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # 4. UNFREEZE all model parameters
    print("Unfreezing all model parameters for full-parameter fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e9:.2f}B")
    
    # Preprocessing function
    def preprocess_function(examples):
        # IndicTrans2 tokenizer expects: "src_lang tgt_lang actual_text"
        if direction == "bn-en":
            inputs = [f"{src_lang} {tgt_lang} {text}" for text in examples["bengali"]]
            targets = examples["english"]
        else:
            inputs = [f"{src_lang} {tgt_lang} {text}" for text in examples["english"]]
            targets = examples["bengali"]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=512, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"Tokenizing dataset for {direction}...")
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    # Data Collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=128, 
        learning_rate=1e-4, 
        weight_decay=0.0001, 
        max_steps=5000, 
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        predict_with_generate=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="inverse_sqrt",
        label_smoothing_factor=0.1,
        warmup_steps=4000,
    )
    
    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print(f"Starting training Stage 1 for {direction}...")
    trainer.train()
    
    # Save Stage 1 results
    stage1_output = os.path.join(output_dir, "stage1")
    trainer.save_model(stage1_output)
    print(f"Stage 1 {direction} complete. Model saved to {stage1_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune IndicTrans2 for legal judgments.")
    parser.add_argument("--direction", type=str, default="bn-en", choices=["bn-en", "en-bn"], help="Translation direction to fine-tune.")
    args = parser.parse_args()
    
    finetune(direction=args.direction)

if __name__ == "__main__":
    finetune()
