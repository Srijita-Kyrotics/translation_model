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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

def finetune():
    model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
    dataset_path = os.path.join("data", "final", "hf_dataset")
    output_dir = "./indictrans2-finetuned-court"
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. Load Model in fp16 (no quantization to avoid deepcopy crash)
    print(f"Loading model {model_name} in fp16...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 3. Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # 4. Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    
    # 5. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocessing function
    src_lang = "ben_Beng"
    tgt_lang = "eng_Latn"
    
    def preprocess_function(examples):
        # IndicTrans2 tokenizer expects: "src_lang tgt_lang actual_text"
        inputs = [f"{src_lang} {tgt_lang} {text}" for text in examples["bengali"]]
        targets = examples["english"]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=512, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
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
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=1000,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
        label_smoothing_factor=0.0,
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
    
    print("Starting training...")
    trainer.train()
    
    # Save the adapter
    trainer.save_model(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")

if __name__ == "__main__":
    finetune()
