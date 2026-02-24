import torch
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    
    # 2. BitsAndBytes Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 3. Load Model in 4-bit
    print(f"Loading model {model_name} in 4-bit...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 4. Prepare for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # 5. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # standard for transformer layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocessing function
    def preprocess_function(examples):
        inputs = examples["bengali"]
        targets = examples["english"]
        
        # Note: IndicTrans2 might need specific prefixes/tags if not using IndicProcessor
        # But here we are fine-tuning the base model directly.
        # Standard IndicTrans2 prompt format: "ben_Beng: <sentence>"
        # However, many rotary variants handle this via special tokens or direct strings.
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
        per_device_train_batch_size=1, # Crucial for 6GB VRAM
        gradient_accumulation_steps=8,  # Effective batch size of 8
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True, # Use half-precision
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" # Save memory on optimizer states
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
