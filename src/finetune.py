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
from transformers.trainer_utils import get_last_checkpoint

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs (e.g. RTX 4090)
torch.backends.cuda.matmul.allow_tf32 = True

class StableSeq2SeqTrainer(Seq2SeqTrainer):
    """
    A custom trainer that handles sequence length mismatches between logits and labels,
    which is common in custom Rotary architectures.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs)
        
        if labels is not None:
            logits = outputs.get("logits")
            # --- CUSTOM ALIGNMENT LOGIC ---
            # If logits and labels have different sequence lengths (dim 1), 
            # we trim the labels to match the logits.
            if logits.shape[1] != labels.shape[1]:
                # Typically the model drops tokens from the head
                # 204 vs 208 means we take labels[:, -204:]
                diff = labels.shape[1] - logits.shape[1]
                if diff > 0:
                    labels = labels[:, diff:]
                else:
                    logits = logits[:, :labels.shape[1]]
            # ------------------------------
            
            if torch.isnan(logits).any():
                print("⚠️ NaN detected in logits")
                
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.args.label_smoothing_factor
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        else:
            loss = outputs.get("loss")
            
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        CUSTOM STREAMING EVALUATION:
        This overrides the default HuggingFace evaluation loop to strictly iterate batch-by-batch
        without storing any logits, predictions, or large tensors in CPU RAM.
        This guarantees stable memory footprint and prevents OOM crashes.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        self.model.eval()
        total_eval_loss = 0.0
        num_batches = 0
        
        # Streaming evaluation loop - process exactly one batch at a time
        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            batch_size = inputs["input_ids"].size(0)
            
            with torch.inference_mode():
                loss, _ = self.compute_loss(self.model, inputs, return_outputs=True)
                
                if loss is not None:
                    total_eval_loss += loss.item() * batch_size
                    num_batches += batch_size
                    
            del inputs
            torch.cuda.empty_cache()
            
        # Compute final average
        avg_loss = total_eval_loss / max(num_batches, 1)
        
        # Return cleanly accumulated scalar metrics
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
        }
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        # Ensure model goes back to training mode
        self.model.train()
        
        return metrics

def finetune(direction="en-bn"):
    if direction == "bn-en":
        model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
        src_lang, tgt_lang = "ben_Beng", "eng_Latn"
        output_dir = "./indictrans2-finetuned-bn-en"
        tokenized_path = "data/final/tokenized_bn_en"
    else:
        model_name = "prajdabre/rotary-indictrans2-en-indic-1B"
        src_lang, tgt_lang = "eng_Latn", "ben_Beng"
        output_dir = "./indictrans2-finetuned-en-bn"
        tokenized_path = "data/final/tokenized_en_bn"

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. Load PRE-TOKENIZED Dataset from disk (Prevents the 79% stall)
    print(f"Loading pre-tokenized dataset from {tokenized_path}...")
    if not os.path.exists(tokenized_path):
        raise FileNotFoundError(f"Tokenized dataset not found at {tokenized_path}. Please run src/cache_tokenized_dataset.py first.")
    tokenized_dataset = load_from_disk(tokenized_path)
    
    # 3. Load Model in bfloat16
    print(f"Loading model {model_name} in bfloat16...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 4. Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # 5. UNFREEZE all model parameters
    print("Unfreezing all model parameters for full-parameter fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e9:.2f}B")
    
    # 6. Data Collator (Standard Seq2Seq collation)
    # The Seq2SeqTrainer uses the collator to shift labels into decoder_input_ids automatically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 7. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,  # Added to strictly bound evaluation RAM
        gradient_accumulation_steps=128, 
        learning_rate=5e-5, 
        weight_decay=0.0001, 
        max_steps=5000, 
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=100,
        prediction_loss_only=True,  # IMPORTANT: Do not accumulate massive logits in RAM, only compute loss
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="inverse_sqrt",
        label_smoothing_factor=0.1,
        warmup_steps=100,
        report_to="none", # Correct way to disable W&B and other integrations
    )
    
    # 8. Trainer
    trainer = StableSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print(f"Starting training Stage 1 for {direction}...")
    
    # Robust Checkpointing
    last_checkpoint = None
    if os.path.exists(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No prior checkpoint found. Starting training from scratch (pretrained weights).")
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

