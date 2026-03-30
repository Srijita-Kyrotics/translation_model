import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

def shift_tokens_right(input_ids_list, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right for a list of sequences.
    """
    shifted_input_ids = []
    for input_ids in input_ids_list:
        # Shift: [start_token, id1, id2, ..., idN-1]
        shifted = [decoder_start_token_id] + input_ids[:-1]
        # Handle -100 (mask) values which are common in labels
        shifted = [i if i != -100 else pad_token_id for i in shifted]
        shifted_input_ids.append(shifted)
    return shifted_input_ids

def prepare_tokenized_dataset(direction="en-bn"):
    if direction == "en-bn":
        model_name = "prajdabre/rotary-indictrans2-en-indic-1B"
        src_lang, tgt_lang = "eng_Latn", "ben_Beng"
        output_path = "data/final/tokenized_en_bn"
    else:
        model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
        src_lang, tgt_lang = "ben_Beng", "eng_Latn"
        output_path = "data/final/tokenized_bn_en"

    print(f"--- Pre-tokenizing dataset for {direction} ---")
    
    # Load raw dataset
    dataset_path = "data/final/hf_dataset"
    print(f"Loading raw dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def preprocess_function(examples):
        if direction == "en-bn":
            inputs = [f"{src_lang} {tgt_lang} {text}" for text in examples["english"]]
            targets = examples["bengali"]
        else:
            inputs = [f"{src_lang} {tgt_lang} {text}" for text in examples["bengali"]]
            targets = examples["english"]
            
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        # Use text_target for modern labels handling
        labels = tokenizer(text_target=targets, max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        
        # MANUALLY CREATE decoder_input_ids for rotary-indictrans architecture
        # Use our list-based shifter to handle variable sequence lengths
        model_inputs["decoder_input_ids"] = shift_tokens_right(
            model_inputs["labels"], 
            tokenizer.pad_token_id, 
            2  # decoder_start_token_id is usually 2 according to our check
        )
        return model_inputs

    print("Starting parallel tokenization (num_proc=16)...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        remove_columns=dataset["train"].column_names,
        desc=f"Tokenizing {direction}"
    )
    
    print(f"Saving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)
    print("Done!")

if __name__ == "__main__":
    prepare_tokenized_dataset(direction="en-bn")
