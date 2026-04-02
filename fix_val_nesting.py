from datasets import load_from_disk
print("Loading dataset...")
ds = load_from_disk('data/final/tokenized_en_bn')

def flatten_if_nested(example):
    # check if input_ids is a list of lists
    if len(example['input_ids']) > 0 and isinstance(example['input_ids'][0], list):
        example['input_ids'] = example['input_ids'][0]
    if len(example['attention_mask']) > 0 and isinstance(example['attention_mask'][0], list):
        example['attention_mask'] = example['attention_mask'][0]
    if len(example['labels']) > 0 and isinstance(example['labels'][0], list):
        example['labels'] = example['labels'][0]
    if 'decoder_input_ids' in example and len(example['decoder_input_ids']) > 0 and isinstance(example['decoder_input_ids'][0], list):
        example['decoder_input_ids'] = example['decoder_input_ids'][0]
    return example

print("Fixing validation split...")
val_ds = ds['validation'].map(flatten_if_nested, desc="Flattening validation nesting")
train_ds = ds['train'].map(flatten_if_nested, desc="Flattening train nesting")

ds['validation'] = val_ds
ds['train'] = train_ds
print("Saving dataset...")
ds.save_to_disk('data/final/tokenized_en_bn_fixed')
print("Replacing old dataset...")
import shutil
shutil.rmtree('data/final/tokenized_en_bn')
import os
os.rename('data/final/tokenized_en_bn_fixed', 'data/final/tokenized_en_bn')
print("Done!")
