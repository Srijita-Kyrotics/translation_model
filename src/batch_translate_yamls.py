import os
import glob
from tqdm import tqdm
from src.structured_translator import StructuredTranslator
import argparse

def batch_process(input_dir, output_dir, src_lang="eng_Latn", tgt_lang="ben_Beng", adapter_path=None):
    """
    Traverses input_dir for .yaml files and translates them using StructuredTranslator.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    yaml_files = glob.glob(os.path.join(input_dir, "**", "*.yaml"), recursive=True)
    print(f"Found {len(yaml_files)} YAML files to translate.")
    
    if not yaml_files:
        print("No YAML files found.")
        return

    # Initialize translator once
    st = StructuredTranslator(
        src_lang=src_lang, 
        tgt_lang=tgt_lang, 
        adapter_path=adapter_path
    )
    print("Loading model weight once for batch processing...")
    st.load_model()
    
    for yaml_path in tqdm(yaml_files, desc="Processing documents"):
        # Determine output path maintaining subfolder structure if needed
        rel_path = os.path.relpath(yaml_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        try:
            st.translate_yaml(yaml_path, out_path)
        except Exception as e:
            print(f"Failed to translate {yaml_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch translate a directory of structured YAML documents.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input YAMLs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save translated YAMLs.")
    parser.add_argument("--src", type=str, default="eng_Latn", help="Source language (default: English).")
    parser.add_argument("--tgt", type=str, default="ben_Beng", help="Target language (default: Bengali).")
    parser.add_argument("--adapter", type=str, default=None, help="Path to fine-tuned LoRA adapter.")
    
    args = parser.parse_args()
    
    batch_process(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        src_lang=args.src, 
        tgt_lang=args.tgt,
        adapter_path=args.adapter
    )
