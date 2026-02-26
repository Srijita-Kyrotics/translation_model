import yaml
from src.translator import Translator
import os

def translate_yaml():
    input_path = "data/mock_yaml_input.yaml"
    output_path = "data/mock_yaml_output.yaml"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    source_text = data.get("natural_text", "")
    if not source_text:
        print("Error: No natural_text found in YAML.")
        return

    # Split into lines to respect formatting and potentially avoid long-sequence issues
    lines = [line.strip() for line in source_text.split("\n") if line.strip()]

    print("Initializing English-to-Bengali Translator...")
    translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
    translator.load_model()

    print("Translating...")
    translated_lines = translator.translate_batch(lines)
    
    translated_text = "\n".join(translated_lines)
    
    # Update YAML data
    data["primary_language"] = "bn"
    data["natural_text"] = translated_text
    
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    
    print("Done.")

if __name__ == "__main__":
    translate_yaml()
