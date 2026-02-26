import yaml
import os
import torch
from tqdm import tqdm
from src.translator import Translator

class StructuredTranslator:
    def __init__(self, src_lang="ben_Beng", tgt_lang="eng_Latn", adapter_path=None, use_correction=True):
        self.translator = Translator(
            src_lang=src_lang, 
            tgt_lang=tgt_lang, 
            adapter_path=adapter_path,
            use_correction=use_correction
        )
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def load_model(self):
        """Initializes the underlying translation model."""
        self.translator.load_model()

    def translate_yaml(self, input_path, output_path):
        """Translates a structured YAML file and saves the result."""
        print(f"Loading YAML from {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 1. Collect all translatable strings
        sentences_to_translate = []
        sentence_refs = [] # (type, index, subkey)
        
        # Collect from sentences list
        if "sentences" in data and isinstance(data["sentences"], list):
            for i, item in enumerate(data["sentences"]):
                if "text" in item:
                    sentences_to_translate.append(item["text"])
                    sentence_refs.append(("sentence", i, "text"))

        # Collect from tables
        if "tables" in data and isinstance(data["tables"], list):
            for t_idx, table in enumerate(data["tables"]):
                # Headers
                if "headers" in table and isinstance(table["headers"], list):
                    for h_idx, header in enumerate(table["headers"]):
                        sentences_to_translate.append(header)
                        sentence_refs.append(("table_header", t_idx, h_idx))
                
                # Rows
                if "rows" in table and isinstance(table["rows"], list):
                    for r_idx, row in enumerate(table["rows"]):
                        for c_idx, cell in enumerate(row):
                            sentences_to_translate.append(str(cell))
                            sentence_refs.append(("table_cell", t_idx, (r_idx, c_idx)))

        if not sentences_to_translate:
            print("No translatable content found.")
            return

        # 2. Batch Translate
        print(f"Translating {len(sentences_to_translate)} items...")
        translated_results = []
        batch_size = 8
        for i in tqdm(range(0, len(sentences_to_translate), batch_size)):
            batch = sentences_to_translate[i:i+batch_size]
            translated_results.extend(self.translator.translate_batch(batch))

        # 3. Reconstruct the data structure
        for i, (ref_type, idx, sub) in enumerate(sentence_refs):
            ref_val = translated_results[i]
            if ref_type == "sentence":
                data["sentences"][idx]["text"] = ref_val
            elif ref_type == "table_header":
                data["tables"][idx]["headers"][sub] = ref_val
            elif ref_type == "table_cell":
                row_idx, col_idx = sub
                data["tables"][idx]["rows"][row_idx][col_idx] = ref_val

        # 4. Update Metadata
        data["original_language"] = self.src_lang
        data["translated_language"] = self.tgt_lang
        # If the input had a primary_language tag, we swap it or update it
        if "primary_language" in data:
             # Logic to determine target language tag (simplified)
             data["primary_language"] = "English" if self.tgt_lang == "eng_Latn" else "Bengali"

        # 5. Save output
        print(f"Saving translated YAML to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Translate structured YAML OCR output.")
    parser.add_argument("--input", type=str, required=True, help="Path to input YAML file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output YAML file.")
    parser.add_argument("--src", type=str, default="ben_Beng", help="Source language code.")
    parser.add_argument("--tgt", type=str, default="eng_Latn", help="Target language code.")
    
    args = parser.parse_args()
    
    st = StructuredTranslator(src_lang=args.src, tgt_lang=args.tgt)
    st.load_model()
    st.translate_yaml(args.input, args.output)
