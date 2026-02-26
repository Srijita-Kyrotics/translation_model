import yaml
import sys
try:
    from src.structured_translator import StructuredTranslator
except ImportError as e:
    # If torch isn't available, we'll just mock the class entirely to test the logic
    print("Mocking complete StructureTranslator due to import error:")
    print(e)
    
    class StructuredTranslator:
        def __init__(self, src_lang="ben_Beng", tgt_lang="eng_Latn"):
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang
            
        def translate_yaml(self, input_path, output_path):
            with open(input_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            sentences_to_translate = []
            sentence_refs = [] 
            
            if "sentences" in data and isinstance(data["sentences"], list):
                for i, item in enumerate(data["sentences"]):
                    if "text" in item:
                        sentences_to_translate.append(item["text"])
                        sentence_refs.append(("sentence", i, "text"))

            if "tables" in data and isinstance(data["tables"], list):
                for t_idx, table in enumerate(data["tables"]):
                    if "headers" in table and isinstance(table["headers"], list):
                        for h_idx, header in enumerate(table["headers"]):
                            sentences_to_translate.append(header)
                            sentence_refs.append(("table_header", t_idx, h_idx))
                    
                    if "rows" in table and isinstance(table["rows"], list):
                        for r_idx, row in enumerate(table["rows"]):
                            for c_idx, cell in enumerate(row):
                                sentences_to_translate.append(str(cell))
                                sentence_refs.append(("table_cell", t_idx, (r_idx, c_idx)))

            print(f"Translating {len(sentences_to_translate)} items...")
            translated_results = [f"[TRANSLATED] {item}" for item in sentences_to_translate]

            for i, (ref_type, idx, sub) in enumerate(sentence_refs):
                ref_val = translated_results[i]
                if ref_type == "sentence":
                    data["sentences"][idx]["text"] = ref_val
                elif ref_type == "table_header":
                    data["tables"][idx]["headers"][sub] = ref_val
                elif ref_type == "table_cell":
                    row_idx, col_idx = sub
                    data["tables"][idx]["rows"][row_idx][col_idx] = ref_val

            data["original_language"] = self.src_lang
            data["translated_language"] = self.tgt_lang
            if "primary_language" in data:
                 data["primary_language"] = "English" if self.tgt_lang == "eng_Latn" else "Bengali"

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

st = StructuredTranslator()

st.translate_yaml("data/sample_ocr.yaml", "data/sample_ocr_out.yaml")

print("\n--- Output YAML ---")
with open("data/sample_ocr_out.yaml", "r") as f:
    print(f.read())
