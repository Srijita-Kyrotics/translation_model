from src.structured_translator import StructuredTranslator
import os
import yaml

def verify():
    input_yaml = os.path.join("data", "mock_ocr.yaml")
    output_yaml = os.path.join("data", "translated_ocr.yaml")
    
    if not os.path.exists(input_yaml):
        print(f"Input file {input_yaml} not found.")
        return

    print("Initializing Structured Translator...")
    st = StructuredTranslator(use_correction=False) # Disable correction for faster testing if needed
    st.load_model()
    
    print("\nProcessing Verification...")
    st.translate_yaml(input_yaml, output_yaml)
    
    print("\nVerification Output:")
    if os.path.exists(output_yaml):
        with open(output_yaml, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Error: Output file not generated.")

if __name__ == "__main__":
    verify()
