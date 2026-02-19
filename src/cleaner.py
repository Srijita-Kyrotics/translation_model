from indicnlp.normalize.indic_normalize import DevanagariNormalizer, IndicNormalizerFactory
import os
import glob
import re

def clean_text(text):
    """
    Basic cleaning:
    - Remove disclaimers
    - Remove separator lines (________)
    - Remove multiple whitespaces
    """
    # Remove disclaimer blocks (case insensitive)
    disclaimer_pattern = re.compile(r"DISCLAIMER.*?(দাবিত্যাগ|execution and implementation\.?)", re.DOTALL | re.IGNORECASE)
    text = disclaimer_pattern.sub("", text)
    
    # Remove separator lines
    text = re.sub(r"_{3,}", "", text)
    
    # Remove Bengali disclaimer if missed by regex
    text = text.replace("দাবিত্যাগ", "")
    
    lines = text.split("\n")
    cleaned_lines = []
    
    skip_keywords = ["সুপ্রিম কোর্ট", "SUPREME", "S.C.R."]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header-like lines containing SCR or Supreme Court in Bengali/English
        if any(kw in line for kw in skip_keywords):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def normalize_bengali(text):
    """Normalizes Bengali text using indic-nlp-library."""
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("bn")
    return normalizer.normalize(text)

def process_processed_directory(input_dir, output_dir):
    """Cleans and normalizes all .txt files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Cleaning {len(files)} files...")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. Clean structure
        cleaned = clean_text(content)
        
        # 2. Normalize Bengali
        normalized = normalize_bengali(cleaned)
        
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(normalized)
            
    print("Cleaning and normalization complete.")

if __name__ == "__main__":
    RAW_DIR = os.path.join("data", "raw")
    PROCESSED_DIR = os.path.join("data", "processed")
    process_processed_directory(RAW_DIR, PROCESSED_DIR)
