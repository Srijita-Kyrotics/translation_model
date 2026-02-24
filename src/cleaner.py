from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
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
    
    # Supreme Court specific headers to skip
    skip_keywords = ["সুপ্রিম কোর্ট", "SUPREME", "S.C.R.", "REPORTABLE"]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header-like lines
        if any(kw in line.upper() for kw in skip_keywords):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def normalize_bengali(text):
    """Normalizes Bengali text using indic-nlp-library."""
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("bn")
    return normalizer.normalize(text)

def process_cleaning(input_dir, output_dir, is_bengali=True):
    """Cleans and optionally normalizes all .txt files in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Cleaning {len(files)} files in {input_dir} (Bengali: {is_bengali})...")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. Clean structure (common for both EN and BN)
        cleaned = clean_text(content)
        
        # 2. Normalize if Bengali
        if is_bengali:
            processed = normalize_bengali(cleaned)
        else:
            processed = cleaned
            
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed)
            
    print(f"Finished processing {input_dir}")

if __name__ == "__main__":
    RAW_BASE = os.path.join("data", "raw")
    PROCESSED_BASE = os.path.join("data", "processed")
    
    # 1. Original docx extracted data (all Bengali)
    process_cleaning(RAW_BASE, PROCESSED_BASE, is_bengali=True)
    
    # 2. New Bengali PDF data
    raw_bn_pdf = os.path.join(RAW_BASE, "judgments_bn")
    proc_bn_pdf = os.path.join(PROCESSED_BASE, "judgments_bn")
    if os.path.exists(raw_bn_pdf):
        process_cleaning(raw_bn_pdf, proc_bn_pdf, is_bengali=True)
        
    # 3. New English PDF data
    raw_en_pdf = os.path.join(RAW_BASE, "judgments_en")
    proc_en_pdf = os.path.join(PROCESSED_BASE, "judgments_en")
    if os.path.exists(raw_en_pdf):
        # We save English directly to 'processed' but it won't be translated later
        process_cleaning(raw_en_pdf, proc_en_pdf, is_bengali=False)
