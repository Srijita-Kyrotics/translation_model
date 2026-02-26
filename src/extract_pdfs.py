"""
Step 1: Extract text from all downloaded Calcutta HC judgment PDFs.
Saves extracted text as .txt files in data/extracted/bengali/ and data/extracted/english/.
Alignment will be done separately.
"""
import os
import glob
import fitz  # PyMuPDF
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, return list of cleaned lines."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 3]
        return lines
    except:
        return []

def main():
    bn_pdf_dir = os.path.join("data", "raw", "judgments", "bengali")
    en_pdf_dir = os.path.join("data", "raw", "judgments", "english")
    bn_txt_dir = os.path.join("data", "extracted", "bengali")
    en_txt_dir = os.path.join("data", "extracted", "english")
    
    os.makedirs(bn_txt_dir, exist_ok=True)
    os.makedirs(en_txt_dir, exist_ok=True)
    
    # Process Bengali PDFs
    bn_pdfs = sorted(glob.glob(os.path.join(bn_pdf_dir, "*.pdf")))
    print(f"Extracting text from {len(bn_pdfs)} Bengali PDFs...")
    bn_ok, bn_empty = 0, 0
    for pdf_path in tqdm(bn_pdfs, desc="Bengali"):
        name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        out_path = os.path.join(bn_txt_dir, name)
        if os.path.exists(out_path):
            bn_ok += 1
            continue
        lines = extract_text_from_pdf(pdf_path)
        if lines:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            bn_ok += 1
        else:
            bn_empty += 1
    print(f"Bengali: {bn_ok} extracted, {bn_empty} empty/failed")
    
    # Process English PDFs
    en_pdfs = sorted(glob.glob(os.path.join(en_pdf_dir, "*.pdf")))
    print(f"\nExtracting text from {len(en_pdfs)} English PDFs...")
    en_ok, en_empty = 0, 0
    for pdf_path in tqdm(en_pdfs, desc="English"):
        name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        out_path = os.path.join(en_txt_dir, name)
        if os.path.exists(out_path):
            en_ok += 1
            continue
        lines = extract_text_from_pdf(pdf_path)
        if lines:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            en_ok += 1
        else:
            en_empty += 1
    print(f"English: {en_ok} extracted, {en_empty} empty/failed")
    
    print(f"\nDone! Text files saved to data/extracted/")

if __name__ == "__main__":
    main()
