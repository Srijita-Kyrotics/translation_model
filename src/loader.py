import docx
import fitz  # PyMuPDF
import os
import glob

def extract_text_from_docx(file_path):
    """Extracts all text from a .docx file and returns it as a list of paragraphs."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    return full_text

def extract_text_from_pdf(file_path):
    """Extracts all text from a .pdf file and returns it as a list of lines."""
    doc = fitz.open(file_path)
    full_text = []
    for page in doc:
        text = page.get_text("text")
        # Split into lines and filter empty ones
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text.extend(lines)
    return full_text

def process_directory(input_dir, output_dir, file_ext="*.docx"):
    """Processes all files with given extension in the input directory and saves them as .txt in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(input_dir, file_ext))
    print(f"Found {len(files)} files with {file_ext} in {input_dir}")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        if file_ext.endswith(".docx"):
            text_lines = extract_text_from_docx(file_path)
        elif file_ext.endswith(".pdf"):
            text_lines = extract_text_from_pdf(file_path)
        else:
            print(f"Unsupported file extension: {file_ext}")
            continue
            
        output_file_name = os.path.splitext(file_name)[0] + ".txt"
        
        # Renaming logic: Strip prefixes like '1950~scr_' to maintain consistency with EBMT 1
        if "~scr_" in output_file_name:
            output_file_name = output_file_name.split("~scr_")[-1].strip()
            
        output_path = os.path.join(output_dir, output_file_name)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for line in text_lines:
                f.write(line + "\n")
        
    print(f"Extraction for {file_ext} complete.")

if __name__ == "__main__":
    # Internal paths as configured in task
    DOCX_INPUT = "EBMT 1"
    PDF_INPUT_BN = os.path.join("data", "raw", "judgments", "bengali")
    PDF_INPUT_EN = os.path.join("data", "raw", "judgments", "english")
    
    RAW_DATA_FOLDER = os.path.join("data", "raw")
    
    # Process original docx files
    process_directory(DOCX_INPUT, RAW_DATA_FOLDER, "*.docx")
    
    # Process new Bengali PDF judgments
    if os.path.exists(PDF_INPUT_BN):
        process_directory(PDF_INPUT_BN, os.path.join(RAW_DATA_FOLDER, "judgments_bn"), "*.pdf")
    
    # Process new English PDF judgments
    if os.path.exists(PDF_INPUT_EN):
        process_directory(PDF_INPUT_EN, os.path.join(RAW_DATA_FOLDER, "judgments_en"), "*.pdf")
