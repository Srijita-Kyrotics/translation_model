import docx
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

def process_directory(input_dir, output_dir):
    """Processes all .docx files in the input directory and saves them as .txt in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(input_dir, "*.docx"))
    print(f"Found {len(files)} files in {input_dir}")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        text_lines = extract_text_from_docx(file_path)
        
        output_file_name = file_name.replace(".docx", ".txt")
        output_path = os.path.join(output_dir, output_file_name)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for line in text_lines:
                f.write(line + "\n")
        
    print("Extraction complete.")

if __name__ == "__main__":
    # Internal paths as configured in task
    INPUT_FOLDER = "EBMT 1"
    RAW_DATA_FOLDER = os.path.join("data", "raw")
    
    process_directory(INPUT_FOLDER, RAW_DATA_FOLDER)
