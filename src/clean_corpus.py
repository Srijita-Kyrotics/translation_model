import pandas as pd
import re

def clean_corpus(input_file, output_file):
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    original_len = len(df)
    
    print(f"Original Row Count: {original_len}")
    
    # 1. Replace internal newlines (\n), carriage returns (\r) with a single space
    print("Scrubbing internal page breaks and newlines...")
    df['english'] = df['english'].astype(str).str.replace(r'[\n\r]+', ' ', regex=True)
    df['bengali_original'] = df['bengali_original'].astype(str).str.replace(r'[\n\r]+', ' ', regex=True)
    
    # 2. Collapse multiple spaces into a single space
    print("Removing excessive whitespace...")
    df['english'] = df['english'].str.replace(r'\s{2,}', ' ', regex=True).str.strip()
    df['bengali_original'] = df['bengali_original'].str.replace(r'\s{2,}', ' ', regex=True).str.strip()
    
    # 3. Purge OCR Metadata Artifacts (olmocr metadata)
    print("Purging olmocr metadata artifacts...")
    metadata_patterns = ['PAGE BREAK', 'primary_language:', 'is_rotation_valid:', 'rotation_correction:', 'is_table:', 'is_diagram:']
    pattern_regex = '|'.join(metadata_patterns)
    df = df[~df['english'].str.contains(pattern_regex, case=False, na=False)]
    df = df[~df['bengali_original'].str.contains(pattern_regex, case=False, na=False)]
    
    # 4. Drop any rows that became completely empty after cleaning
    df = df.dropna(subset=['english', 'bengali_original'])
    df = df[(df['english'].str.strip() != '') & (df['bengali_original'].str.strip() != '')]
    
    print(f"Final Cleaned Row Count: {len(df)}")
    
    # 4. Save the cleaned CSV
    print(f"Saving cleaned dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    print("✅ Cleanup Complete!")

if __name__ == "__main__":
    input_path = "data/final/parallel_corpus_v5_labse_gold.csv"
    output_path = "data/final/parallel_corpus_v5_labse_gold_clean.csv"
    clean_corpus(input_path, output_path)
