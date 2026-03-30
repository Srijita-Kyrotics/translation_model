import pandas as pd
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Remove leading numbering like "1. ", "1) ", "[1] ", "1.  ", etc.
    text = re.sub(r'^\s*\[?\d+\]?[\.\)\s-]+\s*', '', text)
    return text.strip()

def merge_datasets():
    csv_path = "data/final/parallel_corpus_v5_labse_gold.csv"
    excel_path = "all_pairs_converted.xlsx"
    output_path = "data/final/parallel_corpus_v6_combined_clean.csv"

    print("Loading existing CSV dataset...")
    if os.path.exists(csv_path):
        # The existing CSV seems to have no header
        df_csv = pd.read_csv(csv_path, header=None, names=['english', 'bengali_original'])
        print(f"Loaded {len(df_csv)} rows from CSV.")
    else:
        print(f"Warning: CSV not found at {csv_path}. Creating empty DataFrame.")
        df_csv = pd.DataFrame(columns=['english', 'bengali_original'])

    print("Loading new Excel dataset...")
    if os.path.exists(excel_path):
        df_excel = pd.read_excel(excel_path)
        print(f"Loaded {len(df_excel)} rows from Excel.")
        
        # Mapping: Source -> english, Target -> bengali_original
        print("Mapping columns...")
        df_excel = df_excel.rename(columns={
            'Source': 'english',
            'Target': 'bengali_original'
        })
        
        # Drop Sl. no if it exists
        if 'Sl. no' in df_excel.columns:
            df_excel = df_excel.drop(columns=['Sl. no'])
            
        # Ensure only the two target columns remain
        df_excel = df_excel[['english', 'bengali_original']]
    else:
        print(f"Error: Excel file not found at {excel_path}!")
        return

    print("Cleaning text (removing unnecessary numbering)...")
    df_csv['english'] = df_csv['english'].apply(clean_text)
    df_csv['bengali_original'] = df_csv['bengali_original'].apply(clean_text)
    df_excel['english'] = df_excel['english'].apply(clean_text)
    df_excel['bengali_original'] = df_excel['bengali_original'].apply(clean_text)

    print("Merging datasets...")
    combined_df = pd.concat([df_csv, df_excel], ignore_index=True)
    
    # Final cleaning: drop empty or null rows
    combined_df = combined_df.dropna(subset=['english', 'bengali_original'])
    combined_df = combined_df[combined_df['english'].str.strip() != ""]
    combined_df = combined_df[combined_df['bengali_original'].str.strip() != ""]
    
    print(f"Total rows after merge and cleaning: {len(combined_df)}")
    
    print(f"Saving combined dataset to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    print("✅ Merge & Cleaning Complete!")

if __name__ == "__main__":
    merge_datasets()
