import os
import glob
import collections

def audit_moibake(bn_dir):
    files = glob.glob(os.path.join(bn_dir, "*.txt"))
    total_files = len(files)
    corrupted_files = 0
    
    # Ranges common in Moibake for these PDFs
    # Greek: 0370-03FF
    # Cyrillic: 0400-04FF
    
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content: continue
            
            greek_chars = sum(1 for c in content if '\u0370' <= c <= '\u03FF')
            cyrillic_chars = sum(1 for c in content if '\u0400' <= c <= '\u04FF')
            bn_chars = sum(1 for c in content if '\u0980' <= c <= '\u09FF')
            
            total_chars = len(content)
            if total_chars == 0: continue
            
            moibake_ratio = (greek_chars + cyrillic_chars) / total_chars
            
            if moibake_ratio > 0.05 or (bn_chars / total_chars < 0.1 and total_chars > 100):
                corrupted_files += 1
                if corrupted_files <= 10:
                    print(f"Corrupted: {os.path.basename(f_path)} | Greek: {greek_chars}, Cyrillic: {cyrillic_chars}, BN: {bn_chars} | Total: {total_chars}")
        except:
            continue
            
    print(f"\nAudit Results:")
    print(f"Total files: {total_files}")
    print(f"Corrupted/Failed files: {corrupted_files} ({corrupted_files/total_files:.2%})")

if __name__ == "__main__":
    audit_moibake("data/raw/judgments_bn/")
