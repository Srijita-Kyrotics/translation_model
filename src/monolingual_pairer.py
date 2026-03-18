import os
import glob
import pandas as pd
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from indicnlp.tokenize import sentence_tokenize

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class MonolingualPairer:
    def __init__(self, src_en_dir, src_bn_dir, trans_bn_dir, threshold=0.75):
        self.src_en_dir = src_en_dir
        self.src_bn_dir = src_bn_dir
        self.trans_bn_dir = trans_bn_dir
        self.threshold = threshold
        # Use a token pattern that supports Unicode characters (Bengali)
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")

    def get_bn_ratio(self, text):
        if not text: return 0
        bn_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        return bn_chars / len(text)

    def align_files(self, en_filename, bn_filename, trans_bn_filename):
        """Aligns sentences between a translated BN file and an original BN file."""
        try:
            with open(en_filename, 'r', encoding='utf-8') as f:
                en_text = f.read().strip()
            with open(bn_filename, 'r', encoding='utf-8') as f:
                bn_text = f.read().strip()
            with open(trans_bn_filename, 'r', encoding='utf-8') as f:
                trans_bn_text = f.read().strip()

            if not en_text or not bn_text or not trans_bn_text:
                return []
            
            # Check if original Bengali is garbage
            bn_ratio = self.get_bn_ratio(bn_text)
            if bn_ratio < 0.1:
                print(f"Skipping {os.path.basename(en_filename)}: Low BN ratio ({bn_ratio:.2%})")
                return []

            # Split into sentences
            # The translated file is SALTED (one sentence per line)
            trans_bn_sents = trans_bn_text.splitlines()
            # Original English must be split the SAME way translation was done
            en_sents = nltk.sent_tokenize(en_text)
            # Original Bengali is split using indicnlp
            orig_bn_sents = sentence_tokenize.sentence_split(bn_text, lang='bn')

            # Sanity check: len(en_sents) should be exact same as len(trans_bn_sents)
            if len(en_sents) != len(trans_bn_sents):
                print(f"Skipping {os.path.basename(en_filename)}: Sentence count mismatch (EN: {len(en_sents)}, Trans BN: {len(trans_bn_sents)})")
                return []

            if not trans_bn_sents or not orig_bn_sents:
                print(f"Skipping {os.path.basename(en_filename)}: Empty sentence lists")
                return []

            # Vectorize
            all_bn = trans_bn_sents + orig_bn_sents
            tfidf_matrix = self.vectorizer.fit_transform(all_bn)
            
            trans_vectors = tfidf_matrix[:len(trans_bn_sents)]
            orig_vectors = tfidf_matrix[len(trans_bn_sents):]

            sim_matrix = cosine_similarity(trans_vectors, orig_vectors)

            pairs = []
            max_scores = []
            for i, score_row in enumerate(sim_matrix):
                best_match_idx = score_row.argmax()
                best_score = score_row[best_match_idx]
                max_scores.append(best_score)

                if best_score >= self.threshold:
                    pairs.append({
                        "english": en_sents[i],
                        "bengali": orig_bn_sents[best_match_idx],
                        "similarity": best_score,
                        "source_file": os.path.basename(en_filename)
                    })

            if max_scores:
                avg_max = sum(max_scores) / len(max_scores)
                print(f"Processed {os.path.basename(en_filename)}: {len(pairs)}/{len(trans_bn_sents)} pairs found. Avg Max Sim: {avg_max:.3f}")

            return pairs

        except Exception as e:
            print(f"Error aligning {en_filename}: {e}")
            return []

def run_pairing():
    # Paths
    SRC_EN_DIR = "data/raw/judgments_en/"
    SRC_BN_DIR = "data/raw/judgments_bn/"
    TRANS_BN_DIR = "data/raw/judgments_en_translated/"
    OUTPUT_CSV = "data/final/parallel_corpus_gold.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    pairer = MonolingualPairer(SRC_EN_DIR, SRC_BN_DIR, TRANS_BN_DIR)
    
    # Find all translated files
    trans_files = glob.glob(os.path.join(TRANS_BN_DIR, "*.txt"))
    print(f"Found {len(trans_files)} translated files to pair.")

    all_gold_pairs = []

    for trans_path in tqdm(trans_files, desc="Pairing documents"):
        filename = os.path.basename(trans_path)
        
        # English source path
        en_path = os.path.join(SRC_EN_DIR, filename)
        
        # Bengali source path: replace _e.txt with _b.txt
        bn_filename = filename.replace("_e.txt", "_b.txt")
        bn_path = os.path.join(SRC_BN_DIR, bn_filename)

        if os.path.exists(en_path) and os.path.exists(bn_path):
            file_pairs = pairer.align_files(en_path, bn_path, trans_path)
            all_gold_pairs.extend(file_pairs)

    if all_gold_pairs:
        df = pd.DataFrame(all_gold_pairs)
        # Drop duplicates based on the pair
        df = df.drop_duplicates(subset=["english", "bengali"])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully saved {len(df)} gold pairs to {OUTPUT_CSV}")
    else:
        print("No gold pairs found.")

if __name__ == "__main__":
    run_pairing()
