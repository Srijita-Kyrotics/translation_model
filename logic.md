# Project Logic and Progress

This document provides a detailed technical breakdown of the logic, algorithms, and strategies applied in the Multilingual Translation Dataset Pipeline.

## 🧠 Pipeline Logic

### 1. Extraction Strategy (`loader.py`)
- **Philosophy**: Minimize formatting noise while preserving semantic integrity.
- **Logic**: The script iterates through every paragraph in the `.docx` file. It ignores empty strings and uses a paragraph-to-line mapping.
- **Why**: Maintaining paragraphs as individual lines is crucial for sentence-level alignment in downstream tasks.

### 2. Cleaning and Normalization (`cleaner.py`)
- **Structural Cleaning**: We use compiled regular expressions to strip multi-line "DISCLAIMER" blocks and horizontal separators (`____`).
- **Keyword Filtering**: Lines containing "Supreme Court" or standard case headers are discarded to focus the dataset on legal narrative/precedents rather than administrative metadata.
- **Indic Normalization**: We utilize the `IndicNormalizerFactory` (source: `indic-nlp-library`) to standardize Bengali Unicode representations. This handles varying representations of the same character, ensuring consistency for the translation model.

### 3. Translation Strategy (`translator.py`)
- **Model Choice**: `Helsinki-NLP/opus-mt-bn-en`. This is a SOTA (State-of-the-art) MarianMT model trained on various parallel corpora, offering a balance between accuracy and performance.
- **Batch Processing**: To optimize throughput, we implement batching (default: 8 sentences per batch). This significantly reduces the overhead of constant GPU/CPU context switching.
- **Beam Search**: The model uses a beam size of 4 during generation to explore more probable translation paths, leading to more natural phrasing.

### 4. Pairing and Alignment Logic (`pairer.py`)
- **Deterministic Rationale**: We opted for **line-index-based alignment**. 
- **The Logic**: Since `cleaner.py` and `translator.py` process files line-by-line without reordering, there is a 1-to-1 correspondence between line $i$ in the processed source and line $i$ in the translated target.
- **Robustness**: If there is a count mismatch (e.g., a line failed translation), the script aligns only up to the `min(source_length, target_length)` to prevent index errors.

## 📈 Progress Report

### What We Have Built
1. **Modular Infrastructure**: A complete `src/` directory with independent, executable modules.
2. **Structured Storage**: A standardized `data/` hierarchy that tracks the lifecycle of a sentence from raw input to final CSV.
3. **Environment Setup**: A verified dependency list in `requirements.txt`.
4. **Documentation**: Comprehensive `README.md` and `logic.md` for team onboarding and reproducibility.

### Current Output
- **Final Product**: `data/final/parallel_corpus.csv`.
- **Content**: A dataset containing columns: `source_file`, `line_index`, `bengali`, and `english`.
- **Quality**: The output is standardized, normalized, and ready for use in training or fine-tuning Large Language Models (LLMs) or NMT systems.

## 🚀 Future Roadmap
- Integration of **IndicTrans2** for potentially higher-quality legal translations.
- Multi-threaded file processing for the `loader` and `cleaner`.
- PDF support in addition to DOCX.
