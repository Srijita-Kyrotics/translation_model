# Multilingual Translation Dataset Pipeline

A modular and scalable pipeline designed to build high-quality parallel corpora for Indian language translation tasks (specifically Bengali-English). This system automates the extraction, cleaning, translation, and pairing of sentences from documents.

##  Overview

The pipeline transforms raw `.docx` documents into a structured, CSV-formatted parallel corpus. It avoids complex embedding-based matching in favor of a deterministic, modular approach.

##  Overview

The pipeline transforms raw `.docx` documents into a structured, CSV-formatted parallel corpus. It avoids complex embedding-based matching in favor of a deterministic, modular approach.

### Key Features
- **Modular Design**: Separate layers for data loading, cleaning, translation, and pairing.
- **Indic-Specific Processing**: Uses `indic-nlp-library` for Bengali text normalization.
- **Automated Translation**: Integrates `Helsinki-NLP/opus-mt-bn-en` for high-quality sentence-level translation.
- **Deterministic Alignment**: Index-based pairing ensuring alignment between source and target corpora.

##  Project Structure

```text
translation_model/
├── data/
│   ├── raw/             # Extracted text from .docx files
│   ├── processed/       # Cleaned and normalized source text (Bengali)
│   ├── translated/      # Machine-translated target text (English)
│   └── final/           # Final parallel_corpus.csv
├── src/
│   ├── loader.py        # Extracts text from EBMT 1 (.docx)
│   ├── cleaner.py       # Cleans and normalizes Bengali text
│   ├── translator.py    # Translates Bengali to English
│   └── pairer.py        # Aligns and pairs text into final CSV
├── EBMT 1/              # Source directory for raw .docx files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

##  Pipeline Stages

1.  **Loading (`loader.py`)**: Reads `.docx` files from the input folder and converts them to plain `.txt` in `data/raw`.
2.  **Cleaning (`cleaner.py`)**: 
    - Removes disclaimers, separator lines, and excessive whitespace.
    - Normalizes Bengali characters using the Indic NLP library.
    - Saves output to `data/processed`.
3.  **Translation (`translator.py`)**: 
    - Uses a Transformer-based model to translate cleaned Bengali text to English.
    - Implements batch processing and GPU acceleration (if available).
    - Saves translated files to `data/translated`.
4.  **Pairing (`pairer.py`)**: 
    - Align sentences from `data/processed` and `data/translated` by line index.
    - Generates a `parallel_corpus.csv` with source file metadata.

##  Installation

```bash
pip install -r requirements.txt
```

##  Usage

Run the scripts in order:

```bash
python src/loader.py
python src/cleaner.py
python src/translator.py
python src/pairer.py
```

##  Dependencies
- `python-docx`: For Word document parsing.
- `indic-nlp-library`: For Indian language normalization.
- `transformers`, `torch`: For neural machine translation.
- `pandas`: For dataset management and CSV generation.
- `tqdm`: For progress tracking.
