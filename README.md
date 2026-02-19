# Multilingual Translation Dataset Pipeline

A modular and scalable pipeline designed to build high-quality parallel corpora for Indian language translation tasks (specifically Bengali-English). This system automates the extraction, cleaning, translation, and pairing of sentences from documents.

##  Overview

The pipeline transforms raw `.docx` documents into a structured, CSV-formatted parallel corpus. It avoids complex embedding-based matching in favor of a deterministic, modular approach.

## Architecture & Model


We utilize the **Rotary IndicTrans2** architecture, specifically the `prajdabre/rotary-indictrans2-indic-en-1B` model.
- **Architecture**: Transformer-based encoder-decoder model with **Rotary Positional Embeddings (RoPE)**. 
- **Why RoPE?**: It allows the model to handle variable sequence lengths better and captures relative positions more effectively than standard sinusoidal embeddings, leading to improved translation quality for complex languages like Bengali.
- **Toolkit**: Preprocessing and tokenization are handled by the `IndicTransToolkit`, ensuring correct handling of Indic scripts.

## Pairing Logic
The pipeline uses a **Deterministic Line-Index Alignment** strategy:
1.  **Source Preservation**: The `cleaner.py` script processes files line-by-line, maintaining the exact order of sentences.
2.  **Sequential Translation**: The `translator.py` script translates these lines sequentially, preserving the order in the output.
3.  **Index Matching**: The `pairer.py` script assumes that line $N$ in the source file corresponds exactly to line $N$ in the translated file. 
    - *Safety Check*: If line counts mismatch (rare), usage is truncated to the minimum length to prevent misalignment.

## Output
The final output is generated at `data/final/parallel_corpus.csv` containing:
- `source_file`: Name of the original document.
- `line_index`: Line number in the source file.
- `bengali`: Original cleaned Bengali text.
- `english`: Generated English translation.

### Key Features
- **Modular Design**: Separate layers for data loading, cleaning, translation, and pairing.
- **Indic-Specific Processing**: Uses `indic-nlp-library` and `IndicTransToolkit` for Bengali text normalization and preprocessing.
- **State-of-the-Art Translation**: Integrates `prajdabre/rotary-indictrans2-indic-en-1B` (a Rotary Positional Embedding version of IndicTrans2) for superior context handling and accuracy.
- **Deterministic Alignment**: 1-to-1 line-index based pairing ensuring strict alignment between source and target corpora.


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
    - Uses the **Rotary IndicTrans2** model (`prajdabre/rotary-indictrans2-indic-en-1B`) for Bengali to English translation.
    - Optimized with **Flash Attention 2** for efficiency.
    - Supports extended context window (1024 tokens) to prevent truncation of long paragraphs.
    - Implements `IndicTransToolkit` preprocessing (normalization) and postprocessing.
    - Uses batch processing (default batch size: 4) and GPU acceleration.
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
- `IndicTransToolkit`: For Indic script preprocessing and tokenization.
- `transformers`, `torch`: For the neural machine translation model.
- `einops`: Required for Rotary Positional Embeddings.
- `scipy`: Secondary dependency for advanced operations.
- `pandas`: For dataset management and CSV generation.
- `tqdm`: For progress tracking.
