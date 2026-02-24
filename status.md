# Project Status Report: Bengali-English Translation Pipeline

This document summarizes all work completed and the current status of the fine-tuning phase.

## ✅ Completed Milestones

### 1. Infrastructure & Research
- **Requirements**: Integrated `PyMuPDF`, `IndicTransToolkit`, `transformers`, and `peft`.
- **Hardware Profile**: Verified RTX 4050 (6GB VRAM) and confirmed Windows compatibility for 4-bit quantization with `bitsandbytes`.

### 2. Large-Scale Data Acquisition (`downloader.py`)
- **Scraping**: Fetched **9,084** PDF links from the Calcutta High Court.
- **Concurrency**: Implemented multi-threading (5 workers), reducing download time from **7 hours to 1 hour**.
- **Dataset**: Successfully acquired **~4,500 pairs** of bilingual judicial judgments.

### 3. Data Processing Suite
- **Extraction (`loader.py`)**: Converted PDFs to text and implemented logic to strip website-specific prefixes (`~scr_`) to maintain consistency with the `EBMT 1` dataset.
- **Cleaning (`cleaner.py`)**: Automated structural cleaning and Bengali normalization using `indic-nlp-library`.
- **Alignment (`pairer.py`)**: Performed deterministic line-index pairing, generating a massive 780K+ pair CSV.

### 4. Training Data Preparation (`exporter.py`, `prepare_dataset.py`)
- **Quality Filtering**: Filtered the corpus down to **605,070 high-quality pairs** (removing short fragments/noise).
- **Formats**: Generated standard training files (`corpus.bn`, `corpus.en`) and a side-by-side verification file.
- **HF Integration**: Converted data into a Hugging Face `DatasetDict` with a **582K Train / 12K Val** split.
- **Structured OCR Reader**: Developed `src/structured_translator.py` to handle YAML-based OCR schema (sentence lists and cell-level table translation).

## 🔄 Current Phase: Model Fine-tuning

### Status: In Progress
- **Script**: `src/finetune.py` is currently executing.
- **Optimization Strategy**:
    - **Model**: `prajdabre/rotary-indictrans2-indic-en-1B`.
    - **Memory**: Using **QLoRA (4-bit)**, **Gradient Checkpointing**, and **Paged AdamW 8-bit**.
- **YAML Integration**: Ready for hand-off to the OCR pipeline teammate.
- **Next Step**: Configure the environment (WSL2/Linux) to support `IndicTransToolkit` for final integration tests.

---
**Last Updated**: 2026-02-23 23:35
**Overall Progress**: ~90% (Infrastructure & Data Complete; Training Phase Started)
