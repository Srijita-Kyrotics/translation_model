# Project Schedule & Progress Report

## Project Overview
This project aims to enhance translation quality for legal judgments between Bengali and English by building a large-scale parallel corpus and fine-tuning an IndicTrans2-based model.

---

##  Completed Milestones

### Phase 1: Data Acquisition & Preprocessing
*   **Judgment Scraping**: Successfully scraped ~10,000 PDF links from Calcutta High Court (Supreme Court, High Court, and Historically Important categories).
*   **Text Extraction**: Extracted and cleaned text from over 5,000 judgment PDFs using PyMuPDF.
*   **Semantic Alignment**: Utilized LaBSE to create a high-quality parallel corpus of **155,468 sentence pairs**.
*   **Dataset Preparation**: Created a refined HuggingFace `DatasetDict` from the aligned corpus with proper shuffling and deduplication.

### Phase 2: Model Fine-tuning
*   **Adapter Training**: Fine-tuned the `rotary-indictrans2-indic-en-1B` model using **LoRA** (Rank 16, Alpha 32) for 1000 steps on the full 155k dataset.
*   **Evaluation Baseline**: Established a standardized evaluation framework with batch size 8 on a consistent 200-sample validation subset.
*   **Performance Gain**: Achieved a **+8.47** increase in the **chrF** score (from 17.10 to 25.57), indicating significant character-level improvement in legal terminology and morphology.

### Phase 3: System Enhancements
*   **Bidirectional Support**: Updated the `Translator` suite to support both **BN-to-EN** (Fine-tuned) and **EN-to-BN** (Base) directions.
*   **Table Translation**: Implemented structured translation support via `StructuredTranslator` to preserve table layouts (headers/rows) during translation.
*   **Bug Resolution**: Resolved a critical model cache error in the `en-indic` model by implementing a dynamic KV-cache toggle.
*   **Demo Assets**: Generated mock/real YAML translation pairs (English source -> Bengali translation) to demonstrate production readiness.

---

##  Current Status
*   **System State**: Operational with Bidirectional Support.
*   **Data Health**:  155k high-quality aligned pairs available for future training.
*   **Core Metrics**: chrF (25.57) | SacreBLEU (1.62).

---

## Future Schedule / Next Steps

| Milestone | Target | Description |
| :--- | :--- | :--- |
| **Extended Training** | 1-2 Epochs | Full training on the 155k corpus (estimated 15-20 hours) to maximize SacreBLEU gains. |
| **Model Scaling** | 5B/13B | Upgrade the base model for deeper semantic understanding of complex legal text. |
| **Legal Glossary** | Continuous | Integration of a domain-specific lexicon for post-processing correction. |
| **UI Integration** | Beta | Prototype a side-by-side legal document comparison tool. |
