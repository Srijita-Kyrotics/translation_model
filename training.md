# Legal Translation Training Methodology: English ↔ Bengali

This document outlines the end-to-end methodology for building a high-precision, bidirectional legal translation system based on the **IndicTrans2** architecture.

## 1. Project Objective
To develop a translation model specialized for the Indian legal domain, specifically for translating court judgments and documents between **English (eng_Latn)** and **Bengali (ben_Beng)** while preserving the formal tone and complex structure (tables/formatting).

---

## 2. Data Acquisition & Processing

### 📥 Data Collection
- **Source**: Calcutta High Court Judgment Portal.
- **Scope**: Supreme Court (SCR), High Court (HC), and Historically Important judgments.
- **Initial Dataset**: Identifiied and downloaded **10,742 PDFs**, forming **5,371 perfectly matched English-Bengali pairs**.
- **Scanned Status**: Full scan reveals **7 files (0.06%)** are truly scanned images; the remaining 10,735 are digital/searchable PDFs.

### 🔍 OCR & Extraction
- **Tool**: **OLM OCR** (specifically optimized for judicial layout).
- **Process**: Scanned images in PDFs are converted to raw text.
- **Sentencing**: Raw text is cleaned and split into individual sentences while removing boilerplate headers (e.g., "SUPREME COURT OF INDIA").

### ⚖️ Semantic Alignment (LaBSE)
- **Tool**: `sentence-transformers/LaBSE` (Language-Agnostic BERT Sentence Embedding).
- **Threshold**: **0.65 cosine similarity**.
- **Result**: Sentences are semantically paired even if word counts or structures differ, ensuring a high-quality "Gold" parallel corpus of **~155,000 pairs**.

---

## 3. Training Methodology (IndicTrans2 Alignment)

We follow the methodology defined in the **IndicTrans2 paper (2305.16307)** to maximize domain-specific performance.

### 🔄 Phase 1: Back-Translation (Data Augmentation)
- **Concept**: To supplement the 155k Gold pairs, we generate **272,000 synthetic pairs** from monolingual legal text.
- **Ratio**: Targeted **1:1.75** ratio of Human-Gold to Synthetic-Silver data.
- **Tagging**: All synthetic sentences are prepended with a **`<BT>` token** to notify the model of their synthetic nature, preventing "hallucination loop."

### 🏗️ Phase 2: Bidirectional Training Architecture
- **Dual Models**: We fine-tune **two separate 1.1 Billion parameter models** (one for each direction) instead of a single multilingual model to ensure maximum capacity for each language pair.
- **Full-Parameter Fine-tuning**: Unlike standard LoRA (which trains ~1% of weights), we perform **full-parameter optimization** to capture the deep linguistic nuances of legal Bengali.

### 💎 Phase 3: Two-Stage Training
- **Stage 1 (Generalization)**: Training on the massive combined pool of 427,000 sentences (Gold + BT).
- **Stage 2 (Refinement)**: A secondary fine-tuning stage on high-quality **Supreme Court Gold judgments** only, acting as a "polish" pass for formal legal style.

---

## 4. Technical Specifications

### 💻 Hardware & Environment
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM).
- **Optimization**: Mixed-precision (`fp16`) and **Gradient Checkpointing** to fit 1.1B parameters into 24GB VRAM.

### ⚙️ Hyperparameters (Synced with Paper)
| Parameter | Value |
| :--- | :--- |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 0.0001 |
| **Label Smoothing** | 0.1 |
| **LR Scheduler** | `inverse_sqrt` |
| **Warmup Steps** | 4000 |
| **Effective Batch Size** | 128 (via Gradient Accumulation) |

---

## 5. Inference & Results

### 📄 Structured Translation (YAML)
The system is designed to process **YAML outputs from OCR**.
- **Tool**: `src/structured_translator.py`
- **Features**: Translates individual sentences as well as **complex table headers and cells** while maintaining the original YAML structure.
- **Batching**: Support for translating hundreds of documents in a single command using `src/batch_translate_yamls.py`.

### 📈 Expected Results
- **Higher BLEU/chrF scores** on legal texts compared to general-purpose models (NMT/Google/ChatGPT).
- **Improved Alignment**: Better handling of complex legal cross-references and citations.
- **Bidirectional Fidelity**: Equal accuracy in both English-to-Bengali and Bengali-to-English directions.
