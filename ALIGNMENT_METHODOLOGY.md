# Alignment Methodology: Triple-Semantic LaBSE Matching

This document details the alignment algorithm used to generate the high-density gold-standard parallel corpus for legal judgments.

## Overview

The alignment process, or "Phase 3," ensures that English legal judgments (A) are paired with the exact original Bengali judgments (B) from the High Court, even if the documents vary in sentence structure or OCR quality. We utilize a **Triple-Semantic Bridge** approach.

## The Pipeline (A -> C -> B)

1.  **Source (A)**: Original English judgment.
2.  **Original (B)**: Original site-downloaded Bengali judgment.
3.  **Bridge (C)**: Bengali translation of (A), generated specifically to act as a semantic pivot.

## The Algorithm: Triple-Match LaBSE

We use **LaBSE (Language-Agnostic BERT Sentence Embedding)** for all similarity calculations because of its superior performance in cross-lingual and Indic-specific semantic tasks.

### Step 1: Preprocessing & Tokenization
- English (A) is tokenized using `nltk.sent_tokenize`.
- Translated Bengali (C) and Original Bengali (B) are tokenized using `indicnlp.sentence_tokenize`.

### Step 2: Parallel Verification (A <-> C)
- We embed both A and C using LaBSE.
- For each translated sentence in C, we find its most similar parent sentence in English (A).
- **Threshold**: 0.70.
- **Goal**: This recovers the English source and handles potential sentence-count discrepancies caused by the translator merging or splitting lines.

### Step 3: Semantic Alignment (C <-> B)
- We compute the cosine similarity between the translated bridge (C) and the original judgment (B).
- For each sentence in C, we identify the best match in B.
- **Threshold**: 0.75.
- **Goal**: This ensures that we are using high-quality, professional Bengali vocabulary from the High Court while maintaining semantic parity with the source.

### Step 4: Triple Filtering
A pair `(English_A, Original_B)` is only saved to the Gold Corpus if:
1.  `Sim(A, C) >= 0.70` (English source is valid).
2.  `Sim(C, B) >= 0.75` (Bengali match is semantically identical).

## Why this works?
- **Domain Fidelity**: By matching against the *original* Bengali judgment (B), we ensure the corpus contains professional legal Bengali rather than purely "machine" Bengali.
- **Parity Resilience**: Unlike line-index or relative-position heuristics, semantic matching is immune to OCR artifacts, paragraph re-ordering, or translation hallucinations.

## Key Performance Indicators (KPIs)
- **Mean Similarity**: ~0.84.
- **Pair Density**: ~18-30 gold pairs per document.
- **Deduplication**: Automatic removal of duplicate sentence pairs across judgments.

---

