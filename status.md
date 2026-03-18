# Project Status Report: Legal Translation Model (EN ↔ BN)

**Date**: March 05, 2026  
**Current Phase**: Phase 1 - Back-Translation (Data Augmentation)

## Back-Translation is a way to "artificially" grow your dataset using the legal documents you already have.

Since you have many Bengali court judgments but only a limited number of high-quality English translations, we use Back-Translation to bridge that gap. Here is its primary function:

 1. Data Augmentation (Quantity)
AI models (like humans) learn better with more examples. If we only have 155,000 "Gold" (human) pairs, the model might not see enough legal scenarios. Back-translation allows us to take a raw Bengali judgment and "guess" its English version. This generates the 272,000 "Silver" pairs we are currently creating.

 2. Domain Adaptation (Specific Knowledge)
Generic models (like Google Translate) know "common" English, but they don't know "Judicial" English well. By back-translating actual court judgments, we force the model to practice translating technical legal phrases (like "mutatis mutandis" or "habeas corpus") over and over again until it becomes a specialist.

 3. Regularization (Focus)
By adding the <BT> tag to these synthetic sentences, we tell the model: "This is a practice exercise, not the absolute truth." This prevents the model from "hallucinating" or making up facts, while still allowing it to learn the complex sentence structures used in judgments.

 4. Reaching the "Magic Ratio"
The IndicTrans2 researchers found that a model performs best when it has a 1 : 1.75 ratio of Human data to Synthetic data. Back-translation is the engine that generates that exact ratio for us.

Summary: Back-translation is like a flight simulator for the model. It allows the model to "practice" on hundreds of thousands of legal sentences that don't have human translations yet, so that when it sees a real document from your teammate, it has already "seen" something similar before.

---

## 1. Current Progress: 🟢 Resumed - 74% Overall Milestone (Phase 2)
The system has been safely resumed from yesterday's shutdown. Progress is being tracked in real-time.

- **Overall Project**: **74.0%** (201,257 / 272,000 sentences).
- **BN ➔ EN Direction**: 🟢 **100% Complete** (108,449 sentences).
- **EN ➔ BN Direction**: 🔵 **57% Complete** (92,808 / 163,552 sentences).
- **Current Status**: **Running - Active Generation.**
- **Hardware**: NVIDIA RTX 4090 (24GB) active with ~4GB VRAM.
- **Hardware**: NVIDIA RTX 4090 (24GB) performing at ~160 sentences/min (~9,600/hr).
- **Estimated Time to Data Readiness**: ~19-20 hours.

# After the Back-Translation (the "Data Generation" phase) is finished, the system will automatically proceed through these three steps:

1. Dataset Packaging (Merging)
The script 'src/prepare_dataset.py'  will combine the 155,000 "Human-Gold" pairs and the 272,000 "Synthetic-Silver" pairs into one massive folder (HuggingFace format).

Result: A single, balanced training file containing ~427,000 legal sentence pairs.

 2. Stage 1 Fine-Tuning (The "Broad Learning" Phase)
This is where the actual "Training" begins. The GPU will stop generating data and start updating the model's brain.

Goal: Teach the model the technical language of both English and Bengali legal systems.
Bidirectional: It will train the English ↔ Bengali and Bengali ↔ English models sequentially.
Duration: This is the longest part, likely taking 12-15 hours on your RTX 4090.

3. Stage 2 Fine-Tuning (The "Judicial Refinement" Phase)
Once the model knows the "jargon," we give it one final, high-quality "polish."

Data: We filter out all the synthetic data and train only on the Supreme Court judgments.
## Result: The model loses any "artificial" stiffness from the back-translation and learns the exact formal tone used by top judges.

---

## 2. Multi-Stage Training Strategy
To ensure maximum legal fidelity, we are executing a **Two-Stage Fine-Tuning** process for **both** translation directions (English ↔ Bengali):

### 🏛️ Stage 1: Knowledge Expansion (Quantity)
*   **Focus**: Broad legal domain expertise and terminology.
*   **Data**: 427,000 sentence pairs (Human-Gold + Synthetic-Silver).
*   **Purpose**: Teach the model the complex grammar and technical jargon of the Indian legal system using a massive data pool.

### 💎 Stage 2: Judicial Refinement (Quality)
*   **Focus**: Precision, formal style, and "judicial tone."
*   **Data**: 155,470 high-quality **Supreme Court Gold** pairs (Human-only).
*   **Purpose**: A secondary "polish" pass to eliminate any synthetic noise from Stage 1 and ensure the final output matches the formal style of the Supreme Court.

---

## 3. Methodology: IndicTrans2 Technical Alignment
We have synchronized our pipeline with the exact technical specifications of the **IndicTrans2 (2305.16307)** research paper:

### A. Data Augmentation (Back-Translation)
*   We use the base model to "back-translate" monolingual legal text, reaching the paper's recommended **1:1.75 ratio** of original to synthetic data.
*   **Tagging**: All synthetic samples are prepended with a **`<BT>` token**. This prevents the model from being overconfident on synthetic data and maintains high "Gold" standard accuracy.

### B. Full-Parameter Fine-Tuning (The "Big Brain")
*   Unlike typical LoRA/PEFT (which trains only ~1% of the model), we are performing **Full-Parameter Optimization**. We are unfreezing and training all **1.1 Billion parameters** to deeply internalize legal Bengali.

### C. Hyperparameter Synchronization
We have precisely matched the paper's optimized training arguments:
*   **Label Smoothing (0.1)**: Prevents the model from becoming over-confident and improves generalization.
*   **Weight Decay (0.0001)**: Ensures stability during the training of 1.1 billion parameters.
*   **Inverse Square Root Scheduler**: A specialized learning rate decay standard for Transformer-based architectures.
*   **Warmup (4,000 steps)**: Allows the model to graduate into the high learning rate without "exploding" gradients.

---

## 4. High-Level Roadmap

| Phase | Status | Activity |
| :--- | :--- | :--- |
| **Phase 1: Augmentation** | 🔄 In-Progress | Generating 272k synthetic legal sentence pairs via back-translation. |
| **Phase 2: Preparation** | 📝 Queued | Automatic packaging into a unified HuggingFace dataset format. |
| **Phase 3: Stage 1 Training** | 📝 Queued | Full-parameter training (AdamW, label smoothing=0.1, weight decay=0.0001). |
| **Phase 4: Stage 2 Training** | 📝 Queued | Final refinement pass on the cleanest Supreme Court data. |
| **Phase 5: Evaluation** | 📝 Queued | Automated BLEU/chrF benchmarking using the 200-sample legal gold set. |

---

## 5. Summary
We are currently performing **Full-Parameter Fine-tuning** on 1.1 billion parameters. This allows the model to deeply internalize legal terminology. 

The pipeline is **fully automated**—once the back-translation finishes tonight, the Stage 1 training will launch immediately without manual intervention.

We are currently in the pre-training preparation phase:

Phase 1: Back-Translation (In Progress - 13%): The GPU is currently "writing" the synthetic textbooks (the Silver data).
Phase 2: The Training (Queued): The actual "learning" part where the model's weights change (the Stage 1 Fine-Tuning) will start the second the data generation finishes.

