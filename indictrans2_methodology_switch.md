# Roadmap: Switching to Full IndicTrans2 Methodology

To move from our current domain-specific LoRA approach to the "gold standard" methodology described in the IndicTrans2 paper (2305.16307), we need to implement the following three pillars.

---

## 1. Massive Data Augmentation (Back-Translation)
The core of IndicTrans2's performance is **Back-Translation (BT)**. They use ~1.75x more synthetic data than original parallel data.

*   **Step A: Monolingual Collection**: Gather millions of lines of Bengali legal text (judgments, statutes) and English legal text.
*   **Step B: Synthetic Pair Generation**:
    *   Use the existing `indic-en` model to translate Bengali legal text to English.
    *   Use the `en-indic` model to translate English legal text to Bengali.
*   **Step C: Tagging**: Prepend a special token (e.g., `<BT>`) to synthetic samples so the model learns to prioritize "Gold" human data while still learning the legal structure from the "Silver" BT data.

## 2. Two-Stage Training Pipeline
IndicTrans2 doesn't just fine-tune; it uses a multi-stage approach:

*   **Stage 1: Domain-Specific Denoising**:
    *   Train the model to "reconstruct" corrupted legal sentences (e.g., span masking) using your monolingual legal corpus. This adapts the encoder/decoder to the legal vocabulary before translation even starts.
*   **Stage 2: Multilingual Fine-tuning**:
    *   Train on the combined dataset (Gold Parallel + Synthetic BT).

## 3. Training Scale & Parameters
IndicTrans2 uses full-parameter optimization rather than LoRA for their final models.

*   **Move to Full Fine-tuning**: Instead of LoRA (Rank 16), unfreeze the entire 1.1 billion parameters. This requires significantly more VRAM (multi-GPU setup with DeepSpeed/FSDP).
*   **Hyperparameter Alignment**:
    *   **Large Batch Sizes**: They use effective batch sizes of 1000+ sequences via gradient accumulation.
    *   **Label Smoothing**: Set `label_smoothing_factor=0.1` (our current is 0.0) to prevent the model from becoming too overconfident.
    *   **Max Seq Length**: Increase from 512 to 1024 or higher to handle long legal clauses.

---

## Technical Checklist for Switch

| Task | Current State | IndicTrans2 Methodology |
| :--- | :--- | :--- |
| **Data Types** | Parallel Only | **Parallel + Monolingual (BT)** |
| **Training Steps** | 1,000 | **~50,000+** |
| **Parameter Tuning** | LoRA (PEFT) | **Full Model (1.1B parameters)** |
| **Loss Function** | Standard CrossEntropy | **CrossEntropy + Label Smoothing** |
| **Learning Rate** | 2e-4 | **Inverse Square Root Schedule** |

---

## Recommended Next Step
If you want to start this transition, we should begin by **scraping more monolingual legal Bengali text** (even if we don't have English translations for them yet) to prepare for Stage 1 Denoising and Back-translation.
