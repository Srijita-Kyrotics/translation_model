# Training Methodology: English to Bengali Legal Translation

This document outlines the fine-tuning methodology, dataset composition, and hyperparameters used for our IndicTrans2 translation model.

## 1. Methodology Overview
We are following the **IndicTrans2** (arXiv:2305.16307) Stage 1 training methodology. Specifically, we are performing **Full-Parameter Fine-Tuning** on a high-fidelity parallel corpus of legal judgments and general translations.

- **Objective**: Optimize the model's ability to translate complex legal terminology and sentence structures from English to Bengali.
- **Base Model**: `prajdabre/rotary-indictrans2-en-indic-1B` (1.1 Billion Parameters).
- **Technique**: Full-parameter unfreezing with Gradient Checkpointing to maximize performance while managing VRAM.

## 2. Dataset Composition
The dataset is a combination of your existing "gold" corpus and newly added judicial translation pairs.

| Metric | Detail |
| :--- | :--- |
| **Total Processed Rows** | 510,819 |
| **Unique Parallel Pairs** | 487,366 |
| **Training Split (95%)** | 462,997 |
| **Validation Split (5%)** | 24,369 |
| **Source Language** | English (`eng_Latn`) |
| **Target Language** | Bengali (`ben_Beng`) |

### Data Cleaning Strategy:
1. **Numbering Removal**: Strict regex patterns removed all leading prefixes like "1. ", "1) ", "[1] ", "2: ", etc.
2. **Deduplication**: Exact identical pairs (boilerplate legal text) were dropped to prevent overfitting.
3. **Validation Split**: A 5% hold-out set was created to monitor validation loss and prevent catastrophic forgetting.

## 3. Training Parameters
Aligned with the IndicTrans2 paper's high-quality benchmark settings:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | `5e-4` | Standard for IndicTrans2 fine-tuning. |
| **LR Scheduler** | `inverse_sqrt` | Decays the learning rate after initial warmup. |
| **Warmup Steps** | `4,000` | Ensures stability in the initial optimization steps. |
| **Weight Decay** | `0.0001` | Prevents over-regularization on domain-specific data. |
| **Label Smoothing** | `0.1` | Improves generalization on noisy/complex parallel data. |
| **Batch Size** | `1` | Per-device batch size (optimized for VRAM). |
| **Gradient Accumulation** | `128` | Simulates an effective batch size of 128 per step. |
| **Optimizer** | `AdamW` | Standard optimizer for Seq2Seq transformers. |
| **FP16 Mixed Precision** | Enabled | Speeds up training on modern GPUs. |

## 4. How to Check Progress
Training logs are generated every **50 steps**. Each log includes the training loss, learning rate, and current epoch.
The validation set is evaluated every **500 steps** to track model quality.
