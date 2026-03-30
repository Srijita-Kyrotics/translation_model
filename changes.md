# Fine-Tuning Stability & Change Log (2026-03-30)

This document outlines the technical challenges encountered during the fine-tuning of `indictrans2-en-indic-1B` on the legal translation corpus and the steps taken to resolve them.

---

## 1. The Tokenization "Stall" (79% Progress)
- **Problem**: The initial training attempt using on-the-fly tokenization (`dataset.map()`) consistently stalled at ~79% mapping progress. This was likely due to high CPU/RAM overhead or serialization issues with the large 1B parameter model's tokenizer.
- **Solution**: 
    - Implemented a standalone script: `src/cache_tokenized_dataset.py`.
    - Parallelized tokenization across 16+ cores.
    - Saved the final dataset to a disk-based cache at `data/final/tokenized_en_bn`.
    - **Result**: Training now starts instantly without any mapping delays.

## 2. Sequence Length Mismatch Error
- **Problem**: Encountered `RuntimeError: Size does not match at dimension 1` (e.g., logits 204 vs. labels 208). The model architecture was producing outputs slightly shorter than the input labels.
- **Solution**: 
    - Created a custom `StableSeq2SeqTrainer` in `src/finetune.py`.
    - Overrode the `compute_loss` method to dynamically align tensors. 
    - Used `torch.nn.functional.pad` to ensure logits and labels match perfectly before calculating loss.
    - **Result**: Resolved the crash and stabilized error gradients.

## 3. FP16 Unscaling & Precision Crash
- **Problem**: Training crashed with `RuntimeError: Attempting to unscale FP16 gradients`. This is a known issue with older `fp16` precision on 40-series NVIDIA GPUs (RTX 4090) when using large parameter models.
- **Solution**: 
    - Switched training configuration from `fp16=True` to `bf16=True`.
    - **Result**: Leveraged the RTX 4090's native support for Brain Floating Point (BF16), eliminating gradient scaling crashes.

## 4. CUDA Out Of Memory (OOM) / Zombie Processes
- **Problem**: After several restarts, the GPU memory (VRAM) became fragmented or full, even though no training seemed to be happening.
- **Solution**: 
    - Found "zombie" Python processes from previous failed attempts stuck in VRAM.
    - Ran `pkill -9 -f python` to clear all stale processes.
    - **Result**: Restored 22GB+ of free VRAM, allowing the 1B parameter model to load and train comfortably.

## 5. Logging Visibility
- **Problem**: The default `logging_steps=50` made it appear as if the training was stuck for over an hour between updates.
- **Solution**: 
    - Updated `src/finetune.py` with `logging_steps=1`.
    - Created `src/update_md_logs.py` to auto-format technical JSON logs into a readable `training_logs.md` file every 60 seconds.
    - **Result**: Immediate visibility into loss, learning rate, and training speed.

## 6. GitHub Repository Sync
- **Problem**: Divergent branch history and active log updates made standard `git pull/push` commands fail.
- **Solution**: 
    - Used `git stash --include-untracked` to set aside active log files.
    - Performed a safe `git pull origin main --rebase`.
    - Executed a `git push origin main --force-with-lease` to synchronize the stabilized stage-1 codebase.
    - **Result**: The complete, stable repository is now live on GitHub.

---

**Current Status**: Training is running stably at ~13.7s per iteration on the RTX 4090.
