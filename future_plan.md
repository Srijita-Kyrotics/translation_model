# Future Plan: IndicTrans2 Integration

This document outlines the technical steps, environment requirements, and code modifications required to migrate the current translation pipeline from OPUS-MT to the **IndicTrans2** toolkit (specifically for Bengali to English translation).

## 🌍 Environment Requirements

IndicTrans2 is highly optimized for Linux environments. To run it on Windows:
1.  **Windows Subsystem for Linux (WSL)**: Install Ubuntu via WSL2.
2.  **GPU Drivers**: Ensure NVIDIA drivers are mapped to WSL if using CUDA.
3.  **Dependencies**:
    - Install `IndicTransToolkit`: `pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git`
    - Install `CTranslate2` or `Fairseq` (depending on the chosen model variant).

## 🛠️ Code Modifications (`src/translator.py`)

To include the IndicTrans2 toolkit, the following architectural changes are needed:

### 1. Model Initialization
Switch from `Helsinki-NLP` to `ai4bharat/indictrans2-indic-en-1b` (or the 200M version for lower VRAM).

```python
# Updated initialization logic
from IndicTransToolkit import IndicProcessor

class IndicTranslator:
    def __init__(self):
        self.ip = IndicProcessor(inference_label="translation")
        self.model_name = "ai4bharat/indictrans2-indic-en-1b"
        # ... load model using transformers ...
```

### 2. Pre-processing and Post-processing
IndicTrans2 requires specific scripts for normalization and placeholder handling.
- **Pre-processing**: Use `ip.preprocess_batch(sentences, src_lang="ben_Beng", tgt_lang="eng_Latn")`.
- **Post-processing**: Use `ip.postprocess_batch(translations, lang="eng_Latn")`.

### 3. Batching Logic
IndicTrans2 models are larger. We may need to:
- Reduce `batch_size` from 8 to 4 (or 2) depending on GPU memory.
- Implement more robust memory clearing (`torch.cuda.empty_cache()`).

## 📊 Expected Improvements
- **Contextual Accuracy**: Better handling of legal and formal Bengali vocabulary.
- **Gender and Cultural Nuance**: Improved pronoun handling and cultural context sensitivity.
- **Higher BLEU Scores**: IndicTrans2 consistently outperforms older MarianMT models like OPUS-MT on the Flores-200 benchmark.

## 🏁 Progress Checklist
- [x] Rationale for migration documented.
- [ ] Setup WSL/Linux environment.
- [ ] Benchmarking current OPUS-MT output for baseline comparison.
- [ ] Implementation of `IndicTranslator` class.
- [ ] Full pipeline validation on a subset of `EBMT 1` data.
