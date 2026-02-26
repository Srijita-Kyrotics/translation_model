# Translation API Integration Guide

This guide is for the Integration Lead to connect the raw OCR JSON/YAML output to the Translation Module. 

The translation pipeline uses the **Rotary IndicTrans2 1B** model, natively optimized for Bengali-to-English legal text, bundled with a T5-based grammar correction module.

## Prerequisites
Ensure the environment is set up according to `README.md` and `status.md`. You must run the API server within the exact Conda environment (`bhasantar_env`) to ensure access to PyTorch, transformers, and IndicTransToolkit.

---

## Option 1: Running the REST API (Recommended)
We have wrapped the translation logic in a high-performance **FastAPI** server, allowing you to treat translation as an independent microservice. It loads the massive model into VRAM once upon startup.

### 1. Start the Server
Navigate to the project root and start the server:
```bash
conda activate bhasantar_env
python src/api.py --host 0.0.0.0 --port 8000
```
> [!NOTE]  
> The server will print `"Model loaded successfully. Ready to translate."` once it has fully loaded the model weights into memory. Do not send requests before this message appears.

### 2. Send an OCR Payload
The API accepts a single POST request at the `/translate` endpoint. 

The payload must be a JSON dictionary matching the OCR structured format. The API will identify all `text` arrays inside `sentences` and all specific cells inside `tables`, translate them, and return the identical structure.

**Curl Example**:
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
           "document_id": "doc_123",
           "is_table": false,
           "sentences": [
             {"id": 1, "text": "The quick brown fox jumps over the lazy dog."}
           ]
         }'
```

**Python `requests` Example**:
```python
import requests

ocr_payload = {
    "document_id": "1960~scr_1960_2_100_110_e_page_4",
    "primary_language": "ben_Beng",
    "is_rotation_valid": True,
    "sentences": [
        {"id": 1, "text": "বাংলা পাঠ্য এখানে"}
    ]
}

response = requests.post("http://localhost:8000/translate", json=ocr_payload)
translated_data = response.json()
print(translated_data)
```

---

## Option 2: Direct Python Import
If you prefer not to use an HTTP server and want to run everything in a single synchronous pipeline script, you can import the class directly.

```python
from src.structured_translator import StructuredTranslator
import yaml

# 1. Initialize and load model (Do this ONCE outside of any loops)
translator = StructuredTranslator(
    model_name="prajdabre/rotary-indictrans2-indic-en-1B",
    src_lang="ben_Beng",
    tgt_lang="eng_Latn",
    use_correction=True
)
translator.load_model()

# 2. Input and Output paths must be files (YAML/JSON format)
translator.translate_yaml("path/to/extracted_ocr.yaml", "path/to/translated_output.yaml")

# 3. Read it back if you need it as a dict
with open("path/to/translated_output.yaml", "r") as f:
    final_dict = yaml.safe_load(f)
```
> [!WARNING]  
> `StructuredTranslator.translate_yaml()` intrinsically reads and writes from disk. If your pipeline operates entirely in-memory using dictionaries, you must use **Option 1 (FastAPI)**, as `api.py` handles the underlying temporary file conversions for you automatically.
