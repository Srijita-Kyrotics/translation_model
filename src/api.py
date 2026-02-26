from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tempfile
import yaml
import os
import uvicorn
from typing import Dict, Any

from src.structured_translator import StructuredTranslator

app = FastAPI(
    title="Indic Translation Service",
    description="A service that translates structured OCR data (JSON/YAML format) to English using the Rotary-IndicTrans2 model.",
    version="1.0.0"
)

# Global model variable to hold our loaded translation model
translator_instance = None

@app.on_event("startup")
async def startup_event():
    """Load the translation model upon server startup to ensure fast inference."""
    global translator_instance
    print("Initializing translation model... This may take a moment.")
    try:
        translator_instance = StructuredTranslator(
            model_name="prajdabre/rotary-indictrans2-indic-en-1B",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
            use_correction=True
        )
        translator_instance.load_model()
        print("Model loaded successfully. Ready to translate.")
    except Exception as e:
        print(f"Failed to load translation model: {e}")
        # Do not raise here; allow the server to start but endpoints will fail gracefully if model is missing.

@app.post("/translate", response_model=Dict[str, Any])
async def translate_endpoint(payload: Dict[str, Any]):
    """
    Accepts a structured JSON payload representing the OCR output,
    translates all translatable fields (sentences, table cells) into English,
    and returns the identical structure with updated values and metadata.
    """
    if translator_instance is None:
        raise HTTPException(status_code=503, detail="Translation model is not currently loaded or available.")

    try:
        # Create temporary files to adapt the JSON payload to the existing YAML-based StructuredTranslator
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8") as temp_in:
            yaml.dump(payload, temp_in, allow_unicode=True, sort_keys=False, default_flow_style=False)
            temp_in_path = temp_in.name
            
        temp_out_path = temp_in_path.replace(".yaml", "_out.yaml")
        
        # Execute the translation passing the temp file paths
        translator_instance.translate_yaml(temp_in_path, temp_out_path)
        
        # Read the translated output back
        with open(temp_out_path, "r", encoding="utf-8") as f:
            translated_payload = yaml.safe_load(f)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

    return translated_payload

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the target Translation API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    args = parser.parse_args()
    
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)
