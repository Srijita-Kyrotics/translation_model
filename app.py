from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForSeq2SeqLM
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading logic
model_dir = "./indictrans2-finetuned-en-bn/stage1"
logger.info(f"Loading IndicTrans2 processor and model from {model_dir}")

try:
    from indicTrans.inference.engine import Model
    logger.info("Importing Custom Model Wrapper...")
    
    # Check if GPU is available to fallback to CPU if not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We load it using the Model engine structure defined in the github repo.
    # It initializes the processor, tokenizer, and transformers logic automatically.
    translator = Model(model_dir, model_type="fairseq") 
    # Oh wait! We fine-tuned it natively using Huggingface RotaryIndicTransForConditionalGeneration
    # test_finetuned.py implies we used `IndicProcessor` instead! Let me fix this down below.
except Exception as e:
    logger.error(f"Failed to load via Engine: {e}")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

logger.info("Loading via raw Transformers as verified in earlier tests...")

processor = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_dir, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
device = model.device
model.eval()

import re

def chunk_text(text, max_words=80):
    # Split text into segments based on periods or newlines to ensure clean chunks
    # Replace multiple newlines with a single space or delimiter to avoid tokenizer shock
    text = re.sub(r'\n+', '\n', text.strip())
    # Split on punctuation (.!?) or newlines
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        # Sanitize whitespace for the model
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if not sentence: continue
        
        words_in_sentence = len(sentence.split())
        
        # If a single sentence is massive, we can't break it further safely here,
        # but we definitely don't add more to it.
        if current_word_count + words_in_sentence > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence
        else:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    english_text = data.get('text', '')
    
    if not english_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        chunks = chunk_text(english_text, max_words=80) 
        final_translations = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Preprocess input directly
            batch = processor.preprocess_batch([chunk], src_lang="eng_Latn", tgt_lang="ben_Beng")
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate with kwargs verified earlier
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512, 
                    num_beams=5,
                    repetition_penalty=1.15, # Penalizes greedy stuck loops
                    no_repeat_ngram_size=4,  # Hard blocks repetitive hallucinated n-grams
                )
                
            # Decode Output
            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            chunk_trans = processor.postprocess_batch(decoded_output, lang="ben_Beng")[0]
            final_translations.append(chunk_trans)
            
        final_text = " ".join(final_translations)
        
        return jsonify({'translation': final_text})
        
    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
