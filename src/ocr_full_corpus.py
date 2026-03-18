import os
import glob
import torch
from vllm import LLM, SamplingParams
from PIL import Image
import fitz
import io
import sys
from tqdm import tqdm
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from transformers import AutoProcessor

# Force Offline Mode for HF to avoid network hangs
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def get_pdf_page_image(pdf_path, page_num):
    """Render a PDF page to a PIL Image at OLM-recommended resolution."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # olmOCR recommends target longest dimension to be 1288
        max_dim = max(img.width, img.height)
        if max_dim > 1288:
            scale = 1288 / max_dim
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
        else:
            new_w = img.width
            new_h = img.height

        # Qwen2.5-VL requires dimensions to be multiples of 28
        new_w = max(28, (new_w // 28) * 28)
        new_h = max(28, (new_h // 28) * 28)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        doc.close()
        return img
    except Exception as e:
        print(f"Error rendering {pdf_path} page {page_num}: {e}")
        return None

def process_corpus():
    # Directories
    EN_PDF_DIR = "data/raw/judgments/english/"
    BN_PDF_DIR = "data/raw/judgments/bengali/"
    EN_TXT_DIR = "data/raw/judgments_en/"
    BN_TXT_DIR = "data/raw/judgments_bn/"

    for d in [EN_TXT_DIR, BN_TXT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    print("Loading vLLM model (allenai/olmOCR-2-7B-1025-FP8)...", flush=True)
    llm = LLM(
        model="allenai/olmOCR-2-7B-1025-FP8",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        # OLM-specific mm config
        limit_mm_per_prompt={"image": 1}
    )

    print("Loading processor for chat template...", flush=True)
    # Using local cache via TRANSFORMERS_OFFLINE=1
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    prompt_text = build_no_anchoring_v4_yaml_prompt()

    # Build the full chat-template prompt string
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": "image"}},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)

    # Gather all PDFs
    en_pdfs = glob.glob(os.path.join(EN_PDF_DIR, "*.pdf"))
    bn_pdfs = glob.glob(os.path.join(BN_PDF_DIR, "*.pdf"))
    
    all_tasks = []
    for pdf in en_pdfs:
        out_name = os.path.basename(pdf).replace(".pdf", ".txt")
        all_tasks.append((pdf, os.path.join(EN_TXT_DIR, out_name)))
    for pdf in bn_pdfs:
        out_name = os.path.basename(pdf).replace(".pdf", ".txt")
        all_tasks.append((pdf, os.path.join(BN_TXT_DIR, out_name)))

    print(f"Total PDFs to process: {len(all_tasks)}", flush=True)

    # Process the entire corpus sequentially to avoid OOM
    for pdf_path, txt_path in tqdm(all_tasks, desc="OCR Progress"):
        # Resume logic
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 10:
            continue

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            doc_text_parts = []
            inputs = []
            
            # Prepare all pages for the document
            for page_idx in range(num_pages):
                img = get_pdf_page_image(pdf_path, page_idx)
                if img is not None:
                    inputs.append({
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": img}
                    })
            
            # Batch generate all pages for this PDF
            if inputs:
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                for output in outputs:
                    doc_text_parts.append(output.outputs[0].text)

            # Combine all pages
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(doc_text_parts)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # print(f"  Saved to {os.path.basename(txt_path)}", flush=True)

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}", flush=True)
            if "EngineCore encountered an issue" in str(e):
                print("EngineCore crashed. Exiting to allow for clean restart.", flush=True)
                sys.exit(1)

if __name__ == "__main__":
    process_corpus()
