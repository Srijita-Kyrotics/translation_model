from src.translator import Translator
import torch

def verify_flash_attention():
    print("Initializing Translator...")
    translator = Translator()
    translator.load_model()
    
    # Check model configuration
    print(f"\nModel Configuration:")
    print(f"Attention Implementation: {translator.model.config._attn_implementation}")
    print(f"Torch Dtype: {translator.model.dtype}")
    
    # Explicit check
    if translator.model.config._attn_implementation != "flash_attention_2":
        print("WARNING: Flash Attention 2 is NOT active.")
    else:
        print("SUCCESS: Flash Attention 2 is active.")

    # Test run
    print("\nRunning test translation...")
    sentences = ["This is a test of the Flash Attention system."]
    translation = translator.translate_batch(sentences)
    print(f"Output: {translation}")

if __name__ == "__main__":
    verify_flash_attention()
