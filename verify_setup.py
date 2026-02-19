from src.translator import Translator
import os

def test_translation():
    print("Initializing Translator...")
    translator = Translator()
    translator.load_model()
    
    sentences = [
        "আমি বাংলায় কথা বলি।",
        "আজকের দিনটি খুব সুন্দর।",
        "Machine learning is fascinating."
    ]
    
    print("\nOriginal Sentences:")
    for s in sentences:
        print(f"- {s}")
        
    print("\nTranslating...")
    translations = translator.translate_batch(sentences)
    
    print("\nTranslations:")
    for t in translations:
        print(f"- {t}")

if __name__ == "__main__":
    test_translation()
