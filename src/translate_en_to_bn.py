from src.translator import Translator
import yaml

def translate_en_to_bn():
    # User's English sample
    sample_en = """F.M.A. - 739 of 2019, decided on 20/12/2022
Constitution of India, Art.226— Contractual service - Non-renewal of Petitioner was engaged as Sikha Shahayika on contractual basis - Her service contract came to end by efflux of time and not by premature termination - Upon expiry of contract period, management resolved not to renew her service contract since management did not find her service to be satisfactory - Management was perfectly within its right to do so - No opportunity of hearing was required to be given before management resolved not to renew service contract, either under terms of contract of service or otherwise - Managing Committee was empowered and competent to assess performance of petitioner and if not satisfied, could refuse to renew her service contract - Nothing was produced to show that petitioner was victimised or other employee was shown undue favour - It was also not demonstrated that decision of Managing Committee was arbitrary or mala fide - Non-renewal of service contract could not be termed as illegal.
W.P. 6294(W) of 2008, D/- 9.08.2018 (Cal)-Affirmed (Para 18, 19, 20, 21)
Case Referred :
Chronological Paras
W.P. 6294(W) of 2008, D/-09-08-2018 Para No.( 1, 8 )
AIR 1980 SC 42 Para No.( 12 )
AIR 1978 SC 597 Para No.( 12 )
Name of Advocates
Sardar Amjad Ali, Ld. Sr. Adv. Samirul Sardar, Masum Ali Sardar, Md. Abdul Alim, Ms. Sucharita Ray for Petitioner; Rezaul Hossain for Respondent."""

    # Initialize Translator for EN -> BN
    translator = Translator(src_lang="eng_Latn", tgt_lang="ben_Beng", use_correction=False)
    translator.load_model()

    # Split into lines for better translation
    sentences = [s.strip() for s in sample_en.split('\n') if s.strip()]
    
    print("\nTranslating...")
    translated_sentences = translator.translate_batch(sentences)
    
    bn_text = "\n".join(translated_sentences)
    
    # Create Bengali YAML
    bn_yaml = {
        "document_id": "2019~fma_739_b_page_1",
        "primary_language": "bn",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": bn_text
    }
    
    output_path = "data/sample_ocr_bn.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(bn_yaml, f, allow_unicode=True, sort_keys=False)
    
    print(f"\nBengali translation saved to {output_path}")
    print("\n--- BENGALI TRANSLATION ---")
    print(bn_text)

if __name__ == "__main__":
    translate_en_to_bn()
