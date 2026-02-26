import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from src.translator import Translator, IndicProcessor
import os

def run_comparison():
    base_model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
    adapter_path = "./indictrans2-finetuned-court"
    
    sample_bn = """এফ.এম.এ. - ২০১৯-এর ৭৩৯, ২০/১২/২০২২ তারিখে মীমাংসিত
ভারতের সংবিধান, অনুচ্ছেদ ২২৬ — চুক্তিবদ্ধ পরিষেবা - নবায়ন না করা। পিটিশনকারী চুক্তিবদ্ধ ভিত্তিতে শিক্ষা সহায়িকা হিসাবে নিযুক্ত ছিলেন - তার পরিষেবার চুক্তি সময়ের অতিবাহিত হওয়ার ফলে শেষ হয়েছে এবং অকাল সমাপ্তির মাধ্যমে নয় - চুক্তির মেয়াদ শেষ হওয়ার পরে, পরিচালনা কমিটি তার পরিষেবার চুক্তি নবায়ন না করার সিদ্ধান্ত নিয়েছে কারণ পরিচালনা কমিটি তার পরিষেবাকে সন্তোষজনক বলে মনে করেনি - পরিচালনা কমিটি এটি করার জন্য সম্পূর্ণভাবে তার অধিকারের মধ্যে ছিল - পরিচালনা কমিটি পরিষেবার চুক্তি নবায়ন না করার সিদ্ধান্ত নেওয়ার আগে শুনানির কোনো সুযোগ দেওয়ার প্রয়োজন ছিল না, পরিষেবার চুক্তির শর্তাবলী বা অন্য কোনোভাবে - পরিচালনা কমিটি পিটিশনকারীর কর্মদক্ষতা মূল্যায়ন করার জন্য ক্ষমতাবান এবং সক্ষম ছিল এবং সন্তুষ্ট না হলে তার পরিষেবার চুক্তি নবায়ন করতে অস্বীকার করতে পারত - পিটিশনকারী ক্ষতিগ্রস্থ হয়েছে বা অন্য কর্মচারীকে অন্যায় সুবিধা দেওয়া হয়েছে তা দেখানোর জন্য কিছুই দাখিল করা হয়নি - এটিও প্রদর্শিত হয়নি যে পরিচালনা কমিটির সিদ্ধান্ত স্বৈরাচারী বা অসৎ উদ্দেশ্যমূলক ছিল - পরিষেবার চুক্তি নবায়ন না করাকে অবৈধ বলা যাবে না।"""

    print("--- Loading Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    translator_base = Translator(base_model_name, "ben_Beng", "eng_Latn", False)
    translator_base.model = base_model
    translator_base.tokenizer = tokenizer
    translator_base.ip = IndicProcessor(inference=True)
    
    print("Translating with Base Model...")
    base_out = translator_base.translate_batch([sample_bn])[0]

    print("\n--- Loading Fine-Tuned Adapter ---")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    
    translator_ft = Translator(base_model_name, "ben_Beng", "eng_Latn", False)
    translator_ft.model = ft_model
    translator_ft.tokenizer = tokenizer
    translator_ft.ip = IndicProcessor(inference=True)
    
    print("Translating with Fine-Tuned Model...")
    ft_out = translator_ft.translate_batch([sample_bn])[0]

    print("\n" + "="*80)
    print("BENGALI SOURCE:")
    print(sample_bn)
    print("-" * 80)
    print("BASE MODEL OUTPUT:")
    print(base_out)
    print("-" * 80)
    print("FINE-TUNED MODEL OUTPUT:")
    print(ft_out)
    print("="*80)

if __name__ == "__main__":
    run_comparison()
