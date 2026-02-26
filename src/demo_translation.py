import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from src.translator import Translator, IndicProcessor
import os

def run_demo():
    base_model_name = "prajdabre/rotary-indictrans2-indic-en-1B"
    adapter_path = "./indictrans2-finetuned-court"
    
    print("Loading model and adapter...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Initialize Translator manually with the loaded model
    translator = Translator(
        model_name=base_model_name,
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        use_correction=False
    )
    translator.model = model
    translator.tokenizer = tokenizer
    translator.ip = IndicProcessor(inference=True)

    sample_bn = """এফ.এম.এ. - ২০১৯-এর ৭৩৯, ২০/১২/২০২২ তারিখে মীমাংসিত
ভারতের সংবিধান, অনুচ্ছেদ ২২৬ — চুক্তিবদ্ধ পরিষেবা - নবায়ন না করা। পিটিশনকারী চুক্তিবদ্ধ ভিত্তিতে শিক্ষা সহায়িকা হিসাবে নিযুক্ত ছিলেন - তার পরিষেবার চুক্তি সময়ের অতিবাহিত হওয়ার ফলে শেষ হয়েছে এবং অকাল সমাপ্তির মাধ্যমে নয় - চুক্তির মেয়াদ শেষ হওয়ার পরে, পরিচালনা কমিটি তার পরিষেবার চুক্তি নবায়ন না করার সিদ্ধান্ত নিয়েছে কারণ পরিচালনা কমিটি তার পরিষেবাকে সন্তোষজনক বলে মনে করেনি - পরিচালনা কমিটি এটি করার জন্য সম্পূর্ণভাবে তার অধিকারের মধ্যে ছিল - পরিচালনা কমিটি পরিষেবার চুক্তি নবায়ন না করার সিদ্ধান্ত নেওয়ার আগে শুনানির কোনো সুযোগ দেওয়ার প্রয়োজন ছিল না, পরিষেবার চুক্তির শর্তাবলী বা অন্য কোনোভাবে - পরিচালনা কমিটি পিটিশনকারীর কর্মদক্ষতা মূল্যায়ন করার জন্য ক্ষমতাবান এবং সক্ষম ছিল এবং সন্তুষ্ট না হলে তার পরিষেবার চুক্তি নবায়ন করতে অস্বীকার করতে পারত - পিটিশনকারী ক্ষতিগ্রস্থ হয়েছে বা অন্য কর্মচারীকে অন্যায় সুবিধা দেওয়া হয়েছে তা দেখানোর জন্য কিছুই দাখিল করা হয়নি - এটিও প্রদর্শিত হয়নি যে পরিচালনা কমিটির সিদ্ধান্ত স্বৈরাচারী বা অসৎ উদ্দেশ্যমূলক ছিল - পরিষেবার চুক্তি নবায়ন না করাকে অবৈধ বলা যাবে না।
ডব্লিউ.পি. ২০০৮-এর ৬২৯৪(ডব্লিউ), তারিখ ৯.০৮.২০১৮ (ক্যাল)-নিশ্চিত (অনুচ্ছেদ ১৮, ১৯, ২০, ২১)
রেফার করা মামলা:
কালানুক্রমিক অনুচ্ছেদ
ডব্লিউ.পি. ২০০৮-এর ৬২৯৪(ডব্লিউ), তারিখ ০৯-০৮-২০১৮ অনুচ্ছেদ নং (১, ৮)
এআইআর ১৯৮০ এসসি ৪২ অনুচ্ছেদ নং (১২)
এআইআর ১৯৭৮ এসসি ৫৯৭ অনুচ্ছেদ নং (১২)
আইনজীবীদের নাম
পিটিশনকারীর পক্ষে সরদার আমজাদ আলী, লে. সিনিয়র এডভোকেট সমিরুল সরদার, মাসুম আলী সরদার, মোঃ আবদুল আলিম, মিস সুচরিতা রায়; বিবাদীর পক্ষে রেজাউল হোসেন।"""

    print("Translating...")
    # Translate single string
    result = translator.translate_batch([sample_bn])
    
    print("\n" + "="*50)
    print("SOURCE (BENGALI):")
    print(sample_bn)
    print("="*50)
    print("TRANSLATION (ENGLISH):")
    print(result[0])
    print("="*50)

if __name__ == "__main__":
    run_demo()
