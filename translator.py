from typing import Iterable
from transformers import MarianMTModel, MarianTokenizer

tr2en_model_name_finetuned = "ckartal/turkish-to-english-finetuned-model"
tr_tokenizer_finetuned = MarianTokenizer.from_pretrained(tr2en_model_name_finetuned)
tr2en_model_finetuned = MarianMTModel.from_pretrained(tr2en_model_name_finetuned)

en2tr_model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
en_tokenizer = MarianTokenizer.from_pretrained(en2tr_model_name)
en2tr_model = MarianMTModel.from_pretrained(en2tr_model_name)

def translate_to_en_finetuned(tr_text):
    tr_text = [tr_text]
    translated = tr2en_model_finetuned.generate(**tr_tokenizer_finetuned(tr_text, return_tensors="pt", padding=True))
    return [ tr_tokenizer_finetuned.decode(t, skip_special_tokens=True) for t in translated ][0]

def translate_to_tr(en_text):
    en_text = [en_text]
    translated = en2tr_model.generate(**en_tokenizer(en_text, return_tensors="pt", padding=True))
    return [ en_tokenizer.decode(t, skip_special_tokens=True) for t in translated ][0]

def translate_from_tr(query):
    query = translate_to_en_finetuned(query)
    print("Translated query:" + query)
    return query

if __name__ == "__main__":
    text = "kullanıcı faturaları veritabanında kaç giriş var?"
    print(translate_to_en_finetuned(text))

    text = "In what year did Ataturk die?"
    print(translate_to_tr(text))
