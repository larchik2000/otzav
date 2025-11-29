import re
import nltk
nltk.download('punkt')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^а-яА-Я0-9ё ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
