from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

def predict_sentiment(text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=200)
    with torch.no_grad():
        outputs = model(**inputs)
    return int(outputs.logits.argmax())
