from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    return {
        "label_id": label_id,
        "label": labels_map[label_id],
        "confidence": float(probs[0][label_id])
    }