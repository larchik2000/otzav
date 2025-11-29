from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

# Загружаем модель ОДИН раз при старте сервера
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Определим устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels_map = {0: "0", 1: "1", 2: "2"}


def predict_batch(texts, batch_size=32):
    """
    Быстрое предсказание тональности батчами.
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # переносим на GPU/CPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        preds = torch.argmax(logits, dim=1).cpu().numpy()

        results.extend([labels_map[p] for p in preds])

    return results
