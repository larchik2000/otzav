from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "best_model"

# Загружаем токенайзер и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3  # классы: 0, 1, 2
)

model.eval()

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Маппинг предсказаний
labels_map = {0: "0", 1: "1", 2: "2"}


def predict_batch(texts, batch_size=16):
    """
    Предсказание батчами для ускорения.
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

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        preds = torch.argmax(logits, dim=1).cpu().tolist()

        results.extend([labels_map[p] for p in preds])

    return results