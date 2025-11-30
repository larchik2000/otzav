from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

MODEL_PATH = "./best_model"

# Загружаем модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict_sentiment(texts):
    """Предсказание sentiment для списка текстов"""
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.cpu().numpy().tolist()


def predict_batch(texts):
    """Баточное предсказание (для API)"""
    results = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_preds = predict_sentiment(batch_texts)

        for text, pred in zip(batch_texts, batch_preds):
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            results.append({
                "text": text,
                "sentiment": sentiment_map[pred],
                "label": int(pred)
            })

    return results


def calculate_metrics(true_labels, pred_labels):
    """Расчет метрик качества"""
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    f1_per_class = f1_score(true_labels, pred_labels, average=None)

    # Матрица ошибок
    cm = confusion_matrix(true_labels, pred_labels)

    # Classification report
    report = classification_report(true_labels, pred_labels, output_dict=True)

    return {
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'f1_per_class': {
            'negative': round(f1_per_class[0], 4),
            'neutral': round(f1_per_class[1], 4),
            'positive': round(f1_per_class[2], 4)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }