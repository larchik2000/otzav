import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_PATH = "best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

df = pd.read_csv("data/test.csv")
texts = df["text"].astype(str).tolist()
ids = df["ID"].tolist()

results = []

batch_size = 64

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]

    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    preds = torch.argmax(logits, dim=1).cpu().numpy()

    for _id, p in zip(ids[i:i+batch_size], preds):
        results.append({"ID": _id, "label": int(p)})

pd.DataFrame(results).to_csv("submission.csv", index=False)

print("Saved submission.csv")
