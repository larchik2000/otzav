import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np

MODEL_NAME = "DeepPavlov/rubert-base-cased"

print("Loading data...")
df = pd.read_csv("data/train.csv")

# Extract data
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.enc["input_ids"])


train_ds = MyDataset(train_texts, train_labels)
val_ds = MyDataset(val_texts, val_labels)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)


# Metric: macro F1
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(pred.label_ids, preds, average="macro")
    return {"macro_f1": f1}


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

    fp16=False,                         # CPU → выключаем FP16
    ddp_find_unused_parameters=False,   # выключаем DDP-проверки
    optim="adamw_torch"                 # отключаем accelerate
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()

print("Saving best model...")
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")

print("DONE!")
