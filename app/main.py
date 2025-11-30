from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import asyncio
import io
from datetime import datetime

from app.model import predict_batch, calculate_metrics

app = FastAPI(title="Sentiment Analysis Dashboard")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

TEXT_COLUMNS = ["text", "comment", "review", "message", "content"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/predict-stream")
async def predict_stream(file: UploadFile):
    file.file.seek(0)

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type="text/event-stream"
        )

    # Ищем текстовую колонку
    text_col = None
    for col in df.columns:
        if col.lower() in TEXT_COLUMNS:
            text_col = col
            break

    if text_col is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'Не найдена текстовая колонка'})}\n\n"]),
            media_type="text/event-stream"
        )

    # Ищем колонку с истинными метками (если есть)
    true_labels_col = None
    for col in df.columns:
        if col.lower() in ['label', 'sentiment', 'true_label']:
            true_labels_col = col
            break

    # Ищем ID колонку
    id_col = None
    for col in df.columns:
        if col.lower() == "id":
            id_col = col
            break

    if id_col is None:
        ids = list(range(len(df)))
    else:
        ids = df[id_col].tolist()

    texts = df[text_col].astype(str).tolist()

    # Истинные метки (если есть)
    true_labels = None
    if true_labels_col and true_labels_col in df.columns:
        true_labels = df[true_labels_col].astype(int).tolist()

    async def streamer():
        total = len(texts)
        batch_size = 256
        results = []
        pred_labels = []

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            preds = predict_batch(batch_texts)

            for _id, pred in zip(batch_ids, preds):
                results.append({
                    "id": _id,
                    "text": pred["text"],
                    "sentiment": pred["sentiment"],
                    "label": pred["label"]
                })
                pred_labels.append(pred["label"])

            progress = int((i + len(batch_texts)) / total * 100)

            yield f"data: {json.dumps({'progress': progress, 'current': i + len(batch_texts), 'total': total})}\n\n"
            await asyncio.sleep(0)

        # Расчет метрик если есть истинные метки
        metrics = None
        if true_labels and len(true_labels) == len(pred_labels):
            metrics = calculate_metrics(true_labels, pred_labels)

        yield f"data: {json.dumps({'done': True, 'result': results, 'metrics': metrics})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/evaluate")
async def evaluate_model(file: UploadFile):
    """Endpoint для оценки модели с истинными метками"""
    try:
        df = pd.read_csv(file.file)

        # Ищем текстовую колонку и колонку с метками
        text_col = None
        label_col = None

        for col in df.columns:
            if col.lower() in TEXT_COLUMNS:
                text_col = col
            if col.lower() in ['label', 'sentiment', 'true_label']:
                label_col = col

        if not text_col or not label_col:
            return JSONResponse(
                {"error": "Не найдены текстовая колонка или колонка с метками"},
                status_code=400
            )

        texts = df[text_col].astype(str).tolist()
        true_labels = df[label_col].astype(int).tolist()

        # Предсказания
        predictions = predict_batch(texts)
        pred_labels = [p["label"] for p in predictions]

        # Расчет метрик
        metrics = calculate_metrics(true_labels, pred_labels)

        # Статистика по классам
        class_stats = {
            'true_distribution': pd.Series(true_labels).value_counts().to_dict(),
            'pred_distribution': pd.Series(pred_labels).value_counts().to_dict()
        }

        return JSONResponse({
            "metrics": metrics,
            "statistics": class_stats,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/metrics")
async def get_model_info():
    """Информация о модели"""
    return JSONResponse({
        "model_name": "RuBERT Sentiment Analysis",
        "classes": {
            0: "negative",
            1: "neutral",
            2: "positive"
        },
        "supported_languages": ["russian"],
        "max_length": 128
    })