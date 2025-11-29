from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import asyncio

from app.model import predict_batch

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

TEXT_COLUMNS = ["text", "comment", "review", "message", "content"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-stream")
async def predict_stream(file: UploadFile):
    file.file.seek(0)

    # читаем CSV
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type="text/event-stream"
        )

    # ищем текстовую колонку
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

    # 🔥 Находим колонку ID в любом варианте регистра
    id_col = None
    for col in df.columns:
        if col.lower() == "id":
            id_col = col
            break

    # 🔥 Если id нет — создаём автоматически
    if id_col is None:
        ids = list(range(len(df)))
    else:
        ids = df[id_col].tolist()

    texts = df[text_col].astype(str).tolist()

    async def streamer():
        total = len(texts)
        batch_size = 256
        results = []

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            preds = predict_batch(batch_texts)

            for _id, txt, label in zip(batch_ids, batch_texts, preds):
                results.append({
                    "id": _id,
                    "text": txt,
                     "label": label})


            progress = int((i + len(batch_texts)) / total * 100)

            yield f"data: {json.dumps({'progress': progress, 'current': i + len(batch_texts), 'total': total})}\n\n"
            await asyncio.sleep(0)

        yield f"data: {json.dumps({'done': True, 'result': results})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")
