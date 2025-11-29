from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from app.model import predict_sentiment

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

TEXT_COLUMNS = ["text", "comment", "review", "message", "content"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-file")
async def predict_file(file: UploadFile):
    file.file.seek(0)  # важно сбросить поток

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"result": [], "error": f"Ошибка чтения CSV: {e}"}

    # Определяем текстовую колонку
    text_col = None
    for col in df.columns:
        if col.lower() in TEXT_COLUMNS:
            text_col = col
            break

    if text_col is None:
        return {
            "result": [],
            "error": "Не найдена текстовая колонка. Ожидается одна из: text, comment, review, message, content"
        }

    # Применяем модель
    try:
        df["label"] = df[text_col].apply(lambda x: predict_sentiment(str(x))["label"])
    except Exception as e:
        return {"result": [], "error": f"Ошибка обработки модели: {e}"}

    return {"result": df.to_dict(orient="records"), "text_col": text_col}