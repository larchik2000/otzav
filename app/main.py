from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.model import predict_sentiment
from app.preprocess import clean_text
from sklearn.metrics import f1_score
from io import StringIO
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="Sentiment Classifier API")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    
    df["clean_text"] = df["text"].apply(clean_text)
    df["pred"] = df["clean_text"].apply(predict_sentiment)

    return df.to_dict(orient="records")

@app.post("/evaluate")
async def evaluate(
    file_with_preds: UploadFile = File(...),
    validation_file: UploadFile = File(...)
):
    preds_content = await file_with_preds.read()
    val_content = await validation_file.read()

    preds_df = pd.read_csv(StringIO(preds_content.decode("utf-8")))
    val_df = pd.read_csv(StringIO(val_content.decode("utf-8")))

    score = f1_score(val_df["label"], preds_df["pred"], average="macro")

    return {"macro_f1": score}