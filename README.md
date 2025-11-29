## 🚀 Запуск проекта

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Windows

``` bat
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

После запуска открой в браузере:

👉 http://127.0.0.1:8000
