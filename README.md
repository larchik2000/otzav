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
 
### Структура
```
otzav/
├── app/                          # Основное приложение FastAPI
│   ├── main.py                   # Главный файл FastAPI приложения
│   ├── model.py                  # Модель для анализа тональности
│   ├── templates/                # HTML шаблоны
│   │   ├── index.html            # Главная страница с загрузкой файлов
│   │   └── dashboard.html        # Дашборд с метриками
│   └── static/                   # Статические файлы (CSS, JS)
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── dashboard.js
├── best_model/                   # Папка с моделью (В .gitignore)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   ├── training_args.bin
│   └── runs/                     # Логи тренировки
├── training.py                   # Скрипт обучения модели (В .gitignore)
├── predict.py                    # Скрипт предсказаний (В .gitignore)
├── requirements.txt              # Зависимости Python
├── README.md                     # Документация проекта
└── .gitignore                    # Игнорируемые 
```
файлы
### Команда FIX IT
```
Комарницкий Андрей: ML, backend, frontend

Бурейко Александр: ML

Ларичева Екатерина: backend, designer
 ```