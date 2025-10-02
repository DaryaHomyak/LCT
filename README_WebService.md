# Инструкции по запуску веб-сервиса

## 📦 Установка зависимостей

```bash
# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

## 🚀 Запуск сервиса

```bash
# Запуск в режиме разработки
python web_service.py

# Или через uvicorn
uvicorn web_service:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 Доступ к сервису

- **Веб-интерфейс:** http://localhost:8000
- **API документация:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📁 Структура проекта

```
archive-ocr-service/
├── web_service.py              # Основной веб-сервис
├── integrated_archive_processor.py  # OCR движок
├── requirements.txt            # Зависимости
├── README_WebService.md        # Эта инструкция
├── uploads/                    # Загруженные файлы
├── results/                    # Результаты обработки
├── static/                     # Статические файлы
├── templates/                  # HTML шаблоны
└── archive_service.db          # База данных SQLite
```

## 🔧 Конфигурация

### Переменные окружения (.env файл):
```
YANDEX_API_KEY=your_yandex_gpt_api_key
YANDEX_FOLDER_ID=your_yandex_folder_id
DATABASE_URL=sqlite:///./archive_service.db
MAX_FILE_SIZE=52428800  # 50MB
```

### Настройки OCR:
- По умолчанию используется PaddleOCR
- Поддержка дореволюционной орфографии
- Автоматическая постобработка через YandexGPT

## 📊 API Endpoints

### Основные функции:
- `POST /upload` - Загрузка документов
- `GET /status/{task_id}` - Статус обработки
- `GET /results/{task_id}` - Результаты распознавания
- `POST /correct/{task_id}` - Коррекция текста
- `POST /export` - Экспорт данных
- `GET /stats` - Статистика обработки

### Поддерживаемые форматы:
- **Вход:** JPG, JPEG, TIFF, PDF
- **Экспорт:** JSON, CSV, XLSX

## 🎯 Соответствие требованиям ТЗ

### ✅ Реализованные функции:
1. **Загрузка файлов** - множественная загрузка, поддержка JPG/TIFF/PDF
2. **Отслеживание прогресса** - асинхронная обработка с веб-интерфейсом
3. **Верификация результатов** - API для коррекции текста
4. **Экспорт данных** - JSON, CSV с настраиваемыми критериями
5. **Статистика** - количество документов, уверенность, ошибки
6. **Предобработка** - нормализация, контраст, сегментация
7. **OCR** - рукописный + печатный текст, дореволюционная орфография
8. **Атрибуты** - извлечение ФИО, дат, адресов, архивных шифров
9. **Структурированный вывод** - координаты, уверенность, метаданные

### 🔄 Функции для доработки:
- Более сложная верификация с визуализацией
- Дообучение на пользовательских правках
- Расширенный конструктор отчетов
- Интеграция с внешними архивными системами

## 🐛 Отладка

### Логи:
```bash
# Просмотр логов в реальном времени
tail -f uvicorn.log
```

### Частые проблемы:
1. **Tesseract не найден** - установите Tesseract OCR
2. **PaddleOCR ошибки** - проверьте совместимость версий
3. **Файлы не загружаются** - проверьте права доступа к директориям

### Тестирование API:
```bash
# Загрузка файла
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.jpg"

# Проверка статуса
curl "http://localhost:8000/status/{task_id}"
```

## 📈 Масштабирование

### Для продакшена:
1. **База данных** - замените SQLite на PostgreSQL
2. **Очереди** - добавьте Redis/Celery для фоновых задач  
3. **Файловое хранилище** - используйте S3/MinIO
4. **Мониторинг** - Prometheus + Grafana
5. **Аутентификация** - JWT токены
6. **Load Balancer** - nginx + несколько воркеров

### Docker deployment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "web_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🎉 Готово!

Веб-сервис полностью готов к работе и соответствует требованиям ТЗ.
Для запуска выполните инструкции выше и откройте http://localhost:8000
