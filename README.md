# 🏛️ Архивный OCR Сервис

**Веб-сервис для обработки архивных документов с ИИ-распознаванием**

Система автоматического извлечения и индексирования информации из образов архивных документов с поддержкой дореволюционной орфографии.

## 📋 Описание

Веб-сервис обеспечивает:
- 📸 **Предварительная обработка изображений** (выравнивание, контраст, шумы)
- 🤖 **OCR распознавание** рукописного и печатного текста
- 📜 **Поддержка дореволюционной орфографии** (ѣ, ѳ, і, ѵ, ъ)
- 🎯 **Атрибутивное распознавание** (ФИО, даты, адреса, архивные шифры)
- ✏️ **Верификация и коррекция** результатов пользователем
- 📊 **Экспорт данных** в различных форматах
- 📈 **Статистика и мониторинг** качества обработки

### Поддерживаемые форматы
- **Входные**: JPG, JPEG, TIFF, PDF
- **Выходные**: JSON, CSV, XML
- **Максимальный размер файла**: 100MB

## 🚀 Быстрый запуск

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd archive-ocr-service
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения
Создайте файл `.env`:
```env
# Яндекс API (опционально для расширенных функций)
YANDEX_API_KEY=your_yandex_api_key_here
YANDEX_FOLDER_ID=your_folder_id_here

# Настройки базы данных
DATABASE_URL=sqlite:///./archive_service.db

# Настройки файлов
MAX_FILE_SIZE=104857600
UPLOAD_DIR=uploads
```

### 4. Запуск сервиса
```bash
python web_service.py
```

### 5. Открытие в браузере
```
http://localhost:8000
```

## 📁 Структура проекта

```
archive-ocr-service/
├── web_service.py                    # Основной веб-сервис FastAPI
├── integrated_archive_processor.py   # OCR процессор
├── requirements.txt                  # Python зависимости
├── .env                             # Переменные окружения
├── Dockerfile                       # Docker образ
├── docker-compose.yml              # Docker Compose
├── README.md                        # Документация
├── docs/                           # Документация
│   ├── deployment-guide.pdf        # Руководство по развертыванию
│   └── api-docs.md                 # API документация
├── uploads/                        # Загруженные файлы
├── static/                         # Статические файлы
└── tests/                          # Тесты
```

## ⚙️ Конфигурация

### Основные настройки
```python
# В web_service.py
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
LOW_CONFIDENCE_THRESHOLD = 0.75     # Порог низкой уверенности
DATABASE_URL = "sqlite:///./archive_service.db"
```

### OCR настройки
```python
# В integrated_archive_processor.py
use_angle_cls=True                  # Классификатор углов поворота
det_db_thresh=0.2                   # Порог детекции
det_db_box_thresh=0.3              # Порог боксов
max_side_len=4096                  # Максимальное разрешение
```

## 🐳 Docker развертывание

### Сборка образа
```bash
docker build -t archive-ocr-service .
```

### Запуск контейнера
```bash
docker run -d \
  --name archive-ocr \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/.env:/app/.env \
  archive-ocr-service
```

### Используя Docker Compose
```bash
docker-compose up -d
```

## 📊 API Endpoints

### Основные операции
- `GET /` - Веб-интерфейс
- `POST /upload` - Загрузка документа
- `GET /status/{task_id}` - Статус обработки
- `GET /results/{task_id}` - Результаты OCR
- `GET /stats` - Статистика

### Верификация и коррекция
- `POST /correct/{task_id}` - Коррекция текста
- `POST /verify/{task_id}` - Верификация сегмента

### Экспорт данных
- `GET /export/{task_id}` - Экспорт документа
- `POST /export` - Расширенный экспорт

### Служебные
- `GET /health` - Проверка здоровья сервиса
- `GET /docs` - Swagger документация

## 🔧 Требования к системе

### Минимальные требования
- **ОС**: Linux Ubuntu 20.04+, Windows 10+, macOS 10.15+
- **Python**: 3.8+
- **RAM**: 4GB (рекомендуется 8GB)
- **Диск**: 10GB свободного места
- **Интернет**: для скачивания OCR моделей при первом запуске

### Рекомендуемые требования
- **CPU**: 4+ ядер
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU с поддержкой CUDA (опционально)
- **SSD**: для быстрого доступа к файлам

## 🧪 Тестирование

### Запуск тестов
```bash
python -m pytest tests/
```

### Проверка работоспособности
```bash
curl http://localhost:8000/health
```

### Пример использования API
```bash
# Загрузка файла
curl -X POST "http://localhost:8000/upload" \
     -F "file=@document.jpg"

# Получение статуса
curl "http://localhost:8000/status/{task_id}"

# Получение результатов
curl "http://localhost:8000/results/{task_id}"
```

## 🔐 Безопасность

- ✅ Валидация типов файлов
- ✅ Ограничение размера файлов (100MB)
- ✅ Санитизация входных данных
- ✅ CORS настройки
- ✅ Логирование операций

## 📈 Мониторинг и логирование

### Логи
```bash
tail -f web_service.log
```

### Метрики
- Количество обработанных документов
- Средняя уверенность распознавания
- Время обработки
- Статистика ошибок

## 🔄 Обновление

```bash
git pull
pip install -r requirements.txt --upgrade
python web_service.py
```


## 👥 Авторы

- **Разработчик**: Команда CUphoria
- **Организатор**: Лидеры Цифровой Трансформации
- **Год**: 2025

## 🔗 Ссылки

- [Презентация проекта](https://docs.google.com/presentation/d/1dJifst9VjbArZirZcbZkCZyHtp2pWOD1eORliVm9i34/edit?usp=sharing)