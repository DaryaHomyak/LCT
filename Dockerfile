# Официальный Python образ
FROM python:3.11-slim

# Метаданные образа
LABEL maintainer="LCT CUphoria Team"
LABEL description="Архивный OCR Сервис для обработки дореволюционных документов"
LABEL version="1.0.0"

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app

# Рабочая директория
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y \
    # OpenCV и GUI-библиотеки (X stack)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    # OpenGL/GLX
    libgl1 \
    # PDF utils
    poppler-utils \
    # Утилиты
    curl \
    wget \
    git \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Создание необходимых директорий
RUN mkdir -p uploads results static logs temp

# Копирование исходного кода
COPY web_service.py .
COPY integrated_archive_processor.py .
COPY .env.example .env

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app $APP_HOME
USER app

# Порт приложения
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Команда запуска
CMD ["python", "web_service.py"]